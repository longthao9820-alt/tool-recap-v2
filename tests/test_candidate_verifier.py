"""Tests for candidate zero-output and low-coverage verifier (Objective rev 8).

Covers:
1. Verification models, enums, serialization, and package exports.
2. Deterministic classification for all ZeroOutputReason codes (PARSER_FAILURE,
   MALFORMED_AI_RESPONSE, CANDIDATE_DISCOVERY_FAILED, FINALIZER_FAILED,
   COVERAGE_INCOMPLETE, INSUFFICIENT_EVIDENCE, CANDIDATES_REJECTED_BY_VALIDATION,
   LOW_COVERAGE_SUSPECT, NO_ELIGIBLE_CANDIDATES).
3. Parser failure can NEVER map to NO_ELIGIBLE_CANDIDATES.
4. Genuine zero validation rules (is_genuine_zero_valid) across pipeline health states.
5. Relative coverage breadth ratio check for suspicious low coverage (not fixed quota).
6. Single strong season arc exemption (valid, does not trigger suspicious check).
7. Healthy non-suspicious candidate sets do not trigger AI verification.
8. Schema validation for AI verification response:
   - Non-empty recovered_candidates OR explicit confirm_no_eligible boolean.
   - Empty candidates list without confirm_no_eligible is malformed.
9. Malformed AI response rejected, returns MALFORMED_AI_RESPONSE, and is NOT cached.
10. Domain cache verification data validation: invalid data rejected, no cache write.
11. Grounding check: recovered candidate uses existing refs; hallucinated refs stripped;
    candidate with zero matching existing refs rejected.
12. Audit verifier does NOT synthesize fallback: generic fallback proposals rejected;
    verifier never invents synthetic candidates when zero or empty.
13. Duration clamping, unknown episodes stripping, scope auto-correction.
14. Cache key determinism, secret isolation, and prompt/policy/model invalidation.
15. Cache resume skips AI call; empty confirmation cached and reloaded.
16. Cancellation token: raises before call, and saves valid result before raising after save.
17. Bounded adaptive payload compaction (FULL -> TRIMMED -> PRIORITY -> SKELETON).
18. Phase callbacks emit AnalysisPhase.ZERO_OUTPUT_VERIFICATION.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from toolrecap_v2.analyzer import (
    CandidateVerifier,
    verify_zero_or_low_output,
)
from toolrecap_v2.analyzer.candidates import (
    HARD_PAYLOAD_CEILING,
    TARGET_PAYLOAD_CEILING,
    VERIFIER_ALGO_VERSION,
    VERIFIER_PROMPT_VERSION,
    classify_deterministic_reason,
    compute_health_hash,
    estimate_verification_request_size,
    format_verification_user_text,
    is_genuine_zero_valid,
    is_suspicious_low_coverage,
    should_trigger_verification,
    validate_recovered_candidates,
    validate_verification_response_schema,
)
from toolrecap_v2.analyzer.errors import AnalysisCancelledError, AnalysisError
from toolrecap_v2.analyzer.phases import AnalysisPhase
from toolrecap_v2.analyzer.prompts import ZERO_OUTPUT_VERIFICATION_SYSTEM_PROMPT
from toolrecap_v2.domain import (
    VERIFICATION_SCHEMA_VERSION,
    CandidateDirective,
    CandidateProposal,
    CandidateScope,
    CandidateSourceRange,
    CandidateStatus,
    CompactionLevel,
    ConsolidationAction,
    ConsolidationDecision,
    EditorialPolicy,
    HierarchyCacheManager,
    PipelineHealth,
    VerificationResult,
    ZeroOutputReason,
    compute_verification_cache_key,
    validate_verification_cache_data,
)
from toolrecap_v2.settings import AppSettings


class MockVerifierAIClient:
    """Mock AI client for candidate verification tests."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.call_history: list[dict[str, Any]] = []

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user_text: str,
        thinking: str = "auto",
        images: Any = (),
        max_tokens: int = 32000,
        cancel_event: threading.Event | None = None,
        phase: Any = None,
    ) -> dict[str, Any]:
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("API call cancelled.")

        self.call_history.append({
            "model": model,
            "system": system,
            "user_text": user_text,
            "thinking": thinking,
            "phase": phase,
        })

        if self.responses:
            resp = self.responses.pop(0)
            if callable(resp):
                resp = resp()
            if isinstance(resp, Exception):
                raise resp
            return resp

        return {"confirm_no_eligible": True, "rationale": "Default mock empty confirmation."}


def _make_candidate(
    cid: str,
    title: str = "Sample Arc",
    episodes: list[str] | None = None,
    thesis: str = "Candidate thesis statement.",
    ranges: list[CandidateSourceRange] | None = None,
    evidence: list[str] | None = None,
    scope: str = CandidateScope.SINGLE_EPISODE.value,
) -> CandidateProposal:
    eps = episodes or ["E01"]
    rngs = ranges or [
        CandidateSourceRange(
            episode_id=eps[0],
            start_seconds=10.0,
            end_seconds=30.0,
            evidence_ref=f"ref_{eps[0]}_01",
        )
    ]
    ev = evidence if evidence is not None else [r.evidence_ref for r in rngs if r.evidence_ref]
    return CandidateProposal(
        proposal_id=cid,
        title=title,
        episodes=eps,
        central_thesis=thesis,
        source_ranges=rngs,
        supporting_evidence=ev,
        candidate_scope=scope,
        status=CandidateStatus.KEEP.value,
    )


# ===========================================================================
# 1. Models, Enums, Serialization, and Exports
# ===========================================================================

def test_zero_output_reason_all_nine_codes() -> None:
    expected_codes = {
        "NO_ELIGIBLE_CANDIDATES",
        "INSUFFICIENT_EVIDENCE",
        "CANDIDATE_DISCOVERY_FAILED",
        "CANDIDATES_REJECTED_BY_VALIDATION",
        "FINALIZER_FAILED",
        "MALFORMED_AI_RESPONSE",
        "PARSER_FAILURE",
        "COVERAGE_INCOMPLETE",
        "LOW_COVERAGE_SUSPECT",
    }
    actual_codes = {r.value for r in ZeroOutputReason}
    assert expected_codes == actual_codes
    for code in expected_codes:
        assert ZeroOutputReason(code).value == code


def test_verification_result_serialization_roundtrip() -> None:
    cand = _make_candidate("c1", "Arc One", episodes=["E01", "E02"], scope=CandidateScope.CROSS_EPISODE.value)
    res = VerificationResult(
        reason=ZeroOutputReason.LOW_COVERAGE_SUSPECT,
        is_valid_zero=False,
        recovered_candidates=[cand],
        diagnostics={"trigger": "low_ratio", "ratio": 0.05},
        completed=True,
        rationale="Recovered 1 cross-episode arc.",
    )

    d = res.to_dict()
    assert d["reason"] == "LOW_COVERAGE_SUSPECT"
    assert d["is_valid_zero"] is False
    assert len(d["recovered_candidates"]) == 1
    assert d["completed"] is True
    assert d["rationale"] == "Recovered 1 cross-episode arc."

    rebuilt = VerificationResult.from_dict(d)
    assert rebuilt.reason == ZeroOutputReason.LOW_COVERAGE_SUSPECT
    assert rebuilt.is_valid_zero is False
    assert len(rebuilt.recovered_candidates) == 1
    assert rebuilt.recovered_candidates[0].proposal_id == "c1"
    assert rebuilt.diagnostics["ratio"] == 0.05
    assert rebuilt.completed is True


def test_pipeline_health_serialization_roundtrip() -> None:
    cand = _make_candidate("c1", "Title 1")
    dec = ConsolidationDecision(
        candidate_id="c1",
        action=ConsolidationAction.REJECT.value,
        reason_code="weak_evidence",
        reason="Not enough evidence",
    )
    health = PipelineHealth(
        coverage_ledgers=[{"episode_id": "E01", "coverage_status": "COMPLETE"}],
        discovery_completed=True,
        discovered_count=1,
        consolidation_completed=True,
        consolidation_decisions=[dec],
        consolidated_candidates=[cand],
        total_evidence_count=15,
    )

    h_dict = health.to_dict()
    assert h_dict["discovered_count"] == 1
    assert len(h_dict["consolidation_decisions"]) == 1

    rebuilt = PipelineHealth.from_dict(h_dict)
    assert rebuilt.discovered_count == 1
    assert len(rebuilt.consolidation_decisions) == 1
    assert rebuilt.consolidation_decisions[0].candidate_id == "c1"


def test_package_exports() -> None:
    from toolrecap_v2.analyzer.candidates import (
        CandidateVerifier as CV1,
        classify_deterministic_reason as CDR,
        verify_zero_or_low_output as VZ1,
    )
    from toolrecap_v2.analyzer import (
        CandidateVerifier as CV2,
        verify_zero_or_low_output as VZ2,
    )
    from toolrecap_v2.domain import (
        PipelineHealth as PH,
        VerificationResult as VR,
        ZeroOutputReason as ZOR,
        compute_verification_cache_key as CVCK,
        validate_verification_cache_data as VVCD,
    )
    assert CV1 is CV2
    assert VZ1 is VZ2
    assert CDR is not None
    assert PH is not None
    assert VR is not None
    assert ZOR is not None
    assert CVCK is not None
    assert VVCD is not None


# ===========================================================================
# 2. Deterministic Classification for All Reason Codes
# ===========================================================================

def test_deterministic_reason_parser_failure() -> None:
    health = PipelineHealth(parser_failed=True, parser_error="SyntaxError: invalid token")
    reason, diag = classify_deterministic_reason(health)
    assert reason == ZeroOutputReason.PARSER_FAILURE
    assert diag["parser_failed"] is True

    # Check verifier integration: parser failure NEVER maps to NO_ELIGIBLE
    verifier = CandidateVerifier()
    res = verifier.verify("p1", health)
    assert res.reason == ZeroOutputReason.PARSER_FAILURE
    assert res.is_valid_zero is False
    assert res.completed is False


def test_deterministic_reason_schema_rejected() -> None:
    # 1. Direct schema rejected flag
    health1 = PipelineHealth(schema_rejected=True, schema_error="Invalid JSON structure")
    reason1, _ = classify_deterministic_reason(health1)
    assert reason1 == ZeroOutputReason.MALFORMED_AI_RESPONSE

    # 2. Discovery schema invalid
    health2 = PipelineHealth(discovery_schema_valid=False)
    reason2, _ = classify_deterministic_reason(health2)
    assert reason2 == ZeroOutputReason.MALFORMED_AI_RESPONSE

    # 3. Consolidation schema invalid
    health3 = PipelineHealth(consolidation_schema_valid=False)
    reason3, _ = classify_deterministic_reason(health3)
    assert reason3 == ZeroOutputReason.MALFORMED_AI_RESPONSE


def test_deterministic_reason_discovery_failed() -> None:
    # Discovery incomplete
    health1 = PipelineHealth(discovery_completed=False)
    reason1, _ = classify_deterministic_reason(health1)
    assert reason1 == ZeroOutputReason.CANDIDATE_DISCOVERY_FAILED

    # Discovery error
    health2 = PipelineHealth(discovery_completed=True, discovery_error="RateLimitError")
    reason2, _ = classify_deterministic_reason(health2)
    assert reason2 == ZeroOutputReason.CANDIDATE_DISCOVERY_FAILED


def test_deterministic_reason_finalizer_failed() -> None:
    health = PipelineHealth(finalizer_attempted=True, finalizer_error="AI connection timeout")
    reason, _ = classify_deterministic_reason(health)
    assert reason == ZeroOutputReason.FINALIZER_FAILED


def test_deterministic_reason_coverage_incomplete() -> None:
    # Ledger with partial status
    health1 = PipelineHealth(
        coverage_ledgers=[{"episode_id": "E01", "coverage_status": "PARTIAL"}],
        total_evidence_count=10,
    )
    reason1, _ = classify_deterministic_reason(health1)
    assert reason1 == ZeroOutputReason.COVERAGE_INCOMPLETE

    # Ledger with actionable gaps
    health2 = PipelineHealth(
        coverage_ledgers=[{"episode_id": "E01", "coverage_status": "INCOMPLETE", "actionable_gaps": True}],
        total_evidence_count=10,
    )
    reason2, _ = classify_deterministic_reason(health2)
    assert reason2 == ZeroOutputReason.COVERAGE_INCOMPLETE


def test_deterministic_reason_insufficient_evidence() -> None:
    health = PipelineHealth(total_evidence_count=0, coverage_ledgers=[])
    reason, _ = classify_deterministic_reason(health)
    assert reason == ZeroOutputReason.INSUFFICIENT_EVIDENCE


def test_deterministic_reason_candidates_rejected_by_validation() -> None:
    dec1 = ConsolidationDecision(candidate_id="c1", action=ConsolidationAction.REJECT.value, reason_code="weak_evidence")
    dec2 = ConsolidationDecision(candidate_id="c2", action=ConsolidationAction.REJECT.value, reason_code="weak_evidence")
    health = PipelineHealth(
        discovered_count=2,
        consolidation_decisions=[dec1, dec2],
        consolidated_candidates=[],
        total_evidence_count=20,
        coverage_ledgers=[{"episode_id": "E01", "coverage_status": "COMPLETE"}],
    )
    reason, diag = classify_deterministic_reason(health)
    assert reason == ZeroOutputReason.CANDIDATES_REJECTED_BY_VALIDATION
    assert diag["rejected_count"] == 2


# ===========================================================================
# 3. Genuine Zero Validation (Test J)
# ===========================================================================

def test_genuine_zero_valid_only_when_fully_healthy() -> None:
    # Fully healthy with 0 candidates
    health_valid = PipelineHealth(
        discovery_completed=True,
        discovery_schema_valid=True,
        consolidation_completed=True,
        consolidation_schema_valid=True,
        consolidated_candidates=[],
        coverage_ledgers=[{"episode_id": "E01", "coverage_status": "COMPLETE", "actionable_gaps": False}],
        total_evidence_count=10,
    )
    assert is_genuine_zero_valid(health_valid) is True

    # Candidates > 0 -> False
    h_cands = copy.deepcopy(health_valid)
    h_cands.consolidated_candidates = [_make_candidate("c1")]
    assert is_genuine_zero_valid(h_cands) is False

    # Parser failed -> False
    h_parser = copy.deepcopy(health_valid)
    h_parser.parser_failed = True
    assert is_genuine_zero_valid(h_parser) is False

    # Schema rejected -> False
    h_schema = copy.deepcopy(health_valid)
    h_schema.schema_rejected = True
    assert is_genuine_zero_valid(h_schema) is False

    # Discovery error -> False
    h_disc = copy.deepcopy(health_valid)
    h_disc.discovery_error = "Error"
    assert is_genuine_zero_valid(h_disc) is False

    # Consolidation error -> False
    h_cons = copy.deepcopy(health_valid)
    h_cons.consolidation_error = "Error"
    assert is_genuine_zero_valid(h_cons) is False

    # Incomplete coverage ledger -> False
    h_cov = copy.deepcopy(health_valid)
    h_cov.coverage_ledgers = [{"episode_id": "E01", "coverage_status": "PARTIAL"}]
    assert is_genuine_zero_valid(h_cov) is False

    # Zero evidence count and no ledgers -> False
    h_no_ev = copy.deepcopy(health_valid)
    h_no_ev.total_evidence_count = 0
    h_no_ev.coverage_ledgers = []
    assert is_genuine_zero_valid(h_no_ev) is False


# ===========================================================================
# 4. Suspicious Low Coverage vs Strong Season Arc (Test K)
# ===========================================================================

def test_suspicious_low_coverage_relative_ratio_trigger() -> None:
    # 10 episodes and 100 evidence items, but candidate only covers 1 episode and 1 evidence ref
    c1 = _make_candidate("c1", episodes=["E01"], evidence=["ref_E01_01"])
    is_susp, diag = is_suspicious_low_coverage(
        [c1],
        total_episodes=10,
        total_evidence_items=100,
        min_episode_ratio=0.40,
        min_evidence_ratio=0.10,
    )
    assert is_susp is True
    assert diag["episode_coverage_ratio"] == 0.10
    assert "Episode ratio" in diag["suspicion_trigger"]


def test_strong_season_arc_exemption_no_trigger() -> None:
    # Single candidate, but it is a strong SEASON_ARC spanning multiple episodes
    c_arc = _make_candidate(
        "arc_season",
        "Epic Season Arc",
        episodes=["E01", "E02", "E03", "E04", "E05"],
        scope=CandidateScope.SEASON_ARC.value,
        evidence=["ref_01", "ref_02", "ref_03", "ref_04", "ref_05"],
    )
    is_susp, diag = is_suspicious_low_coverage(
        [c_arc],
        total_episodes=5,
        total_evidence_items=50,
    )
    assert is_susp is False
    assert diag["reason"] == "strong_season_arc"
    assert diag["season_arc_candidate"] == "arc_season"


def test_healthy_candidates_not_suspicious_no_verification_trigger() -> None:
    c1 = _make_candidate("c1", episodes=["E01", "E02"], evidence=["ref_1", "ref_2"])
    c2 = _make_candidate("c2", episodes=["E03", "E04"], evidence=["ref_3", "ref_4"])
    health = PipelineHealth(
        discovered_count=2,
        consolidated_candidates=[c1, c2],
        total_evidence_count=10,
        coverage_ledgers=[
            {"episode_id": f"E0{i}", "coverage_status": "COMPLETE"}
            for i in range(1, 5)
        ],
    )
    should_trig, reason, _ = should_trigger_verification(health, [c1, c2], total_episodes=4, total_evidence_items=10)
    assert should_trig is False
    assert reason is None


# ===========================================================================
# 5. Schema Validation & Malformed Domain Response No Cache (Test L)
# ===========================================================================

def test_schema_validation_valid_cases() -> None:
    # Case 1: Non-empty recovered_candidates
    resp1 = {
        "recovered_candidates": [
            {"proposal_id": "r1", "title": "Recovered Arc", "episodes": ["E01"]}
        ]
    }
    assert validate_verification_response_schema(resp1, raise_error=True) is True

    # Case 2: Explicit confirm_no_eligible boolean
    resp2 = {"confirm_no_eligible": True, "rationale": "Genuinely no narrative candidate found."}
    assert validate_verification_response_schema(resp2, raise_error=True) is True

    # Case 3: Both present (e.g. confirm_no_eligible: False with candidates)
    resp3 = {
        "confirm_no_eligible": False,
        "recovered_candidates": [{"proposal_id": "r1", "title": "Arc", "episodes": ["E01"]}],
    }
    assert validate_verification_response_schema(resp3, raise_error=True) is True


def test_schema_validation_empty_confirmation_schema_explicit() -> None:
    # Empty candidates list without confirm_no_eligible MUST fail validation
    resp_empty = {"recovered_candidates": []}
    assert validate_verification_response_schema(resp_empty, raise_error=False) is False
    with pytest.raises(AnalysisError, match="must contain non-empty 'recovered_candidates' or explicit 'confirm_no_eligible' boolean"):
        validate_verification_response_schema(resp_empty, raise_error=True)

    # Empty dict without confirm_no_eligible MUST fail
    assert validate_verification_response_schema({}, raise_error=False) is False

    # String confirm_no_eligible MUST fail
    assert validate_verification_response_schema({"confirm_no_eligible": "true"}, raise_error=False) is False

    # Non-dict MUST fail
    assert validate_verification_response_schema("not a dict", raise_error=False) is False


def test_malformed_ai_response_not_cached(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    # Mock AI returning malformed schema (empty candidates without confirmation)
    mock_client = MockVerifierAIClient(responses=[
        {"recovered_candidates": []}  # Malformed: no confirm_no_eligible!
    ])
    health = PipelineHealth(
        discovered_count=0,
        consolidated_candidates=[],
        total_evidence_count=10,
        coverage_ledgers=[{"episode_id": "E01", "coverage_status": "COMPLETE"}],
    )

    verifier = CandidateVerifier(client=mock_client, cache=cache)
    res = verifier.verify("scope_malformed", health)

    assert res.reason == ZeroOutputReason.MALFORMED_AI_RESPONSE
    assert res.is_valid_zero is False
    assert res.completed is False
    assert "malformed" in res.rationale.lower()

    # Verify that cache file was NEVER written
    cached_files = list(cache.verification_dir.glob("*.verification.json"))
    assert len(cached_files) == 0


def test_domain_validate_verification_cache_data() -> None:
    # Valid: non-empty recovered candidates
    assert validate_verification_cache_data({"recovered_candidates": [{"title": "t"}]}) is True
    # Valid: explicit confirm_no_eligible boolean
    assert validate_verification_cache_data({"confirm_no_eligible": True}) is True
    assert validate_verification_cache_data({"confirm_no_eligible": False, "recovered_candidates": []}) is True

    # Invalid: empty candidates list without confirm_no_eligible
    assert validate_verification_cache_data({"recovered_candidates": []}) is False
    assert validate_verification_cache_data({"candidates": []}) is False
    assert validate_verification_cache_data({}) is False
    assert validate_verification_cache_data({"confirm_no_eligible": "yes"}) is False
    assert validate_verification_cache_data("string") is False


def test_save_verification_result_rejects_malformed(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="Invalid verification cache data"):
        cache.save_verification_result(
            scope_id="scope1",
            cache_key="key123456789012345678901234",
            data={"recovered_candidates": []},  # Empty without confirm_no_eligible
        )


# ===========================================================================
# 6. Candidate Grounding & Audit Verifier Does NOT Synthesize Fallback
# ===========================================================================

def test_recovered_candidate_grounding_strips_hallucinated_refs() -> None:
    allowed_refs = {"ref_E01_01", "ref_E01_02"}
    raw_candidate = {
        "proposal_id": "rec_1",
        "title": "Grounded Recovery",
        "episodes": ["E01"],
        "central_thesis": "Strong narrative grounding.",
        "source_ranges": [
            {"episode_id": "E01", "start_seconds": 10.0, "end_seconds": 20.0, "evidence_ref": "ref_E01_01"},
            {"episode_id": "E01", "start_seconds": 30.0, "end_seconds": 40.0, "evidence_ref": "hallucinated_ref_99"},
        ],
        "supporting_evidence": ["ref_E01_01", "hallucinated_ref_99"],
    }

    validated = validate_recovered_candidates(
        [raw_candidate],
        allowed_episode_ids={"E01"},
        existing_evidence_refs=allowed_refs,
        episode_durations={"E01": 100.0},
    )

    assert len(validated) == 1
    cand = validated[0]
    assert "hallucinated_ref_99" not in cand.supporting_evidence
    assert cand.supporting_evidence == ["ref_E01_01"]
    # Source range with hallucinated ref must have evidence_ref stripped to None
    assert cand.source_ranges[0].evidence_ref == "ref_E01_01"
    assert cand.source_ranges[1].evidence_ref is None


def test_recovered_candidate_with_zero_matching_refs_rejected() -> None:
    allowed_refs = {"ref_valid_1", "ref_valid_2"}
    raw_candidate = {
        "proposal_id": "rec_fake",
        "title": "Ungrounded Candidate",
        "episodes": ["E01"],
        "central_thesis": "Completely made up refs.",
        "source_ranges": [
            {"episode_id": "E01", "start_seconds": 10.0, "end_seconds": 20.0, "evidence_ref": "invented_ref_A"}
        ],
        "supporting_evidence": ["invented_ref_A"],
    }

    validated = validate_recovered_candidates(
        [raw_candidate],
        allowed_episode_ids={"E01"},
        existing_evidence_refs=allowed_refs,
    )
    # Must be completely rejected
    assert len(validated) == 0


def test_verifier_rejects_generic_fallback_candidates() -> None:
    generic_samples = [
        {"proposal_id": "g1", "title": "Generic Fallback Candidate", "central_thesis": "A standard recap"},
        {"proposal_id": "g2", "title": "Main Recap", "central_thesis": "Default recap fallback."},
        {"proposal_id": "g3", "title": "Placeholder Candidate", "central_thesis": "Temporary summary."},
    ]

    validated = validate_recovered_candidates(generic_samples)
    assert len(validated) == 0


def test_verifier_never_synthesizes_fallback_when_empty_or_confirmed(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    # AI confirms genuine zero
    mock_client = MockVerifierAIClient(responses=[
        {"confirm_no_eligible": True, "rationale": "No viable narrative candidate."}
    ])
    health = PipelineHealth(
        discovered_count=0,
        consolidated_candidates=[],
        total_evidence_count=10,
        coverage_ledgers=[{"episode_id": "E01", "coverage_status": "COMPLETE"}],
    )

    verifier = CandidateVerifier(client=mock_client, cache=cache)
    res = verifier.verify("scope_zero", health)

    assert res.reason == ZeroOutputReason.NO_ELIGIBLE_CANDIDATES
    assert res.is_valid_zero is True
    # Zero candidates synthesized
    assert len(res.recovered_candidates) == 0
    assert res.recovered_candidates == []


def test_duration_clamping_unknown_episodes_and_scope_autocorrect() -> None:
    raw = {
        "proposal_id": "c_auto",
        "title": "Auto Scope Arc",
        "episodes": ["E01", "E99"],  # E99 is unknown
        "central_thesis": "Thesis",
        "source_ranges": [
            {"episode_id": "E01", "start_seconds": 50.0, "end_seconds": 200.0, "evidence_ref": "ref_1"},  # Clamped to 120
            {"episode_id": "E99", "start_seconds": 10.0, "end_seconds": 20.0, "evidence_ref": "ref_99"},   # Dropped
        ],
        "supporting_evidence": ["ref_1"],
        "candidate_scope": CandidateScope.CROSS_EPISODE.value,  # Only E01 remains -> auto-correct to SINGLE_EPISODE
    }

    validated = validate_recovered_candidates(
        [raw],
        allowed_episode_ids={"E01"},
        episode_durations={"E01": 120.0},
        existing_evidence_refs={"ref_1"},
    )

    assert len(validated) == 1
    cand = validated[0]
    assert cand.episodes == ["E01"]
    assert len(cand.source_ranges) == 1
    assert cand.source_ranges[0].end_seconds == 120.0
    assert cand.candidate_scope == CandidateScope.SINGLE_EPISODE.value


# ===========================================================================
# 7. Cache Key Determinism, Isolation, and Invalidation
# ===========================================================================

def test_cache_key_determinism_and_isolation() -> None:
    health = PipelineHealth(discovered_count=2, total_evidence_count=20)
    health_hash = compute_health_hash(health)
    dir_hash = hashlib.sha256(b"directive_v1").hexdigest()

    key1 = compute_verification_cache_key(
        scope_id="season_1",
        health_hash=health_hash,
        candidate_directive_hash=dir_hash,
        model="gpt-4o",
        thinking="auto",
    )
    key2 = compute_verification_cache_key(
        scope_id="season_1",
        health_hash=health_hash,
        candidate_directive_hash=dir_hash,
        model="gpt-4o",
        thinking="auto",
    )
    assert key1 == key2

    # Invalidation by directive change
    dir_hash2 = hashlib.sha256(b"directive_v2").hexdigest()
    key_dir = compute_verification_cache_key(
        scope_id="season_1",
        health_hash=health_hash,
        candidate_directive_hash=dir_hash2,
        model="gpt-4o",
    )
    assert key1 != key_dir

    # Invalidation by model change
    key_mod = compute_verification_cache_key(
        scope_id="season_1",
        health_hash=health_hash,
        candidate_directive_hash=dir_hash,
        model="gpt-5-preview",
    )
    assert key1 != key_mod

    # Invalidation by health change
    health2 = PipelineHealth(discovered_count=5, total_evidence_count=50)
    key_h = compute_verification_cache_key(
        scope_id="season_1",
        health_hash=compute_health_hash(health2),
        candidate_directive_hash=dir_hash,
        model="gpt-4o",
    )
    assert key1 != key_h

    # Ensure no secrets or API keys in key
    assert len(key1) == 64
    assert int(key1, 16) > 0


# ===========================================================================
# 8. Cache Resume & Cancellation
# ===========================================================================

def test_cache_resume_skips_ai_call(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    mock_client = MockVerifierAIClient(responses=[
        {"confirm_no_eligible": True, "rationale": "Zero candidate verified."}
    ])
    health = PipelineHealth(
        discovered_count=0,
        consolidated_candidates=[],
        total_evidence_count=10,
        coverage_ledgers=[{"episode_id": "E01", "coverage_status": "COMPLETE"}],
    )

    verifier = CandidateVerifier(client=mock_client, cache=cache)
    # First call: calls AI
    res1 = verifier.verify("scope_resume", health, model="test-model")
    assert res1.completed is True
    assert len(mock_client.call_history) == 1

    # Second call: loads from cache, AI not called
    res2 = verifier.verify("scope_resume", health, model="test-model")
    assert res2.completed is True
    assert len(mock_client.call_history) == 1
    assert res2.rationale == res1.rationale


def test_cancellation_before_ai_call_raises(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    mock_client = MockVerifierAIClient()
    health = PipelineHealth(discovered_count=0, consolidated_candidates=[])

    token = threading.Event()
    token.set()  # Already cancelled

    verifier = CandidateVerifier(client=mock_client, cache=cache)
    with pytest.raises(AnalysisCancelledError, match="cancelled"):
        verifier.verify("scope_cancel", health, cancellation_token=token)

    assert len(mock_client.call_history) == 0


def test_cancellation_after_save_persists_valid_result(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    token = threading.Event()

    def _ai_response() -> dict[str, Any]:
        # Cancel event set during AI execution
        token.set()
        return {"confirm_no_eligible": True, "rationale": "Confirmed zero before cancel."}

    mock_client = MockVerifierAIClient(responses=[_ai_response])
    health = PipelineHealth(
        discovered_count=0,
        consolidated_candidates=[],
        total_evidence_count=10,
        coverage_ledgers=[{"episode_id": "E01", "coverage_status": "COMPLETE"}],
    )

    verifier = CandidateVerifier(client=mock_client, cache=cache)
    # Must save valid result and then raise AnalysisCancelledError
    with pytest.raises(AnalysisCancelledError, match="cancelled after save"):
        verifier.verify("scope_save_cancel", health, model="test-model", cancellation_token=token)

    # Verify cache file exists
    cached_files = list(cache.verification_dir.glob("*.verification.json"))
    assert len(cached_files) == 1

    # Next call without cancellation resumes from cache without AI call!
    token.clear()
    res_resumed = verifier.verify("scope_save_cancel", health, model="test-model", cancellation_token=token)
    assert res_resumed.completed is True
    assert res_resumed.rationale == "Confirmed zero before cancel."
    # AI client was only called once total
    assert len(mock_client.call_history) == 1


# ===========================================================================
# 9. Bounded Payload & Compaction
# ===========================================================================

def test_format_verification_user_text_compaction_levels() -> None:
    health = PipelineHealth(
        discovered_count=10,
        consolidation_decisions=[
            ConsolidationDecision(candidate_id=f"c_{i}", action=ConsolidationAction.REJECT.value, reason_code="weak_evidence")
            for i in range(50)
        ],
        coverage_ledgers=[
            {"episode_id": f"E{i:02d}", "coverage_status": "COMPLETE"}
            for i in range(30)
        ],
        total_evidence_count=100,
    )

    text_full = format_verification_user_text("scope", health, compaction_level=CompactionLevel.FULL)
    text_skeleton = format_verification_user_text("scope", health, compaction_level=CompactionLevel.SKELETON)

    assert len(text_skeleton) < len(text_full)
    # Skeleton bounds rejected decisions to 2
    assert text_skeleton.count("- ID: c_") == 2
    assert text_full.count("- ID: c_") == 30


def test_estimate_verification_request_size_bounds() -> None:
    text = "Short verification user text."
    size = estimate_verification_request_size("gpt-4o", text, thinking="off", max_tokens=2000)
    assert size > 0
    assert size < HARD_PAYLOAD_CEILING


# ===========================================================================
# 10. Phase Callbacks and Functional verify_zero_or_low_output
# ===========================================================================

def test_phase_callbacks_emitted(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    mock_client = MockVerifierAIClient(responses=[
        {"confirm_no_eligible": True, "rationale": "Verified."}
    ])
    health = PipelineHealth(
        discovered_count=0,
        consolidated_candidates=[],
        total_evidence_count=10,
        coverage_ledgers=[{"episode_id": "E01", "coverage_status": "COMPLETE"}],
    )

    phases: list[tuple[AnalysisPhase, str, dict[str, Any]]] = []

    def callback(phase: AnalysisPhase, msg: str, user_data: dict[str, Any] | None = None) -> None:
        phases.append((phase, msg, user_data or {}))

    res = verify_zero_or_low_output(
        scope_id="scope_cb",
        health=health,
        client=mock_client,
        cache=cache,
        phase_callback=callback,
    )

    assert res.completed is True
    assert len(phases) >= 2
    assert all(p[0] == AnalysisPhase.ZERO_OUTPUT_VERIFICATION for p in phases)


def test_low_coverage_recovery_recovers_candidates(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    # Low coverage suspect triggers AI, AI recovers viable candidate
    recovered_dict = {
        "proposal_id": "rec_arc",
        "title": "Recovered Detective Investigation",
        "episodes": ["E01", "E02"],
        "central_thesis": "Investigation reveals suspect.",
        "source_ranges": [
            {"episode_id": "E01", "start_seconds": 10.0, "end_seconds": 30.0, "evidence_ref": "ref_ev_1"},
            {"episode_id": "E02", "start_seconds": 15.0, "end_seconds": 45.0, "evidence_ref": "ref_ev_2"},
        ],
        "supporting_evidence": ["ref_ev_1", "ref_ev_2"],
    }
    mock_client = MockVerifierAIClient(responses=[
        {"recovered_candidates": [recovered_dict], "confirm_no_eligible": False}
    ])

    c_weak = _make_candidate("c_weak", episodes=["E01"], evidence=["ref_ev_1"])
    health = PipelineHealth(
        discovered_count=1,
        consolidated_candidates=[c_weak],
        total_evidence_count=50,
        coverage_ledgers=[{"episode_id": f"E0{i}", "coverage_status": "COMPLETE"} for i in range(1, 6)],
    )

    verifier = CandidateVerifier(client=mock_client, cache=cache)
    res = verifier.verify(
        "scope_rec",
        health,
        allowed_episode_ids={"E01", "E02", "E03", "E04", "E05"},
        existing_evidence_refs={"ref_ev_1", "ref_ev_2", "ref_ev_3"},
    )

    assert res.reason == ZeroOutputReason.LOW_COVERAGE_SUSPECT
    assert res.completed is True
    assert len(res.recovered_candidates) == 1
    rec = res.recovered_candidates[0]
    assert rec.proposal_id == "rec_arc"
    assert rec.candidate_scope == CandidateScope.CROSS_EPISODE.value
