"""Tests for candidate-consolidation-r8 (Objective rev 8).

Covers:
1. ConsolidationDecision and ConsolidatedCandidateSet models, serialization, properties.
2. ConsolidationAction and ConsolidationReason enums.
3. Cache key determinism, content addressing, policy invalidation, no secrets.
4. Duplicates across E01/E05 merge cross episode into unified candidate.
5. Same character two theses remain separate (distinct_thesis).
6. Supporting character strong proposal remains despite many main character proposals.
7. Weak candidate rejected with explicit reason code (weak_evidence).
8. All 25 candidates retained if distinct (no arbitrary quota or cap).
9. Oversized candidate set recursively bounded, adaptive payload, no singleton loop.
10. Malformed AI response rejected and not cached.
11. Resume skips AI call; cancellation saves valid node before raising.
12. No silent IDs: all input candidate IDs accounted for; missing default KEEP conservatively.
13. Source ref integrity: AI cannot invent source refs or ranges, merged candidate only union.
14. Phase CANDIDATE_CONSOLIDATION callbacks emit counts (merged, rejected, eligible, cache).
15. Legacy connection seed candidates consumed seamlessly.
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

from toolrecap_v2.analyzer.candidates import (
    CONSOLIDATION_ALGO_VERSION,
    CONSOLIDATION_PROMPT_VERSION,
    CandidateConsolidator,
    compact_candidate_payload,
    compute_candidate_similarity,
    compute_candidates_payload_hash,
    consolidate_candidates,
    estimate_consolidation_request_size,
    format_consolidation_node_user_text,
    pre_group_candidates,
    reconcile_and_validate_decisions,
    validate_consolidation_response_schema,
)
from toolrecap_v2.analyzer.errors import AnalysisCancelledError, AnalysisError
from toolrecap_v2.analyzer.phases import AnalysisPhase
from toolrecap_v2.analyzer.prompts import CANDIDATE_CONSOLIDATION_SYSTEM_PROMPT
from toolrecap_v2.domain import (
    CONSOLIDATION_SCHEMA_VERSION,
    CandidateDirective,
    CandidateProposal,
    CandidateScope,
    CandidateSourceRange,
    CandidateStatus,
    CompactionLevel,
    ConsolidatedCandidateSet,
    ConsolidationAction,
    ConsolidationDecision,
    ConsolidationReason,
    EditorialPolicy,
    HierarchyCacheManager,
    compute_consolidation_cache_key,
    validate_consolidation_cache_data,
)
from toolrecap_v2.settings import AppSettings


class MockConsolidationAIClient:
    """Mock AI client for candidate consolidation tests."""

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
    ) -> dict[str, Any]:
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("API call cancelled.")

        self.call_history.append({
            "model": model,
            "system": system,
            "user_text": user_text,
            "thinking": thinking,
        })

        if self.responses:
            resp = self.responses.pop(0)
            if callable(resp):
                resp = resp()
            if isinstance(resp, Exception):
                raise resp
            return resp

        return {"consolidated_candidates": [], "decisions": []}


def _make_candidate(
    cid: str,
    title: str = "Sample Title",
    episodes: list[str] | None = None,
    character: str = "Alice",
    thesis: str = "Central narrative thesis.",
    ranges: list[CandidateSourceRange] | None = None,
    evidence: list[str] | None = None,
    tags: list[str] | None = None,
    confidence: float = 0.90,
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
    ev = evidence or [f"ref_{eps[0]}_01"]
    return CandidateProposal(
        proposal_id=cid,
        title=title,
        candidate_scope=scope,
        episodes=eps,
        characters=[character],
        primary_character=character,
        central_thesis=thesis,
        source_ranges=rngs,
        supporting_evidence=ev,
        overlap_tags=tags or ["drama"],
        confidence=confidence,
        why=f"Why {cid}",
        status=CandidateStatus.KEEP.value,
    )


# ===========================================================================
# 1. Models, Enums & Serialization Tests
# ===========================================================================

def test_consolidation_decision_roundtrip() -> None:
    dec = ConsolidationDecision(
        candidate_id="cand_01",
        action=ConsolidationAction.MERGE.value,
        reason_code=ConsolidationReason.CROSS_EPISODE_MERGE.value,
        reason="Merged cross episode E01 and E05",
        target_id="cand_root",
    )
    data = dec.to_dict()
    assert data["candidate_id"] == "cand_01"
    assert data["action"] == "MERGE"
    assert data["reason_code"] == "cross_episode_merge"
    assert data["target_id"] == "cand_root"

    restored = ConsolidationDecision.from_dict(data)
    assert restored.candidate_id == dec.candidate_id
    assert restored.action == dec.action
    assert restored.reason_code == dec.reason_code
    assert restored.target_id == dec.target_id


def test_consolidated_candidate_set_properties_and_json() -> None:
    c1 = _make_candidate("c1", "Title 1")
    c2 = _make_candidate("c2", "Title 2")
    d1 = ConsolidationDecision(candidate_id="c1", action=ConsolidationAction.KEEP.value)
    d2 = ConsolidationDecision(
        candidate_id="c2",
        action=ConsolidationAction.MERGE.value,
        reason_code=ConsolidationReason.CROSS_EPISODE_MERGE.value,
        target_id="c1",
    )
    d3 = ConsolidationDecision(
        candidate_id="c3",
        action=ConsolidationAction.REJECT.value,
        reason_code=ConsolidationReason.WEAK_EVIDENCE.value,
    )

    cset = ConsolidatedCandidateSet(
        candidates=[c1],
        decisions=[d1, d2, d3],
        schema_version=CONSOLIDATION_SCHEMA_VERSION,
        metadata={"pass": "final"},
    )
    assert cset.eligible_count == 1
    assert cset.kept_count == 1
    assert cset.merged_count == 1
    assert cset.rejected_count == 1
    assert cset.get_decision("c2") == d2
    assert cset.get_decision("unknown") is None

    json_str = cset.to_json()
    restored = ConsolidatedCandidateSet.from_json(json_str)
    assert restored.eligible_count == 1
    assert len(restored.decisions) == 3
    assert restored.schema_version == CONSOLIDATION_SCHEMA_VERSION


# ===========================================================================
# 2. Cache Key Determinism, Policy Invalidation & No Secrets
# ===========================================================================

def test_consolidation_cache_key_determinism_and_no_secrets() -> None:
    k1 = compute_consolidation_cache_key(
        node_id="node_01",
        candidate_directive_hash="dir_abc",
        input_payload_hash="input_123",
        model="gemini-2.5-flash",
        thinking="auto",
    )
    k2 = compute_consolidation_cache_key(
        node_id="node_01",
        candidate_directive_hash="dir_abc",
        input_payload_hash="input_123",
        model="gemini-2.5-flash",
        thinking="auto",
    )
    assert k1 == k2
    assert len(k1) == 64

    # Secret isolation
    raw_key = compute_consolidation_cache_key(
        node_id="node_01",
        candidate_directive_hash="dir_abc",
        input_payload_hash="input_123",
        model="gemini-2.5-flash",
    )
    assert "sk-" not in raw_key
    assert "token" not in raw_key


def test_consolidation_cache_policy_invalidation() -> None:
    k1 = compute_consolidation_cache_key(
        node_id="node_01",
        candidate_directive_hash="dir_hash_v1",
        input_payload_hash="input_123",
        model="gemini-2.5-flash",
    )
    k2 = compute_consolidation_cache_key(
        node_id="node_01",
        candidate_directive_hash="dir_hash_v2",  # policy changed
        input_payload_hash="input_123",
        model="gemini-2.5-flash",
    )
    assert k1 != k2


# ===========================================================================
# 3. Duplicates Across E01/E05 Merge Cross-Episode
# ===========================================================================

def test_duplicates_across_e01_e05_merge_cross_episode(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    c1 = _make_candidate(
        "cand_e01",
        title="Alice's Rebellion in Sector 4",
        episodes=["E01"],
        character="Alice",
        thesis="Alice rebels against the authoritarian regime.",
        ranges=[
            CandidateSourceRange(
                episode_id="E01",
                start_seconds=100.0,
                end_seconds=200.0,
                evidence_ref="ref_e01_reb",
            )
        ],
        evidence=["ref_e01_reb"],
    )
    c2 = _make_candidate(
        "cand_e05",
        title="Alice's Continued Rebellion",
        episodes=["E05"],
        character="Alice",
        thesis="Alice rebels against the authoritarian regime.",
        ranges=[
            CandidateSourceRange(
                episode_id="E05",
                start_seconds=300.0,
                end_seconds=400.0,
                evidence_ref="ref_e05_reb",
            )
        ],
        evidence=["ref_e05_reb"],
    )

    ai_resp = {
        "consolidated_candidates": [
            {
                "proposal_id": "cand_e01",
                "title": "Alice's Full Rebellion Arc",
                "candidate_scope": "CROSS_EPISODE",
                "episodes": ["E01", "E05"],
                "characters": ["Alice"],
                "primary_character": "Alice",
                "central_thesis": "Alice rebels against the authoritarian regime.",
                "source_ranges": [
                    {"episode_id": "E01", "start_seconds": 100.0, "end_seconds": 200.0, "evidence_ref": "ref_e01_reb"},
                    {"episode_id": "E05", "start_seconds": 300.0, "end_seconds": 400.0, "evidence_ref": "ref_e05_reb"},
                ],
                "supporting_evidence": ["ref_e01_reb", "ref_e05_reb"],
                "confidence": 0.95,
                "status": "keep",
            }
        ],
        "decisions": [
            {
                "candidate_id": "cand_e01",
                "action": "KEEP",
                "reason_code": "cross_episode_merge",
                "reason": "Surviving unified cross-episode arc.",
                "target_id": "",
            },
            {
                "candidate_id": "cand_e05",
                "action": "MERGE",
                "reason_code": "cross_episode_merge",
                "reason": "Merged into cand_e01 across E01 and E05.",
                "target_id": "cand_e01",
            },
        ],
    }

    client = MockConsolidationAIClient(responses=[ai_resp])
    consolidator = CandidateConsolidator(client=client, cache=cache)
    result = consolidator.consolidate([c1, c2])

    assert result.eligible_count == 1
    assert result.merged_count == 1
    survivor = result.candidates[0]
    assert survivor.proposal_id == "cand_e01"
    assert sorted(survivor.episodes) == ["E01", "E05"]
    assert survivor.candidate_scope == CandidateScope.CROSS_EPISODE.value
    assert len(survivor.source_ranges) == 2
    assert "ref_e01_reb" in survivor.supporting_evidence
    assert "ref_e05_reb" in survivor.supporting_evidence

    dec_e05 = result.get_decision("cand_e05")
    assert dec_e05 is not None
    assert dec_e05.action == "MERGE"
    assert dec_e05.reason_code == "cross_episode_merge"
    assert dec_e05.target_id == "cand_e01"


# ===========================================================================
# 4. Same Character Two Distinct Theses Remain Separate
# ===========================================================================

def test_same_character_two_theses_remain_separate(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    c1 = _make_candidate(
        "alice_moral",
        title="Alice's Moral Decline",
        character="Alice",
        thesis="Alice compromises her ethical principles for power.",
    )
    c2 = _make_candidate(
        "alice_rebel",
        title="Alice's Secret Whistleblowing",
        character="Alice",
        thesis="Alice leaks corporate secrets to protect innocent workers.",
    )

    ai_resp = {
        "consolidated_candidates": [c1.to_dict(), c2.to_dict()],
        "decisions": [
            {
                "candidate_id": "alice_moral",
                "action": "KEEP",
                "reason_code": "distinct_thesis",
                "reason": "Explores internal corruption.",
            },
            {
                "candidate_id": "alice_rebel",
                "action": "KEEP",
                "reason_code": "distinct_thesis",
                "reason": "Explores external whistleblowing.",
            },
        ],
    }

    client = MockConsolidationAIClient(responses=[ai_resp])
    consolidator = CandidateConsolidator(client=client, cache=cache)
    result = consolidator.consolidate([c1, c2])

    assert result.eligible_count == 2
    assert result.kept_count == 2
    assert result.merged_count == 0
    cids = {c.proposal_id for c in result.candidates}
    assert cids == {"alice_moral", "alice_rebel"}
    assert result.get_decision("alice_moral").reason_code == "distinct_thesis"
    assert result.get_decision("alice_rebel").reason_code == "distinct_thesis"


def test_validator_separates_distinct_theses_if_ai_attempts_merge() -> None:
    c1 = _make_candidate(
        "bob_alcohol",
        character="Bob",
        thesis="Bob struggles with alcoholism and family neglect.",
    )
    c2 = _make_candidate(
        "bob_patent",
        character="Bob",
        thesis="Bob battles corporate spies to register his quantum patent.",
    )

    # Erroneous AI decision attempting to merge completely different theses
    ai_decisions = [
        ConsolidationDecision(
            candidate_id="bob_patent",
            action=ConsolidationAction.MERGE.value,
            target_id="bob_alcohol",
        ),
        ConsolidationDecision(
            candidate_id="bob_alcohol",
            action=ConsolidationAction.KEEP.value,
        ),
    ]

    cset = reconcile_and_validate_decisions(
        input_candidates=[c1, c2],
        ai_consolidated=[c1],
        ai_decisions=ai_decisions,
    )

    # Validator must separate distinct theses
    assert cset.eligible_count == 2
    d_patent = cset.get_decision("bob_patent")
    assert d_patent.action == ConsolidationAction.KEEP.value
    assert d_patent.reason_code == ConsolidationReason.DISTINCT_THESIS.value


# ===========================================================================
# 5. Supporting Character Strong Remains Despite Many Main Character Proposals
# ===========================================================================

def test_supporting_character_strong_remains_despite_many_main(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    main_cands = [
        _make_candidate(f"main_{i}", f"Protagonist Arc {i}", character="Alice", thesis=f"Protagonist thesis {i}")
        for i in range(5)
    ]
    supp_cand = _make_candidate(
        "supp_01",
        title="Sidekick Charlie's Sacrifice",
        character="Charlie",
        thesis="Charlie risks everything to safeguard the evidence.",
        confidence=0.98,
        tags=["supporting_arc", "sacrifice"],
    )

    all_inputs = main_cands + [supp_cand]

    ai_resp = {
        "consolidated_candidates": [c.to_dict() for c in all_inputs],
        "decisions": [
            {
                "candidate_id": c.proposal_id,
                "action": "KEEP",
                "reason_code": "distinct_thesis",
                "reason": "Strong grounded arc.",
            }
            for c in all_inputs
        ],
    }

    client = MockConsolidationAIClient(responses=[ai_resp])
    consolidator = CandidateConsolidator(client=client, cache=cache)
    result = consolidator.consolidate(all_inputs)

    assert result.eligible_count == 6
    supp_in_result = any(c.proposal_id == "supp_01" for c in result.candidates)
    assert supp_in_result
    dec_supp = result.get_decision("supp_01")
    assert dec_supp.action == "KEEP"


# ===========================================================================
# 6. Weak Candidate Rejected with Explicit Reason
# ===========================================================================

def test_weak_candidate_rejected_with_explicit_reason(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    strong = _make_candidate("strong_cand", "Strong Arc", confidence=0.95)
    weak = _make_candidate(
        "weak_cand",
        "Ungrounded Rumor",
        confidence=0.20,
        ranges=[],
        evidence=[],
    )

    ai_resp = {
        "consolidated_candidates": [strong.to_dict()],
        "decisions": [
            {
                "candidate_id": "strong_cand",
                "action": "KEEP",
                "reason_code": "distinct_thesis",
                "reason": "Strong verified evidence.",
            },
            {
                "candidate_id": "weak_cand",
                "action": "REJECT",
                "reason_code": "weak_evidence",
                "reason": "Low confidence and no verified evidence refs.",
            },
        ],
    }

    client = MockConsolidationAIClient(responses=[ai_resp])
    consolidator = CandidateConsolidator(client=client, cache=cache)
    result = consolidator.consolidate([strong, weak])

    assert result.eligible_count == 1
    assert result.rejected_count == 1
    assert result.candidates[0].proposal_id == "strong_cand"
    dec_weak = result.get_decision("weak_cand")
    assert dec_weak.action == "REJECT"
    assert dec_weak.reason_code == "weak_evidence"
    assert bool(dec_weak.reason)


# ===========================================================================
# 7. All 25 Retained No Quota If Distinct
# ===========================================================================

def test_all_25_retained_no_quota_if_distinct(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    cands = [
        _make_candidate(
            f"distinct_{i:02d}",
            title=f"Unique Story {i}",
            character=f"Character_{i}",
            thesis=f"Unique thesis statement {i} about distinctive events.",
            confidence=0.90,
        )
        for i in range(25)
    ]

    # Mock client returns KEEP for all 25
    def make_resp(user_text: str) -> dict[str, Any]:
        # Parse inputs in this node
        data = json.loads(user_text.split("INPUT PROPOSALS TO CONSOLIDATE:")[1].split("MANDATES:")[0])
        return {
            "consolidated_candidates": data,
            "decisions": [
                {
                    "candidate_id": item["proposal_id"],
                    "action": "KEEP",
                    "reason_code": "distinct_thesis",
                    "reason": "Unique narrative arc retained.",
                }
                for item in data
            ],
        }

    class DynamicMockClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            return make_resp(kwargs["user_text"])

    client = DynamicMockClient()
    consolidator = CandidateConsolidator(client=client, cache=cache)
    result = consolidator.consolidate(cands)

    assert result.eligible_count == 25
    assert len(result.candidates) == 25
    assert result.rejected_count == 0
    assert result.merged_count == 0


# ===========================================================================
# 8. Oversized Recursively Bounded & No Singleton Loop
# ===========================================================================

def test_oversized_recursively_bounded_and_no_singleton_loop(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    # Generate 15 candidates connected into an oversized ambiguous group
    cands = [
        _make_candidate(
            f"cand_{i:02d}",
            title=f"Ambiguous Episode {i}",
            character="Alice",
            thesis="Alice pursues corporate espionage.",
            tags=["espionage", "alice"],
            confidence=0.90,
        )
        for i in range(15)
    ]

    call_counts = {"calls": 0}

    class BoundedMockClient:
        def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            call_counts["calls"] += 1
            u_text = kwargs["user_text"]
            raw_data = json.loads(u_text.split("INPUT PROPOSALS TO CONSOLIDATE:")[1].split("MANDATES:")[0])
            # Each node handles <= MAX_CANDIDATES_PER_NODE
            assert len(raw_data) <= 10
            # Keep first, merge rest
            survivor = raw_data[0]
            decs = [
                {"candidate_id": survivor["proposal_id"], "action": "KEEP", "reason_code": "distinct_thesis", "reason": "Kept"},
            ]
            for other in raw_data[1:]:
                decs.append({
                    "candidate_id": other["proposal_id"],
                    "action": "MERGE",
                    "reason_code": "cross_episode_merge",
                    "reason": "Merged",
                    "target_id": survivor["proposal_id"],
                })
            return {"consolidated_candidates": [survivor], "decisions": decs}

    client = BoundedMockClient()
    consolidator = CandidateConsolidator(client=client, cache=cache)
    result = consolidator.consolidate(cands)

    assert call_counts["calls"] >= 2
    assert result.eligible_count >= 1
    # Check all 15 candidates accounted for
    for c in cands:
        assert result.get_decision(c.proposal_id) is not None


def test_singleton_group_no_ai_loop(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    c1 = _make_candidate("singleton_1", "Solitary Arc")

    client = MockConsolidationAIClient(responses=[])
    consolidator = CandidateConsolidator(client=client, cache=cache)
    result = consolidator.consolidate([c1])

    # Singleton must be evaluated directly without calling AI client (no singleton loop)
    assert len(client.call_history) == 0
    assert result.eligible_count == 1
    assert result.candidates[0].proposal_id == "singleton_1"
    assert result.get_decision("singleton_1").action == "KEEP"


# ===========================================================================
# 9. Malformed Response Not Cached
# ===========================================================================

def test_malformed_response_not_cached(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    c1 = _make_candidate("c1", "Candidate 1", tags=["t1"])
    c2 = _make_candidate("c2", "Candidate 2", tags=["t1"])

    # Malformed response: missing 'decisions'
    malformed = {"consolidated_candidates": [c1.to_dict()]}
    client = MockConsolidationAIClient(responses=[malformed])
    consolidator = CandidateConsolidator(client=client, cache=cache)

    with pytest.raises(AnalysisError, match="thiếu danh sách 'decisions'"):
        consolidator.consolidate([c1, c2])

    # Verify no cache file was written in consolidation_dir
    cached_files = list(cache.consolidation_dir.glob("*.consolidation.json"))
    assert len(cached_files) == 0


# ===========================================================================
# 10. Resume from Cache & Cancel Saves Valid Completed Node
# ===========================================================================

def test_resume_from_cache_skips_ai_call(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    c1 = _make_candidate("c1", "Title 1", tags=["shared"])
    c2 = _make_candidate("c2", "Title 2", tags=["shared"])

    ai_resp = {
        "consolidated_candidates": [c1.to_dict()],
        "decisions": [
            {"candidate_id": "c1", "action": "KEEP", "reason_code": "distinct_thesis", "reason": "Kept"},
            {"candidate_id": "c2", "action": "MERGE", "reason_code": "cross_episode_merge", "reason": "Merged", "target_id": "c1"},
        ],
    }

    # First run saves to cache
    client1 = MockConsolidationAIClient(responses=[ai_resp])
    consolidator1 = CandidateConsolidator(client=client1, cache=cache)
    res1 = consolidator1.consolidate([c1, c2])
    assert len(client1.call_history) == 1

    # Second run loads from cache without calling AI
    client2 = MockConsolidationAIClient(responses=[])
    consolidator2 = CandidateConsolidator(client=client2, cache=cache)
    res2 = consolidator2.consolidate([c1, c2])
    assert len(client2.call_history) == 0
    assert res2.eligible_count == res1.eligible_count


def test_cancel_saves_valid_node_before_raising(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    c1 = _make_candidate("c1", "Title 1", tags=["tag1"])
    c2 = _make_candidate("c2", "Title 2", tags=["tag1"])

    cancel_event = threading.Event()

    def set_cancel_and_return() -> dict[str, Any]:
        cancel_event.set()
        return {
            "consolidated_candidates": [c1.to_dict()],
            "decisions": [
                {"candidate_id": "c1", "action": "KEEP", "reason_code": "distinct_thesis", "reason": "Kept"},
                {"candidate_id": "c2", "action": "MERGE", "reason_code": "cross_episode_merge", "reason": "Merged", "target_id": "c1"},
            ],
        }

    client = MockConsolidationAIClient(responses=[set_cancel_and_return])
    consolidator = CandidateConsolidator(client=client, cache=cache)

    with pytest.raises(AnalysisCancelledError):
        consolidator.consolidate([c1, c2], cancellation_token=cancel_event)

    # Valid node must have been persisted to cache before raising cancellation
    cached_files = list(cache.consolidation_dir.glob("*.consolidation.json"))
    assert len(cached_files) == 1


# ===========================================================================
# 11. No Silent IDs: Missing Decisions Default to KEEP Conservatively
# ===========================================================================

def test_no_silent_ids_missing_decisions_default_keep_conservatively() -> None:
    c1 = _make_candidate("c1", "First")
    c2 = _make_candidate("c2", "Second")
    c3 = _make_candidate("c3", "Third")

    # AI returned decisions only for c1 and c2, omitting c3
    ai_decisions = [
        ConsolidationDecision(candidate_id="c1", action=ConsolidationAction.KEEP.value),
        ConsolidationDecision(candidate_id="c2", action=ConsolidationAction.MERGE.value, target_id="c1"),
    ]

    cset = reconcile_and_validate_decisions(
        input_candidates=[c1, c2, c3],
        ai_consolidated=[c1],
        ai_decisions=ai_decisions,
    )

    # All 3 input IDs must be accounted for
    d3 = cset.get_decision("c3")
    assert d3 is not None
    assert d3.action == ConsolidationAction.KEEP.value
    assert d3.reason_code == ConsolidationReason.CONSERVATIVE_KEEP.value

    # c3 must be preserved in surviving candidates list
    survivor_ids = {c.proposal_id for c in cset.candidates}
    assert "c3" in survivor_ids
    assert "c1" in survivor_ids


# ===========================================================================
# 12. Source Ref Integrity: AI Cannot Invent Source Ranges or Evidence
# ===========================================================================

def test_source_ref_integrity_ai_cannot_invent_source_refs() -> None:
    c1 = _make_candidate(
        "c1",
        "Original Arc",
        ranges=[CandidateSourceRange(episode_id="E01", start_seconds=10.0, end_seconds=20.0, evidence_ref="ref_orig_1")],
        evidence=["ref_orig_1"],
    )

    # AI outputs candidate with hallucinated source range and evidence
    ai_cand = CandidateProposal(
        proposal_id="c1",
        title="Original Arc",
        source_ranges=[
            CandidateSourceRange(episode_id="E01", start_seconds=10.0, end_seconds=20.0, evidence_ref="ref_orig_1"),
            CandidateSourceRange(episode_id="E99", start_seconds=999.0, end_seconds=1000.0, evidence_ref="hallucinated_ref"),
        ],
        supporting_evidence=["ref_orig_1", "hallucinated_ref"],
    )

    cset = reconcile_and_validate_decisions(
        input_candidates=[c1],
        ai_consolidated=[ai_cand],
        ai_decisions=[ConsolidationDecision(candidate_id="c1", action=ConsolidationAction.KEEP.value)],
    )

    survivor = cset.candidates[0]
    # Hallucinated range and ref must be stripped
    assert len(survivor.source_ranges) == 1
    assert survivor.source_ranges[0].episode_id == "E01"
    assert "hallucinated_ref" not in survivor.supporting_evidence
    assert survivor.supporting_evidence == ["ref_orig_1"]


# ===========================================================================
# 13. Phase CANDIDATE_CONSOLIDATION Callbacks
# ===========================================================================

def test_phase_callbacks_emit_counts(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    c1 = _make_candidate("c1", "Title 1", tags=["groupA"])
    c2 = _make_candidate("c2", "Title 2", tags=["groupA"])

    ai_resp = {
        "consolidated_candidates": [c1.to_dict()],
        "decisions": [
            {"candidate_id": "c1", "action": "KEEP", "reason_code": "distinct_thesis", "reason": "Kept"},
            {"candidate_id": "c2", "action": "MERGE", "reason_code": "cross_episode_merge", "reason": "Merged", "target_id": "c1"},
        ],
    }

    phases_called: list[tuple[AnalysisPhase, str, dict[str, Any]]] = []

    def on_phase(phase: AnalysisPhase, msg: str, data: dict[str, Any]) -> None:
        phases_called.append((phase, msg, data))

    client = MockConsolidationAIClient(responses=[ai_resp])
    consolidator = CandidateConsolidator(client=client, cache=cache)
    consolidator.consolidate([c1, c2], phase_callback=on_phase)

    assert len(phases_called) >= 1
    for ph, msg, data in phases_called:
        assert ph == AnalysisPhase.CANDIDATE_CONSOLIDATION
        assert "merged" in data
        assert "rejected" in data
        assert "eligible" in data
        assert "cache" in data


# ===========================================================================
# 14. Legacy Seeds Consumed Cleanly
# ===========================================================================

def test_legacy_connection_seeds_consumed_cleanly(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path / "cache")
    # CandidateProposal created with legacy dict fields
    legacy_data = {
        "proposal_id": "legacy_01",
        "title": "Legacy Connection Proposal",
        "candidate_scope": "CROSS_EPISODE",
        "episodes": ["E01", "E02"],
        "characters": ["Bob"],
        "primary_character": "Bob",
        "central_thesis": "Bob bridges the gap between factions.",
        "supporting_evidence_refs": ["ref_b1", "ref_b2"],  # legacy field alias
        "status": "keep",
        "confidence": 0.88,
    }
    legacy_cand = CandidateProposal.from_dict(legacy_data)
    assert legacy_cand.supporting_evidence == ["ref_b1", "ref_b2"]

    client = MockConsolidationAIClient(responses=[])
    consolidator = CandidateConsolidator(client=client, cache=cache)
    result = consolidator.consolidate([legacy_cand])

    assert result.eligible_count == 1
    assert result.candidates[0].proposal_id == "legacy_01"
    assert result.get_decision("legacy_01").action == "KEEP"
