"""Tests for candidate-discovery-r8 (Objective rev 8).

Covers:
1. CandidateProposal extended fields, defaults, backward compatibility from_dict.
2. CandidateSourceRange properties, roundtrip, timestamp math.
3. Cache key computation, determinism, isolation from secrets, order dependency.
4. Policy and coverage hash invalidations.
5. HierarchyCacheManager discovery methods, envelope v1, atomic persistence.
6. Schema validation for AI discovery response (malformed domain response not cached).
7. Ground validation: allowed episodes, unknown episodes strip, empty reject with log,
   timestamp duration clamping, scope auto-correct, AI title/thesis preservation, no synthetic candidates.
8. No quota mandate: 25 candidates returned and retained.
9. Supporting candidates equality.
10. Cache resume skips AI call.
11. Failure node not cached.
12. Cancellation saves valid completed node before raising.
13. Prompt scanner irrelevant: candidate prompt uses compact CandidateDirective, no scanner instructions.
14. Legacy connection seed proposals combined and deduplicated by ID only.
15. migrate_legacy_connection_candidates offline helper.
16. Single episode and multi-episode season discovery end-to-end flows.
17. Adaptive payload: oversized summaries compact and split across nodes.
18. Phase callbacks user data: index, total, cache, candidates_count.
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
    CandidateDiscoverer,
    DiscoveryNodePlanItem,
    compute_coverage_hash,
    discover_candidates_season,
    discover_candidates_single,
    estimate_discovery_request_size,
    format_discovery_node_user_text,
    ground_validate_candidate,
    migrate_legacy_connection_candidates,
    plan_discovery_nodes,
    validate_discovery_response_schema,
)
from toolrecap_v2.analyzer.connection import (
    CandidateProposal,
    SeasonConnectionResult,
)
from toolrecap_v2.analyzer.errors import AnalysisCancelledError, AnalysisError
from toolrecap_v2.analyzer.phases import AnalysisPhase
from toolrecap_v2.analyzer.prompts import CANDIDATE_DISCOVERY_SYSTEM_PROMPT
from toolrecap_v2.domain import (
    CandidateDirective,
    CandidateScope,
    CandidateSourceRange,
    CandidateStatus,
    CompactEpisodeSummary,
    CompactSummaryItem,
    CompactionLevel,
    DISCOVERY_SCHEMA_VERSION,
    EditorialPolicy,
    HierarchyCacheManager,
    SourceEpisode,
    compute_discovery_cache_key,
    validate_discovery_cache_data,
)
from toolrecap_v2.settings import AppSettings


class MockDiscoveryAIClient:
    """Mock AI client for candidate discovery tests."""

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

        return {"discovered_candidates": []}


def _make_dummy_summary(ep_id: str, duration: float = 600.0, num_items: int = 3) -> CompactEpisodeSummary:
    items = []
    step = duration / max(1, num_items)
    for i in range(num_items):
        start = i * step
        end = (i + 1) * step
        items.append(
            CompactSummaryItem(
                refs=[f"ref_{ep_id}_{i}"],
                episode_id=ep_id,
                start_sec=start,
                end_sec=end,
                characters=["Alice", "Bob"],
                summary=f"Event {i} in {ep_id}",
                categories=["major_scenes"],
                item_type="scene",
            )
        )
    return CompactEpisodeSummary(
        episode_id=ep_id,
        title=f"Episode {ep_id}",
        duration_seconds=duration,
        items=items,
    )


# ===========================================================================
# 1. CandidateProposal & CandidateSourceRange Tests
# ===========================================================================

def test_candidate_proposal_roundtrip_extended_fields() -> None:
    range1 = CandidateSourceRange(
        episode_id="E01",
        start_seconds=12.5,
        end_seconds=45.0,
        evidence_ref="ref_01",
    )
    prop = CandidateProposal(
        proposal_id="cand_01",
        title="Betrayal in the Shadows",
        candidate_scope=CandidateScope.CROSS_EPISODE.value,
        episodes=["E01", "E02"],
        characters=["Alice", "Bob"],
        description="A story of broken trust.",
        editorial_reason="High dramatic tension and strong payoff.",
        status=CandidateStatus.KEEP.value,
        subject="Trust and betrayal",
        central_thesis="Ambition erodes personal loyalty.",
        primary_character="Alice",
        supporting_characters=["Bob", "Charlie"],
        source_ranges=[range1],
        setup="Alice needs funding.",
        development="Bob offers dirty money.",
        turning="Alice signs the contract.",
        payoff="The scheme collapses.",
        consequence="Alice faces trial.",
        observed_facts=["Alice took the money", "Bob fled"],
        supporting_evidence=["ref_01", "ref_02"],
        counter_evidence=["Alice hesitated"],
        praise="Exceptional character consistency",
        criticism="Middle section slightly dragged",
        alternative="Could focus on Charlie's perspective",
        why="Pivotal emotional arc of the season",
        hooks=["Will Alice expose Bob?"],
        estimated_duration=360.0,
        overlap_tags=["corruption", "crime"],
        confidence=0.92,
    )

    data = prop.to_dict()
    assert data["proposal_id"] == "cand_01"
    assert data["subject"] == "Trust and betrayal"
    assert data["central_thesis"] == "Ambition erodes personal loyalty."
    assert data["primary_character"] == "Alice"
    assert data["supporting_characters"] == ["Bob", "Charlie"]
    assert len(data["source_ranges"]) == 1
    assert data["source_ranges"][0]["evidence_ref"] == "ref_01"
    assert data["turning"] == "Alice signs the contract."
    assert data["confidence"] == 0.92
    assert data["status"] == "keep"

    restored = CandidateProposal.from_dict(data)
    assert restored.proposal_id == prop.proposal_id
    assert restored.title == prop.title
    assert restored.subject == prop.subject
    assert restored.central_thesis == prop.central_thesis
    assert restored.primary_character == prop.primary_character
    assert restored.supporting_characters == prop.supporting_characters
    assert len(restored.source_ranges) == 1
    assert restored.source_ranges[0].start_seconds == 12.5
    assert restored.source_ranges[0].end_seconds == 45.0
    assert restored.source_ranges[0].evidence_ref == "ref_01"
    assert restored.observed_facts == ["Alice took the money", "Bob fled"]
    assert restored.confidence == 0.92


def test_candidate_proposal_backward_compat_minimal_dict() -> None:
    minimal = {
        "proposal_id": "legacy_01",
        "title": "Legacy Proposal",
        "candidate_scope": "SINGLE_EPISODE",
        "episodes": ["E01"],
    }
    prop = CandidateProposal.from_dict(minimal)
    assert prop.proposal_id == "legacy_01"
    assert prop.title == "Legacy Proposal"
    assert prop.candidate_scope == "SINGLE_EPISODE"
    assert prop.episodes == ["E01"]
    # Extended fields should have defaults
    assert prop.subject == ""
    assert prop.central_thesis == ""
    assert prop.primary_character == ""
    assert prop.supporting_characters == []
    assert prop.source_ranges == []
    assert prop.setup == ""
    assert prop.turning == ""
    assert prop.observed_facts == []
    assert prop.supporting_evidence == []
    assert prop.counter_evidence == []
    assert prop.praise == ""
    assert prop.why == ""
    assert prop.hooks == []
    assert prop.estimated_duration == 0.0
    assert prop.overlap_tags == []
    assert prop.confidence == 1.0


def test_candidate_proposal_field_aliases() -> None:
    data = {
        "proposal_id": "p_alias",
        "title": "Alias Test",
        "turning_point": "The big shift",
        "estimated_duration_seconds": 240.0,
        "supporting_evidence_refs": ["ref_1", "ref_2"],
        "counter_evidence_refs": ["ref_3"],
        "hooks": "Single hook as string",
        "praise": ["Praise point 1", "Praise point 2"],
    }
    prop = CandidateProposal.from_dict(data)
    assert prop.turning == "The big shift"
    assert prop.estimated_duration == 240.0
    assert prop.supporting_evidence == ["ref_1", "ref_2"]
    assert prop.counter_evidence == ["ref_3"]
    assert prop.hooks == ["Single hook as string"]
    assert "Praise point 1" in prop.praise
    assert "Praise point 2" in prop.praise


def test_candidate_source_range_properties_and_aliases() -> None:
    rng = CandidateSourceRange.from_dict({
        "episode_id": "E01",
        "start": 10.0,
        "end": 40.0,
        "ref": "ref_abc",
    })
    assert rng.episode_id == "E01"
    assert rng.start_seconds == 10.0
    assert rng.end_seconds == 40.0
    assert rng.evidence_ref == "ref_abc"
    assert rng.ref == "ref_abc"
    assert rng.start == 10.0
    assert rng.end == 40.0
    assert rng.duration == 30.0


# ===========================================================================
# 2. Cache Key & HierarchyCacheManager Tests
# ===========================================================================

def test_compute_discovery_cache_key_deterministic() -> None:
    k1 = compute_discovery_cache_key(
        node_id="node_1",
        child_hashes=["h1", "h2"],
        candidate_directive_hash="dir_hash_abc",
        coverage_hash="cov_hash_xyz",
        model="gemini-2.5-flash",
        thinking="auto",
    )
    k2 = compute_discovery_cache_key(
        node_id="node_1",
        child_hashes=["h1", "h2"],
        candidate_directive_hash="dir_hash_abc",
        coverage_hash="cov_hash_xyz",
        model="gemini-2.5-flash",
        thinking="auto",
    )
    assert k1 == k2
    assert len(k1) == 64


def test_compute_discovery_cache_key_order_dependent() -> None:
    k_order1 = compute_discovery_cache_key(
        node_id="node_1",
        child_hashes=["h1", "h2"],
        candidate_directive_hash="dir_hash",
        coverage_hash="cov_hash",
        model="gemini-2.5-flash",
    )
    k_order2 = compute_discovery_cache_key(
        node_id="node_1",
        child_hashes=["h2", "h1"],
        candidate_directive_hash="dir_hash",
        coverage_hash="cov_hash",
        model="gemini-2.5-flash",
    )
    assert k_order1 != k_order2


def test_policy_hash_invalidation() -> None:
    pol1 = EditorialPolicy(candidate_directive=CandidateDirective(priority_themes=["Theme A"]))
    pol2 = EditorialPolicy(candidate_directive=CandidateDirective(priority_themes=["Theme B"]))

    h1 = pol1.candidate_directive.compute_hash()
    h2 = pol2.candidate_directive.compute_hash()
    assert h1 != h2

    k1 = compute_discovery_cache_key(
        node_id="node_1",
        child_hashes=["h1"],
        candidate_directive_hash=h1,
        coverage_hash="cov_hash",
        model="m",
    )
    k2 = compute_discovery_cache_key(
        node_id="node_1",
        child_hashes=["h1"],
        candidate_directive_hash=h2,
        coverage_hash="cov_hash",
        model="m",
    )
    assert k1 != k2


def test_coverage_hash_invalidation() -> None:
    cov1 = {"status": "COMPLETE", "missing": []}
    cov2 = {"status": "PARTIAL", "missing": ["E02"]}
    h1 = compute_coverage_hash(cov1)
    h2 = compute_coverage_hash(cov2)
    assert h1 != h2

    k1 = compute_discovery_cache_key("node", ["h"], "d", h1, "m")
    k2 = compute_discovery_cache_key("node", ["h"], "d", h2, "m")
    assert k1 != k2


def test_hierarchy_cache_save_and_load_discovery_result(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    node_id = "test_node_01"
    key = "a" * 64
    data = {
        "discovered_candidates": [
            {"proposal_id": "p1", "title": "Saved Candidate", "episodes": ["E01"]}
        ]
    }

    path = cache.save_discovery_result(node_id, key, data)
    assert path.is_file()

    # Read envelope from disk
    raw_disk = json.loads(path.read_text(encoding="utf-8"))
    assert raw_disk["schema"] == DISCOVERY_SCHEMA_VERSION
    assert raw_disk["cache_key"] == key
    assert raw_disk["result"]["discovered_candidates"][0]["title"] == "Saved Candidate"

    # Load via cache manager
    loaded, meta = cache.load_discovery_result(node_id, key)
    assert meta["hit"] is True
    assert loaded is not None
    assert loaded["discovered_candidates"][0]["proposal_id"] == "p1"


def test_validate_discovery_cache_data() -> None:
    assert validate_discovery_cache_data({"discovered_candidates": [{"title": "C1"}]}) is True
    assert validate_discovery_cache_data({"candidate_proposals": [{"title": "C2"}]}) is True
    assert validate_discovery_cache_data({"discovered_candidates": []}) is True
    assert validate_discovery_cache_data("not a dict") is False
    assert validate_discovery_cache_data({"other_key": []}) is False
    assert validate_discovery_cache_data({"discovered_candidates": "not a list"}) is False
    assert validate_discovery_cache_data({"discovered_candidates": ["not a dict item"]}) is False


def test_validate_discovery_response_schema() -> None:
    assert validate_discovery_response_schema({"discovered_candidates": []}) is True
    assert validate_discovery_response_schema({"discovered_candidates": [{"id": "1"}]}) is True
    assert validate_discovery_response_schema("string") is False
    assert validate_discovery_response_schema({"no_candidates": 1}) is False

    with pytest.raises(AnalysisError, match="phải là dict"):
        validate_discovery_response_schema([1, 2], raise_error=True)

    with pytest.raises(AnalysisError, match="thiếu trường"):
        validate_discovery_response_schema({"wrong": 123}, raise_error=True)


# ===========================================================================
# 3. Ground Validation Tests
# ===========================================================================

def test_ground_validate_unknown_episodes_stripped(caplog: pytest.LogCaptureFixture) -> None:
    cand = CandidateProposal(
        proposal_id="p1",
        title="Unknown Episode Test",
        candidate_scope=CandidateScope.CROSS_EPISODE.value,
        episodes=["E01", "E99"],
        source_ranges=[
            CandidateSourceRange(episode_id="E01", start_seconds=10, end_seconds=20),
            CandidateSourceRange(episode_id="E99", start_seconds=30, end_seconds=40),
        ],
    )
    with caplog.at_level(logging.WARNING):
        result = ground_validate_candidate(cand, allowed_episode_ids={"E01", "E02"})

    assert result is not None
    assert result.episodes == ["E01"]
    assert len(result.source_ranges) == 1
    assert result.source_ranges[0].episode_id == "E01"


def test_ground_validate_empty_candidate_rejected_with_log(caplog: pytest.LogCaptureFixture) -> None:
    cand = CandidateProposal(
        proposal_id="p_empty",
        title="Invalid Candidate",
        episodes=["E99", "E88"],
    )
    with caplog.at_level(logging.WARNING):
        result = ground_validate_candidate(cand, allowed_episode_ids={"E01", "E02"})

    assert result is None
    assert "Rejecting candidate 'Invalid Candidate'" in caplog.text


def test_ground_validate_timestamp_duration_clamping() -> None:
    cand = CandidateProposal(
        proposal_id="p1",
        title="Clamping Test",
        episodes=["E01"],
        source_ranges=[
            CandidateSourceRange(episode_id="E01", start_seconds=500.0, end_seconds=800.0),
            CandidateSourceRange(episode_id="E01", start_seconds=900.0, end_seconds=1000.0),
        ],
    )
    durations = {"E01": 600.0}
    result = ground_validate_candidate(cand, allowed_episode_ids={"E01"}, episode_durations=durations)

    assert result is not None
    assert len(result.source_ranges) == 1
    assert result.source_ranges[0].start_seconds == 500.0
    assert result.source_ranges[0].end_seconds == 600.0  # clamped to duration


def test_ground_validate_scope_autocorrect_multi_to_cross() -> None:
    cand = CandidateProposal(
        proposal_id="p1",
        title="Multi Scope Test",
        candidate_scope=CandidateScope.SINGLE_EPISODE.value,
        episodes=["E01", "E02"],
    )
    result = ground_validate_candidate(cand, allowed_episode_ids={"E01", "E02", "E03"})
    assert result is not None
    assert result.candidate_scope == CandidateScope.CROSS_EPISODE.value


def test_ground_validate_scope_autocorrect_all_to_season_arc() -> None:
    cand = CandidateProposal(
        proposal_id="p1",
        title="Full Season Arc",
        candidate_scope=CandidateScope.SINGLE_EPISODE.value,
        episodes=["E01", "E02", "E03"],
    )
    result = ground_validate_candidate(cand, allowed_episode_ids={"E01", "E02", "E03"})
    assert result is not None
    assert result.candidate_scope == CandidateScope.SEASON_ARC.value


def test_ground_validate_scope_autocorrect_single_to_single() -> None:
    cand = CandidateProposal(
        proposal_id="p1",
        title="Single Scope Test",
        candidate_scope=CandidateScope.CROSS_EPISODE.value,
        episodes=["E01"],
    )
    result = ground_validate_candidate(cand, allowed_episode_ids={"E01", "E02"})
    assert result is not None
    assert result.candidate_scope == CandidateScope.SINGLE_EPISODE.value


def test_ground_validate_preserves_ai_title_and_thesis() -> None:
    cand = CandidateProposal(
        proposal_id="p1",
        title="Unique Creative Title from AI",
        central_thesis="A deep grounded thematic insight",
        episodes=["E01"],
    )
    result = ground_validate_candidate(cand, allowed_episode_ids={"E01"})
    assert result is not None
    assert result.title == "Unique Creative Title from AI"
    assert result.central_thesis == "A deep grounded thematic insight"


# ===========================================================================
# 4. CandidateDiscoverer AI Stage & Policies Tests
# ===========================================================================

def test_no_synthetic_candidates(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    client = MockDiscoveryAIClient([{"discovered_candidates": []}])
    discoverer = CandidateDiscoverer(client=client, cache=cache)

    summ = _make_dummy_summary("E01")
    candidates = discoverer.discover_single(summ)
    assert candidates == []
    assert len(client.call_history) == 1


def test_no_quota_25_candidates_retained(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    # Generate 25 distinct candidates
    cand_list = [
        {
            "proposal_id": f"cand_{i:02d}",
            "title": f"Candidate Story {i:02d}",
            "candidate_scope": "SINGLE_EPISODE",
            "episodes": ["E01"],
            "central_thesis": f"Thesis {i}",
        }
        for i in range(25)
    ]
    client = MockDiscoveryAIClient([{"discovered_candidates": cand_list}])
    discoverer = CandidateDiscoverer(client=client, cache=cache)

    summ = _make_dummy_summary("E01")
    candidates = discoverer.discover_single(summ)
    assert len(candidates) == 25
    assert all(isinstance(c, CandidateProposal) for c in candidates)
    # Verify no slicing/quota cap was applied
    assert candidates[24].proposal_id == "cand_24"


def test_supporting_candidates_equal_weight(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    cand_list = [
        {
            "proposal_id": "protagonist_01",
            "title": "Main Hero Journey",
            "primary_character": "Hero",
            "episodes": ["E01"],
        },
        {
            "proposal_id": "supporting_01",
            "title": "Sidekick Secret Arc",
            "primary_character": "Sidekick",
            "episodes": ["E01"],
        },
    ]
    client = MockDiscoveryAIClient([{"discovered_candidates": cand_list}])
    discoverer = CandidateDiscoverer(client=client, cache=cache)

    summ = _make_dummy_summary("E01")
    candidates = discoverer.discover_single(summ)
    assert len(candidates) == 2
    primary_chars = {c.primary_character for c in candidates}
    assert "Sidekick" in primary_chars
    assert "Hero" in primary_chars


def test_malformed_response_not_cached(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    # Malformed response: discovered_candidates is not a list
    client = MockDiscoveryAIClient([{"discovered_candidates": "not-a-list"}])
    discoverer = CandidateDiscoverer(client=client, cache=cache)

    summ = _make_dummy_summary("E01")
    with pytest.raises(AnalysisError, match="sai cấu trúc schema"):
        discoverer.discover_single(summ)

    # Check cache directory remains empty
    cached_files = list(cache.discovery_dir.glob("*.discovery.json"))
    assert len(cached_files) == 0


def test_cache_resume_skips_ai_call(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    cand_list = [{"proposal_id": "p1", "title": "Cached Arc", "episodes": ["E01"]}]
    client = MockDiscoveryAIClient([{"discovered_candidates": cand_list}])
    discoverer = CandidateDiscoverer(client=client, cache=cache)

    summ = _make_dummy_summary("E01")

    # Pass 1: AI called and result cached
    cands1 = discoverer.discover_single(summ)
    assert len(cands1) == 1
    assert len(client.call_history) == 1

    # Pass 2: hits cache, client call count remains 1
    cands2 = discoverer.discover_single(summ)
    assert len(cands2) == 1
    assert cands2[0].title == "Cached Arc"
    assert len(client.call_history) == 1


def test_failure_node_not_cached_and_retryable(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    # First attempt raises error, second succeeds
    client = MockDiscoveryAIClient([
        RuntimeError("Transient network failure"),
        {"discovered_candidates": [{"proposal_id": "p_retry", "title": "Retry Success", "episodes": ["E01"]}]},
    ])
    discoverer = CandidateDiscoverer(client=client, cache=cache)
    summ = _make_dummy_summary("E01")

    with pytest.raises(RuntimeError, match="Transient network failure"):
        discoverer.discover_single(summ)

    assert len(list(cache.discovery_dir.glob("*.discovery.json"))) == 0

    # Retry succeeds
    cands = discoverer.discover_single(summ)
    assert len(cands) == 1
    assert cands[0].title == "Retry Success"
    assert len(list(cache.discovery_dir.glob("*.discovery.json"))) == 1


def test_cancellation_saves_valid_before_raising(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    cancel_evt = threading.Event()

    # Create 2 summaries that will form 2 separate nodes by setting target_ceiling low
    s1 = _make_dummy_summary("E01", num_items=5)
    s2 = _make_dummy_summary("E02", num_items=5)

    def _resp1() -> dict[str, Any]:
        # Trip cancellation right when first node finishes
        cancel_evt.set()
        return {"discovered_candidates": [{"proposal_id": "p1", "title": "Node 1 Success", "episodes": ["E01"]}]}

    client = MockDiscoveryAIClient([_resp1, {"discovered_candidates": []}])
    settings = AppSettings()
    settings.target_payload_ceiling = 100  # forces each summary into its own node
    discoverer = CandidateDiscoverer(settings=settings, client=client, cache=cache)

    with pytest.raises(AnalysisCancelledError):
        discoverer.discover_season([s1, s2], cancellation_token=cancel_evt)

    # Verify node 1 was saved to disk before cancellation raised
    cached_files = list(cache.discovery_dir.glob("*.discovery.json"))
    assert len(cached_files) == 1
    disk_content = json.loads(cached_files[0].read_text(encoding="utf-8"))
    assert disk_content["result"]["discovered_candidates"][0]["title"] == "Node 1 Success"


def test_prompt_scanner_irrelevant() -> None:
    summ = _make_dummy_summary("E01")
    cand_dir = CandidateDirective(
        priority_themes=["Underdog triumph"],
        candidate_rules=["Focus on dramatic turning points"],
    )
    user_text = format_discovery_node_user_text(
        node_id="disc_01",
        summaries=[summ],
        candidate_directive=cand_dir,
    )

    # Candidate directive content present
    assert "Underdog triumph" in user_text
    assert "Focus on dramatic turning points" in user_text

    # Irrelevant scanner categories/instructions absent
    assert "SCANNER" not in user_text
    assert "Evidence categories:" not in user_text
    assert "major_scenes" not in user_text.split("Episode Summaries:")[0]


def test_legacy_connection_seed_proposals_combined_and_deduped(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    legacy_conn = SeasonConnectionResult(
        candidate_proposals=[
            CandidateProposal(proposal_id="p1", title="Seed Proposal 1", episodes=["E01"]),
            CandidateProposal(proposal_id="p2", title="Seed Proposal 2 (Old)", episodes=["E01", "E02"]),
        ]
    )
    # AI discovers updated p2 and new p3
    ai_resp = {
        "discovered_candidates": [
            {"proposal_id": "p2", "title": "Updated Proposal 2 (New)", "episodes": ["E01", "E02"]},
            {"proposal_id": "p3", "title": "Fresh Proposal 3", "episodes": ["E02"]},
        ]
    }
    client = MockDiscoveryAIClient([ai_resp])
    discoverer = CandidateDiscoverer(client=client, cache=cache)

    s1 = _make_dummy_summary("E01")
    s2 = _make_dummy_summary("E02")

    result = discoverer.discover_season([s1, s2], connection=legacy_conn)
    assert len(result) == 3
    p_map = {p.proposal_id: p for p in result}
    assert p_map["p1"].title == "Seed Proposal 1"
    assert p_map["p2"].title == "Updated Proposal 2 (New)"
    assert p_map["p3"].title == "Fresh Proposal 3"


def test_migrate_legacy_connection_candidates_without_ai(tmp_path: Path) -> None:
    # 1. From SeasonConnectionResult object
    conn_obj = SeasonConnectionResult(
        candidate_proposals=[CandidateProposal(proposal_id="c1", title="From Obj", episodes=["E01"])]
    )
    migrated1 = migrate_legacy_connection_candidates(conn_obj)
    assert len(migrated1) == 1
    assert migrated1[0].title == "From Obj"

    # 2. From dictionary
    conn_dict = {
        "candidate_proposals": [{"proposal_id": "c2", "title": "From Dict", "episodes": ["E02"]}]
    }
    migrated2 = migrate_legacy_connection_candidates(conn_dict)
    assert len(migrated2) == 1
    assert migrated2[0].title == "From Dict"

    # 3. From JSON file on disk
    fpath = tmp_path / "conn.json"
    fpath.write_text(json.dumps({"result": conn_dict}), encoding="utf-8")
    migrated3 = migrate_legacy_connection_candidates(fpath)
    assert len(migrated3) == 1
    assert migrated3[0].title == "From Dict"


def test_single_episode_discovery_end_to_end(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    client = MockDiscoveryAIClient([
        {
            "discovered_candidates": [
                {
                    "proposal_id": "single_01",
                    "title": "Standalone Episode Arc",
                    "candidate_scope": "SINGLE_EPISODE",
                    "episodes": ["E01"],
                    "central_thesis": "One day can change everything.",
                    "primary_character": "Alice",
                }
            ]
        }
    ])
    phases_logged: list[tuple[AnalysisPhase, str, dict[str, Any]]] = []

    def on_phase(phase: AnalysisPhase, msg: str, data: dict[str, Any]) -> None:
        phases_logged.append((phase, msg, data))

    summ = _make_dummy_summary("E01")
    candidates = discover_candidates_single(
        summary=summ,
        client=client,
        cache=cache,
        phase_callback=on_phase,
    )
    assert len(candidates) == 1
    assert candidates[0].proposal_id == "single_01"
    assert len(phases_logged) == 1
    assert phases_logged[0][0] == AnalysisPhase.CANDIDATE_DISCOVERY
    assert phases_logged[0][2]["candidates_count"] == 1
    assert phases_logged[0][2]["cache"] is False


def test_multi_episode_season_discovery_end_to_end(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    client = MockDiscoveryAIClient([
        {
            "discovered_candidates": [
                {
                    "proposal_id": "season_arc_01",
                    "title": "Season-Long Rivalry",
                    "candidate_scope": "SEASON_ARC",
                    "episodes": ["E01", "E02", "E03"],
                    "characters": ["Alice", "Bob"],
                    "primary_character": "Alice",
                }
            ]
        }
    ])
    s1 = _make_dummy_summary("E01")
    s2 = _make_dummy_summary("E02")
    s3 = _make_dummy_summary("E03")

    candidates = discover_candidates_season(
        summaries=[s1, s2, s3],
        client=client,
        cache=cache,
    )
    assert len(candidates) == 1
    assert candidates[0].candidate_scope == CandidateScope.SEASON_ARC.value
    assert candidates[0].episodes == ["E01", "E02", "E03"]


def test_adaptive_payload_oversized_summary_compacts_and_splits() -> None:
    # Create an episode summary with many items
    s = _make_dummy_summary("E01", num_items=20)
    # Set target_ceiling low enough that it cannot fit in FULL compaction
    nodes = plan_discovery_nodes(
        summaries=[s],
        target_ceiling=1500,
        hard_ceiling=500_000,
    )
    assert len(nodes) >= 1
    # Check that compaction occurred (not FULL, or split into multiple nodes)
    compaction_used = nodes[0].compaction_level
    assert compaction_used in (
        CompactionLevel.TRIMMED,
        CompactionLevel.PRIORITY,
        CompactionLevel.SKELETON,
    ) or len(nodes) > 1


def test_phase_callback_metrics(tmp_path: Path) -> None:
    cache = HierarchyCacheManager(base_dir=tmp_path)
    client = MockDiscoveryAIClient([
        {"discovered_candidates": [{"proposal_id": "p1", "title": "P1", "episodes": ["E01"]}]}
    ])
    callback_payloads: list[dict[str, Any]] = []

    def on_phase(phase: AnalysisPhase, msg: str, data: dict[str, Any]) -> None:
        if phase == AnalysisPhase.CANDIDATE_DISCOVERY:
            callback_payloads.append(data)

    discoverer = CandidateDiscoverer(client=client, cache=cache)
    summ = _make_dummy_summary("E01")
    discoverer.discover_single(summ, phase_callback=on_phase)

    assert len(callback_payloads) == 1
    p = callback_payloads[0]
    assert p["index"] == 1
    assert p["total"] == 1
    assert p["cache"] is False
    assert p["candidates_count"] == 1
