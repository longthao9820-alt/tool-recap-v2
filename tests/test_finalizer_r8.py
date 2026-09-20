"""Tests for finalizer-api-tests-r8 (Objective rev 8).

Survey matrix & scenarios:
Legacy & New CommentaryOutput Serialization:
1. Legacy CommentaryOutput serialization without new fields preserves exact keys and structure.
2. New fields (candidate_id, source_candidate_id) roundtrip cleanly in CommentaryOutput to_dict/from_dict/json.
3. CommentaryOutput constructor directly accepts candidate_id and source_candidate_id without error.
4. AnalysisManifest serialization and validation with mixed legacy/new CommentaryOutput items.

Exact 7 scenarios from previous:
1. Multi-group proposal partitioning with all calls <= 500,000 bytes and <= 480,000 bytes, no outputs lost.
2. Huge single field cap (600KB field capped safely without breaking structure or exceeding ceiling).
3. Dense 1.2MB single evidence: no raw evidence categories/transcript in prompt, all calls bounded.
4. Small connection: single finalizer call without unnecessary capping.
5. Mid-group cancellation handling (cancellation before group 2 stops execution, raises AnalysisCancelledError).
6. Finalizer group cache and resume on retry (reuse group 1 from cache, run group 2, all outputs returned).
7. Final plan cache and resume consideration with AnalysisEngine.

Rev 8 Candidate Finalization API:
8. finalize_candidates candidate tracking (attempted, accepted, rejected) and output source_candidate_id matching.
9. finalize_candidates candidate ID omission behavior: single candidate defaults safely, multiple candidates rejected with error.
10. finalize_candidates unknown candidate ID rejection.
11. finalize_candidates cache key determinism and cache resume.
12. finalize_candidates cancellation handling.
13. finalize_candidates offline mode with legacy wrapper vs disabled.
14. FinalizationResult dataclass methods (to_dict, from_dict, to_json, from_json) and count properties.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from typing import Any

import pytest

from toolrecap_v2.analyzer.connection import (
    HARD_PAYLOAD_CEILING,
    TARGET_PAYLOAD_CEILING,
    SeasonConnectionResult,
)
from toolrecap_v2.analyzer.engine import AnalysisEngine
from toolrecap_v2.analyzer.errors import AnalysisCancelledError, AnalysisError
from toolrecap_v2.analyzer.finalizer import (
    CandidateFinalizer,
    FinalizationResult,
    FinalizerGroupPlan,
    SeasonFinalizer,
    estimate_finalizer_request_size,
    finalize_candidates,
    format_finalizer_user_text,
    plan_finalizer_groups,
)
from toolrecap_v2.analyzer.prompts import FINALIZER_SYSTEM_PROMPT
from toolrecap_v2.api_client import APIError
from toolrecap_v2.domain.cache import (
    EvidenceCacheManager,
    HierarchyCacheManager,
    compute_finalizer_cache_key,
    validate_finalizer_cache_data,
)
from toolrecap_v2.domain.enums import AnalysisScope, AudioPolicy, CandidateScope, OutputStatus
from toolrecap_v2.domain.models import (
    AnalysisManifest,
    CandidateProposal,
    CommentaryOutput,
    EpisodeEvidence,
    Segment,
    SourceClip,
    SourceEpisode,
    ValidationError,
)
from toolrecap_v2.domain.policy import EditorialPolicy, OutputDirective
from toolrecap_v2.settings import AppSettings


class MockFinalizerClient:
    """Mock OpenAICompatibleClient for finalizer R8 testing."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.call_history: list[dict[str, Any]] = []
        self.endpoint = "http://mock-ai:20128/v1"
        self.api_key = "mock-key"

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("Mock API call cancelled.")

        call_record = {
            "model": model,
            "system": system,
            "user_text": user_text,
            "thinking": thinking,
        }
        self.call_history.append(call_record)

        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if callable(resp):
                return resp(call_record)
            return resp

        return {"outputs": []}


def _make_episodes(tmp_path: Path, count: int = 2) -> list[SourceEpisode]:
    episodes = []
    for i in range(1, count + 1):
        ep_id = f"E{i:02d}"
        v_path = tmp_path / f"{ep_id}.mp4"
        if not v_path.exists():
            v_path.write_bytes(b"dummy video data")
        episodes.append(
            SourceEpisode(
                episode_id=ep_id,
                source_video=str(v_path),
                duration_seconds=300.0,
                title=f"Episode {i}",
            )
        )
    return episodes


# ===========================================================================
# Legacy & New CommentaryOutput Serialization Tests
# ===========================================================================

def test_legacy_commentary_output_serialization_no_new_fields_break() -> None:
    """Legacy CommentaryOutput serialization without candidate fields produces exact legacy keys without breaking."""
    seg = Segment(
        segment_id="seg_01",
        source_clips=[SourceClip(episode_id="E01", source_video="e01.mp4", start=0.0, end=10.0)],
        narration="Narration text",
        audio_policy="duck",
    )
    legacy_out = CommentaryOutput(
        output_id="out_leg_01",
        title="Legacy Recap Title",
        sanitized_title="Legacy Recap Title",
        candidate_scope=CandidateScope.SINGLE_EPISODE.value,
        segments=[seg],
        status=OutputStatus.WAITING.value,
    )
    d = legacy_out.to_dict()

    # Legacy keys must be preserved; candidate_id and source_candidate_id omitted when empty
    expected_legacy_keys = {
        "output_id",
        "title",
        "sanitized_title",
        "candidate_scope",
        "segments",
        "status",
        "publication_video_path",
        "publication_original_srt_path",
        "publication_narration_srt_path",
        "error",
        "progress",
    }
    assert set(d.keys()) == expected_legacy_keys
    assert "candidate_id" not in d
    assert "source_candidate_id" not in d

    # Deserializing from legacy dict without new fields works cleanly
    restored = CommentaryOutput.from_dict(d)
    assert restored.output_id == "out_leg_01"
    assert restored.title == "Legacy Recap Title"
    assert restored.candidate_id == ""
    assert restored.source_candidate_id == ""
    assert len(restored.segments) == 1

    # Re-serialization of restored legacy object also omits candidate fields
    d2 = restored.to_dict()
    assert set(d2.keys()) == expected_legacy_keys


def test_commentary_output_new_fields_roundtrip() -> None:
    """CommentaryOutput with candidate_id and source_candidate_id roundtrips through to_dict, from_dict, to_json, from_json."""
    seg = Segment(
        segment_id="seg_01",
        source_clips=[SourceClip(episode_id="E01", source_video="e01.mp4", start=0.0, end=10.0)],
        narration="Narration text",
        audio_policy="duck",
    )
    # Direct constructor call with new fields
    out = CommentaryOutput(
        output_id="out_new_01",
        title="Modern Arc Recap",
        candidate_id="prop_hero_arc",
        source_candidate_id="prop_hero_arc",
        segments=[seg],
    )
    assert out.candidate_id == "prop_hero_arc"
    assert out.source_candidate_id == "prop_hero_arc"

    d = out.to_dict()
    assert d["candidate_id"] == "prop_hero_arc"
    assert d["source_candidate_id"] == "prop_hero_arc"

    restored = CommentaryOutput.from_dict(d)
    assert restored.candidate_id == "prop_hero_arc"
    assert restored.source_candidate_id == "prop_hero_arc"
    assert restored.title == "Modern Arc Recap"

    # JSON roundtrip
    json_str = out.to_json()
    assert "prop_hero_arc" in json_str
    from_json_out = CommentaryOutput.from_json(json_str)
    assert from_json_out.output_id == "out_new_01"
    assert from_json_out.candidate_id == "prop_hero_arc"
    assert from_json_out.source_candidate_id == "prop_hero_arc"


def test_commentary_output_single_candidate_field_sync() -> None:
    """Setting either candidate_id or source_candidate_id synchronizes both in to_dict and from_dict."""
    # Only source_candidate_id in dict
    d1 = {"output_id": "out_01", "title": "Title 1", "source_candidate_id": "cand_99"}
    obj1 = CommentaryOutput.from_dict(d1)
    assert obj1.source_candidate_id == "cand_99"
    assert obj1.candidate_id == "cand_99"
    assert obj1.to_dict()["source_candidate_id"] == "cand_99"
    assert obj1.to_dict()["candidate_id"] == "cand_99"

    # Only candidate_id in dict
    d2 = {"output_id": "out_02", "title": "Title 2", "candidate_id": "cand_88"}
    obj2 = CommentaryOutput.from_dict(d2)
    assert obj2.candidate_id == "cand_88"
    assert obj2.source_candidate_id == "cand_88"
    assert obj2.to_dict()["candidate_id"] == "cand_88"
    assert obj2.to_dict()["source_candidate_id"] == "cand_88"


def test_manifest_serialization_mixed_outputs(tmp_path: Path) -> None:
    """AnalysisManifest serializes and validates cleanly with both legacy and new CommentaryOutputs."""
    ep = SourceEpisode(episode_id="E01", source_video=str(tmp_path / "e01.mp4"), duration_seconds=120.0)
    seg1 = Segment(
        segment_id="seg_01",
        source_clips=[SourceClip(episode_id="E01", source_video=ep.source_video, start=0.0, end=10.0)],
        narration="Legacy narrative",
    )
    seg2 = Segment(
        segment_id="seg_02",
        source_clips=[SourceClip(episode_id="E01", source_video=ep.source_video, start=10.0, end=20.0)],
        narration="Modern narrative",
    )
    legacy_out = CommentaryOutput(output_id="out_leg", title="Legacy Title", segments=[seg1])
    modern_out = CommentaryOutput(
        output_id="out_mod",
        title="Modern Title",
        candidate_id="prop_modern",
        source_candidate_id="prop_modern",
        segments=[seg2],
    )

    manifest = AnalysisManifest(
        project_id="proj_mixed_test",
        analysis_scope=AnalysisScope.SINGLE_EPISODE.value,
        source_episodes=[ep],
        outputs=[legacy_out, modern_out],
    )
    manifest.validate()

    manifest_dict = manifest.to_dict()
    assert len(manifest_dict["outputs"]) == 2
    assert "candidate_id" not in manifest_dict["outputs"][0]
    assert manifest_dict["outputs"][1]["candidate_id"] == "prop_modern"

    restored_manifest = AnalysisManifest.from_dict(manifest_dict)
    restored_manifest.validate()
    assert restored_manifest.outputs[0].candidate_id == ""
    assert restored_manifest.outputs[1].candidate_id == "prop_modern"


# ===========================================================================
# Exact 7 Scenarios from Previous (Rev 7 Matrix)
# ===========================================================================

def test_multigroup_proposal_partitioning_all_calls_bounded_no_outputs_lost(tmp_path: Path) -> None:
    """Scenario 1: Multiple proposals exceeding target ceiling partition into >= 2 groups; all calls <= 500k, no outputs lost."""
    episodes = _make_episodes(tmp_path, count=3)
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
        finalizer_model="prime",
        finalizer_thinking="high",
    )

    # Create 12 large proposals that exceed 480k when formatted together
    proposals = []
    for i in range(1, 13):
        proposals.append(
            CandidateProposal(
                proposal_id=f"prop_{i:02d}",
                title=f"Story Arc Proposal {i}",
                candidate_scope="CROSS_EPISODE",
                episodes=["E01", "E02", "E03"],
                characters=[f"Char_{i}_A", f"Char_{i}_B"],
                description="Detailed narrative description of events " * 800,  # ~32KB each
                editorial_reason="Compelling reason for inclusion " * 400,
                status="accepted",
            )
        )

    conn_res = SeasonConnectionResult(
        cross_episode_links=[
            {
                "link_id": "link_01",
                "episodes": ["E01", "E02"],
                "theme": "Betrayal",
                "summary": "Alice betrays Bob in dramatic fashion" * 100,
            }
        ],
        candidate_proposals=proposals,
        supporting_character_arcs=[],
        rejected_or_merged=[],
    )

    groups = plan_finalizer_groups(episodes, conn_res, settings)
    assert len(groups) >= 2

    # Each group request size must be <= 500,000 and <= 480,000
    for grp in groups:
        assert grp.estimated_bytes <= HARD_PAYLOAD_CEILING
        assert grp.estimated_bytes <= TARGET_PAYLOAD_CEILING

    # Mock client returns distinct outputs for each group
    group_responses = []
    expected_output_count = 0
    for idx, grp in enumerate(groups, start=1):
        resp_outputs = [
            {
                "output_id": f"out_g{idx}_{j}",
                "title": f"Recap Arc Group {idx} Item {j}",
                "candidate_scope": "CROSS_EPISODE",
                "segments": [
                    {
                        "segment_id": f"seg_{idx}_{j}",
                        "narration": f"Narration for group {idx} item {j}",
                        "audio_policy": "duck",
                        "source_clips": [
                            {
                                "episode_id": "E01",
                                "start": 0.0,
                                "end": 15.0,
                            }
                        ],
                    }
                ],
            }
            for j in range(1, 3)
        ]
        expected_output_count += len(resp_outputs)
        group_responses.append({"outputs": resp_outputs})

    mock_client = MockFinalizerClient(group_responses)
    finalizer = CandidateFinalizer(settings=settings, client=mock_client)

    outputs = finalizer.finalize_from_connection(episodes, conn_res)

    # All outputs preserved, none lost
    assert len(outputs) == expected_output_count
    assert len(mock_client.call_history) == len(groups)

    # Check that each actual call was bounded
    for call in mock_client.call_history:
        size = estimate_finalizer_request_size(settings.finalizer_model, call["user_text"], settings.finalizer_thinking)
        assert size <= HARD_PAYLOAD_CEILING

    # Title deduplication across all groups
    unique_titles = {out.sanitized_title for out in outputs}
    assert len(unique_titles) == len(outputs)
    for out in outputs:
        assert out.sanitized_title
        assert len(out.sanitized_title) <= 120


def test_huge_single_field_capped_safely(tmp_path: Path) -> None:
    """Scenario 2: A proposal with a 600KB field is deterministically capped safely without breaking structure or exceeding ceiling."""
    episodes = _make_episodes(tmp_path, count=2)
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
        finalizer_model="prime",
        finalizer_thinking="high",
    )

    huge_text = "Huge description block " * 30_000  # ~690KB
    prop = CandidateProposal(
        proposal_id="prop_huge_01",
        title="Epic Hero Journey",
        candidate_scope="CROSS_EPISODE",
        episodes=["E01", "E02"],
        characters=["Protagonist", "Antagonist"],
        description=huge_text,
        editorial_reason=huge_text,
        status="accepted",
    )

    conn_res = SeasonConnectionResult(
        cross_episode_links=[],
        candidate_proposals=[prop],
        supporting_character_arcs=[],
        rejected_or_merged=[],
    )

    groups = plan_finalizer_groups(episodes, conn_res, settings)
    assert len(groups) >= 1

    for grp in groups:
        assert grp.estimated_bytes <= HARD_PAYLOAD_CEILING
        assert grp.estimated_bytes <= TARGET_PAYLOAD_CEILING
        # Proposal identity preserved
        assert len(grp.proposals) == 1
        p = grp.proposals[0]
        assert p["proposal_id"] == "prop_huge_01"
        assert p["title"] == "Epic Hero Journey"
        assert p["episodes"] == ["E01", "E02"]
        assert "Protagonist" in p["characters"]
        # Description was capped
        assert len(p["description"]) <= 120


def test_dense_single_evidence_no_raw_evidence_in_prompt(tmp_path: Path) -> None:
    """Scenario 3: Dense 1.2MB single evidence does not send raw evidence categories or transcripts in finalizer prompt."""
    episodes = _make_episodes(tmp_path, count=1)
    ep = episodes[0]
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
        finalizer_model="prime",
        finalizer_thinking="high",
    )

    # 1.2MB dense evidence
    heavy_scenes = [
        {
            "start_sec": float(i * 10),
            "end_sec": float((i + 1) * 10),
            "summary": f"Dense scene description number {i} with extensive details " * 20,
            "dialogue_evidence": [f"Spoken line {i} with rich context " * 10],
            "characters": ["Alice", "Bob"],
        }
        for i in range(100)
    ]

    ev = EpisodeEvidence(
        episode_id=ep.episode_id,
        source_video=ep.source_video,
        duration_seconds=1000.0,
        coverage={"ratio": 1.0},
        major_scenes=heavy_scenes,
        dialogue=heavy_scenes,
        character_decisions=heavy_scenes,
    )

    finalizer_resp = {
        "outputs": [
            {
                "output_id": "out_single_01",
                "title": "Alice's Story",
                "candidate_scope": "SINGLE_EPISODE",
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "narration": "Alice starts her quest.",
                        "audio_policy": "duck",
                        "source_clips": [
                            {
                                "episode_id": ep.episode_id,
                                "start": 0.0,
                                "end": 20.0,
                            }
                        ],
                    }
                ],
            }
        ]
    }

    mock_client = MockFinalizerClient([finalizer_resp])
    finalizer = CandidateFinalizer(settings=settings, client=mock_client)

    outputs = finalizer.finalize_single(ep, ev)

    assert len(outputs) == 1
    assert outputs[0].title == "Alice's Story"
    assert len(mock_client.call_history) == 1

    finalizer_call = mock_client.call_history[0]
    user_text = finalizer_call["user_text"]

    # Finalizer prompt must NOT contain raw evidence categories or huge scene transcripts
    assert "major_scenes" not in user_text
    assert "evidence_categories" not in user_text
    assert "dialogue_evidence" not in user_text
    assert "Spoken line 0 with rich context" not in user_text

    # Prompt request size must be bounded
    size = estimate_finalizer_request_size(settings.finalizer_model, user_text, settings.finalizer_thinking)
    assert size <= HARD_PAYLOAD_CEILING


def test_small_connection_single_call_no_unnecessary_capping(tmp_path: Path) -> None:
    """Scenario 4: Small connection produces exactly 1 finalizer group call without unnecessary capping."""
    episodes = _make_episodes(tmp_path, count=2)
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
        finalizer_model="prime",
        finalizer_thinking="high",
    )

    original_desc = "A concise, complete description of the story arc."
    conn_res = SeasonConnectionResult(
        cross_episode_links=[
            {
                "link_id": "link_01",
                "episodes": ["E01", "E02"],
                "theme": "Trust",
                "summary": "Friendship tested.",
            }
        ],
        candidate_proposals=[
            CandidateProposal(
                proposal_id="prop_01",
                title="Friendship Arc",
                candidate_scope="CROSS_EPISODE",
                episodes=["E01", "E02"],
                characters=["Alice", "Bob"],
                description=original_desc,
                editorial_reason="Good pacing.",
                status="accepted",
            )
        ],
        supporting_character_arcs=[],
        rejected_or_merged=[],
    )

    groups = plan_finalizer_groups(episodes, conn_res, settings)
    assert len(groups) == 1
    assert groups[0].group_id == "group_01"

    # Description is NOT capped because FULL compaction fits comfortably
    prop_in_group = groups[0].proposals[0]
    assert prop_in_group["description"] == original_desc


def test_cancellation_before_each_group(tmp_path: Path) -> None:
    """Scenario 5: Cancellation before group 2 stops execution and raises AnalysisCancelledError."""
    episodes = _make_episodes(tmp_path, count=3)
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
        finalizer_model="prime",
        finalizer_thinking="high",
    )

    proposals = [
        CandidateProposal(
            proposal_id=f"prop_{i:02d}",
            title=f"Proposal {i}",
            candidate_scope="CROSS_EPISODE",
            episodes=["E01", "E02"],
            characters=[f"Char_{i}"],
            description="Detailed description " * 3000,  # Forces multi-group
            status="accepted",
        )
        for i in range(1, 10)
    ]

    conn_res = SeasonConnectionResult(
        cross_episode_links=[],
        candidate_proposals=proposals,
        supporting_character_arcs=[],
        rejected_or_merged=[],
    )

    groups = plan_finalizer_groups(episodes, conn_res, settings)
    assert len(groups) >= 2

    cancel_event = threading.Event()

    def _group_1_response(call_record: dict[str, Any]) -> dict[str, Any]:
        # Cancel right after group 1 completes
        cancel_event.set()
        return {
            "outputs": [
                {
                    "output_id": "out_01",
                    "title": "Group 1 Output",
                    "candidate_scope": "CROSS_EPISODE",
                    "segments": [
                        {
                            "segment_id": "seg_01",
                            "narration": "Narration 1",
                            "audio_policy": "duck",
                            "source_clips": [{"episode_id": "E01", "start": 0.0, "end": 10.0}],
                        }
                    ],
                }
            ]
        }

    mock_client = MockFinalizerClient([_group_1_response])
    h_cache = HierarchyCacheManager(tmp_path / "hierarchy")
    finalizer = CandidateFinalizer(settings=settings, client=mock_client, hierarchy_cache=h_cache)

    with pytest.raises(AnalysisCancelledError):
        finalizer.finalize_from_connection(episodes, conn_res, cancel_event=cancel_event)

    # Group 1 was called and completed
    assert len(mock_client.call_history) == 1

    # Group 1 result was safely cached in HierarchyCacheManager
    cached_files = list(h_cache.finalizer_dir.glob("*.finalizer.json"))
    assert len(cached_files) == 1


def test_finalizer_group_cache_and_resume_on_retry(tmp_path: Path) -> None:
    """Scenario 6: When group 1 succeeds and group 2 is cancelled, retry reuses group 1 from cache and runs only group 2."""
    episodes = _make_episodes(tmp_path, count=3)
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
        finalizer_model="prime",
        finalizer_thinking="high",
    )

    proposals = [
        CandidateProposal(
            proposal_id=f"prop_{i:02d}",
            title=f"Proposal {i}",
            candidate_scope="CROSS_EPISODE",
            episodes=["E01", "E02"],
            characters=[f"Char_{i}"],
            description="Detailed proposal text " * 3000,  # Forces multi-group
            status="accepted",
        )
        for i in range(1, 10)
    ]

    conn_res = SeasonConnectionResult(
        cross_episode_links=[],
        candidate_proposals=proposals,
        supporting_character_arcs=[],
        rejected_or_merged=[],
    )

    groups = plan_finalizer_groups(episodes, conn_res, settings)
    assert len(groups) >= 2

    h_cache = HierarchyCacheManager(tmp_path / "hierarchy")

    # --- RUN 1: Group 1 succeeds, Group 2 cancelled ---
    cancel_event = threading.Event()

    def _group_1_handler(call_record: dict[str, Any]) -> dict[str, Any]:
        cancel_event.set()
        return {
            "outputs": [
                {
                    "output_id": "out_grp1",
                    "title": "Group 1 Story",
                    "candidate_scope": "CROSS_EPISODE",
                    "segments": [
                        {
                            "segment_id": "seg_01",
                            "narration": "Group 1 narration.",
                            "audio_policy": "duck",
                            "source_clips": [{"episode_id": "E01", "start": 0.0, "end": 10.0}],
                        }
                    ],
                }
            ]
        }

    client_run1 = MockFinalizerClient([_group_1_handler])
    finalizer_run1 = CandidateFinalizer(settings=settings, client=client_run1, hierarchy_cache=h_cache)

    with pytest.raises(AnalysisCancelledError):
        finalizer_run1.finalize_from_connection(episodes, conn_res, cancel_event=cancel_event)

    assert len(client_run1.call_history) == 1
    # Verify group 1 is in finalizer cache
    cached_files = list(h_cache.finalizer_dir.glob("*.finalizer.json"))
    assert len(cached_files) == 1

    # --- RUN 2 (Retry): Group 1 should be loaded from cache, Group 2 should execute ---
    group_2_resp = {
        "outputs": [
            {
                "output_id": "out_grp2",
                "title": "Group 2 Story",
                "candidate_scope": "CROSS_EPISODE",
                "segments": [
                    {
                        "segment_id": "seg_02",
                        "narration": "Group 2 narration.",
                        "audio_policy": "duck",
                        "source_clips": [{"episode_id": "E02", "start": 0.0, "end": 10.0}],
                    }
                ],
            }
        ]
    }
    remaining_responses = [group_2_resp] * (len(groups) - 1)

    client_run2 = MockFinalizerClient(remaining_responses)
    finalizer_run2 = CandidateFinalizer(settings=settings, client=client_run2, hierarchy_cache=h_cache)

    outputs_run2 = finalizer_run2.finalize_from_connection(episodes, conn_res)

    # Exactly len(groups) - 1 calls made on retry (Group 1 was NOT called because of cache hit!)
    assert len(client_run2.call_history) == len(groups) - 1

    # Outputs from BOTH Group 1 and remaining groups are present!
    output_titles = [out.title for out in outputs_run2]
    assert "Group 1 Story" in output_titles
    assert "Group 2 Story" in output_titles
    assert len(outputs_run2) >= 2


def test_final_plan_cache_and_resume_with_engine(tmp_path: Path) -> None:
    """Scenario 7: End-to-end AnalysisEngine caches final plan; subsequent run reuses plan cache without AI calls."""
    episodes = _make_episodes(tmp_path, count=2)
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
        finalizer_model="prime",
        finalizer_thinking="high",
        scanner_model="sub",
        scanner_thinking="auto",
    )

    ev_cache = EvidenceCacheManager(tmp_path / "evidence")
    h_cache = HierarchyCacheManager(tmp_path / "hierarchy")

    scanner_resp = {
        "major_scenes": [
            {
                "start_sec": 0.0,
                "end_sec": 10.0,
                "summary": "Scene 1",
                "dialogue_evidence": ["Hello"],
                "characters": ["Alice"],
            }
        ]
    }
    conn_resp = {
        "candidate_proposals": [
            {
                "proposal_id": "prop_01",
                "title": "Season Arc",
                "candidate_scope": "SEASON_ARC",
                "episodes": ["E01", "E02"],
                "characters": ["Alice"],
                "description": "Alice's whole story.",
                "status": "accepted",
            }
        ],
        "cross_episode_links": [],
        "supporting_character_arcs": [],
        "rejected_or_merged": [],
    }
    finalizer_resp = {
        "outputs": [
            {
                "output_id": "out_season",
                "title": "Complete Season Recap",
                "candidate_scope": "SEASON_ARC",
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "narration": "The journey begins.",
                        "audio_policy": "duck",
                        "source_clips": [
                            {"episode_id": "E01", "start": 0.0, "end": 10.0},
                            {"episode_id": "E02", "start": 0.0, "end": 10.0},
                        ],
                    }
                ],
            }
        ]
    }

    from tests.helpers_editorial import stage_response

    disc_resp = stage_response("candidate discoverer", episodes=episodes, default=finalizer_resp)
    cons_resp = stage_response("candidate consolidator", episodes=episodes, default=finalizer_resp)

    # RUN 1: Full pipeline execution
    client1 = MockFinalizerClient([scanner_resp, scanner_resp, conn_resp, disc_resp, cons_resp, finalizer_resp])
    engine1 = AnalysisEngine(
        settings=settings,
        client=client1,
        cache_manager=ev_cache,
        hierarchy_cache=h_cache,
    )

    manifest1 = engine1.analyze(
        project_id="proj_plan_cache_test_r8",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        use_final_plan_cache=True,
    )

    assert len(manifest1.outputs) == 1
    assert manifest1.outputs[0].title == "Complete Season Recap"
    assert len(client1.call_history) == 6  # 2 scanner + 1 conn + 1 disc + 1 cons + 1 finalizer

    plan_files = list(engine1.plan_cache_dir.glob("*.json"))
    assert len(plan_files) == 1

    # RUN 2: Immediate plan cache hit with 0 client calls
    client2 = MockFinalizerClient([])
    engine2 = AnalysisEngine(
        settings=settings,
        client=client2,
        cache_manager=ev_cache,
        hierarchy_cache=h_cache,
    )

    manifest2 = engine2.analyze(
        project_id="proj_plan_cache_test_r8_reloaded",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        use_final_plan_cache=True,
    )

    assert len(manifest2.outputs) == 1
    assert manifest2.outputs[0].title == "Complete Season Recap"
    assert manifest2.project_id == "proj_plan_cache_test_r8_reloaded"
    assert len(client2.call_history) == 0


# ===========================================================================
# Rev 8 Candidate Finalization API Tests
# ===========================================================================

def test_finalize_candidates_exact_candidate_tracking(tmp_path: Path) -> None:
    """Scenario 8: finalize_candidates tracks attempted, accepted, and rejected candidate IDs accurately."""
    episodes = _make_episodes(tmp_path, count=2)
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
        finalizer_model="prime",
        finalizer_thinking="high",
    )

    cands = [
        CandidateProposal(
            proposal_id="cand_alpha",
            title="Alpha Hero Journey",
            episodes=["E01", "E02"],
            characters=["Alice"],
            description="Alpha journey description",
            status="keep",
        ),
        CandidateProposal(
            proposal_id="cand_beta",
            title="Beta Supporting Arc",
            episodes=["E01"],
            characters=["Bob"],
            description="Beta journey description",
            status="keep",
        ),
        CandidateProposal(
            proposal_id="cand_gamma",
            title="Gamma Dropped Arc",
            episodes=["E02"],
            characters=["Charlie"],
            description="Gamma journey description",
            status="keep",
        ),
    ]

    # AI returns outputs for cand_alpha and cand_beta only (cand_gamma is not finalized)
    ai_resp = {
        "outputs": [
            {
                "output_id": "out_01",
                "source_candidate_id": "cand_alpha",
                "title": "Alpha Journey Recap",
                "candidate_scope": "CROSS_EPISODE",
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "narration": "Alpha narration.",
                        "audio_policy": "duck",
                        "source_clips": [{"episode_id": "E01", "start": 0.0, "end": 10.0}],
                    }
                ],
            },
            {
                "output_id": "out_02",
                "source_candidate_id": "cand_beta",
                "title": "Beta Arc Recap",
                "candidate_scope": "SINGLE_EPISODE",
                "segments": [
                    {
                        "segment_id": "seg_02",
                        "narration": "Beta narration.",
                        "audio_policy": "duck",
                        "source_clips": [{"episode_id": "E01", "start": 10.0, "end": 20.0}],
                    }
                ],
            },
        ]
    }

    mock_client = MockFinalizerClient([ai_resp])
    result = finalize_candidates(
        candidates=cands,
        source_episodes=episodes,
        settings=settings,
        client=mock_client,
        return_diagnostics=True,
    )

    assert isinstance(result, FinalizationResult)
    assert result.completed is True
    assert result.attempted_candidates == ["cand_alpha", "cand_beta", "cand_gamma"]
    assert result.accepted_candidate_ids == ["cand_alpha", "cand_beta"]
    assert result.rejected_candidate_ids == ["cand_gamma"]
    assert result.accepted_count == 2
    assert result.rejected_count == 1
    assert result.attempted_count == 3
    assert result.outputs_count == 2

    # Check output properties
    out_alpha = next(o for o in result.outputs if o.source_candidate_id == "cand_alpha")
    assert out_alpha.candidate_id == "cand_alpha"
    assert out_alpha.title == "Alpha Journey Recap"

    out_beta = next(o for o in result.outputs if o.source_candidate_id == "cand_beta")
    assert out_beta.candidate_id == "cand_beta"
    assert out_beta.title == "Beta Arc Recap"


def test_finalize_candidates_omitted_cid_single_vs_multi_candidate(tmp_path: Path) -> None:
    """Scenario 9: Omitted source_candidate_id defaults cleanly for single-candidate group, but rejects for multi-candidate group."""
    episodes = _make_episodes(tmp_path, count=1)
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
        finalizer_model="prime",
    )

    # Case A: Single candidate in group - AI omitted source_candidate_id
    cand_single = [
        CandidateProposal(
            proposal_id="cand_only_one",
            title="Sole Candidate",
            episodes=["E01"],
            characters=["Solo"],
            description="Description",
        )
    ]
    resp_no_cid = {
        "outputs": [
            {
                "output_id": "out_solo",
                "title": "Solo Journey",
                # source_candidate_id omitted!
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "narration": "Solo narration.",
                        "source_clips": [{"episode_id": "E01", "start": 0.0, "end": 10.0}],
                    }
                ],
            }
        ]
    }
    client_a = MockFinalizerClient([resp_no_cid])
    res_a = finalize_candidates(
        candidates=cand_single,
        source_episodes=episodes,
        settings=settings,
        client=client_a,
    )
    assert res_a.accepted_candidate_ids == ["cand_only_one"]
    assert len(res_a.outputs) == 1
    assert res_a.outputs[0].source_candidate_id == "cand_only_one"

    # Case B: Multiple candidates in group - AI omitted source_candidate_id
    cand_multi = [
        CandidateProposal(proposal_id="cand_one", title="One", episodes=["E01"]),
        CandidateProposal(proposal_id="cand_two", title="Two", episodes=["E01"]),
    ]
    client_b = MockFinalizerClient([resp_no_cid])
    res_b = finalize_candidates(
        candidates=cand_multi,
        source_episodes=episodes,
        settings=settings,
        client=client_b,
    )
    # Output rejected because candidate reference was ambiguous
    assert len(res_b.outputs) == 0
    assert len(res_b.errors) >= 1
    assert "omitted source_candidate_id" in res_b.errors[0]
    assert "cand_one" in res_b.rejected_candidate_ids
    assert "cand_two" in res_b.rejected_candidate_ids


def test_finalize_candidates_unknown_cid_rejection(tmp_path: Path) -> None:
    """Scenario 10: AI response referencing unknown candidate ID is rejected with error."""
    episodes = _make_episodes(tmp_path, count=1)
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
    )
    cands = [CandidateProposal(proposal_id="cand_valid", title="Valid", episodes=["E01"])]
    resp_unknown = {
        "outputs": [
            {
                "output_id": "out_hallucinated",
                "source_candidate_id": "cand_hallucinated_ghost",
                "title": "Ghost Story",
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "narration": "Ghost narration.",
                        "source_clips": [{"episode_id": "E01", "start": 0.0, "end": 10.0}],
                    }
                ],
            }
        ]
    }
    client = MockFinalizerClient([resp_unknown])
    res = finalize_candidates(
        candidates=cands,
        source_episodes=episodes,
        settings=settings,
        client=client,
    )
    assert len(res.outputs) == 0
    assert "cand_valid" in res.rejected_candidate_ids
    assert len(res.errors) >= 1
    assert "unknown candidate_id 'cand_hallucinated_ghost'" in res.errors[0]


def test_finalize_candidates_cache_key_and_resume(tmp_path: Path) -> None:
    """Scenario 11: finalize_candidates caches successful group result with candidate IDs in key; subsequent run resumes from cache."""
    episodes = _make_episodes(tmp_path, count=2)
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
        finalizer_model="prime",
        finalizer_thinking="high",
    )
    h_cache = HierarchyCacheManager(tmp_path / "hierarchy")

    cands = [
        CandidateProposal(
            proposal_id="cand_res_01",
            title="Resume Arc",
            episodes=["E01", "E02"],
            characters=["Hero"],
            description="Resume test description",
        )
    ]

    ai_resp = {
        "outputs": [
            {
                "output_id": "out_res",
                "source_candidate_id": "cand_res_01",
                "title": "Resume Arc Story",
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "narration": "Heroic journey resume.",
                        "audio_policy": "duck",
                        "source_clips": [{"episode_id": "E01", "start": 0.0, "end": 15.0}],
                    }
                ],
            }
        ]
    }

    # RUN 1: AI called and result cached
    client1 = MockFinalizerClient([ai_resp])
    res1 = finalize_candidates(
        candidates=cands,
        source_episodes=episodes,
        settings=settings,
        client=client1,
        hierarchy_cache=h_cache,
    )
    assert len(client1.call_history) == 1
    assert res1.accepted_candidate_ids == ["cand_res_01"]

    cached_files = list(h_cache.finalizer_dir.glob("*.finalizer.json"))
    assert len(cached_files) == 1

    # RUN 2: Cache hit; AI not called
    client2 = MockFinalizerClient([])
    res2 = finalize_candidates(
        candidates=cands,
        source_episodes=episodes,
        settings=settings,
        client=client2,
        hierarchy_cache=h_cache,
    )
    assert len(client2.call_history) == 0  # 0 AI calls!
    assert res2.accepted_candidate_ids == ["cand_res_01"]
    assert res2.outputs[0].title == "Resume Arc Story"


def test_finalize_candidates_cancellation(tmp_path: Path) -> None:
    """Scenario 12: finalize_candidates raises AnalysisCancelledError when cancel_event is set."""
    episodes = _make_episodes(tmp_path, count=2)
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        api_key="secret",
    )
    cands = [CandidateProposal(proposal_id="cand_cancel", title="Cancel Me", episodes=["E01"])]

    cancel_event = threading.Event()
    cancel_event.set()

    mock_client = MockFinalizerClient([])
    with pytest.raises(AnalysisCancelledError):
        finalize_candidates(
            candidates=cands,
            source_episodes=episodes,
            settings=settings,
            client=mock_client,
            cancel_event=cancel_event,
        )


def test_finalize_candidates_offline_mode(tmp_path: Path) -> None:
    """Scenario 13: finalize_candidates offline mode handles legacy_wrapper flag properly."""
    episodes = _make_episodes(tmp_path, count=2)
    settings = AppSettings(
        gateway_enabled=False,  # Offline
        api_endpoint="",
    )
    cands = [
        CandidateProposal(proposal_id="cand_off_1", title="Off 1", episodes=["E01"]),
        CandidateProposal(proposal_id="cand_off_2", title="Off 2", episodes=["E02"]),
    ]

    # With legacy_wrapper=True, produces deterministic offline outputs with candidate IDs assigned
    res_offline = finalize_candidates(
        candidates=cands,
        source_episodes=episodes,
        settings=settings,
        client=None,
        legacy_wrapper=True,
    )
    assert isinstance(res_offline, FinalizationResult)
    assert res_offline.completed is True
    assert len(res_offline.outputs) >= 1
    assert res_offline.diagnostics.get("offline") is True
    assert res_offline.outputs[0].source_candidate_id != ""

    # With legacy_wrapper=False, returns clean completed result with offline_disabled diagnostic
    res_disabled = finalize_candidates(
        candidates=cands,
        source_episodes=episodes,
        settings=settings,
        client=None,
        legacy_wrapper=False,
    )
    assert isinstance(res_disabled, FinalizationResult)
    assert len(res_disabled.outputs) == 0
    assert res_disabled.rejected_candidate_ids == ["cand_off_1", "cand_off_2"]
    assert res_disabled.diagnostics.get("offline_disabled") is True


def test_finalization_result_dataclass_methods_and_properties() -> None:
    """Scenario 14: FinalizationResult to_dict, from_dict, to_json, from_json, and count properties work cleanly."""
    seg = Segment(
        segment_id="seg_01",
        source_clips=[SourceClip(episode_id="E01", source_video="e01.mp4", start=0.0, end=10.0)],
        narration="Narration",
    )
    out = CommentaryOutput(
        output_id="out_01",
        title="Title 1",
        candidate_id="cand_01",
        source_candidate_id="cand_01",
        segments=[seg],
    )
    res = FinalizationResult(
        outputs=[out],
        attempted_candidates=["cand_01", "cand_02"],
        accepted_candidate_ids=["cand_01"],
        rejected_candidate_ids=["cand_02"],
        errors=["Candidate 2 omitted"],
        completed=True,
        diagnostics={"test": True},
    )

    assert res.accepted_count == 1
    assert res.rejected_count == 1
    assert res.attempted_count == 2
    assert res.outputs_count == 1

    d = res.to_dict()
    assert d["completed"] is True
    assert len(d["outputs"]) == 1
    assert d["outputs"][0]["candidate_id"] == "cand_01"

    restored = FinalizationResult.from_dict(d)
    assert restored.completed is True
    assert restored.accepted_candidate_ids == ["cand_01"]
    assert restored.rejected_candidate_ids == ["cand_02"]
    assert restored.outputs[0].candidate_id == "cand_01"

    json_str = res.to_json()
    from_json_res = FinalizationResult.from_json(json_str)
    assert from_json_res.accepted_candidate_ids == ["cand_01"]
    assert from_json_res.errors == ["Candidate 2 omitted"]
