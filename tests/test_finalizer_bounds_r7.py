"""Tests for finalizer-tests-r7 (Objective rev 7).

Matrix:
1. Multi-group proposal partitioning with all calls <= 500,000 bytes and <= 480,000 bytes, no outputs lost.
2. Huge single field cap (600KB field capped safely without breaking structure or exceeding ceiling).
3. Dense 1.2MB single evidence: no raw evidence categories/transcript in prompt, all calls bounded.
4. Small connection: single finalizer call without unnecessary capping.
5. Mid-group cancellation handling (cancellation before group 2 stops execution, raises AnalysisCancelledError).
6. Finalizer group cache and resume on retry (reuse group 1 from cache, run group 2, all outputs returned).
7. Final plan cache and resume consideration with AnalysisEngine.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any

import pytest

from toolrecap_v2.analyzer.connection import (
    HARD_PAYLOAD_CEILING,
    TARGET_PAYLOAD_CEILING,
    CandidateProposal,
    SeasonConnectionResult,
)
from toolrecap_v2.analyzer.engine import AnalysisEngine
from toolrecap_v2.analyzer.errors import AnalysisCancelledError, AnalysisError
from toolrecap_v2.analyzer.finalizer import (
    CandidateFinalizer,
    FinalizerGroupPlan,
    estimate_finalizer_request_size,
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
from toolrecap_v2.domain.enums import AnalysisScope, CandidateScope
from toolrecap_v2.domain.models import (
    CommentaryOutput,
    EpisodeEvidence,
    Segment,
    SourceClip,
    SourceEpisode,
)
from toolrecap_v2.settings import AppSettings


class MockFinalizerClient:
    """Mock OpenAICompatibleClient for finalizer R7 testing."""

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


def test_multigroup_proposal_partitioning_all_calls_bounded_no_outputs_lost(tmp_path: Path) -> None:
    """Multiple proposals exceeding target ceiling partition into >= 2 groups; all calls <= 500k, no outputs lost."""
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

    # All outputs preserved, none lost!
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
    """A proposal with a 600KB field is deterministically capped safely without breaking structure or exceeding ceiling."""
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
    """Dense 1.2MB single evidence does not send raw evidence categories or transcripts in finalizer prompt."""
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

    # Mock gateway client for connector + finalizer
    conn_resp = {
        "candidate_proposals": [
            {
                "proposal_id": "prop_01",
                "title": "Single Episode Journey",
                "candidate_scope": "SINGLE_EPISODE",
                "episodes": [ep.episode_id],
                "characters": ["Alice"],
                "description": "Short summary of Alice's journey",
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
    """Small connection produces exactly 1 finalizer group call without unnecessary capping."""
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
    """Cancellation before group 2 stops execution and raises AnalysisCancelledError."""
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
    """When group 1 succeeds and group 2 is cancelled, retry reuses group 1 from cache and runs only group 2."""
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
    # Provide responses ONLY for the remaining groups (e.g. len(groups) - 1 responses)
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
    """End-to-end AnalysisEngine caches final plan; subsequent run reuses plan cache without AI calls."""
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

    # Scanner responses for E01 and E02
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
    # Connection response
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
    # Finalizer response
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
        project_id="proj_plan_cache_test",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        use_final_plan_cache=True,
    )

    assert len(manifest1.outputs) == 1
    assert manifest1.outputs[0].title == "Complete Season Recap"
    assert len(client1.call_history) == 6  # 2 scanner + 1 conn + 1 disc + 1 cons + 1 finalizer

    # Verify plan cache file exists
    plan_files = list(engine1.plan_cache_dir.glob("*.json"))
    assert len(plan_files) == 1

    # RUN 2: Immediate plan cache hit with 0 client calls
    client2 = MockFinalizerClient([])  # Empty responses; any call will fail!
    engine2 = AnalysisEngine(
        settings=settings,
        client=client2,
        cache_manager=ev_cache,
        hierarchy_cache=h_cache,
    )

    manifest2 = engine2.analyze(
        project_id="proj_plan_cache_test_reloaded",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        use_final_plan_cache=True,
    )

    assert len(manifest2.outputs) == 1
    assert manifest2.outputs[0].title == "Complete Season Recap"
    assert manifest2.project_id == "proj_plan_cache_test_reloaded"
    # Zero calls made!
    assert len(client2.call_history) == 0
