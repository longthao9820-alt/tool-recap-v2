"""Comprehensive tests for episode evidence scanner, season connection, and candidate finalizer.

Covers all mandatory invariants:
- Single 0/1/multiple outputs without quota
- Simulated E01 setup, E02 development, E03 supporting, E04 consequence, E05 payoff
- Cross output E01/E03/E05
- Supporting candidate retained despite protagonist frequency
- All outputs parsed without slicing
- Incomplete coverage: default fail vs allow_incomplete=True
- Selective cache invalidation (modifying E05 reuses E01-E04)
- Cancellation during Scanner, Season Connection, and Finalizer
- Rejection of malformed/hallucinated references
- Strictly no network calls
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
import pytest

from toolrecap_v2.analyzer import (
    AnalysisCancelledError,
    AnalysisEngine,
    AnalysisError,
    AnalysisPhase,
    CandidateFinalizer,
    CandidateProposal,
    CoverageIncompleteError,
    EVIDENCE_CATEGORIES,
    EvidenceScanner,
    FINALIZER_SYSTEM_PROMPT,
    SCANNER_SYSTEM_PROMPT,
    SEASON_CONNECTION_SYSTEM_PROMPT,
    SeasonConnectionResult,
    SeasonConnector,
    compute_scanner_config_version,
    run_analysis,
)
from toolrecap_v2.analyzer.connection import (
    check_payload_size,
    partition_season_batches,
    validate_batch_response_schema,
)
from toolrecap_v2.analyzer.engine import compute_final_plan_cache_key
from toolrecap_v2.api_client import APIError, OpenAICompatibleClient
from toolrecap_v2.domain import (
    AnalysisManifest,
    AnalysisScope,
    AudioPolicy,
    CandidateScope,
    CommentaryOutput,
    EpisodeEvidence,
    EvidenceCacheManager,
    HierarchyCacheManager,
    Segment,
    SourceClip,
    SourceEpisode,
    ValidationError,
    compute_cache_key,
)
from toolrecap_v2.domain.cache import (
    compute_batch_cache_key,
    compute_connection_cache_key,
    compute_merge_cache_key,
    compute_summary_cache_key,
)
from toolrecap_v2.settings import AppSettings
from toolrecap_v2.subtitles.models import SubtitleCue


# ---------------------------------------------------------------------------
# Helpers and Mock Client
# ---------------------------------------------------------------------------

class MockAIClient:
    """Mock OpenAICompatibleClient for testing without network."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
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
            return resp

        # Default fallback response depending on prompt
        if "evidence scanner" in system.lower():
            return {
                "major_scenes": [{"start_ms": 0, "end_ms": 10000, "summary": "Scene 1"}],
                "dialogue": [{"start_ms": 2000, "end_ms": 8000, "speaker": "A", "quote": "Hello"}],
                "character_decisions": [],
                "supporting_developments": [],
                "relationships": [],
                "reveals": [],
                "reversals": [],
                "failures": [],
                "consequences": [],
                "performance_moments": [],
                "setup_payoff": [],
                "unresolved": [],
                "conflicts": [],
                "subplots": [],
                "strengths_weaknesses": [],
            }
        elif "season narrative architect" in system.lower():
            return {
                "cross_episode_links": [{"thread_id": "t1", "episodes": ["E01", "E02"]}],
                "candidate_proposals": [
                    {
                        "proposal_id": "prop_01",
                        "title": "Season Journey",
                        "candidate_scope": "SEASON_ARC",
                        "episodes": ["E01", "E02"],
                        "status": "keep",
                    }
                ],
                "supporting_character_arcs": [],
                "rejected_or_merged": [],
            }
        else:
            return {"outputs": []}


def make_dummy_episode(ep_id: str, duration: float = 60.0, tmp_path: Path | None = None) -> SourceEpisode:
    if tmp_path:
        f = tmp_path / f"{ep_id.lower()}.mp4"
        if not f.exists():
            f.write_bytes(b"dummy_video_bytes")
        v_path = str(f)
    else:
        v_path = f"C:/media/{ep_id.lower()}.mp4"

    return SourceEpisode(
        episode_id=ep_id,
        source_video=v_path,
        season_number=1,
        episode_number=int(ep_id.replace("E", "") or "1"),
        title=f"Episode {ep_id}",
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Test 1: Single Episode 0 / 1 / Multiple Outputs (No Quota Slicing)
# ---------------------------------------------------------------------------

def test_single_episode_zero_outputs(tmp_path: Path) -> None:
    """Empty outputs [] is valid when no candidate meets quality."""
    ep = make_dummy_episode("E01", duration=100.0, tmp_path=tmp_path)
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")

    # Scanner returns evidence, Connector returns batch proposals, Finalizer returns empty outputs []
    scanner_resp = {cat: [] for cat in EVIDENCE_CATEGORIES}
    conn_resp = {
        "cross_episode_links": [],
        "candidate_proposals": [],
        "supporting_character_arcs": [],
        "rejected_or_merged": [],
    }
    finalizer_resp = {"outputs": []}

    mock_client = MockAIClient([scanner_resp, conn_resp, finalizer_resp])
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")

    engine = AnalysisEngine(settings=settings, client=mock_client, cache_manager=cache_mgr)
    manifest = engine.analyze(
        project_id="test_proj",
        episodes=[ep],
        scope=AnalysisScope.SINGLE_EPISODE,
    )

    assert manifest.project_id == "test_proj"
    assert manifest.analysis_scope == AnalysisScope.SINGLE_EPISODE.value
    assert len(manifest.outputs) == 0
    # Strict validation passes
    manifest.validate()


def test_single_episode_one_output(tmp_path: Path) -> None:
    """Single episode generating exactly 1 output."""
    ep = make_dummy_episode("E01", duration=100.0, tmp_path=tmp_path)
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")

    scanner_resp = {cat: [] for cat in EVIDENCE_CATEGORIES}
    conn_resp = {
        "cross_episode_links": [],
        "candidate_proposals": [
            {
                "proposal_id": "prop_01",
                "title": "The Awakening",
                "candidate_scope": "SINGLE_EPISODE",
                "episodes": ["E01"],
                "status": "keep",
            }
        ],
        "supporting_character_arcs": [],
        "rejected_or_merged": [],
    }
    finalizer_resp = {
        "outputs": [
            {
                "output_id": "out_01",
                "title": "The Awakening",
                "candidate_scope": "SINGLE_EPISODE",
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "source_clips": [
                            {"episode_id": "E01", "source_video": ep.source_video, "start": 0.0, "end": 20.0}
                        ],
                        "narration": "The story opens with a mysterious discovery.",
                        "audio_policy": "mute",
                    }
                ],
            }
        ]
    }

    mock_client = MockAIClient([scanner_resp, conn_resp, finalizer_resp])
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")

    engine = AnalysisEngine(settings=settings, client=mock_client, cache_manager=cache_mgr)
    manifest = engine.analyze(
        project_id="test_proj",
        episodes=[ep],
        scope=AnalysisScope.SINGLE_EPISODE,
    )

    assert len(manifest.outputs) == 1
    assert manifest.outputs[0].output_id == "out_01"
    assert manifest.outputs[0].title == "The Awakening"
    assert manifest.outputs[0].sanitized_title == "The Awakening"
    assert len(manifest.outputs[0].segments) == 1
    manifest.validate()


def test_single_episode_multiple_outputs_no_quota_slicing(tmp_path: Path) -> None:
    """Single episode generating 4 outputs; all 4 must be preserved without artificial slicing."""
    ep = make_dummy_episode("E01", duration=200.0, tmp_path=tmp_path)
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")

    scanner_resp = {cat: [] for cat in EVIDENCE_CATEGORIES}
    conn_resp = {
        "cross_episode_links": [],
        "candidate_proposals": [
            {
                "proposal_id": f"prop_{i:02d}",
                "title": f"Candidate Story {i}",
                "candidate_scope": "SINGLE_SCENE" if i % 2 == 0 else "SINGLE_EPISODE",
                "episodes": ["E01"],
                "status": "keep",
            }
            for i in range(1, 5)
        ],
        "supporting_character_arcs": [],
        "rejected_or_merged": [],
    }
    finalizer_resp = {
        "outputs": [
            {
                "output_id": f"out_{i:02d}",
                "title": f"Candidate Story {i}",
                "candidate_scope": "SINGLE_SCENE" if i % 2 == 0 else "SINGLE_EPISODE",
                "segments": [
                    {
                        "segment_id": f"seg_{i}_01",
                        "source_clips": [
                            {
                                "episode_id": "E01",
                                "source_video": ep.source_video,
                                "start": float(i * 10),
                                "end": float(i * 10 + 15),
                            }
                        ],
                        "narration": f"Narration for candidate {i}.",
                        "audio_policy": "mute",
                    }
                ],
            }
            for i in range(1, 5)  # 4 outputs
        ]
    }

    mock_client = MockAIClient([scanner_resp, conn_resp, finalizer_resp])
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")

    engine = AnalysisEngine(settings=settings, client=mock_client, cache_manager=cache_mgr)
    manifest = engine.analyze(
        project_id="test_proj",
        episodes=[ep],
        scope=AnalysisScope.SINGLE_EPISODE,
    )

    assert len(manifest.outputs) == 4
    for i, out in enumerate(manifest.outputs, start=1):
        assert out.output_id == f"out_{i:02d}"
        assert out.title == f"Candidate Story {i}"
    manifest.validate()


# ---------------------------------------------------------------------------
# Test 2: Simulated E01-E05 Season Connection Prompt and Setup/Payoff
# ---------------------------------------------------------------------------

def test_simulated_e01_to_e05_season_connection_prompt(tmp_path: Path) -> None:
    """Simulate E01 setup, E02 development, E03 supporting, E04 consequence, E05 payoff.

    Verifies:
    - All 5 episodes scanned first.
    - Barrier holds before season connection pass.
    - Season connection prompt receives all 5 episodes' evidence.
    - Prompts mandate cross-episode links, supporting character equal consideration, candidate scopes, no quota.
    """
    episodes = [make_dummy_episode(f"E0{i}", duration=300.0, tmp_path=tmp_path) for i in range(1, 6)]
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")

    # Scanner responses for E01-E05
    scanner_responses = [
        # E01: setup
        {
            "major_scenes": [{"start_ms": 10000, "end_ms": 30000, "summary": "Setup of the conspiracy"}],
            "setup_payoff": [{"start_ms": 15000, "end_ms": 25000, "type": "setup", "detail": "Hidden ledger planted"}],
            "dialogue": [{"start_ms": 20000, "end_ms": 25000, "speaker": "Protagonist", "quote": "We must hide the ledger"}],
        },
        # E02: development
        {
            "major_scenes": [{"start_ms": 10000, "end_ms": 30000, "summary": "Investigation develops"}],
            "character_decisions": [{"start_ms": 12000, "end_ms": 20000, "character": "Protagonist", "decision": "Search the archive"}],
        },
        # E03: supporting storyline
        {
            "major_scenes": [{"start_ms": 10000, "end_ms": 30000, "summary": "Deputy Adams discovers discrepancy"}],
            "supporting_developments": [{"start_ms": 11000, "end_ms": 28000, "character": "Deputy Adams", "development": "Uncovers shadow accounts"}],
        },
        # E04: consequence
        {
            "major_scenes": [{"start_ms": 10000, "end_ms": 30000, "summary": "Consequences of the discovery"}],
            "consequences": [{"start_ms": 12000, "end_ms": 25000, "consequence": "The precinct is attacked"}],
            "failures": [{"start_ms": 15000, "end_ms": 22000, "character": "Protagonist", "failure": "Failed to protect backup records"}],
        },
        # E05: payoff
        {
            "major_scenes": [{"start_ms": 10000, "end_ms": 30000, "summary": "Final confrontation and resolution"}],
            "setup_payoff": [{"start_ms": 15000, "end_ms": 25000, "type": "payoff", "detail": "Hidden ledger from E01 exposes the villain"}],
            "reveals": [{"start_ms": 18000, "end_ms": 24000, "reveal": "Villain identity confirmed"}],
        },
    ]

    # Batch 1 response (E01-E03)
    batch1_resp = {
        "batch_id": "batch_E01_E03",
        "cross_episode_links": [
            {
                "thread_id": "thread_ledger_early",
                "theme": "The Hidden Ledger Setup",
                "episodes": ["E01", "E03"],
                "summary": "Setup in E01, uncovered in E03.",
            }
        ],
        "candidate_proposals": [
            {
                "proposal_id": "prop_ledger_part1",
                "title": "The Ledger Setup",
                "candidate_scope": "CROSS_EPISODE",
                "episodes": ["E01", "E03"],
                "editorial_reason": "Setup and early investigation.",
                "status": "keep",
            }
        ],
        "supporting_character_arcs": [
            {
                "character": "Deputy Adams",
                "arc_summary": "Uncovers critical evidence in E03.",
                "episodes": ["E03"],
                "has_dedicated_candidate": True,
            }
        ],
        "rejected_or_merged": [],
    }

    # Batch 2 response (E04-E05)
    batch2_resp = {
        "batch_id": "batch_E04_E05",
        "cross_episode_links": [
            {
                "thread_id": "thread_ledger_late",
                "theme": "The Ledger Payoff",
                "episodes": ["E04", "E05"],
                "summary": "Consequences and payoff.",
            }
        ],
        "candidate_proposals": [
            {
                "proposal_id": "prop_ledger_part2",
                "title": "The Ledger Payoff",
                "candidate_scope": "CROSS_EPISODE",
                "episodes": ["E04", "E05"],
                "editorial_reason": "Consequence and resolution.",
                "status": "keep",
            }
        ],
        "supporting_character_arcs": [],
        "rejected_or_merged": [],
    }

    # Cross-batch merge response
    merge_resp = {
        "cross_episode_links": [
            {
                "thread_id": "thread_ledger",
                "theme": "The Hidden Ledger Arc",
                "episodes": ["E01", "E03", "E05"],
                "summary": "Setup in E01, uncovered in E03, paid off in E05.",
            }
        ],
        "candidate_proposals": [
            {
                "proposal_id": "prop_ledger",
                "title": "The Ledger Conspiracy",
                "candidate_scope": "CROSS_EPISODE",
                "episodes": ["E01", "E03", "E05"],
                "editorial_reason": "Clear setup to payoff narrative arc.",
                "status": "keep",
            }
        ],
        "supporting_character_arcs": [
            {
                "character": "Deputy Adams",
                "arc_summary": "Uncovers critical evidence in E03.",
                "episodes": ["E03"],
                "has_dedicated_candidate": True,
            }
        ],
        "rejected_or_merged": [],
    }

    # Finalizer response
    finalizer_resp = {
        "outputs": [
            {
                "output_id": "out_ledger_arc",
                "title": "The Ledger Conspiracy",
                "candidate_scope": "CROSS_EPISODE",
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "source_clips": [
                            {"episode_id": "E01", "source_video": episodes[0].source_video, "start": 10.0, "end": 30.0}
                        ],
                        "narration": "In the beginning, the ledger is concealed.",
                        "audio_policy": "mute",
                    },
                    {
                        "segment_id": "seg_02",
                        "source_clips": [
                            {"episode_id": "E03", "source_video": episodes[2].source_video, "start": 10.0, "end": 30.0}
                        ],
                        "narration": "Deputy Adams discovers the discrepancies.",
                        "audio_policy": "mute",
                    },
                    {
                        "segment_id": "seg_03",
                        "source_clips": [
                            {"episode_id": "E05", "source_video": episodes[4].source_video, "start": 15.0, "end": 28.0}
                        ],
                        "narration": "Finally, the ledger delivers total exposure.",
                        "audio_policy": "mute",
                    },
                ],
            }
        ]
    }

    mock_client = MockAIClient(scanner_responses + [batch1_resp, batch2_resp, merge_resp, finalizer_resp])
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")

    phase_log: list[tuple[AnalysisPhase, str]] = []

    def on_phase(phase: AnalysisPhase, identifier: str, details: dict[str, Any]) -> None:
        phase_log.append((phase, identifier))

    manifest = run_analysis(
        project_id="season_conspiracy",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        settings=settings,
        client=mock_client,
        cache_manager=cache_mgr,
        on_phase=on_phase,
    )

    # 1. Verify phase sequence: all 5 episodes scanned before barrier and connection pass
    scanner_phases = [p for p in phase_log if p[0] == AnalysisPhase.SCANNER]
    assert len(scanner_phases) >= 5

    barrier_idx = [i for i, p in enumerate(phase_log) if p[0] == AnalysisPhase.SEASON_BARRIER]
    connecting_idx = [i for i, p in enumerate(phase_log) if p[0] == AnalysisPhase.SEASON_CONNECTING]
    assert barrier_idx and connecting_idx
    assert barrier_idx[0] < connecting_idx[0]

    # 2. Inspect the Season Connection calls (hierarchical: batch 1, batch 2, and cross-batch merge)
    conn_calls = [c for c in mock_client.call_history if "season narrative architect" in c["system"].lower()]
    assert len(conn_calls) == 3  # Batch 1 (E01-E03), Batch 2 (E04-E05), Merge (Round final)

    # Batch 1 call: compact summaries, no raw evidence_categories
    b1_call = conn_calls[0]
    assert "CROSS-EPISODE LINKS" in b1_call["system"]
    assert "SUPPORTING/MINOR CHARACTERS" in b1_call["system"]
    assert "NO QUOTA" in b1_call["system"]
    assert "E01" in b1_call["user_text"]
    assert "E02" in b1_call["user_text"]
    assert "E03" in b1_call["user_text"]
    assert "E05" not in b1_call["user_text"]
    assert "evidence_categories" not in b1_call["user_text"]
    assert "conspiracy" in b1_call["user_text"] or "ledger" in b1_call["user_text"]

    # Batch 2 call: remaining episode summary
    b2_call = conn_calls[1]
    assert "E05" in b2_call["user_text"]
    assert '"episode_id": "E01"' not in b2_call["user_text"]
    assert "evidence_categories" not in b2_call["user_text"]

    # Merge call: contains batch results from both batches
    merge_call = conn_calls[2]
    assert "batch_results" in merge_call["user_text"]
    assert "evidence_categories" not in merge_call["user_text"]

    # 3. Output validation
    assert len(manifest.outputs) == 1
    out = manifest.outputs[0]
    assert out.candidate_scope == CandidateScope.CROSS_EPISODE.value
    assert len(out.segments) == 3
    manifest.validate()


# ---------------------------------------------------------------------------
# Test 3: Cross Output Clipping E01, E03, and E05
# ---------------------------------------------------------------------------

def test_cross_output_clipping_e01_e03_e05(tmp_path: Path) -> None:
    """Verify an output containing source clips from E01, E03, and E05 is properly parsed and validated."""
    ep1 = make_dummy_episode("E01", duration=200.0, tmp_path=tmp_path)
    ep3 = make_dummy_episode("E03", duration=250.0, tmp_path=tmp_path)
    ep5 = make_dummy_episode("E05", duration=300.0, tmp_path=tmp_path)
    episodes = [ep1, ep3, ep5]

    finalizer = CandidateFinalizer()
    raw_ai = {
        "outputs": [
            {
                "output_id": "out_cross_135",
                "title": "Cross Episode Investigation Arc",
                "candidate_scope": "CROSS_EPISODE",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [
                            {"episode_id": "E01", "source_video": ep1.source_video, "start": 5.0, "end": 25.0}
                        ],
                        "narration": "Clip from Episode 1.",
                        "audio_policy": "mute",
                    },
                    {
                        "segment_id": "s2",
                        "source_clips": [
                            {"episode_id": "E03", "source_video": ep3.source_video, "start": 50.0, "end": 75.0}
                        ],
                        "narration": "Clip from Episode 3.",
                        "audio_policy": "duck",
                    },
                    {
                        "segment_id": "s3",
                        "source_clips": [
                            {"episode_id": "E05", "source_video": ep5.source_video, "start": 100.0, "end": 140.0}
                        ],
                        "narration": "Clip from Episode 5.",
                        "audio_policy": "mute",
                    },
                ],
            }
        ]
    }

    outputs = finalizer.parse_outputs(raw_ai, episodes)
    assert len(outputs) == 1
    out = outputs[0]
    assert out.candidate_scope == CandidateScope.CROSS_EPISODE.value
    assert len(out.segments) == 3

    # Check clips
    c1 = out.segments[0].source_clips[0]
    assert c1.episode_id == "E01"
    assert c1.start == 5.0
    assert c1.end == 25.0

    c2 = out.segments[1].source_clips[0]
    assert c2.episode_id == "E03"
    assert c2.start == 50.0
    assert c2.end == 75.0

    c3 = out.segments[2].source_clips[0]
    assert c3.episode_id == "E05"
    assert c3.start == 100.0
    assert c3.end == 140.0

    # Build manifest and validate
    manifest = AnalysisManifest(
        project_id="test_proj",
        analysis_scope=AnalysisScope.SEASON.value,
        source_episodes=episodes,
        outputs=outputs,
    )
    manifest.validate()


# ---------------------------------------------------------------------------
# Test 4: Supporting Candidate Retained Despite Protagonist Frequency
# ---------------------------------------------------------------------------

def test_supporting_candidate_retained_despite_protagonist_frequency(tmp_path: Path) -> None:
    """Protagonist appears in almost all events, while minor character has key moments.

    Ensure supporting character candidate is retained and parsed.
    """
    episodes = [make_dummy_episode(f"E0{i}", duration=200.0, tmp_path=tmp_path) for i in range(1, 4)]

    # Finalizer returns two outputs: one for protagonist arc, one dedicated to supporting character
    raw_ai = {
        "outputs": [
            {
                "output_id": "out_protagonist",
                "title": "The Detective's Hunt",
                "candidate_scope": "SEASON_ARC",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [
                            {"episode_id": "E01", "source_video": episodes[0].source_video, "start": 0.0, "end": 30.0}
                        ],
                        "narration": "The Detective starts the hunt.",
                        "audio_policy": "mute",
                    }
                ],
            },
            {
                "output_id": "out_supporting",
                "title": "The Informant's Dilemma",
                "candidate_scope": "CROSS_EPISODE",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [
                            {"episode_id": "E02", "source_video": episodes[1].source_video, "start": 40.0, "end": 60.0}
                        ],
                        "narration": "The Informant risks everything to pass the note.",
                        "audio_policy": "mute",
                    },
                    {
                        "segment_id": "s2",
                        "source_clips": [
                            {"episode_id": "E03", "source_video": episodes[2].source_video, "start": 10.0, "end": 35.0}
                        ],
                        "narration": "The Informant faces the consequence.",
                        "audio_policy": "mute",
                    },
                ],
            },
        ]
    }

    finalizer = CandidateFinalizer()
    outputs = finalizer.parse_outputs(raw_ai, episodes)

    assert len(outputs) == 2
    titles = [o.title for o in outputs]
    assert "The Detective's Hunt" in titles
    assert "The Informant's Dilemma" in titles

    supp_out = [o for o in outputs if o.title == "The Informant's Dilemma"][0]
    assert supp_out.candidate_scope == CandidateScope.CROSS_EPISODE.value
    assert len(supp_out.segments) == 2
    assert supp_out.segments[0].source_clips[0].episode_id == "E02"
    assert supp_out.segments[1].source_clips[0].episode_id == "E03"


# ---------------------------------------------------------------------------
# Test 5: All Outputs Parsed Without Quota Slicing
# ---------------------------------------------------------------------------

def test_all_outputs_parsed_without_quota_slicing(tmp_path: Path) -> None:
    """AI returns 5 distinct candidate outputs of different scopes; all 5 are parsed."""
    ep = make_dummy_episode("E01", duration=300.0, tmp_path=tmp_path)
    scopes = [
        CandidateScope.SINGLE_SCENE.value,
        CandidateScope.SINGLE_EPISODE.value,
        CandidateScope.CROSS_EPISODE.value,
        CandidateScope.SEASON_ARC.value,
        CandidateScope.SINGLE_SCENE.value,
    ]

    raw_ai = {
        "outputs": [
            {
                "output_id": f"out_{idx}",
                "title": f"Unique Storyline {idx}",
                "candidate_scope": scope,
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [
                            {
                                "episode_id": "E01",
                                "source_video": ep.source_video,
                                "start": float(idx * 20),
                                "end": float(idx * 20 + 10),
                            }
                        ],
                        "narration": f"Narration for storyline {idx}.",
                        "audio_policy": "mute",
                    }
                ],
            }
            for idx, scope in enumerate(scopes, start=1)
        ]
    }

    finalizer = CandidateFinalizer()
    outputs = finalizer.parse_outputs(raw_ai, [ep])

    assert len(outputs) == 5
    for idx, (out, scope) in enumerate(zip(outputs, scopes), start=1):
        assert out.output_id == f"out_{idx}"
        assert out.candidate_scope == scope
        assert out.title == f"Unique Storyline {idx}"


# ---------------------------------------------------------------------------
# Test 6: Incomplete Coverage - Default Fail vs allow_incomplete=True
# ---------------------------------------------------------------------------

def test_incomplete_coverage_default_fails(tmp_path: Path) -> None:
    """Default season finalization fails when an episode is missing or failed."""
    episodes = [make_dummy_episode(f"E0{i}", duration=100.0, tmp_path=tmp_path) for i in range(1, 6)]
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")

    # Scanner fails on E04
    call_count = 0

    def mock_chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        # Fail when scanning E04
        if "E04" in kwargs.get("user_text", ""):
            raise RuntimeError("Corrupted video file in E04")
        return {cat: [] for cat in EVIDENCE_CATEGORIES}

    client = MockAIClient()
    client.chat_json = mock_chat  # type: ignore[assignment]
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")

    engine = AnalysisEngine(settings=settings, client=client, cache_manager=cache_mgr)

    with pytest.raises(CoverageIncompleteError) as exc_info:
        engine.analyze(
            project_id="test_fail",
            episodes=episodes,
            scope=AnalysisScope.SEASON,
            allow_incomplete=False,  # default
        )

    assert "E04" in str(exc_info.value)
    assert "allow_incomplete=True" in str(exc_info.value)


def test_incomplete_coverage_allowed_informs_ai(tmp_path: Path) -> None:
    """When allow_incomplete=True, analysis proceeds and explicitly warns AI of missing IDs."""
    episodes = [make_dummy_episode(f"E0{i}", duration=100.0, tmp_path=tmp_path) for i in range(1, 5)]
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")

    # E03 fails
    def mock_chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
        user_text = kwargs.get("user_text", "")
        system = kwargs.get("system", "")
        client.call_history.append({"model": kwargs.get("model", ""), "system": system, "user_text": user_text})
        if "E03" in user_text and "scanner" in system.lower():
            raise RuntimeError("Missing subtitle stream in E03")
        if "scanner" in system.lower():
            return {cat: [] for cat in EVIDENCE_CATEGORIES}
        if "season narrative architect" in system.lower():
            return {
                "cross_episode_links": [],
                "candidate_proposals": [
                    {
                        "proposal_id": "prop_partial",
                        "title": "Partial Season Story",
                        "candidate_scope": "CROSS_EPISODE",
                        "episodes": ["E01", "E02"],
                        "status": "keep",
                    }
                ],
            }
        # Finalizer
        return {
            "outputs": [
                {
                    "output_id": "out_01",
                    "title": "Partial Season Recap",
                    "candidate_scope": "CROSS_EPISODE",
                    "segments": [
                        {
                            "segment_id": "s1",
                            "source_clips": [
                                {"episode_id": "E01", "source_video": episodes[0].source_video, "start": 0.0, "end": 10.0}
                            ],
                            "narration": "Recap of available episodes.",
                            "audio_policy": "mute",
                        }
                    ],
                }
            ]
        }

    client = MockAIClient()
    client.chat_json = mock_chat  # type: ignore[assignment]
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")

    engine = AnalysisEngine(settings=settings, client=client, cache_manager=cache_mgr)

    manifest = engine.analyze(
        project_id="test_allowed",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        allow_incomplete=True,
    )

    assert len(manifest.outputs) == 1
    # Check that the connection pass received explicit notice about E03 missing
    conn_calls = [c for c in client.call_history if "season narrative architect" in c["system"].lower()]
    assert len(conn_calls) == 1
    assert "E03" in conn_calls[0]["user_text"]
    assert "PARTIAL season analysis" in conn_calls[0]["user_text"]
    assert "NEVER claim, imply, or hallucinate full season coverage" in conn_calls[0]["user_text"]


# ---------------------------------------------------------------------------
# Test 7: Selective Cache Invalidation
# ---------------------------------------------------------------------------

def test_selective_cache_invalidation_reuses_e01_to_e04(tmp_path: Path) -> None:
    """Modifying E05 reuses E01-E04 scanner caches and only rescans E05, then reruns connection & finalizer."""
    episodes = [make_dummy_episode(f"E0{i}", duration=100.0, tmp_path=tmp_path) for i in range(1, 6)]
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")

    scanner_calls: list[str] = []

    def mock_chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
        user_text = kwargs.get("user_text", "")
        system = kwargs.get("system", "")
        if "evidence scanner" in system.lower():
            for ep in episodes:
                if ep.episode_id in user_text:
                    scanner_calls.append(ep.episode_id)
                    break
            return {cat: [] for cat in EVIDENCE_CATEGORIES}
        if "season narrative architect" in system.lower():
            return {
                "cross_episode_links": [],
                "candidate_proposals": [
                    {
                        "proposal_id": "prop_1",
                        "title": "Season Arc",
                        "candidate_scope": "SEASON_ARC",
                        "episodes": ["E01"],
                        "status": "keep",
                    }
                ],
            }
        return {
            "outputs": [
                {
                    "output_id": "out_01",
                    "title": "Season Arc",
                    "candidate_scope": "SEASON_ARC",
                    "segments": [
                        {
                            "segment_id": "s1",
                            "source_clips": [
                                {"episode_id": "E01", "source_video": episodes[0].source_video, "start": 0.0, "end": 10.0}
                            ],
                            "narration": "Season narration.",
                            "audio_policy": "mute",
                        }
                    ],
                }
            ]
        }

    client = MockAIClient()
    client.chat_json = mock_chat  # type: ignore[assignment]
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")

    engine = AnalysisEngine(settings=settings, client=client, cache_manager=cache_mgr)

    # 1. First run: all 5 episodes are scanned
    engine.analyze(
        project_id="proj_cache_1",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        use_final_plan_cache=False,
    )

    assert set(scanner_calls) == {"E01", "E02", "E03", "E04", "E05"}
    assert len(scanner_calls) == 5

    # 2. Modify E05 (invalidate in cache_mgr)
    cache_mgr.invalidate("E05")
    scanner_calls.clear()

    # 3. Second run: E01-E04 hit cache; ONLY E05 is scanned!
    engine.analyze(
        project_id="proj_cache_2",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        use_final_plan_cache=False,
    )

    assert scanner_calls == ["E05"]  # Only E05 rescanned!


# ---------------------------------------------------------------------------
# Test 8: Cancellation during Scanner, Season Connection, and Finalizer
# ---------------------------------------------------------------------------

def test_cancellation_during_scanner(tmp_path: Path) -> None:
    """Cancellation during scanner phase aborts promptly."""
    ep = make_dummy_episode("E01", duration=100.0, tmp_path=tmp_path)
    cancel_event = threading.Event()
    cancel_event.set()  # Already cancelled

    client = MockAIClient()
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")
    engine = AnalysisEngine(settings=settings, client=client)

    with pytest.raises(AnalysisCancelledError):
        engine.analyze(
            project_id="cancel_scanner",
            episodes=[ep],
            scope=AnalysisScope.SINGLE_EPISODE,
            cancel_event=cancel_event,
        )


def test_cancellation_during_season_connection(tmp_path: Path) -> None:
    """Cancellation during season connection pass stops execution."""
    episodes = [make_dummy_episode("E01", duration=100.0, tmp_path=tmp_path)]
    cancel_event = threading.Event()
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")

    def mock_chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
        system = kwargs.get("system", "")
        if "evidence scanner" in system.lower():
            return {cat: [] for cat in EVIDENCE_CATEGORIES}
        if "season narrative architect" in system.lower():
            cancel_event.set()
            raise AnalysisCancelledError("Cancelled in connection pass")
        return {}

    client = MockAIClient()
    client.chat_json = mock_chat  # type: ignore[assignment]
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")
    engine = AnalysisEngine(settings=settings, client=client, cache_manager=cache_mgr)

    with pytest.raises(AnalysisCancelledError):
        engine.analyze(
            project_id="cancel_conn",
            episodes=episodes,
            scope=AnalysisScope.SEASON,
            cancel_event=cancel_event,
        )


def test_cancellation_during_finalizer(tmp_path: Path) -> None:
    """Cancellation during finalizer phase stops execution."""
    episodes = [make_dummy_episode("E01", duration=100.0, tmp_path=tmp_path)]
    cancel_event = threading.Event()
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")

    def mock_chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
        system = kwargs.get("system", "")
        if "evidence scanner" in system.lower():
            return {cat: [] for cat in EVIDENCE_CATEGORIES}
        if "season narrative architect" in system.lower():
            return {"candidate_proposals": []}
        if "lead editor" in system.lower():
            cancel_event.set()
            raise AnalysisCancelledError("Cancelled in finalizer")
        return {}

    client = MockAIClient()
    client.chat_json = mock_chat  # type: ignore[assignment]
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")
    engine = AnalysisEngine(settings=settings, client=client, cache_manager=cache_mgr)

    with pytest.raises(AnalysisCancelledError):
        engine.analyze(
            project_id="cancel_finalizer",
            episodes=episodes,
            scope=AnalysisScope.SEASON,
            cancel_event=cancel_event,
        )


# ---------------------------------------------------------------------------
# Test 9: Malformed and Hallucinated References Rejected
# ---------------------------------------------------------------------------

def test_reject_hallucinated_episode_id(tmp_path: Path) -> None:
    """Source clip referencing non-existent episode ID must be rejected."""
    ep = make_dummy_episode("E01", duration=100.0, tmp_path=tmp_path)
    finalizer = CandidateFinalizer()

    bad_ai = {
        "outputs": [
            {
                "output_id": "out_01",
                "title": "Bad Episode ID",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [
                            {"episode_id": "E99", "source_video": ep.source_video, "start": 0.0, "end": 10.0}
                        ],
                        "narration": "Hallucinated episode.",
                    }
                ],
            }
        ]
    }

    with pytest.raises(ValidationError, match="Invalid episode_id 'E99'"):
        finalizer.parse_outputs(bad_ai, [ep])


def test_reject_source_video_mismatch(tmp_path: Path) -> None:
    """Source clip referencing wrong video file must be rejected."""
    ep = make_dummy_episode("E01", duration=100.0, tmp_path=tmp_path)
    finalizer = CandidateFinalizer()

    bad_ai = {
        "outputs": [
            {
                "output_id": "out_01",
                "title": "Bad Source File",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [
                            {"episode_id": "E01", "source_video": "completely_different_movie.mp4", "start": 0.0, "end": 10.0}
                        ],
                        "narration": "Hallucinated source file.",
                    }
                ],
            }
        ]
    }

    with pytest.raises(ValidationError, match="does not match episode"):
        finalizer.parse_outputs(bad_ai, [ep])


def test_reject_inverted_and_negative_timestamps(tmp_path: Path) -> None:
    """Negative start or end <= start must be rejected."""
    ep = make_dummy_episode("E01", duration=100.0, tmp_path=tmp_path)
    finalizer = CandidateFinalizer()

    # Negative start
    bad_ai_neg = {
        "outputs": [
            {
                "output_id": "out_01",
                "title": "Negative start",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [
                            {"episode_id": "E01", "source_video": ep.source_video, "start": -5.0, "end": 10.0}
                        ],
                        "narration": "Negative start.",
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValidationError, match="must be >= 0"):
        finalizer.parse_outputs(bad_ai_neg, [ep])

    # End <= Start
    bad_ai_inv = {
        "outputs": [
            {
                "output_id": "out_02",
                "title": "Inverted",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [
                            {"episode_id": "E01", "source_video": ep.source_video, "start": 20.0, "end": 15.0}
                        ],
                        "narration": "Inverted.",
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValidationError, match="must be greater than start"):
        finalizer.parse_outputs(bad_ai_inv, [ep])


def test_reject_timestamp_exceeding_duration(tmp_path: Path) -> None:
    """Clip end timestamp exceeding episode duration must be rejected."""
    ep = make_dummy_episode("E01", duration=50.0, tmp_path=tmp_path)
    finalizer = CandidateFinalizer()

    bad_ai = {
        "outputs": [
            {
                "output_id": "out_01",
                "title": "Exceeds duration",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [
                            {"episode_id": "E01", "source_video": ep.source_video, "start": 10.0, "end": 100.0}
                        ],
                        "narration": "Clip end 100 exceeds 50.",
                    }
                ],
            }
        ]
    }

    with pytest.raises(ValidationError, match="exceeds episode 'E01' duration"):
        finalizer.parse_outputs(bad_ai, [ep])


# ---------------------------------------------------------------------------
# Test 10: Deterministic Offline Mode with Injected Transcripts
# ---------------------------------------------------------------------------

def test_offline_mode_with_injected_transcripts(tmp_path: Path) -> None:
    """Ensure completely offline mode produces valid AnalysisManifest with all categories."""
    ep1 = make_dummy_episode("E01", duration=60.0, tmp_path=tmp_path)
    ep2 = make_dummy_episode("E02", duration=60.0, tmp_path=tmp_path)

    cues_e01 = [
        SubtitleCue(start_ms=0, end_ms=5000, text="First dialogue in E01", source_type="sidecar", source_format="srt"),
        SubtitleCue(start_ms=10000, end_ms=15000, text="Second dialogue in E01", source_type="sidecar", source_format="srt"),
    ]
    cues_e02 = [
        SubtitleCue(start_ms=2000, end_ms=8000, text="Dialogue in E02", source_type="sidecar", source_format="srt"),
    ]

    manifest = run_analysis(
        project_id="offline_season",
        episodes=[ep1, ep2],
        scope=AnalysisScope.SEASON,
        settings=AppSettings(gateway_enabled=False),
        injected_transcripts={"E01": cues_e01, "E02": cues_e02},
    )

    assert manifest.analysis_scope == AnalysisScope.SEASON.value
    # Offline mode without AI does not fabricate fixed outputs or quotas: outputs == []
    assert manifest.outputs == []
    manifest.validate()

    # If explicit legacy_wrapper is requested, legacy offline outputs are returned
    legacy_manifest = run_analysis(
        project_id="offline_season_legacy",
        episodes=[ep1, ep2],
        scope=AnalysisScope.SEASON,
        settings=AppSettings(gateway_enabled=False),
        injected_transcripts={"E01": cues_e01, "E02": cues_e02},
        legacy_wrapper=True,
    )
    assert len(legacy_manifest.outputs) >= 1
    legacy_manifest.validate()


# ---------------------------------------------------------------------------
# Test 11: Cross-Batch Merge Setup E02 / Payoff E08 & Preserves Supporting Arcs
# ---------------------------------------------------------------------------

def test_cross_batch_merge_finds_e02_setup_e08_payoff_preserves_supporting_arcs(tmp_path: Path) -> None:
    """Cross-batch merge identifies E02 setup to E08 payoff arc while preserving supporting character arcs."""
    episodes = [make_dummy_episode(f"E0{i}", duration=300.0, tmp_path=tmp_path) for i in range(1, 9)]
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")

    scanner_responses = [{cat: [] for cat in EVIDENCE_CATEGORIES} for _ in range(8)]

    # Batch 1 (E01-E04)
    batch1_resp = {
        "batch_id": "batch_E01_E04",
        "cross_episode_links": [
            {"thread_id": "thread_early_conspiracy", "episodes": ["E01", "E02"], "summary": "Early conspiracy setup"}
        ],
        "candidate_proposals": [
            {
                "proposal_id": "prop_e02_setup",
                "title": "Conspiracy Setup",
                "candidate_scope": "CROSS_EPISODE",
                "episodes": ["E01", "E02"],
                "editorial_reason": "Setup of the conspiracy in E02.",
                "status": "keep",
            }
        ],
        "supporting_character_arcs": [
            {
                "character": "Officer Miller",
                "arc_summary": "Officer Miller investigates unauthorized transactions in E03.",
                "episodes": ["E03"],
                "has_dedicated_candidate": True,
            }
        ],
        "rejected_or_merged": [],
    }

    # Batch 2 (E05-E08)
    batch2_resp = {
        "batch_id": "batch_E05_E08",
        "cross_episode_links": [
            {"thread_id": "thread_late_payoff", "episodes": ["E07", "E08"], "summary": "Conspiracy reaches climax and payoff"}
        ],
        "candidate_proposals": [
            {
                "proposal_id": "prop_e08_payoff",
                "title": "Conspiracy Payoff",
                "candidate_scope": "CROSS_EPISODE",
                "episodes": ["E07", "E08"],
                "editorial_reason": "Payoff of the conspiracy in E08.",
                "status": "keep",
            }
        ],
        "supporting_character_arcs": [],
        "rejected_or_merged": [],
    }

    # Cross-batch merge response
    merge_resp = {
        "cross_episode_links": [
            {
                "thread_id": "thread_full_conspiracy",
                "episodes": ["E02", "E08"],
                "summary": "Full conspiracy arc connecting E02 setup directly to E08 payoff.",
            }
        ],
        "candidate_proposals": [
            {
                "proposal_id": "prop_full_conspiracy",
                "title": "The Master Conspiracy",
                "candidate_scope": "SEASON_ARC",
                "episodes": ["E02", "E08"],
                "editorial_reason": "Master arc connecting early setup in E02 to resolution in E08.",
                "status": "keep",
            },
            {
                "proposal_id": "prop_officer_miller",
                "title": "Officer Miller's Stand",
                "candidate_scope": "SINGLE_EPISODE",
                "episodes": ["E03"],
                "editorial_reason": "Crucial supporting investigation preserved.",
                "status": "keep",
            },
        ],
        "supporting_character_arcs": [
            {
                "character": "Officer Miller",
                "arc_summary": "Officer Miller's parallel investigation survives cross-batch merge.",
                "episodes": ["E03"],
                "has_dedicated_candidate": True,
            }
        ],
        "rejected_or_merged": [
            {"proposal_id": "prop_e02_setup", "reason": "Merged into prop_full_conspiracy"},
            {"proposal_id": "prop_e08_payoff", "reason": "Merged into prop_full_conspiracy"},
        ],
    }

    finalizer_resp = {
        "outputs": [
            {
                "output_id": "out_conspiracy",
                "title": "The Master Conspiracy",
                "candidate_scope": "SEASON_ARC",
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "source_clips": [{"episode_id": "E02", "source_video": episodes[1].source_video, "start": 10.0, "end": 30.0}],
                        "narration": "The conspiracy takes root in Episode 2.",
                        "audio_policy": "mute",
                    },
                    {
                        "segment_id": "seg_02",
                        "source_clips": [{"episode_id": "E08", "source_video": episodes[7].source_video, "start": 20.0, "end": 45.0}],
                        "narration": "The payoff is revealed in Episode 8.",
                        "audio_policy": "mute",
                    },
                ],
            }
        ]
    }

    mock_client = MockAIClient(scanner_responses + [batch1_resp, batch2_resp, merge_resp, finalizer_resp])
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")

    manifest = run_analysis(
        project_id="season_e02_e08",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        settings=settings,
        client=mock_client,
        cache_manager=cache_mgr,
    )

    # 1. Verify merge call received both batches in user_text
    merge_calls = [c for c in mock_client.call_history if "cross-batch merge synthesizer" in c["system"].lower()]
    assert len(merge_calls) == 1
    m_text = merge_calls[0]["user_text"]
    assert "prop_e02_setup" in m_text
    assert "prop_e08_payoff" in m_text
    assert "Officer Miller" in m_text

    # 2. Verify manifest output spans E02 and E08
    assert len(manifest.outputs) == 1
    out = manifest.outputs[0]
    clip_eps = [c.episode_id for s in out.segments for c in s.source_clips]
    assert "E02" in clip_eps
    assert "E08" in clip_eps
    manifest.validate()


# ---------------------------------------------------------------------------
# Test 12: Resume Exact Batch 3 Failure Reuses Prior Batches and Evidence
# ---------------------------------------------------------------------------

def test_resume_exact_batch3_failure_reuses_prior_batches_and_evidence(tmp_path: Path) -> None:
    """When batch 3 fails on first run, second run reuses evidence and batches 1-2, only executing batch 3 and merge."""
    episodes = [make_dummy_episode(f"E{i:02d}", duration=100.0, tmp_path=tmp_path) for i in range(1, 11)]
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")

    # 10 episodes -> 3 batches: [4, 3, 3] -> batch_E01_E04, batch_E05_E07, batch_E08_E10
    scanner_responses = [{cat: [] for cat in EVIDENCE_CATEGORIES} for _ in range(10)]
    b1_resp = {"candidate_proposals": [{"proposal_id": "p1", "title": "Part 1", "episodes": ["E01", "E02"]}]}
    b2_resp = {"candidate_proposals": [{"proposal_id": "p2", "title": "Part 2", "episodes": ["E05", "E06"]}]}
    b3_fail = APIError("Simulated Batch 3 rate limit failure")

    client1 = MockAIClient(scanner_responses + [b1_resp, b2_resp, b3_fail])
    engine1 = AnalysisEngine(settings=settings, client=client1, cache_manager=cache_mgr)

    with pytest.raises(AnalysisError, match=r"season batch (node_L0_\d+_\d+|batch_)"):
        engine1.analyze(
            project_id="proj_resume",
            episodes=episodes,
            scope=AnalysisScope.SEASON,
            use_final_plan_cache=False,
        )

    # Verify first run scanned 10 episodes and called 3 batches
    scanner_calls_1 = [c for c in client1.call_history if "evidence scanner" in c["system"].lower()]
    assert len(scanner_calls_1) == 10
    batch_calls_1 = [c for c in client1.call_history if "batch connection analyst" in c["system"].lower()]
    assert len(batch_calls_1) == 3

    # Run 2: Provide Batch 3 success, Merge success, and Finalizer success
    b3_success = {"candidate_proposals": [{"proposal_id": "p3", "title": "Part 3", "episodes": ["E08", "E09"]}]}
    merge_resp = {
        "candidate_proposals": [{"proposal_id": "p_full", "title": "Full Season Story", "episodes": ["E01", "E08"]}],
        "cross_episode_links": [],
    }
    finalizer_resp = {
        "outputs": [
            {
                "output_id": "out_resumed",
                "title": "Full Season Story",
                "candidate_scope": "SEASON_ARC",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [{"episode_id": "E01", "source_video": episodes[0].source_video, "start": 0.0, "end": 10.0}],
                        "narration": "Resumed narration.",
                        "audio_policy": "mute",
                    }
                ],
            }
        ]
    }

    client2 = MockAIClient([b3_success, merge_resp, finalizer_resp])
    engine2 = AnalysisEngine(settings=settings, client=client2, cache_manager=cache_mgr)

    manifest = engine2.analyze(
        project_id="proj_resume",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        use_final_plan_cache=False,
    )

    # Verify on run 2:
    # 0 scanner calls (all 10 hit evidence cache)
    scanner_calls_2 = [c for c in client2.call_history if "evidence scanner" in c["system"].lower()]
    assert len(scanner_calls_2) == 0

    # Only Batch 3 was called (Batches 1 & 2 hit cache)
    batch_calls_2 = [c for c in client2.call_history if "batch connection analyst" in c["system"].lower()]
    assert len(batch_calls_2) == 1
    assert "E09" in batch_calls_2[0]["user_text"]
    assert "E10" in batch_calls_2[0]["user_text"]

    # Merge was called once
    merge_calls_2 = [c for c in client2.call_history if "cross-batch merge synthesizer" in c["system"].lower()]
    assert len(merge_calls_2) == 1

    assert len(manifest.outputs) == 1
    manifest.validate()


# ---------------------------------------------------------------------------
# Test 13: Finalizer Fail then Retry Reuses Connection Cache
# ---------------------------------------------------------------------------

def test_finalizer_fail_then_retry_connection_cache_hit_no_connection_calls(tmp_path: Path) -> None:
    """When finalizer fails on first run, retry reuses full connection cache without making connection calls."""
    episodes = [make_dummy_episode("E01", duration=100.0, tmp_path=tmp_path), make_dummy_episode("E02", duration=100.0, tmp_path=tmp_path)]
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")

    # Run 1: Scanner succeeds, Connection succeeds, Finalizer fails
    scanner_resps = [{cat: [] for cat in EVIDENCE_CATEGORIES}, {cat: [] for cat in EVIDENCE_CATEGORIES}]
    conn_resp = {
        "candidate_proposals": [
            {"proposal_id": "prop_1", "title": "Duo Journey", "episodes": ["E01", "E02"], "candidate_scope": "CROSS_EPISODE"}
        ],
        "cross_episode_links": [],
    }
    finalizer_fail = APIError("Simulated finalizer 503 error")

    client1 = MockAIClient(scanner_resps + [conn_resp, finalizer_fail])
    engine1 = AnalysisEngine(settings=settings, client=client1, cache_manager=cache_mgr)

    with pytest.raises(AnalysisError, match="Finalizer"):
        engine1.analyze(
            project_id="proj_fin_retry",
            episodes=episodes,
            scope=AnalysisScope.SEASON,
            use_final_plan_cache=False,
        )

    # Run 2: Finalizer succeeds
    finalizer_success = {
        "outputs": [
            {
                "output_id": "out_01",
                "title": "Duo Journey",
                "candidate_scope": "CROSS_EPISODE",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [{"episode_id": "E01", "source_video": episodes[0].source_video, "start": 0.0, "end": 10.0}],
                        "narration": "Narration.",
                        "audio_policy": "mute",
                    }
                ],
            }
        ]
    }
    client2 = MockAIClient([finalizer_success])
    engine2 = AnalysisEngine(settings=settings, client=client2, cache_manager=cache_mgr)

    manifest = engine2.analyze(
        project_id="proj_fin_retry",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        use_final_plan_cache=False,
    )

    # Verify run 2:
    # 0 scanner calls
    scanner_calls = [c for c in client2.call_history if "evidence scanner" in c["system"].lower()]
    assert len(scanner_calls) == 0

    # 0 connection calls (connection cache hit!)
    conn_calls = [c for c in client2.call_history if "season narrative architect" in c["system"].lower()]
    assert len(conn_calls) == 0

    # Exactly 1 finalizer call
    fin_calls = [c for c in client2.call_history if "lead editor" in c["system"].lower()]
    assert len(fin_calls) == 1

    assert len(manifest.outputs) == 1
    manifest.validate()


# ---------------------------------------------------------------------------
# Test 14: Hierarchy Cache Root Derived from Injected Cache Parent
# ---------------------------------------------------------------------------

def test_hierarchy_cache_root_derived_from_injected_cache_parent(tmp_path: Path) -> None:
    """Hierarchy cache root and plan cache dir must be derived from injected evidence cache parent."""
    custom_root = tmp_path / "custom_isolation"
    ev_cache_dir = custom_root / "evidence_store"
    cache_mgr = EvidenceCacheManager(cache_dir=ev_cache_dir)

    engine = AnalysisEngine(cache_manager=cache_mgr)

    expected_hierarchy = custom_root / "season_hierarchy"
    expected_plan = custom_root / "analysis_plans"

    assert engine.hierarchy_cache.base_dir == expected_hierarchy
    assert engine.plan_cache_dir == expected_plan
    assert engine.hierarchy_cache.summary_dir == expected_hierarchy / "summaries"
    assert engine.hierarchy_cache.batch_dir == expected_hierarchy / "batches"
    assert engine.hierarchy_cache.merge_dir == expected_hierarchy / "merges"
    assert engine.hierarchy_cache.connection_dir == expected_hierarchy / "connections"


# ---------------------------------------------------------------------------
# Test 15: Validate AI Batch Responses Schema Before Cache
# ---------------------------------------------------------------------------

def test_validate_ai_batch_responses_schema_before_cache_malformed_no_cache(tmp_path: Path) -> None:
    """Malformed AI responses raise AnalysisError and are strictly never written to hierarchy cache."""
    # Test validate_batch_response_schema unit behaviors
    with pytest.raises(AnalysisError, match="phải là một dictionary"):
        validate_batch_response_schema("not a dict")

    with pytest.raises(AnalysisError, match="phải là kiểu list"):
        validate_batch_response_schema({"candidate_proposals": "not a list"})

    with pytest.raises(AnalysisError, match="phải là dict"):
        validate_batch_response_schema({"candidate_proposals": ["not a dict item"]})

    with pytest.raises(AnalysisError, match="thiếu các trường danh sách liên kết bắt buộc"):
        validate_batch_response_schema({"irrelevant_field": 123})

    # Test integration: malformed batch is NOT cached
    h_cache = HierarchyCacheManager(tmp_path / "hierarchy")
    episodes = [make_dummy_episode("E01", duration=60.0, tmp_path=tmp_path)]
    evidence_map = {
        "E01": EpisodeEvidence(
            episode_id="E01",
            source_video=episodes[0].source_video,
            duration_seconds=60.0,
            coverage={"ratio": 1.0},
        )
    }

    client = MockAIClient([{"candidate_proposals": "invalid_string_not_list"}])
    connector = SeasonConnector(
        settings=AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234"),
        client=client,
        hierarchy_cache=h_cache,
    )

    with pytest.raises(AnalysisError, match="phải là kiểu list"):
        connector.connect_season(episodes, evidence_map)

    # Verify no batch cache file was created
    batch_files = list(h_cache.batch_dir.glob("*.json"))
    assert len(batch_files) == 0
    conn_files = list(h_cache.connection_dir.glob("*.json"))
    assert len(conn_files) == 0


# ---------------------------------------------------------------------------
# Test 16: Payload Size Protection and Pre-Serialization Reduction (10 eps)
# ---------------------------------------------------------------------------

def test_payload_size_protection_and_pre_serialization_reduction_10_episodes(tmp_path: Path) -> None:
    """Payload size protection triggers on oversized text, and 10-episode batch requests contain <=4 summary IDs."""
    # 1. Payload size check raises AnalysisError
    with pytest.raises(AnalysisError, match="exceeds max allowed limit"):
        check_payload_size("x" * 1000, max_bytes=500, context="test_overflow")

    # 2. 10 episodes analyzed in season mode
    episodes = [make_dummy_episode(f"E{i:02d}", duration=100.0, tmp_path=tmp_path) for i in range(1, 11)]
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")

    scanner_resps = [{cat: [] for cat in EVIDENCE_CATEGORIES} for _ in range(10)]
    batch_resps = [
        {"candidate_proposals": [{"proposal_id": f"p_{i}", "title": f"Batch {i}", "episodes": [f"E{i:02d}"]}]}
        for i in range(1, 4)
    ]
    merge_resp = {
        "candidate_proposals": [{"proposal_id": "p_merge", "title": "Full Story", "episodes": ["E01", "E10"]}],
        "cross_episode_links": [],
    }
    finalizer_resp = {
        "outputs": [
            {
                "output_id": "out_01",
                "title": "Full Story",
                "candidate_scope": "SEASON_ARC",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_clips": [{"episode_id": "E01", "source_video": episodes[0].source_video, "start": 0.0, "end": 10.0}],
                        "narration": "Full season recap narration.",
                        "audio_policy": "mute",
                    }
                ],
            }
        ]
    }

    client = MockAIClient(scanner_resps + batch_resps + [merge_resp, finalizer_resp])
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock:1234")

    manifest = run_analysis(
        project_id="proj_10_episodes",
        episodes=episodes,
        scope=AnalysisScope.SEASON,
        settings=settings,
        client=client,
        cache_manager=cache_mgr,
    )

    batch_calls = [c for c in client.call_history if "batch connection analyst" in c["system"].lower()]
    assert len(batch_calls) == 3

    for call in batch_calls:
        text = call["user_text"]
        assert len(text.encode("utf-8")) <= 500_000
        # Must not contain raw evidence categories or transcripts
        assert "evidence_categories" not in text
        # Must have at most 4 episode summaries per batch
        ep_count = text.count('"episode_id"')
        assert 1 <= ep_count <= 4

    manifest.validate()


# ---------------------------------------------------------------------------
# Test 17: Partition Season Batches N=1 to 24
# ---------------------------------------------------------------------------

def test_partition_season_batches_n1_to_24() -> None:
    """Deterministic partition of N=1 to 24 episodes preserves all items with sizes <= 4 and 3-4 for N>=6."""
    for n in range(1, 25):
        items = [f"E{i:02d}" for i in range(1, n + 1)]
        batches = partition_season_batches(items)

        # 1. Total items preserved
        total_items = sum(len(b[1]) for b in batches)
        assert total_items == n, f"Failed item preservation for N={n}"

        # 2. Every batch size <= 4
        for bid, bitems in batches:
            assert 1 <= len(bitems) <= 4, f"Batch size violation {len(bitems)} for N={n}"
            assert bid.startswith("batch_")

        # 3. For N >= 6, every batch has size 3 or 4
        if n >= 6:
            for bid, bitems in batches:
                assert len(bitems) in (3, 4), f"N={n} produced invalid batch size {len(bitems)} in {bid}"

        # 4. Batch IDs are unique
        bids = [b[0] for b in batches]
        assert len(bids) == len(set(bids)), f"Duplicate batch ID for N={n}: {bids}"


# ---------------------------------------------------------------------------
# Test 18: Cache Keys and Disk Payloads Strictly Exclude Secrets
# ---------------------------------------------------------------------------

def test_cache_keys_and_payloads_strictly_exclude_secrets(tmp_path: Path) -> None:
    """All cache key computations and disk payloads strictly exclude API keys, tokens, or credentials."""
    secret_token = "sk-proj-super-secret-key-12345"

    k_ev = compute_cache_key("E01", "C:/video.mp4", "v1", 100, 1.0)
    k_sum = compute_summary_cache_key("E01", "abc123hash")
    k_batch = compute_batch_cache_key("b1", ["h1", "h2"], "model-x", "auto", "prompt")
    k_merge = compute_merge_cache_key("m1", ["b1", "b2"], "model-x", "auto", "prompt")
    k_conn = compute_connection_cache_key(["h1", "h2"], "model-x", "auto", "prompt")

    ev_map = {
        "E01": EpisodeEvidence(episode_id="E01", source_video="C:/video.mp4", duration_seconds=60.0)
    }
    settings = AppSettings(api_key=secret_token, api_endpoint="http://secret-endpoint/v1")
    k_plan = compute_final_plan_cache_key(ev_map, settings, "SEASON")

    for k in (k_ev, k_sum, k_batch, k_merge, k_conn, k_plan):
        assert isinstance(k, str)
        assert secret_token not in k
        assert "secret" not in k

    # Verify saved cache files do not contain secrets
    h_cache = HierarchyCacheManager(tmp_path / "hierarchy")
    b_path = h_cache.save_batch_result("b1", k_batch, {"candidate_proposals": []})
    m_path = h_cache.save_merge_result("m1", k_merge, {"candidate_proposals": []})
    c_path = h_cache.save_connection_result(k_conn, {"candidate_proposals": []})

    for path in (b_path, m_path, c_path):
        content = path.read_text(encoding="utf-8")
        assert secret_token not in content
        assert "api_key" not in content
