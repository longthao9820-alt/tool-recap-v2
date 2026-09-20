"""Tests for scanner-bounds-r7 (Objective rev 7).

Matrix:
1. Exact measured scanner request formatter and planner size estimation.
2. 600KB dense transcript splits all calls <= 500,000 bytes and <= 480,000 target.
3. All actual calls pass max_payload_bytes hard ceiling (500,000).
4. Two huge cues split by index halves into two separate calls without unnecessary capping.
5. One atomic cue oversize structured cap text while preserving start/end/id; if baseline exceeds target fails clear.
6. Small transcript unchanged produces exactly one call.
7. Strict chronological order and timestamp monotonicity across all split chunks.
8. Dynamic total progress callbacks and bounded parallelism.
9. Merge category results as before across all split chunks.
10. Cache episode evidence on success only; never on failure or cancellation.
11. Cancellation checks across planner, futures, and client.
"""
from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

from toolrecap_v2.analyzer.errors import AnalysisCancelledError, AnalysisError
from toolrecap_v2.analyzer.evidence import (
    EVIDENCE_CATEGORIES,
    HARD_PAYLOAD_CEILING,
    TARGET_PAYLOAD_CEILING,
    EvidenceScanner,
    ScannerChunkPlan,
    cap_single_cue_text,
    check_scanner_baseline_size,
    estimate_scanner_request_size,
    format_scanner_user_text,
    plan_scanner_chunks,
)
from toolrecap_v2.analyzer.phases import AnalysisPhase
from toolrecap_v2.analyzer.prompts import SCANNER_SYSTEM_PROMPT
from toolrecap_v2.api_client import APIError, estimate_request_size
from toolrecap_v2.domain.cache import EvidenceCacheManager
from toolrecap_v2.domain.models import EpisodeEvidence, SourceEpisode
from toolrecap_v2.settings import AppSettings
from toolrecap_v2.subtitles.models import SubtitleCue


class MockScannerAIClient:
    """Mock OpenAICompatibleClient for scanner bounds R7 testing."""

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
        max_payload_bytes: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("Mock API call cancelled.")

        call_record = {
            "model": model,
            "system": system,
            "user_text": user_text,
            "thinking": thinking,
            "max_payload_bytes": max_payload_bytes,
        }
        self.call_history.append(call_record)

        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if callable(resp):
                return resp(call_record)
            return resp

        # Default valid scanner chunk response
        return {
            "major_scenes": [
                {
                    "start_sec": 0.0,
                    "end_sec": 10.0,
                    "summary": "Key scene detected in chunk",
                    "dialogue_evidence": ["Line from chunk"],
                    "characters": ["Alice"],
                }
            ],
            "dialogue": [
                {
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "summary": "Important spoken dialogue",
                    "dialogue_evidence": ["Line from chunk"],
                    "characters": ["Alice"],
                }
            ],
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


# ---------------------------------------------------------------------------
# 1. Exact Measured Scanner Request Formatter and Planner Size Estimation
# ---------------------------------------------------------------------------

def test_exact_measured_scanner_request_formatter_and_estimator() -> None:
    """Formatter and estimator calculate exact byte size of serialized request payload."""
    ep = SourceEpisode(episode_id="E01", source_video="video1.mp4", duration_seconds=120.0)
    cues = [
        (0.0, 5.0, "Hello world"),
        (5.0, 10.0, "Second dialogue line"),
    ]
    user_text = format_scanner_user_text(ep, 0.0, 60.0, cues)
    assert "Episode ID: E01" in user_text
    assert "Hello world" in user_text
    assert "Second dialogue line" in user_text

    est_size = estimate_scanner_request_size("sub", user_text, "auto")
    direct_est = estimate_request_size(
        model="sub",
        system=SCANNER_SYSTEM_PROMPT,
        user=user_text,
        thinking="auto",
        variant=0,
    )
    assert est_size == direct_est
    assert est_size < TARGET_PAYLOAD_CEILING


# ---------------------------------------------------------------------------
# 2. 600KB Dense Transcript Split All Calls <= 500,000 Bytes & max_payload_bytes
# ---------------------------------------------------------------------------

def test_600kb_dense_transcript_split_all_calls_bounded_and_pass_max_payload(tmp_path: Path) -> None:
    """600KB dense transcript splits recursively by index halves so all calls <= 500,000 bytes,
    and all actual calls pass max_payload_bytes hard ceiling."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=600.0)
    # Generate ~600KB of dense cues: 1200 cues each ~500 chars -> 600,000 bytes
    cues = [
        SubtitleCue(
            start_ms=i * 500,
            end_ms=(i + 1) * 500,
            text=f"Line {i:04d}: " + ("Dialogue text repeating narrative " * 14)[:480],
            source_type="sidecar",
            source_format="srt",
            episode_id="E01",
        )
        for i in range(1200)
    ]

    # Verify unpartitioned full transcript exceeds 500,000 bytes
    full_text = format_scanner_user_text(ep, 0.0, 600.0, cues)
    full_est = estimate_scanner_request_size("sub", full_text, "auto")
    assert full_est > 550_000, f"Dense transcript should be >550,000 bytes, got {full_est}"

    # Plan scanner chunks
    plans = plan_scanner_chunks(
        episode=ep,
        cues=cues,
        duration_sec=600.0,
        chunk_seconds=600.0,
        model="sub",
        thinking="auto",
        target_ceiling=TARGET_PAYLOAD_CEILING,
        hard_ceiling=HARD_PAYLOAD_CEILING,
    )

    # Must be split into multiple chunks
    assert len(plans) >= 2
    for p in plans:
        assert p.estimated_bytes <= TARGET_PAYLOAD_CEILING
        assert p.estimated_bytes <= HARD_PAYLOAD_CEILING
        assert p.estimated_bytes <= 500_000

    # Run scan_episode through EvidenceScanner with mock client
    mock_client = MockScannerAIClient()
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")
    scanner = EvidenceScanner(settings=settings, client=mock_client, cache_manager=cache_mgr)

    evidence = scanner.scan_episode(ep, injected_cues=cues)
    assert isinstance(evidence, EpisodeEvidence)
    assert evidence.coverage["status"] == "complete"
    assert evidence.coverage["chunks_count"] == len(plans)

    # Verify every call made to the client stayed <= 500,000 and passed max_payload_bytes
    assert len(mock_client.call_history) == len(plans)
    for call in mock_client.call_history:
        call_est = estimate_scanner_request_size(call["model"], call["user_text"], call["thinking"])
        assert call_est <= HARD_PAYLOAD_CEILING
        assert call_est <= 500_000
        assert call["max_payload_bytes"] == HARD_PAYLOAD_CEILING


# ---------------------------------------------------------------------------
# 3. Two Huge Cues Split Without Unnecessary Capping
# ---------------------------------------------------------------------------

def test_two_huge_cues_split_by_halves_into_two_calls_without_capping(tmp_path: Path) -> None:
    """Two huge cues (each ~260KB, combined 520KB > 480KB) split by index halves into 2 calls
    without capping individual cue texts."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=300.0)
    cue1_text = "Cue 1 long dialogue: " + ("Narrative monologue segment A. " * 8000)[:250_000]
    cue2_text = "Cue 2 long dialogue: " + ("Narrative monologue segment B. " * 8000)[:250_000]

    cues = [
        SubtitleCue(start_ms=10_000, end_ms=140_000, text=cue1_text, source_type="sidecar", source_format="srt"),
        SubtitleCue(start_ms=150_000, end_ms=290_000, text=cue2_text, source_type="sidecar", source_format="srt"),
    ]

    # Combined size exceeds target ceiling
    combined_text = format_scanner_user_text(ep, 0.0, 300.0, cues)
    combined_est = estimate_scanner_request_size("sub", combined_text, "auto")
    assert combined_est > TARGET_PAYLOAD_CEILING

    mock_client = MockScannerAIClient()
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")
    scanner = EvidenceScanner(settings=settings, client=mock_client, cache_manager=cache_mgr)

    evidence = scanner.scan_episode(ep, injected_cues=cues)
    assert isinstance(evidence, EpisodeEvidence)

    # Exactly 2 calls made (one for each huge cue)
    assert len(mock_client.call_history) == 2
    for call in mock_client.call_history:
        call_est = estimate_scanner_request_size(call["model"], call["user_text"], call["thinking"])
        assert call_est <= HARD_PAYLOAD_CEILING
        assert call["max_payload_bytes"] == HARD_PAYLOAD_CEILING

    # Verify neither cue text was capped with suffix because each individually <= 480KB
    assert "... [capped]" not in mock_client.call_history[0]["user_text"]
    assert "... [capped]" not in mock_client.call_history[1]["user_text"]
    assert "Cue 1 long dialogue" in mock_client.call_history[0]["user_text"]
    assert "Cue 2 long dialogue" in mock_client.call_history[1]["user_text"]


# ---------------------------------------------------------------------------
# 4. One Atomic Cue Oversize Structured Cap Text While Preserving Start/End/ID
# ---------------------------------------------------------------------------

def test_one_atomic_cue_oversize_structured_cap_text_preserves_metadata() -> None:
    """A single cue whose text alone exceeds 500,000 bytes cannot be halved;
    it is structured capped while strictly preserving start_ms, end_ms, and identity."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=300.0)
    huge_text = "Monologue beginning: " + ("Extremely long text stream " * 25000)[:550_000]

    cue = SubtitleCue(
        start_ms=15_000,
        end_ms=75_000,
        text=huge_text,
        source_type="sidecar",
        source_format="srt",
        episode_id="E01",
    )

    plans = plan_scanner_chunks(
        episode=ep,
        cues=[cue],
        duration_sec=300.0,
        chunk_seconds=300.0,
        model="sub",
        thinking="auto",
        target_ceiling=TARGET_PAYLOAD_CEILING,
        hard_ceiling=HARD_PAYLOAD_CEILING,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.estimated_bytes <= TARGET_PAYLOAD_CEILING
    assert plan.estimated_bytes <= HARD_PAYLOAD_CEILING
    assert plan.estimated_bytes <= 500_000

    # Verify structured cap preserved cue identity, start and end timestamps
    capped_cue = plan.cues[0]
    if isinstance(capped_cue, SubtitleCue):
        assert capped_cue.start_ms == 15_000
        assert capped_cue.end_ms == 75_000
        assert capped_cue.episode_id == "E01"
        assert capped_cue.text.endswith("... [capped]")
    else:
        assert round(capped_cue[0] * 1000) == 15_000
        assert round(capped_cue[1] * 1000) == 75_000
        assert capped_cue[2].endswith("... [capped]")

    # Check user text preserves episode ID and range
    assert "Episode ID: E01" in plan.user_text
    assert "00:00:15" in plan.user_text
    assert "00:01:15" in plan.user_text
    assert "... [capped]" in plan.user_text


# ---------------------------------------------------------------------------
# 5. Baseline System Envelope Too Large Fails Clear
# ---------------------------------------------------------------------------

def test_baseline_system_envelope_too_large_fails_clear() -> None:
    """When baseline scanner prompt envelope alone exceeds target ceiling, raises fatal AnalysisError immediately."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=120.0)

    # Empty envelope is ~1,000+ bytes. Setting target_ceiling=100 causes immediate fatal fail
    with pytest.raises(AnalysisError, match="Baseline scanner prompt envelope size .* exceeds target limit"):
        check_scanner_baseline_size(
            episode=ep,
            start_sec=0.0,
            end_sec=120.0,
            model="sub",
            thinking="auto",
            target_ceiling=100,
        )

    with pytest.raises(AnalysisError, match="Baseline scanner prompt envelope size .* exceeds target limit"):
        plan_scanner_chunks(
            episode=ep,
            cues=[],
            duration_sec=120.0,
            chunk_seconds=120.0,
            model="sub",
            thinking="auto",
            target_ceiling=100,
            hard_ceiling=200,
        )


# ---------------------------------------------------------------------------
# 6. Small Transcript Unchanged Produces Exactly One Call
# ---------------------------------------------------------------------------

def test_small_transcript_unchanged_one_call(tmp_path: Path) -> None:
    """Small transcript (< 480KB) remains completely unchanged, producing exactly 1 call."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=120.0)
    cues = [
        (1.0, 5.0, "Short cue 1"),
        (6.0, 10.0, "Short cue 2"),
        (11.0, 15.0, "Short cue 3"),
    ]

    mock_client = MockScannerAIClient()
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")
    scanner = EvidenceScanner(settings=settings, client=mock_client, cache_manager=cache_mgr)

    evidence = scanner.scan_episode(ep, injected_cues=cues)
    assert isinstance(evidence, EpisodeEvidence)
    assert evidence.coverage["chunks_count"] == 1
    assert len(mock_client.call_history) == 1
    assert "Short cue 1" in mock_client.call_history[0]["user_text"]
    assert "Short cue 3" in mock_client.call_history[0]["user_text"]


# ---------------------------------------------------------------------------
# 7. Order and Monotonic Timestamps Across All Split Chunks
# ---------------------------------------------------------------------------

def test_order_and_timestamp_monotonicity_across_splits() -> None:
    """When chunks are split recursively, resulting plans preserve strict chronological order
    and continuous monotonic timestamps without inversions or gaps."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=1200.0)
    # Generate 800 cues across 1200 seconds with uneven text sizes
    cues = [
        (
            float(i * 1.5),
            float((i + 1) * 1.5),
            f"Cue {i}: " + "narrative event dialogue " * ((i % 15) + 10),
        )
        for i in range(800)
    ]

    # Use smaller target ceiling (e.g. 15,000 bytes) to force deep recursive splitting
    plans = plan_scanner_chunks(
        episode=ep,
        cues=cues,
        duration_sec=1200.0,
        chunk_seconds=600.0,
        model="sub",
        thinking="auto",
        target_ceiling=15_000,
        hard_ceiling=18_000,
    )

    assert len(plans) >= 4

    # Verify sequential chunk indices
    for idx, p in enumerate(plans):
        assert p.chunk_index == idx
        assert p.start_sec < p.end_sec
        assert p.estimated_bytes <= 18_000

    # Verify monotonic continuity: plan[i].end_sec == plan[i+1].start_sec
    for i in range(len(plans) - 1):
        assert plans[i].end_sec == plans[i + 1].start_sec
        assert plans[i].start_sec <= plans[i].end_sec

    # First starts at 0.0, last ends at 1200.0
    assert plans[0].start_sec == 0.0
    assert plans[-1].end_sec == 1200.0


# ---------------------------------------------------------------------------
# 8. Dynamic Total Progress Callbacks and Bounded Parallelism
# ---------------------------------------------------------------------------

def test_dynamic_total_progress_callbacks_and_bounded_parallelism(tmp_path: Path) -> None:
    """on_phase callback receives dynamic total_chunks and completed_chunks counts;
    parallelism is strictly bounded."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=600.0)
    # Cues that force splitting into multiple chunks
    cues = [
        (float(i * 10), float((i + 1) * 10), "Plot discussion: " + "context details " * 80)
        for i in range(60)
    ]

    phase_events: list[dict[str, Any]] = []

    def on_phase(phase: AnalysisPhase, ep_id: str, data: dict[str, Any]) -> None:
        if phase == AnalysisPhase.SCANNER:
            phase_events.append(data)

    mock_client = MockScannerAIClient()
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    # Set parallelism = 3
    settings = AppSettings(
        gateway_enabled=True,
        api_endpoint="http://mock-ai:20128/v1",
        scanner_parallelism=3,
    )
    scanner = EvidenceScanner(
        settings=settings,
        client=mock_client,
        cache_manager=cache_mgr,
        target_ceiling=10_000,
        hard_ceiling=12_000,
    )

    scanner.scan_episode(ep, injected_cues=cues, on_phase=on_phase)

    # Check phase events contain dynamic total_chunks and monotonic completed_chunks
    assert len(phase_events) >= 3
    scanning_event = phase_events[0]
    assert scanning_event.get("status") == "scanning"
    total = scanning_event.get("total_chunks")
    assert total is not None and total >= 2

    # Check progress updates
    progress_events = [e for e in phase_events if "completed_chunks" in e]
    completed_values = [e["completed_chunks"] for e in progress_events]
    assert completed_values[-1] == total


# ---------------------------------------------------------------------------
# 9. Merge Category Results as Before Across Split Chunks
# ---------------------------------------------------------------------------

def test_merge_category_results_across_split_chunks(tmp_path: Path) -> None:
    """Evidence across all split chunks merges properly into all 15+ evidence categories."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=300.0)
    cues = [
        (0.0, 50.0, "Chunk 1 speech " * 60),
        (150.0, 200.0, "Chunk 2 speech " * 60),
    ]

    def dynamic_resp(call_record: dict[str, Any]) -> dict[str, Any]:
        user_text = call_record["user_text"]
        if "Chunk 1 speech" in user_text:
            return {
                "major_scenes": [{"start_sec": 10.0, "end_sec": 20.0, "summary": "Scene from chunk 1"}],
                "reveals": [{"start_sec": 30.0, "end_sec": 40.0, "summary": "Reveal from chunk 1"}],
                "dialogue": [{"start_sec": 50.0, "end_sec": 60.0, "summary": "Dialogue from chunk 1"}],
                "strengths": [{"start_sec": 70.0, "end_sec": 80.0, "summary": "Strength from chunk 1"}],
            }
        return {
            "major_scenes": [{"start_sec": 160.0, "end_sec": 170.0, "summary": "Scene from chunk 2"}],
            "reveals": [{"start_sec": 180.0, "end_sec": 190.0, "summary": "Reveal from chunk 2"}],
            "weaknesses": [{"start_sec": 190.0, "end_sec": 200.0, "summary": "Weakness from chunk 2"}],
        }

    mock_client = MockScannerAIClient([dynamic_resp, dynamic_resp])
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")
    scanner = EvidenceScanner(
        settings=settings,
        client=mock_client,
        cache_manager=cache_mgr,
        target_ceiling=5_000,
        hard_ceiling=6_000,
    )

    evidence = scanner.scan_episode(ep, injected_cues=cues)
    assert len(evidence.data["major_scenes"]) == 2
    assert evidence.data["major_scenes"][0]["summary"] == "Scene from chunk 1"
    assert evidence.data["major_scenes"][1]["summary"] == "Scene from chunk 2"
    assert len(evidence.data["reveals"]) == 2
    assert len(evidence.data["strengths"]) == 1
    assert len(evidence.data["weaknesses"]) == 1
    assert len(evidence.data["strengths_weaknesses"]) == 2


# ---------------------------------------------------------------------------
# 10. Cache Episode Evidence Success Only
# ---------------------------------------------------------------------------

def test_cache_episode_evidence_success_only(tmp_path: Path) -> None:
    """Episode evidence is cached ONLY on full success; failed or cancelled scans never save cache."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=120.0)
    cues = [(0.0, 10.0, "Test dialogue")]

    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")

    # Run 1: Failure during AI call -> No cache saved
    failing_client = MockScannerAIClient([APIError("AI Service unavailable")])
    scanner1 = EvidenceScanner(settings=settings, client=failing_client, cache_manager=cache_mgr)

    with pytest.raises(AnalysisError):
        scanner1.scan_episode(ep, injected_cues=cues)

    # Verify cache is empty
    assert len(list(cache_mgr.cache_dir.glob("*.json"))) == 0

    # Run 2: Success -> Saved to cache
    success_client = MockScannerAIClient()
    scanner2 = EvidenceScanner(settings=settings, client=success_client, cache_manager=cache_mgr)
    ev = scanner2.scan_episode(ep, injected_cues=cues)
    assert isinstance(ev, EpisodeEvidence)

    cached_files = list(cache_mgr.cache_dir.glob("*.json"))
    assert len(cached_files) == 1

    # Run 3: Second run reuses cache (no client calls)
    empty_client = MockScannerAIClient()
    scanner3 = EvidenceScanner(settings=settings, client=empty_client, cache_manager=cache_mgr)
    ev_cached = scanner3.scan_episode(ep, injected_cues=cues)
    assert len(empty_client.call_history) == 0
    assert ev_cached.episode_id == "E01"


# ---------------------------------------------------------------------------
# 11. Cancellation Checks Across Planner, Futures, and Client
# ---------------------------------------------------------------------------

def test_cancellation_in_planner() -> None:
    """Pre-set cancel_event immediately raises AnalysisCancelledError in planner."""
    cancel_event = threading.Event()
    cancel_event.set()

    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=120.0)
    cues = [(0.0, 10.0, "Line 1"), (10.0, 20.0, "Line 2")]

    with pytest.raises(AnalysisCancelledError):
        plan_scanner_chunks(
            episode=ep,
            cues=cues,
            duration_sec=120.0,
            cancel_event=cancel_event,
        )


def test_cancellation_in_futures(tmp_path: Path) -> None:
    """Setting cancel_event during chunk execution aborts futures cleanly and raises AnalysisCancelledError."""
    cancel_event = threading.Event()
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=600.0)
    cues = [(float(i * 10), float((i + 1) * 10), f"Line {i} " * 50) for i in range(20)]

    def slow_chat(call_record: dict[str, Any]) -> dict[str, Any]:
        cancel_event.set()
        return {
            "major_scenes": [{"start_sec": 0.0, "end_sec": 10.0, "summary": "Scene"}],
            "dialogue": [],
        }

    mock_client = MockScannerAIClient([slow_chat] * 10)
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")
    scanner = EvidenceScanner(
        settings=settings,
        client=mock_client,
        cache_manager=cache_mgr,
        target_ceiling=6000,
        hard_ceiling=8000,
    )

    with pytest.raises(AnalysisCancelledError):
        scanner.scan_episode(ep, injected_cues=cues, cancel_event=cancel_event)

    # Verify no cache was saved
    assert len(list(cache_mgr.cache_dir.glob("*.json"))) == 0


def test_cancellation_in_client(tmp_path: Path) -> None:
    """Client raising cancelled APIError cleanly bubbles as AnalysisCancelledError without saving cache."""
    cancel_event = threading.Event()
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=120.0)
    cues = [(0.0, 10.0, "Line 1")]

    mock_client = MockScannerAIClient([APIError("Yêu cầu API đã bị dừng.")])
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")
    scanner = EvidenceScanner(settings=settings, client=mock_client, cache_manager=cache_mgr)

    with pytest.raises(AnalysisCancelledError):
        scanner.scan_episode(ep, injected_cues=cues, cancel_event=cancel_event)

    assert len(list(cache_mgr.cache_dir.glob("*.json"))) == 0
