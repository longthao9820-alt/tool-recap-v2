"""Tests for coverage-green-r8 (Objective rev 8).

Survey unit matrix:
1. validate_gap_result_schema (valid dict, empty dict, non-dict, non-list, non-dict items, metadata ignored)
2. TimeRange (normalization, duration, contains, intersects, overlap, merge, serialization)
3. merge_intervals and subtract_intervals (merging adjacent/overlapping, disjoint remainder math)
4. TimelineCoverage (attributes, ratio calculations, serialization roundtrip)
5. Audit records (ChunkCoverageRecord, CategoryCoverageRecord, CharacterCoverageRecord, CoverageGap)
6. CoverageStatus & GapType (enum semantics, str equality compatibility, hashing)
7. detect_coverage_gaps (UNOBSERVED_SILENCE, DROPPED_TRANSCRIPT, CHUNK_SPARSITY, CHARACTER_DEFICIT, CATEGORY_DEFICIT)
8. reconcile_coverage_gaps (preserves RESOLVED / PERSISTENT_EMPTY, carries forward, adds new)
9. compute_episode_coverage (status evaluation: EMPTY_NO_DATA, PARTIAL_WITH_GAPS, COMPLETE_TRANSCRIPT_ONLY, COMPLETE)
10. plan_second_pass_requests (gap merging, anchor compaction, cue index half recursion, single cue cap, ceiling bounds)
11. merge_second_pass_results (normalization, TRANSCRIPT_GROUNDED tag, deduplication, character merging)

7 Integration scenarios:
1. Second pass triggered on gap
2. Valid empty {} marks persistent empty and caches
3. Results merge/dedup across multi-fragment envelopes
4. Cache hit skips AI call
5. Malformed schema rejects without caching
6. Response cached before cancellation error raised
7. Truthful offline produces dialogue-only with source_modality="transcript"
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

from toolrecap_v2.analyzer.coverage import (
    HARD_PAYLOAD_CEILING,
    STANDARD_EVIDENCE_CATEGORIES,
    TARGET_PAYLOAD_CEILING,
    CategoryCoverageRecord,
    CharacterCoverageRecord,
    ChunkCoverageRecord,
    CoverageGap,
    CoverageStatus,
    EpisodeCoverage,
    GapType,
    SecondPassRequest,
    TimelineCoverage,
    TimeRange,
    compute_episode_coverage,
    detect_coverage_gaps,
    merge_intervals,
    merge_second_pass_results,
    plan_second_pass_requests,
    reconcile_coverage_gaps,
    subtract_intervals,
    validate_gap_result_schema,
)
from toolrecap_v2.analyzer.errors import AnalysisCancelledError, AnalysisError
from toolrecap_v2.analyzer.evidence import (
    EVIDENCE_CATEGORIES,
    EvidenceScanner,
    ScannerChunkPlan,
)
from toolrecap_v2.analyzer.phases import AnalysisPhase
from toolrecap_v2.domain import (
    EvidenceCacheManager,
    ScannerDirective,
    SourceEpisode,
    compute_gap_cache_key,
)
from toolrecap_v2.domain.models import EpisodeEvidence
from toolrecap_v2.settings import AppSettings
from toolrecap_v2.subtitles.models import SubtitleCue


# ===========================================================================
# Mock AI Client for Integration Scenarios
# ===========================================================================

class CoverageMockAIClient:
    """Mock AI client accounting for first-pass scanner chunks then gap second-pass."""

    def __init__(
        self,
        responses: list[Any] | None = None,
        default_chunk_resp: dict[str, Any] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.default_chunk_resp = default_chunk_resp
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
            raise AnalysisCancelledError("Cancelled in mock client.")

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

        if self.default_chunk_resp is not None:
            return copy.deepcopy(self.default_chunk_resp)

        return {"major_scenes": [{"start_sec": 0.0, "end_sec": 5.0, "summary": "Default scene"}]}


# ===========================================================================
# SURVEY UNIT MATRIX
# ===========================================================================

# 1. validate_gap_result_schema
def test_validate_gap_result_schema() -> None:
    """Schema validation checks dict structure, list types, dict items, and empty validity."""
    # Valid dict with category lists of dicts
    valid = {
        "major_scenes": [{"start_sec": 10.0, "end_sec": 20.0, "summary": "Test scene"}],
        "dialogue": [{"start_sec": 12.0, "end_sec": 18.0, "summary": "Test dialogue"}],
    }
    assert validate_gap_result_schema(valid) is True
    assert validate_gap_result_schema(valid, raise_error=True) is True

    # Empty dict is valid semantic empty
    assert validate_gap_result_schema({}) is True
    assert validate_gap_result_schema({}, raise_error=True) is True

    # Unknown / metadata fields ignored
    with_metadata = {
        "episode_id": "EP01",
        "range_start_ms": 10000,
        "range_end_ms": 20000,
        "gap_id": "gap_01",
        "metadata": {"note": "ignored"},
        "unknown_custom_field": "some string",
        "major_scenes": [{"start_sec": 10.0, "end_sec": 20.0}],
    }
    assert validate_gap_result_schema(with_metadata) is True

    # Non-dict root fails
    assert validate_gap_result_schema(["not", "a", "dict"]) is False
    with pytest.raises(AnalysisError, match="phải là dict"):
        validate_gap_result_schema(["not", "a", "dict"], raise_error=True)

    # Known category non-list fails
    invalid_cat_type = {"major_scenes": "string-not-list"}
    assert validate_gap_result_schema(invalid_cat_type) is False
    with pytest.raises(AnalysisError, match="phải là kiểu list"):
        validate_gap_result_schema(invalid_cat_type, raise_error=True)

    # List items non-dict fails
    invalid_item_type = {"major_scenes": ["string-item-not-dict"]}
    assert validate_gap_result_schema(invalid_item_type) is False
    with pytest.raises(AnalysisError, match="phải là dict"):
        validate_gap_result_schema(invalid_item_type, raise_error=True)


# 2. TimeRange operations and normalization
def test_time_range_operations_and_serialization() -> None:
    """TimeRange normalizes bounds, computes relations, and serializes cleanly."""
    r1 = TimeRange(10.5, 20.25)
    assert r1.start_sec == 10.5
    assert r1.end_sec == 20.25
    assert r1.duration == 9.75
    assert r1.contains_sec(15.0) is True
    assert r1.contains_sec(25.0) is False

    # Negative bounds clamped
    r_neg = TimeRange(-5.0, 15.0)
    assert r_neg.start_sec == 0.0

    # Inverted bounds normalized
    r_inv = TimeRange(30.0, 10.0)
    assert r_inv.start_sec == 30.0
    assert r_inv.end_sec == 30.0

    # Intersection and overlap
    r2 = TimeRange(15.0, 25.0)
    assert r1.intersects(r2) is True
    assert r1.overlap(r2) == 5.25

    r3 = TimeRange(30.0, 40.0)
    assert r1.intersects(r3) is False
    assert r1.overlap(r3) == 0.0

    # Merge
    merged = r1.merge(r2)
    assert merged.start_sec == 10.5
    assert merged.end_sec == 25.0

    # Serialization roundtrip
    d = r1.to_dict()
    assert d == {"start_sec": 10.5, "end_sec": 20.25}
    assert TimeRange.from_dict(d) == r1
    assert TimeRange.from_dict([10.5, 20.25]) == r1
    assert TimeRange.from_dict({"start_ms": 10500, "end_ms": 20250}) == r1
    assert r1.to_list() == [10.5, 20.25]


# 3. merge_intervals and subtract_intervals
def test_intervals_merge_and_subtract() -> None:
    """Interval arithmetic handles empty, adjacent, overlapping, and split remainder cases."""
    assert merge_intervals([]) == []

    # Overlapping and touching
    intervals = [
        TimeRange(0.0, 10.0),
        TimeRange(9.98, 20.0),
        TimeRange(30.0, 40.0),
        TimeRange(40.02, 50.0),
    ]
    merged = merge_intervals(intervals)
    assert len(merged) == 2
    assert merged[0] == TimeRange(0.0, 20.0)
    assert merged[1] == TimeRange(30.0, 50.0)

    # Subtraction: base [0, 100], subtract [20, 40] and [60, 80]
    base = [TimeRange(0.0, 100.0)]
    to_sub = [TimeRange(20.0, 40.0), TimeRange(60.0, 80.0)]
    remainders = subtract_intervals(base, to_sub)
    assert len(remainders) == 3
    assert remainders[0] == TimeRange(0.0, 20.0)
    assert remainders[1] == TimeRange(40.0, 60.0)
    assert remainders[2] == TimeRange(80.0, 100.0)

    # Subtraction: exact match leaves empty
    assert subtract_intervals([TimeRange(10.0, 20.0)], [TimeRange(10.0, 20.0)]) == []


# 4. TimelineCoverage
def test_timeline_coverage_roundtrip() -> None:
    """TimelineCoverage holds intervals, calculates coverage ratios, and roundtrips to dict."""
    tc = TimelineCoverage(
        transcript_ranges=[TimeRange(0.0, 60.0)],
        visual_ranges=[],
        evidence_ranges=[TimeRange(10.0, 50.0)],
        unobserved_ranges=[TimeRange(60.0, 120.0)],
        transcript_coverage_ratio=0.5,
        evidence_coverage_ratio=0.3333,
        visual_coverage_ratio=0.0,
    )
    d = tc.to_dict()
    assert d["transcript_ranges"] == [[0.0, 60.0]]
    assert d["visual_ranges"] == []
    assert d["transcript_coverage_ratio"] == 0.5

    loaded = TimelineCoverage.from_dict(d)
    assert loaded.transcript_coverage_ratio == tc.transcript_coverage_ratio
    assert loaded.evidence_ranges == tc.evidence_ranges
    assert loaded.visual_ranges == []


# 5. Audit records
def test_coverage_audit_records_and_roundtrip() -> None:
    """Audit record dataclasses serialize and deserialize faithfully."""
    # ChunkCoverageRecord
    ch = ChunkCoverageRecord(
        chunk_index=0,
        start_sec=0.0,
        end_sec=60.0,
        cues_count=10,
        items_count=3,
        categories_present=["major_scenes", "dialogue"],
        characters_present=["Alice", "Bob"],
        payload_bytes=4500,
        status="OK",
    )
    ch_d = ch.to_dict()
    assert ChunkCoverageRecord.from_dict(ch_d) == ch

    # CategoryCoverageRecord
    cat = CategoryCoverageRecord(
        category="major_scenes",
        count=4,
        density_per_minute=2.0,
        first_occurrence_sec=5.0,
        last_occurrence_sec=115.0,
        is_empty=False,
        verification_status="PRESENT",
    )
    cat_d = cat.to_dict()
    assert CategoryCoverageRecord.from_dict(cat_d) == cat

    # CharacterCoverageRecord
    char = CharacterCoverageRecord(
        character_name="Alice",
        dialogue_mention_count=12,
        evidence_item_count=5,
        first_seen_sec=2.0,
        last_seen_sec=118.0,
        categories_involved=["major_scenes", "character_decisions"],
        is_supporting=False,
        has_decisions_or_actions=True,
    )
    char_d = char.to_dict()
    assert CharacterCoverageRecord.from_dict(char_d) == char

    # CoverageGap
    gap = CoverageGap(
        gap_id="gap_01",
        gap_type=GapType.DROPPED_TRANSCRIPT,
        start_sec=20.0,
        end_sec=55.0,
        target_categories=["major_scenes", "dialogue"],
        target_characters=["Alice"],
        is_actionable=True,
        status="PENDING",
    )
    gap_d = gap.to_dict()
    assert CoverageGap.from_dict(gap_d) == gap


# 6. CoverageStatus and GapType
def test_coverage_status_and_gap_type_enums() -> None:
    """CoverageStatus supports str equality shortcuts, distinct values, and hashing."""
    # CoverageStatus equality
    assert CoverageStatus.COMPLETE == "COMPLETE"
    assert CoverageStatus.COMPLETE == "complete"
    assert CoverageStatus.COMPLETE_TRANSCRIPT_ONLY == "COMPLETE_TRANSCRIPT_ONLY"
    assert CoverageStatus.COMPLETE_TRANSCRIPT_ONLY == "complete"
    assert CoverageStatus.COMPLETE_TRANSCRIPT_ONLY != CoverageStatus.COMPLETE
    assert CoverageStatus.PARTIAL_WITH_GAPS == "PARTIAL_WITH_GAPS"
    assert CoverageStatus.EMPTY_NO_DATA == "EMPTY_NO_DATA"

    # GapType values
    assert GapType.DROPPED_TRANSCRIPT.value == "DROPPED_TRANSCRIPT"
    assert GapType.CHARACTER_DEFICIT.value == "CHARACTER_DEFICIT"
    assert GapType.CATEGORY_DEFICIT.value == "CATEGORY_DEFICIT"
    assert GapType.CHUNK_SPARSITY.value == "CHUNK_SPARSITY"
    assert GapType.UNOBSERVED_SILENCE.value == "UNOBSERVED_SILENCE"

    # Set and dict hashing
    statuses = {CoverageStatus.COMPLETE, CoverageStatus.COMPLETE_TRANSCRIPT_ONLY}
    assert len(statuses) == 2


# 7. detect_coverage_gaps
def test_detect_coverage_gaps_all_types() -> None:
    """detect_coverage_gaps accurately detects silence, dropped transcript, sparsity, character, and category deficits."""
    # 7.1 UNOBSERVED_SILENCE: 40s gap without cues or visuals
    cues_silence = [(0.0, 10.0, "Hello"), (50.0, 60.0, "World")]
    gaps_silence = detect_coverage_gaps(duration_sec=60.0, cues=cues_silence, evidence_data={})
    silence_gaps = [g for g in gaps_silence if g.gap_type == GapType.UNOBSERVED_SILENCE]
    assert len(silence_gaps) == 1
    assert silence_gaps[0].start_sec == 10.0
    assert silence_gaps[0].end_sec == 50.0
    assert silence_gaps[0].is_actionable is False
    assert silence_gaps[0].status == "RESOLVED"

    # 7.2 DROPPED_TRANSCRIPT: continuous span >= 30s with >= 3 cues and 0 evidence
    cues_dropped = [
        (10.0, 20.0, "Speaker: Dialogue line 1"),
        (20.0, 30.0, "Speaker: Dialogue line 2"),
        (30.0, 45.0, "Speaker: Dialogue line 3"),
    ]
    gaps_dropped = detect_coverage_gaps(duration_sec=60.0, cues=cues_dropped, evidence_data={})
    dropped_gaps = [g for g in gaps_dropped if g.gap_type == GapType.DROPPED_TRANSCRIPT]
    assert len(dropped_gaps) == 1
    assert dropped_gaps[0].is_actionable is True
    assert dropped_gaps[0].status == "PENDING"

    # 7.3 CHUNK_SPARSITY: chunk with >= 15 cues and <= 1 evidence item
    planned_chunk = ScannerChunkPlan(
        chunk_index=0,
        start_sec=0.0,
        end_sec=60.0,
        cues=[(float(i), float(i + 1), f"Line {i}") for i in range(16)],
        user_text="",
        estimated_bytes=2000,
    )
    gaps_sparse = detect_coverage_gaps(
        duration_sec=60.0,
        cues=planned_chunk.cues,
        evidence_data={"major_scenes": [{"start_sec": 0.0, "end_sec": 5.0, "summary": "Single scene"}]},
        planned_chunks=[planned_chunk],
    )
    sparse_gaps = [g for g in gaps_sparse if g.gap_type == GapType.CHUNK_SPARSITY]
    assert len(sparse_gaps) == 1

    # 7.4 CHARACTER_DEFICIT: character with >= 4 speaker tags and <= 1 evidence
    cues_char = [
        (10.0, 15.0, "Bob: Hello there"),
        (20.0, 25.0, "Bob: How are you?"),
        (30.0, 35.0, "Bob: Look at that"),
        (40.0, 45.0, "Bob: Good bye"),
    ]
    gaps_char = detect_coverage_gaps(duration_sec=60.0, cues=cues_char, evidence_data={})
    char_gaps = [g for g in gaps_char if g.gap_type == GapType.CHARACTER_DEFICIT]
    assert len(char_gaps) == 1
    assert "Bob" in char_gaps[0].target_characters

    # 7.5 CATEGORY_DEFICIT: >= 20 cues, >= 300s duration, missing critical categories
    cues_cat = [(float(i * 10), float(i * 10 + 5), f"Speaker: Line {i}") for i in range(25)]
    gaps_cat = detect_coverage_gaps(
        duration_sec=320.0,
        cues=cues_cat,
        evidence_data={"major_scenes": [{"start_sec": 10.0, "end_sec": 20.0, "summary": "Scene"}]},
    )
    cat_gaps = [g for g in gaps_cat if g.gap_type == GapType.CATEGORY_DEFICIT]
    assert len(cat_gaps) >= 1
    assert any("conflicts" in g.target_categories for g in cat_gaps)


# 8. reconcile_coverage_gaps
def test_reconcile_coverage_gaps() -> None:
    """reconcile_coverage_gaps preserves prior RESOLVED and PERSISTENT_EMPTY statuses."""
    prior_resolved = CoverageGap(
        gap_id="gap_dropped_01",
        gap_type=GapType.DROPPED_TRANSCRIPT,
        start_sec=10.0,
        end_sec=45.0,
        target_categories=["major_scenes"],
        target_characters=[],
        is_actionable=True,
        status="RESOLVED",
    )
    prior_persistent = CoverageGap(
        gap_id="gap_cat_01",
        gap_type=GapType.CATEGORY_DEFICIT,
        start_sec=0.0,
        end_sec=300.0,
        target_categories=["conflicts"],
        target_characters=[],
        is_actionable=True,
        status="PERSISTENT_EMPTY",
    )

    # Fresh detection detects same dropped transcript gap and a new sparsity gap
    fresh_detected = [
        CoverageGap(
            gap_id="fresh_01",
            gap_type=GapType.DROPPED_TRANSCRIPT,
            start_sec=10.0,
            end_sec=45.0,
            target_categories=["major_scenes"],
            status="PENDING",
        ),
        CoverageGap(
            gap_id="fresh_02",
            gap_type=GapType.CHUNK_SPARSITY,
            start_sec=100.0,
            end_sec=150.0,
            status="PENDING",
        ),
    ]

    reconciled = reconcile_coverage_gaps(fresh_detected, [prior_resolved, prior_persistent])
    rec_by_id = {g.gap_id: g for g in reconciled}

    # Prior resolved status was preserved on fresh_01
    assert rec_by_id["gap_dropped_01"].status == "RESOLVED"
    # Prior persistent empty gap carried forward
    assert rec_by_id["gap_cat_01"].status == "PERSISTENT_EMPTY"
    # New gap added as pending
    assert rec_by_id["fresh_02"].status == "PENDING"


# 9. compute_episode_coverage
def test_compute_episode_coverage_statuses() -> None:
    """compute_episode_coverage assigns appropriate CoverageStatus based on speech, visual, and gaps."""
    # 9.1 EMPTY_NO_DATA: no duration or no cues and no visuals
    cov_empty = compute_episode_coverage(
        episode_id="E01",
        duration_sec=100.0,
        cues=[],
        evidence_data={},
        visual_spans=[],
    )
    assert cov_empty.status == CoverageStatus.EMPTY_NO_DATA

    # 9.2 PARTIAL_WITH_GAPS: actionable pending gaps exist
    cues_partial = [
        (10.0, 20.0, "Speaker: Speech 1"),
        (20.0, 30.0, "Speaker: Speech 2"),
        (30.0, 45.0, "Speaker: Speech 3"),
    ]
    cov_partial = compute_episode_coverage(
        episode_id="E01",
        duration_sec=60.0,
        cues=cues_partial,
        evidence_data={},
    )
    assert cov_partial.status == CoverageStatus.PARTIAL_WITH_GAPS

    # 9.3 COMPLETE_TRANSCRIPT_ONLY: speech covered, no actionable pending gaps, visual_ranges empty
    cov_transcript = compute_episode_coverage(
        episode_id="E01",
        duration_sec=60.0,
        cues=cues_partial,
        evidence_data={"major_scenes": [{"start_sec": 10.0, "end_sec": 45.0, "summary": "Scene"}]},
    )
    assert cov_transcript.status == CoverageStatus.COMPLETE_TRANSCRIPT_ONLY
    assert cov_transcript.status == "complete"
    assert cov_transcript.timeline.visual_ranges == []

    # 9.4 COMPLETE: visual ranges present with coverage ratio >= 0.85
    cov_full = compute_episode_coverage(
        episode_id="E01",
        duration_sec=60.0,
        cues=cues_partial,
        evidence_data={"major_scenes": [{"start_sec": 10.0, "end_sec": 45.0, "summary": "Scene"}]},
        visual_spans=[(0.0, 55.0)],
    )
    assert cov_full.status == CoverageStatus.COMPLETE


# 10. plan_second_pass_requests
def test_plan_second_pass_requests() -> None:
    """plan_second_pass_requests merges adjacent gaps, splits cues by halves, caps oversized cues, and obeys ceiling."""
    # Adjacent gaps within 15s are merged
    gap1 = CoverageGap(
        gap_id="gap_01",
        gap_type=GapType.DROPPED_TRANSCRIPT,
        start_sec=10.0,
        end_sec=30.0,
        target_categories=["major_scenes"],
        target_characters=["Alice"],
        is_actionable=True,
        status="PENDING",
    )
    gap2 = CoverageGap(
        gap_id="gap_02",
        gap_type=GapType.DROPPED_TRANSCRIPT,
        start_sec=35.0,
        end_sec=60.0,
        target_categories=["dialogue"],
        target_characters=["Bob"],
        is_actionable=True,
        status="PENDING",
    )
    cues = [
        (10.0, 25.0, "Cues in gap 1 " * 10),
        (35.0, 55.0, "Cues in gap 2 " * 10),
    ]

    requests = plan_second_pass_requests(
        episode_id="E01",
        source_video="ep01.mp4",
        duration_seconds=100.0,
        gaps=[gap1, gap2],
        cues=cues,
        existing_evidence={"major_scenes": [{"start_sec": 0.0, "end_sec": 5.0, "summary": "Anchor"}]},
        model="mock-model",
    )
    assert len(requests) == 1
    req = requests[0]
    assert "gap_01" in req.gap_ids
    assert "gap_02" in req.gap_ids
    assert "Alice" in req.target_characters
    assert "Bob" in req.target_characters
    assert req.estimated_bytes <= TARGET_PAYLOAD_CEILING

    # Cue recursion splitting when target_ceiling is tight
    dense_cues = [(float(i * 2), float(i * 2 + 1), "Text " * 20) for i in range(10)]
    gap_split = CoverageGap(
        gap_id="gap_split",
        gap_type=GapType.DROPPED_TRANSCRIPT,
        start_sec=0.0,
        end_sec=20.0,
        target_categories=["major_scenes"],
        is_actionable=True,
        status="PENDING",
    )
    split_reqs = plan_second_pass_requests(
        episode_id="E01",
        source_video="ep01.mp4",
        duration_seconds=30.0,
        gaps=[gap_split],
        cues=dense_cues,
        existing_evidence={},
        model="mock-model",
        target_ceiling=600,
    )
    assert len(split_reqs) > 1
    for r in split_reqs:
        assert r.gap_ids == ["gap_split"]
        assert r.estimated_bytes <= 600

    # Single oversized cue capped
    huge_cue = [(0.0, 10.0, "A" * 5000)]
    capped_reqs = plan_second_pass_requests(
        episode_id="E01",
        source_video="ep01.mp4",
        duration_seconds=20.0,
        gaps=[gap_split],
        cues=huge_cue,
        existing_evidence={},
        model="mock-model",
        target_ceiling=600,
    )
    assert len(capped_reqs) == 1
    assert "[capped]" in capped_reqs[0].cues[0][2]


# 11. merge_second_pass_results
def test_merge_second_pass_results() -> None:
    """merge_second_pass_results normalizes items, sets TRANSCRIPT_GROUNDED, and deduplicates."""
    base_data: dict[str, list[dict[str, Any]]] = {
        "major_scenes": [
            {
                "start_sec": 10.0,
                "end_sec": 20.0,
                "summary": "Alice confronts Bob about the missing documents",
                "characters": ["Alice", "Bob"],
                "source_modality": "transcript",
            }
        ],
        "dialogue": [],
    }

    # Second pass returns a duplicate item (similar text and close time) with extra character,
    # plus a completely new item
    second_pass_res = {
        "major_scenes": [
            {
                "start_sec": 11.0,
                "end_sec": 20.5,
                "summary": "Alice confronts Bob about missing documents",
                "characters": ["Charlie"],
            },
            {
                "start_sec": 50.0,
                "end_sec": 60.0,
                "summary": "Eve discovers the hidden recording device",
                "characters": ["Eve"],
            },
        ]
    }

    merged, added = merge_second_pass_results(
        base_evidence_data=base_data,
        second_pass_results=[second_pass_res],
        episode_id="E01",
        duration_sec=100.0,
    )

    # 1 duplicate merged, 1 new item added
    assert added == 1
    assert len(merged["major_scenes"]) == 2

    # Duplicate merged Charlie into character list
    ex_item = merged["major_scenes"][0]
    assert "Charlie" in ex_item["characters"]
    assert "Alice" in ex_item["characters"]

    # New item has TRANSCRIPT_GROUNDED source modality
    new_item = merged["major_scenes"][1]
    assert new_item["source_modality"] == "TRANSCRIPT_GROUNDED"
    assert new_item["summary"] == "Eve discovers the hidden recording device"


# ===========================================================================
# 7 INTEGRATION SCENARIOS
# ===========================================================================

# Scenario 1: Second pass triggered on gap
def test_integration_scenario_1_second_pass_triggered_on_gap(tmp_path: Path) -> None:
    """Scanner detects actionable gap, plans second-pass request, calls AI, and merges items."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=120.0)
    cues = [
        (0.0, 10.0, "Intro narration"),
        (40.0, 50.0, "Detective: The evidence was moved."),
        (50.0, 60.0, "Suspect: I know nothing about it."),
        (60.0, 75.0, "Detective: Look at these photographs."),
    ]

    # First pass: chunk returns evidence only for 0-10s -> leaves 40-75s dropped transcript gap
    chunk_resp = {
        "major_scenes": [{"start_sec": 0.0, "end_sec": 10.0, "summary": "Intro scene"}],
        "dialogue": [],
    }
    # Second pass: gap call returns evidence for the dropped span
    gap_resp = {
        "dialogue": [
            {"start_sec": 42.0, "end_sec": 70.0, "summary": "Interrogation dialogue", "characters": ["Detective", "Suspect"]}
        ]
    }

    mock_client = CoverageMockAIClient([chunk_resp, gap_resp])
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")
    scanner = EvidenceScanner(settings=settings, client=mock_client, cache_manager=cache_mgr)

    phases_recorded: list[dict[str, Any]] = []

    def on_phase(phase: AnalysisPhase, ep_id: str, data: dict[str, Any]) -> None:
        phases_recorded.append(data)

    evidence = scanner.scan_episode(ep, injected_cues=cues, on_phase=on_phase)

    # 2 calls made: chunk first-pass + gap second-pass
    assert len(mock_client.call_history) == 2
    assert "TARGETED EVIDENCE GAP" in mock_client.call_history[1]["user_text"]

    # Phase callbacks observed
    statuses = [p.get("status") for p in phases_recorded]
    assert "coverage_check" in statuses
    assert "second_pass" in statuses
    assert "complete" in statuses

    # Gap resolved and items merged
    assert len(evidence.data["dialogue"]) == 1
    assert evidence.data["dialogue"][0]["source_modality"] == "TRANSCRIPT_GROUNDED"
    gaps = evidence.coverage.get("gaps", [])
    dropped = [g for g in gaps if g.get("gap_type") == GapType.DROPPED_TRANSCRIPT.value]
    assert len(dropped) == 1
    assert dropped[0]["status"] == "RESOLVED"
    assert evidence.coverage["status"] == "complete"


# Scenario 2: Valid empty {} marks persistent empty and caches
def test_integration_scenario_2_valid_empty_marks_persistent_empty_and_caches(tmp_path: Path) -> None:
    """Gap call returning valid empty {} marks gap PERSISTENT_EMPTY and caches gap file."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=120.0)
    cues = [
        (0.0, 10.0, "Intro narration"),
        (40.0, 50.0, "Whisper in shadows"),
        (50.0, 60.0, "Indistinct murmur"),
        (60.0, 75.0, "Footsteps walking away"),
    ]

    chunk_resp = {
        "major_scenes": [{"start_sec": 0.0, "end_sec": 10.0, "summary": "Intro scene"}],
        "dialogue": [],
    }
    # Second pass confirms nothing of note occurred in this gap
    gap_resp = {}

    mock_client = CoverageMockAIClient([chunk_resp, gap_resp])
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")
    scanner = EvidenceScanner(settings=settings, client=mock_client, cache_manager=cache_mgr)

    evidence = scanner.scan_episode(ep, injected_cues=cues)

    # Gap marked PERSISTENT_EMPTY
    gaps = evidence.coverage.get("gaps", [])
    dropped = [g for g in gaps if g.get("gap_type") == GapType.DROPPED_TRANSCRIPT.value]
    assert len(dropped) == 1
    assert dropped[0]["status"] == "PERSISTENT_EMPTY"

    # Coverage status completes (no pending actionable gaps)
    assert evidence.coverage["status"] == "complete"

    # Gap cache file exists on disk
    gap_cache_files = list(cache_mgr.gap_dir.glob("*.gap.json"))
    assert len(gap_cache_files) == 1
    cached_content = json.loads(gap_cache_files[0].read_text(encoding="utf-8"))
    assert cached_content["result"] == {}


# Scenario 3: Results merge/dedup across multi-fragment envelopes
def test_integration_scenario_3_multi_fragment_merge_and_dedup(tmp_path: Path) -> None:
    """Multi-fragment gap requests deduplicate overlapping items and accumulate resolved counts."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=120.0)
    cues = [
        (0.0, 5.0, "Intro"),
        (30.0, 40.0, "Dense line A " * 100),
        (40.0, 50.0, "Dense line B " * 100),
        (50.0, 60.0, "Dense line C " * 100),
        (60.0, 70.0, "Dense line D " * 100),
    ]

    chunk_resp = {
        "major_scenes": [{"start_sec": 0.0, "end_sec": 5.0, "summary": "Intro"}],
        "dialogue": [],
    }
    # Fragment 1 response
    gap_frag1 = {
        "major_scenes": [
            {"start_sec": 32.0, "end_sec": 45.0, "summary": "Shared conference scene", "characters": ["Alice"]}
        ]
    }
    # Fragment 2 response has duplicate boundary item plus new item
    gap_frag2 = {
        "major_scenes": [
            {"start_sec": 32.0, "end_sec": 45.0, "summary": "Shared conference scene", "characters": ["Bob"]},
            {"start_sec": 55.0, "end_sec": 68.0, "summary": "Signing the treaty", "characters": ["Alice", "Bob"]},
        ]
    }

    def dispatch(call: dict[str, Any]) -> dict[str, Any]:
        if "TARGETED EVIDENCE GAP SECOND PASS" in call["user_text"]:
            dispatch.gap_index += 1
            return gap_frag1 if dispatch.gap_index == 1 else gap_frag2
        dispatch.scan_index += 1
        return chunk_resp if dispatch.scan_index == 1 else {}

    dispatch.gap_index = 0
    dispatch.scan_index = 0
    mock_client = CoverageMockAIClient([dispatch] * 12)
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")
    # Set low target_ceiling to force gap request to split into 2 subchunk fragments
    scanner = EvidenceScanner(
        settings=settings,
        client=mock_client,
        cache_manager=cache_mgr,
        target_ceiling=6500,
        hard_ceiling=7000,
    )

    evidence = scanner.scan_episode(ep, injected_cues=cues)

    # Both fragments merged: total major scenes is 3 (intro + 1 shared deduplicated + 1 treaty)
    assert len(evidence.data["major_scenes"]) == 3
    shared_scene = [s for s in evidence.data["major_scenes"] if "conference" in s["summary"]][0]
    assert "Alice" in shared_scene["characters"]
    assert "Bob" in shared_scene["characters"]

    # Gap status is RESOLVED
    gaps = evidence.coverage.get("gaps", [])
    dropped = [g for g in gaps if g.get("gap_type") == GapType.DROPPED_TRANSCRIPT.value]
    assert len(dropped) == 1
    assert dropped[0]["status"] == "RESOLVED"


# Scenario 4: Cache hit skips AI call
def test_integration_scenario_4_cache_hit_skips_ai_call(tmp_path: Path) -> None:
    """Subsequent scan loads gap result from cache without invoking AI client for second-pass."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=120.0)
    cues = [
        (0.0, 10.0, "Intro narration"),
        (40.0, 50.0, "Line 1"),
        (50.0, 60.0, "Line 2"),
        (60.0, 75.0, "Line 3"),
    ]

    chunk_resp = {
        "major_scenes": [{"start_sec": 0.0, "end_sec": 10.0, "summary": "Intro scene"}],
        "dialogue": [],
    }
    gap_resp = {
        "dialogue": [{"start_sec": 42.0, "end_sec": 70.0, "summary": "Cached dialogue line"}]
    }

    cache_dir = tmp_path / "cache"
    cache_mgr = EvidenceCacheManager(cache_dir)
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")

    # Pass 1: Calls chunk then gap
    mock_client1 = CoverageMockAIClient([chunk_resp, gap_resp])
    scanner1 = EvidenceScanner(settings=settings, client=mock_client1, cache_manager=cache_mgr)
    scanner1.scan_episode(ep, injected_cues=cues)
    assert len(mock_client1.call_history) == 2

    # Invalidate episode evidence cache to force re-scan of episode, but retain gap cache
    ep_cache_files = list(cache_dir.glob("*.evidence.json"))
    for f in ep_cache_files:
        f.unlink()

    # Pass 2: Same cues and video -> chunk executes, but gap is loaded from cache
    mock_client2 = CoverageMockAIClient([chunk_resp])
    scanner2 = EvidenceScanner(settings=settings, client=mock_client2, cache_manager=cache_mgr)
    evidence2 = scanner2.scan_episode(ep, injected_cues=cues)

    # Exactly 1 AI call made (chunk only, gap call skipped)
    assert len(mock_client2.call_history) == 1
    assert "TARGETED EVIDENCE GAP" not in mock_client2.call_history[0]["user_text"]
    assert len(evidence2.data["dialogue"]) == 1
    assert evidence2.data["dialogue"][0]["summary"] == "Cached dialogue line"


# Scenario 5: Malformed schema rejects without caching
def test_integration_scenario_5_malformed_schema_rejects_without_caching(tmp_path: Path) -> None:
    """Malformed gap response raises AnalysisError and is never saved to gap cache."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=120.0)
    cues = [
        (0.0, 10.0, "Intro"),
        (40.0, 50.0, "Line 1"),
        (50.0, 60.0, "Line 2"),
        (60.0, 75.0, "Line 3"),
    ]

    chunk_resp = {
        "major_scenes": [{"start_sec": 0.0, "end_sec": 10.0, "summary": "Intro"}],
        "dialogue": [],
    }
    # Malformed gap response: category maps to a non-list string
    malformed_gap_resp = {"major_scenes": "not-a-list-error"}

    mock_client = CoverageMockAIClient([chunk_resp, malformed_gap_resp])
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")
    scanner = EvidenceScanner(settings=settings, client=mock_client, cache_manager=cache_mgr)

    with pytest.raises(AnalysisError, match="phải là kiểu list"):
        scanner.scan_episode(ep, injected_cues=cues)

    # Gap cache directory remains empty
    assert list(cache_mgr.gap_dir.glob("*.gap.json")) == []
    # Episode evidence was not saved
    assert list(cache_mgr.cache_dir.glob("*.evidence.json")) == []


# Scenario 6: Response cached before cancellation error raised
def test_integration_scenario_6_response_cached_before_cancellation(tmp_path: Path) -> None:
    """When cancellation occurs during second-pass, valid response is saved to disk before raising."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=120.0)
    cues = [
        (0.0, 10.0, "Intro"),
        (40.0, 50.0, "Line 1"),
        (50.0, 60.0, "Line 2"),
        (60.0, 75.0, "Line 3"),
    ]

    cancel_event = threading.Event()

    chunk_resp = {
        "major_scenes": [{"start_sec": 0.0, "end_sec": 10.0, "summary": "Intro"}],
        "dialogue": [],
    }

    def gap_cancelling_call(call_record: dict[str, Any]) -> dict[str, Any]:
        # Cancel right after returning valid gap payload
        cancel_event.set()
        return {
            "dialogue": [{"start_sec": 42.0, "end_sec": 70.0, "summary": "Pre-cancel valid dialogue"}]
        }

    mock_client = CoverageMockAIClient([chunk_resp, gap_cancelling_call])
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://mock-ai:20128/v1")
    scanner = EvidenceScanner(settings=settings, client=mock_client, cache_manager=cache_mgr)

    with pytest.raises(AnalysisCancelledError):
        scanner.scan_episode(ep, injected_cues=cues, cancel_event=cancel_event)

    # Valid gap result was saved to disk before cancellation interrupted flow
    gap_cache_files = list(cache_mgr.gap_dir.glob("*.gap.json"))
    assert len(gap_cache_files) == 1
    cached_content = json.loads(gap_cache_files[0].read_text(encoding="utf-8"))
    assert "dialogue" in cached_content["result"]

    # Overall episode evidence was NOT saved due to cancellation
    assert list(cache_mgr.cache_dir.glob("*.evidence.json")) == []


# Scenario 7: Truthful offline produces dialogue-only with source_modality="transcript"
def test_integration_scenario_7_truthful_offline_dialogue_only(tmp_path: Path) -> None:
    """Offline scanning populates dialogue only with transcript modality, omitting fake visual claims."""
    ep = SourceEpisode(episode_id="E01", source_video="ep01.mp4", duration_seconds=100.0)
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    settings = AppSettings(gateway_enabled=False)
    scanner = EvidenceScanner(settings=settings, client=None, cache_manager=cache_mgr)

    # Case 7.1: With dialogue cues
    cues = [
        (10.0, 20.0, "Alice: I have the key."),
        (30.0, 40.0, "Bob: Let us proceed."),
    ]
    evidence = scanner.scan_episode(ep, injected_cues=cues)

    assert len(evidence.data["dialogue"]) == 2
    for item in evidence.data["dialogue"]:
        assert item["source_modality"] == "transcript"

    # All non-dialogue categories remain completely empty (no fake major scenes or visual anchors)
    for cat in EVIDENCE_CATEGORIES:
        if cat != "dialogue":
            assert evidence.data[cat] == [], f"Category {cat} should be empty in truthful offline mode"

    assert evidence.coverage["timeline"]["visual_ranges"] == []
    assert evidence.coverage["status"] == "complete"

    # Case 7.2: No speech -> EMPTY_NO_DATA
    ep_silent = SourceEpisode(episode_id="E02", source_video="ep02.mp4", duration_seconds=100.0)
    evidence_silent = scanner.scan_episode(ep_silent, injected_cues=[])
    for cat in EVIDENCE_CATEGORIES:
        assert evidence_silent.data[cat] == []
    assert evidence_silent.coverage["status"] == CoverageStatus.EMPTY_NO_DATA
