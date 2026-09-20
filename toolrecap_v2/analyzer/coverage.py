"""Deterministic episode coverage ledger, timeline audit, gap detection, and second-pass planning."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..api_client import estimate_request_size
from ..domain.models import _normalize_tokens
from ..domain.policy import ScannerDirective
from ..subtitles.models import SubtitleCue
from .errors import AnalysisError

HARD_PAYLOAD_CEILING: int = 500_000
TARGET_PAYLOAD_CEILING: int = 480_000

STANDARD_EVIDENCE_CATEGORIES: set[str] = {
    "major_scenes",
    "dialogue",
    "character_decisions",
    "conflicts",
    "reveals",
    "reversals",
    "setup_payoff",
    "character_relationships",
    "relationships",
    "supporting_developments",
    "continuity_anchors",
    "recurring_elements",
    "failures",
    "consequences",
    "performance_moments",
    "unresolved",
    "subplots",
    "strengths_weaknesses",
    "contradictions",
    "dilemmas",
    "visual_storytelling",
    "recurring_behavior",
    "power_shifts",
    "reactions",
    "counter_evidence",
}


def validate_gap_result_schema(
    raw: Any,
    allowed_categories: Sequence[str] | set[str] | None = None,
    *,
    raise_error: bool = False,
) -> bool:
    """Validate second-pass gap result schema.

    Requirements:
    - Must be a dictionary.
    - Empty `{}` is valid semantic empty (confirms persistent empty).
    - Known category keys must map to lists of dicts.
    - Rejects values that are non-list for known category keys.
    - Rejects list items that are non-dict.
    - Unknown/irrelevant data is ignored and does not invalidate the response.
    """
    if not isinstance(raw, dict):
        if raise_error:
            raise AnalysisError(f"Phản hồi gap result phải là dict, nhận được {type(raw).__name__}.")
        return False

    allowed_set = set(allowed_categories) if allowed_categories is not None else STANDARD_EVIDENCE_CATEGORIES

    for key, val in raw.items():
        if key in ("episode_id", "range_start_ms", "range_end_ms", "gap_id", "metadata"):
            continue
        if key not in allowed_set:
            continue

        if not isinstance(val, list):
            if raise_error:
                raise AnalysisError(f"Trường '{key}' phải là kiểu list, nhận được {type(val).__name__}.")
            return False

        for idx, item in enumerate(val):
            if not isinstance(item, dict):
                if raise_error:
                    raise AnalysisError(f"Phần tử {idx} trong '{key}' phải là dict, nhận được {type(item).__name__}.")
                return False

    return True


def _match_gap(det: CoverageGap, prior: CoverageGap) -> bool:
    """Deterministically match newly detected gap with prior gap by id, signature, range, or type."""
    if det.gap_id and prior.gap_id and det.gap_id == prior.gap_id:
        return True
    if det.gap_type != prior.gap_type:
        return False
    if det.gap_type == GapType.CATEGORY_DEFICIT:
        return bool(set(det.target_categories) & set(prior.target_categories))
    if det.gap_type == GapType.CHARACTER_DEFICIT:
        return bool(set(det.target_characters) & set(prior.target_characters))
    # Timeline gaps: DROPPED_TRANSCRIPT, CHUNK_SPARSITY, UNOBSERVED_SILENCE
    det_range = TimeRange(det.start_sec, det.end_sec)
    prior_range = TimeRange(prior.start_sec, prior.end_sec)
    if det_range.intersects(prior_range):
        return True
    if abs(det.start_sec - prior.start_sec) <= 10.0 and abs(det.end_sec - prior.end_sec) <= 10.0:
        return True
    return False


def reconcile_coverage_gaps(
    detected_gaps: Sequence[CoverageGap],
    prior_gaps: Sequence[CoverageGap] | None,
) -> list[CoverageGap]:
    """Reconcile fresh gap detections with prior statuses.

    Resolved or persistent empty gaps are preserved and not recreated as pending.
    New gaps are added.
    """
    if not prior_gaps:
        return list(detected_gaps)

    reconciled: list[CoverageGap] = []
    matched_prior_indices: set[int] = set()

    for det in detected_gaps:
        matched_prior: CoverageGap | None = None
        matched_idx: int | None = None

        for idx, prior in enumerate(prior_gaps):
            if idx in matched_prior_indices:
                continue
            if _match_gap(det, prior):
                matched_prior = prior
                matched_idx = idx
                break

        if matched_prior is not None and matched_idx is not None:
            matched_prior_indices.add(matched_idx)
            reconciled.append(
                CoverageGap(
                    gap_id=matched_prior.gap_id,
                    gap_type=det.gap_type,
                    start_sec=det.start_sec,
                    end_sec=det.end_sec,
                    target_categories=det.target_categories or matched_prior.target_categories,
                    target_characters=det.target_characters or matched_prior.target_characters,
                    is_actionable=matched_prior.is_actionable if matched_prior.status != "PENDING" else det.is_actionable,
                    status=matched_prior.status,
                )
            )
        else:
            reconciled.append(det)

    # Carry forward any prior gaps that were RESOLVED or PERSISTENT_EMPTY but not re-detected
    for idx, prior in enumerate(prior_gaps):
        if idx not in matched_prior_indices:
            if prior.status in ("RESOLVED", "PERSISTENT_EMPTY"):
                reconciled.append(prior)

    return reconciled


class CoverageStatus(str, Enum):
    """Episode coverage completion states."""

    COMPLETE = "COMPLETE"
    COMPLETE_TRANSCRIPT_ONLY = "COMPLETE_TRANSCRIPT_ONLY"
    PARTIAL_WITH_GAPS = "PARTIAL_WITH_GAPS"
    EMPTY_NO_DATA = "EMPTY_NO_DATA"
    FAILED = "FAILED"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            other_clean = other.strip().upper()
            self_clean = self.value.upper()
            if other.strip().lower() == "complete":
                return self_clean in ("COMPLETE", "COMPLETE_TRANSCRIPT_ONLY")
            return self_clean == other_clean
        if isinstance(other, CoverageStatus):
            return self.value == other.value
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value)


class GapType(str, Enum):
    """Types of narrative or timeline gaps."""

    DROPPED_TRANSCRIPT = "DROPPED_TRANSCRIPT"
    CHARACTER_DEFICIT = "CHARACTER_DEFICIT"
    CATEGORY_DEFICIT = "CATEGORY_DEFICIT"
    CHUNK_SPARSITY = "CHUNK_SPARSITY"
    UNOBSERVED_SILENCE = "UNOBSERVED_SILENCE"


@dataclass(frozen=True)
class TimeRange:
    """Immutable normalized time range interval [start_sec, end_sec]."""

    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        s = max(0.0, float(self.start_sec))
        e = max(s, float(self.end_sec))
        object.__setattr__(self, "start_sec", round(s, 3))
        object.__setattr__(self, "end_sec", round(e, 3))

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def intersects(self, other: TimeRange) -> bool:
        return max(self.start_sec, other.start_sec) < min(self.end_sec, other.end_sec)

    def overlap(self, other: TimeRange) -> float:
        return max(0.0, min(self.end_sec, other.end_sec) - max(self.start_sec, other.start_sec))

    def contains_sec(self, t: float) -> bool:
        return self.start_sec <= t <= self.end_sec

    def merge(self, other: TimeRange) -> TimeRange:
        return TimeRange(min(self.start_sec, other.start_sec), max(self.end_sec, other.end_sec))

    def to_dict(self) -> dict[str, float]:
        return {"start_sec": self.start_sec, "end_sec": self.end_sec}

    def to_list(self) -> list[float]:
        return [self.start_sec, self.end_sec]

    @classmethod
    def from_dict(cls, data: Any) -> TimeRange:
        if isinstance(data, TimeRange):
            return data
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            return cls(float(data[0]), float(data[1]))
        if isinstance(data, dict):
            s = float(data.get("start_sec", data.get("start_ms", 0) / 1000.0))
            e = float(data.get("end_sec", data.get("end_ms", 0) / 1000.0))
            return cls(s, e)
        return cls(0.0, 0.0)


def merge_intervals(intervals: Sequence[TimeRange]) -> list[TimeRange]:
    """Deterministically merge overlapping or touching intervals."""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda r: (r.start_sec, r.end_sec))
    merged: list[TimeRange] = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current.start_sec <= last.end_sec + 0.05:
            merged[-1] = TimeRange(last.start_sec, max(last.end_sec, current.end_sec))
        else:
            merged.append(current)
    return merged


def subtract_intervals(
    base_intervals: Sequence[TimeRange],
    to_subtract: Sequence[TimeRange],
) -> list[TimeRange]:
    """Subtract a set of intervals from base intervals, returning remaining disjoint spans."""
    merged_sub = merge_intervals(to_subtract)
    result: list[TimeRange] = list(base_intervals)

    for sub in merged_sub:
        next_result: list[TimeRange] = []
        for base in result:
            if not base.intersects(sub):
                next_result.append(base)
                continue
            # Left remainder
            if base.start_sec < sub.start_sec:
                next_result.append(TimeRange(base.start_sec, sub.start_sec))
            # Right remainder
            if base.end_sec > sub.end_sec:
                next_result.append(TimeRange(sub.end_sec, base.end_sec))
        result = next_result

    return [r for r in result if r.duration > 0.1]


@dataclass
class TimelineCoverage:
    """Timeline intervals and calculated coverage ratios for an episode."""

    transcript_ranges: list[TimeRange] = field(default_factory=list)
    visual_ranges: list[TimeRange] = field(default_factory=list)
    evidence_ranges: list[TimeRange] = field(default_factory=list)
    unobserved_ranges: list[TimeRange] = field(default_factory=list)
    transcript_coverage_ratio: float = 0.0
    evidence_coverage_ratio: float = 0.0
    visual_coverage_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript_ranges": [r.to_list() for r in self.transcript_ranges],
            "visual_ranges": [r.to_list() for r in self.visual_ranges],
            "evidence_ranges": [r.to_list() for r in self.evidence_ranges],
            "unobserved_ranges": [r.to_list() for r in self.unobserved_ranges],
            "transcript_coverage_ratio": round(self.transcript_coverage_ratio, 4),
            "evidence_coverage_ratio": round(self.evidence_coverage_ratio, 4),
            "visual_coverage_ratio": round(self.visual_coverage_ratio, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimelineCoverage:
        return cls(
            transcript_ranges=[TimeRange.from_dict(r) for r in data.get("transcript_ranges", [])],
            visual_ranges=[TimeRange.from_dict(r) for r in data.get("visual_ranges", [])],
            evidence_ranges=[TimeRange.from_dict(r) for r in data.get("evidence_ranges", [])],
            unobserved_ranges=[TimeRange.from_dict(r) for r in data.get("unobserved_ranges", [])],
            transcript_coverage_ratio=float(data.get("transcript_coverage_ratio", 0.0)),
            evidence_coverage_ratio=float(data.get("evidence_coverage_ratio", 0.0)),
            visual_coverage_ratio=float(data.get("visual_coverage_ratio", 0.0)),
        )


@dataclass
class ChunkCoverageRecord:
    """Audit record for a single scanned chunk."""

    chunk_index: int
    start_sec: float
    end_sec: float
    cues_count: int = 0
    items_count: int = 0
    categories_present: list[str] = field(default_factory=list)
    characters_present: list[str] = field(default_factory=list)
    payload_bytes: int = 0
    status: str = "OK"  # OK, EMPTY_TRANSCRIPT, SPARSE_EVIDENCE, NEEDS_SECOND_PASS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkCoverageRecord:
        return cls(
            chunk_index=int(data.get("chunk_index", 0)),
            start_sec=float(data.get("start_sec", 0.0)),
            end_sec=float(data.get("end_sec", 0.0)),
            cues_count=int(data.get("cues_count", 0)),
            items_count=int(data.get("items_count", 0)),
            categories_present=list(data.get("categories_present", [])),
            characters_present=list(data.get("characters_present", [])),
            payload_bytes=int(data.get("payload_bytes", 0)),
            status=str(data.get("status", "OK")),
        )


@dataclass
class CategoryCoverageRecord:
    """Audit record for an individual evidence category."""

    category: str
    count: int = 0
    density_per_minute: float = 0.0
    first_occurrence_sec: float | None = None
    last_occurrence_sec: float | None = None
    is_empty: bool = False
    verification_status: str = "PRESENT"  # PRESENT, VERIFIED_EMPTY, SUSPECT_EMPTY

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "count": self.count,
            "density_per_minute": round(self.density_per_minute, 4),
            "first_occurrence_sec": round(self.first_occurrence_sec, 3) if self.first_occurrence_sec is not None else None,
            "last_occurrence_sec": round(self.last_occurrence_sec, 3) if self.last_occurrence_sec is not None else None,
            "is_empty": self.is_empty,
            "verification_status": self.verification_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoryCoverageRecord:
        return cls(
            category=str(data.get("category", "")),
            count=int(data.get("count", 0)),
            density_per_minute=float(data.get("density_per_minute", 0.0)),
            first_occurrence_sec=float(data["first_occurrence_sec"]) if data.get("first_occurrence_sec") is not None else None,
            last_occurrence_sec=float(data["last_occurrence_sec"]) if data.get("last_occurrence_sec") is not None else None,
            is_empty=bool(data.get("is_empty", False)),
            verification_status=str(data.get("verification_status", "PRESENT")),
        )


@dataclass
class CharacterCoverageRecord:
    """Audit record for character presence and grounding across an episode."""

    character_name: str
    dialogue_mention_count: int = 0
    evidence_item_count: int = 0
    first_seen_sec: float = 0.0
    last_seen_sec: float = 0.0
    categories_involved: list[str] = field(default_factory=list)
    is_supporting: bool = False
    has_decisions_or_actions: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharacterCoverageRecord:
        return cls(
            character_name=str(data.get("character_name", "")),
            dialogue_mention_count=int(data.get("dialogue_mention_count", 0)),
            evidence_item_count=int(data.get("evidence_item_count", 0)),
            first_seen_sec=float(data.get("first_seen_sec", 0.0)),
            last_seen_sec=float(data.get("last_seen_sec", 0.0)),
            categories_involved=list(data.get("categories_involved", [])),
            is_supporting=bool(data.get("is_supporting", False)),
            has_decisions_or_actions=bool(data.get("has_decisions_or_actions", False)),
        )


@dataclass
class CoverageGap:
    """Identified coverage gap requiring targeted action or recording honest silence."""

    gap_id: str
    gap_type: GapType
    start_sec: float
    end_sec: float
    target_categories: list[str] = field(default_factory=list)
    target_characters: list[str] = field(default_factory=list)
    is_actionable: bool = True
    status: str = "PENDING"  # PENDING, RESOLVED, PERSISTENT_EMPTY

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type.value if isinstance(self.gap_type, GapType) else str(self.gap_type),
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "target_categories": list(self.target_categories),
            "target_characters": list(self.target_characters),
            "is_actionable": self.is_actionable,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageGap:
        gtype = data.get("gap_type", GapType.DROPPED_TRANSCRIPT.value)
        return cls(
            gap_id=str(data.get("gap_id", "")),
            gap_type=GapType(gtype) if gtype in GapType._value2member_map_ else GapType.DROPPED_TRANSCRIPT,
            start_sec=float(data.get("start_sec", 0.0)),
            end_sec=float(data.get("end_sec", 0.0)),
            target_categories=list(data.get("target_categories", [])),
            target_characters=list(data.get("target_characters", [])),
            is_actionable=bool(data.get("is_actionable", True)),
            status=str(data.get("status", "PENDING")),
        )


@dataclass
class EpisodeCoverage:
    """Comprehensive structured coverage ledger for an episode."""

    episode_id: str
    duration_seconds: float
    status: CoverageStatus
    timeline: TimelineCoverage = field(default_factory=TimelineCoverage)
    chunks: list[ChunkCoverageRecord] = field(default_factory=list)
    categories: dict[str, CategoryCoverageRecord] = field(default_factory=dict)
    characters: dict[str, CharacterCoverageRecord] = field(default_factory=dict)
    gaps: list[CoverageGap] = field(default_factory=list)
    chunks_count: int = 0
    cues_count: int = 0
    has_speech: bool = False
    category_counts: dict[str, int] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "duration_seconds": round(self.duration_seconds, 3),
            "status": self.status,
            "timeline": self.timeline.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "characters": {k: v.to_dict() for k, v in self.characters.items()},
            "gaps": [g.to_dict() for g in self.gaps],
            "chunks_count": self.chunks_count,
            "cues_count": self.cues_count,
            "has_speech": self.has_speech,
            "category_counts": dict(self.category_counts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeCoverage:
        st_val = data.get("status", CoverageStatus.COMPLETE_TRANSCRIPT_ONLY.value)
        if isinstance(st_val, CoverageStatus):
            status = st_val
        elif st_val in CoverageStatus._value2member_map_:
            status = CoverageStatus(st_val)
        elif str(st_val).upper() == "COMPLETE":
            status = CoverageStatus.COMPLETE
        else:
            status = CoverageStatus.COMPLETE_TRANSCRIPT_ONLY

        timeline_data = data.get("timeline", {})
        timeline = TimelineCoverage.from_dict(timeline_data) if isinstance(timeline_data, dict) else TimelineCoverage()

        chunks = [
            ChunkCoverageRecord.from_dict(c)
            for c in data.get("chunks", [])
            if isinstance(c, dict)
        ]

        categories = {
            k: CategoryCoverageRecord.from_dict(v)
            for k, v in data.get("categories", {}).items()
            if isinstance(v, dict)
        }

        characters = {
            k: CharacterCoverageRecord.from_dict(v)
            for k, v in data.get("characters", {}).items()
            if isinstance(v, dict)
        }

        gaps = [
            CoverageGap.from_dict(g)
            for g in data.get("gaps", [])
            if isinstance(g, dict)
        ]

        category_counts = {str(k): int(v) for k, v in data.get("category_counts", {}).items()}
        if not category_counts and categories:
            category_counts = {k: v.count for k, v in categories.items()}

        return cls(
            episode_id=str(data.get("episode_id", "")),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            status=status,
            timeline=timeline,
            chunks=chunks,
            categories=categories,
            characters=characters,
            gaps=gaps,
            chunks_count=int(data.get("chunks_count", len(chunks))),
            cues_count=int(data.get("cues_count", 0)),
            has_speech=bool(data.get("has_speech", False)),
            category_counts=category_counts,
        )


def _cue_tuple(cue: Any) -> tuple[float, float, str]:
    """Helper to convert any cue representation into (start_sec, end_sec, text)."""
    if isinstance(cue, SubtitleCue):
        return cue.start_sec, cue.end_sec, cue.text
    if isinstance(cue, tuple) and len(cue) >= 3:
        return float(cue[0]), float(cue[1]), str(cue[2])
    if isinstance(cue, dict):
        s = float(cue.get("start_sec", cue.get("start_ms", 0) / 1000.0))
        e = float(cue.get("end_sec", cue.get("end_ms", 0) / 1000.0))
        return s, e, str(cue.get("text", ""))
    return 0.0, 0.0, ""


RE_SPEAKER_NAME = re.compile(r"^(?:\[([^\]]+)\]|([^:]+):)\s*(.*)$")


def _extract_speakers_from_text(text: str) -> list[str]:
    """Extract explicit speaker tags like '[Alice]' or 'Bob:' from cue text."""
    match = RE_SPEAKER_NAME.match(text.strip())
    if match:
        raw_name = match.group(1) or match.group(2)
        if raw_name:
            clean = raw_name.strip()
            if 1 <= len(clean) <= 40 and not any(w in clean.lower() for w in ("line", "cue", "segment", "part")):
                return [clean]
    return []


def detect_coverage_gaps(
    duration_sec: float,
    cues: Sequence[Any],
    evidence_data: dict[str, list[dict[str, Any]]],
    *,
    planned_chunks: Sequence[Any] | None = None,
    visual_spans: Sequence[tuple[float, float]] | Sequence[TimeRange] | None = None,
    directive: ScannerDirective | None = None,
) -> list[CoverageGap]:
    """Deterministically detect timeline, sparsity, category, and character gaps."""
    gaps: list[CoverageGap] = []
    gap_idx = 0

    dur = max(1.0, float(duration_sec))
    tuple_cues = [_cue_tuple(c) for c in cues if _cue_tuple(c)[2].strip()]

    # Collect all evidence items spans
    all_evidence_spans: list[TimeRange] = []
    for cat, items in evidence_data.items():
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            s = float(it.get("start_sec", it.get("start_ms", 0) / 1000.0))
            e = float(it.get("end_sec", it.get("end_ms", 0) / 1000.0))
            if e >= s:
                all_evidence_spans.append(TimeRange(s, e))

    merged_evidence = merge_intervals(all_evidence_spans)

    # 1. Detect UNOBSERVED_SILENCE gaps
    cue_intervals = [TimeRange(s, e) for s, e, _ in tuple_cues]
    merged_cues = merge_intervals(cue_intervals)
    vis_intervals = [
        TimeRange(r[0], r[1]) if isinstance(r, (tuple, list)) else r
        for r in (visual_spans or [])
    ]
    merged_vis = merge_intervals(vis_intervals)

    observed_ranges = merge_intervals(merged_cues + merged_vis)
    unobserved = subtract_intervals([TimeRange(0.0, dur)], observed_ranges)

    for r in unobserved:
        if r.duration >= 30.0:
            gap_idx += 1
            gaps.append(
                CoverageGap(
                    gap_id=f"gap_silence_{gap_idx:02d}",
                    gap_type=GapType.UNOBSERVED_SILENCE,
                    start_sec=r.start_sec,
                    end_sec=r.end_sec,
                    target_categories=[],
                    target_characters=[],
                    is_actionable=False,
                    status="RESOLVED",
                )
            )

    # 2. Detect DROPPED_TRANSCRIPT gaps
    # Continuous speaking intervals (>= 30.0s, >= 3 cues) where zero evidence items exist
    if merged_cues:
        for span in merged_cues:
            if span.duration < 30.0:
                continue
            cues_in_span = [c for c in tuple_cues if c[0] <= span.end_sec and c[1] >= span.start_sec]
            if len(cues_in_span) < 3:
                continue

            has_evidence = any(span.intersects(ev) for ev in merged_evidence)
            if not has_evidence:
                gap_idx += 1
                gaps.append(
                    CoverageGap(
                        gap_id=f"gap_dropped_{gap_idx:02d}",
                        gap_type=GapType.DROPPED_TRANSCRIPT,
                        start_sec=span.start_sec,
                        end_sec=span.end_sec,
                        target_categories=["major_scenes", "dialogue"],
                        target_characters=[],
                        is_actionable=True,
                        status="PENDING",
                    )
                )

    # 3. Detect CHUNK_SPARSITY gaps
    if planned_chunks:
        for ch in planned_chunks:
            c_start = getattr(ch, "start_sec", 0.0)
            c_end = getattr(ch, "end_sec", dur)
            c_cues = getattr(ch, "cues", [])
            c_items = [
                ev for ev in all_evidence_spans
                if (ev.start_sec <= c_end and ev.end_sec >= c_start)
            ]
            if len(c_cues) >= 15 and len(c_items) <= 1:
                gap_idx += 1
                gaps.append(
                    CoverageGap(
                        gap_id=f"gap_sparse_{gap_idx:02d}",
                        gap_type=GapType.CHUNK_SPARSITY,
                        start_sec=c_start,
                        end_sec=c_end,
                        target_categories=["major_scenes", "dialogue", "character_decisions"],
                        target_characters=[],
                        is_actionable=True,
                        status="PENDING",
                    )
                )

    # 4. Detect CHARACTER_DEFICIT gaps
    speaker_counts: dict[str, int] = {}
    speaker_first_seen: dict[str, float] = {}
    speaker_last_seen: dict[str, float] = {}

    for s, e, text in tuple_cues:
        speakers = _extract_speakers_from_text(text)
        for spk in speakers:
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            if spk not in speaker_first_seen or s < speaker_first_seen[spk]:
                speaker_first_seen[spk] = s
            if spk not in speaker_last_seen or e > speaker_last_seen[spk]:
                speaker_last_seen[spk] = e

    for spk, count in speaker_counts.items():
        if count >= 4:
            # Count evidence items mentioning this character
            char_items_count = 0
            has_supporting_or_decisions = False
            for cat, items in evidence_data.items():
                for it in items:
                    c_list = it.get("characters", [])
                    c_single = str(it.get("character", ""))
                    if spk in c_list or spk == c_single:
                        char_items_count += 1
                        if cat in ("supporting_developments", "relationships", "character_decisions"):
                            has_supporting_or_decisions = True

            if char_items_count <= 1 and not has_supporting_or_decisions:
                gap_idx += 1
                gaps.append(
                    CoverageGap(
                        gap_id=f"gap_char_{gap_idx:02d}",
                        gap_type=GapType.CHARACTER_DEFICIT,
                        start_sec=speaker_first_seen.get(spk, 0.0),
                        end_sec=speaker_last_seen.get(spk, dur),
                        target_categories=["supporting_developments", "relationships", "character_decisions"],
                        target_characters=[spk],
                        is_actionable=True,
                        status="PENDING",
                    )
                )

    # 5. Detect CATEGORY_DEFICIT gaps
    # Only triggered if substantial dialogue and duration exist
    if len(tuple_cues) >= 20 and dur >= 300.0:
        critical_categories = [
            "major_scenes",
            "character_decisions",
            "conflicts",
            "reveals",
            "setup_payoff",
        ]
        if directive is not None:
            if directive.requested_categories:
                critical_categories = list(dict.fromkeys(critical_categories + directive.requested_categories))
            elif directive.schema_categories:
                critical_categories = list(dict.fromkeys(critical_categories + directive.schema_categories))

        for cat in critical_categories:
            items = evidence_data.get(cat, [])
            if not items:
                gap_idx += 1
                gaps.append(
                    CoverageGap(
                        gap_id=f"gap_cat_{gap_idx:02d}",
                        gap_type=GapType.CATEGORY_DEFICIT,
                        start_sec=0.0,
                        end_sec=dur,
                        target_categories=[cat],
                        target_characters=[],
                        is_actionable=True,
                        status="PENDING",
                    )
                )

    return gaps


def compute_episode_coverage(
    episode_id: str,
    duration_sec: float,
    cues: Sequence[Any],
    evidence_data: dict[str, list[dict[str, Any]]],
    *,
    planned_chunks: Sequence[Any] | None = None,
    visual_spans: Sequence[tuple[float, float]] | Sequence[TimeRange] | None = None,
    directive: ScannerDirective | None = None,
    gaps: list[CoverageGap] | None = None,
) -> EpisodeCoverage:
    """Deterministically compute complete EpisodeCoverage ledger."""
    dur = max(1.0, float(duration_sec))
    tuple_cues = [_cue_tuple(c) for c in cues if _cue_tuple(c)[2].strip()]
    has_speech = bool(tuple_cues)

    # 1. Timeline intervals
    cue_intervals = [TimeRange(s, e) for s, e, _ in tuple_cues]
    transcript_ranges = merge_intervals(cue_intervals)
    transcript_seconds = sum(r.duration for r in transcript_ranges)
    transcript_coverage_ratio = min(1.0, transcript_seconds / dur) if dur > 0 else 0.0

    vis_intervals = [
        TimeRange(r[0], r[1]) if isinstance(r, (tuple, list)) else r
        for r in (visual_spans or [])
    ]
    visual_ranges = merge_intervals(vis_intervals)
    visual_seconds = sum(r.duration for r in visual_ranges)
    visual_coverage_ratio = min(1.0, visual_seconds / dur) if dur > 0 else 0.0

    # Evidence intervals
    all_evidence_spans: list[TimeRange] = []
    category_counts: dict[str, int] = {}
    for cat, items in sorted(evidence_data.items(), key=lambda x: x[0]):
        if isinstance(items, list):
            category_counts[cat] = len(items)
            for it in items:
                if isinstance(it, dict):
                    s = float(it.get("start_sec", it.get("start_ms", 0) / 1000.0))
                    e = float(it.get("end_sec", it.get("end_ms", 0) / 1000.0))
                    if e >= s:
                        all_evidence_spans.append(TimeRange(s, e))

    evidence_ranges = merge_intervals(all_evidence_spans)
    evidence_seconds = sum(r.duration for r in evidence_ranges)
    evidence_coverage_ratio = min(1.0, evidence_seconds / dur) if dur > 0 else 0.0

    observed_ranges = merge_intervals(transcript_ranges + visual_ranges)
    unobserved_ranges = subtract_intervals([TimeRange(0.0, dur)], observed_ranges)

    timeline = TimelineCoverage(
        transcript_ranges=transcript_ranges,
        visual_ranges=visual_ranges,
        evidence_ranges=evidence_ranges,
        unobserved_ranges=unobserved_ranges,
        transcript_coverage_ratio=transcript_coverage_ratio,
        evidence_coverage_ratio=evidence_coverage_ratio,
        visual_coverage_ratio=visual_coverage_ratio,
    )

    # 2. Gaps detection & reconciliation
    fresh_gaps = detect_coverage_gaps(
        dur,
        tuple_cues,
        evidence_data,
        planned_chunks=planned_chunks,
        visual_spans=visual_ranges,
        directive=directive,
    )
    final_gaps = reconcile_coverage_gaps(fresh_gaps, gaps)

    # 3. Chunk coverage records
    chunks: list[ChunkCoverageRecord] = []
    if planned_chunks:
        for idx, ch in enumerate(planned_chunks):
            c_start = getattr(ch, "start_sec", 0.0)
            c_end = getattr(ch, "end_sec", dur)
            c_cues = getattr(ch, "cues", [])
            c_bytes = getattr(ch, "estimated_bytes", 0)

            c_items: list[dict[str, Any]] = []
            cats_present: set[str] = set()
            chars_present: set[str] = set()

            for cat, it_list in evidence_data.items():
                if not isinstance(it_list, list):
                    continue
                for it in it_list:
                    if not isinstance(it, dict):
                        continue
                    it_s = float(it.get("start_sec", it.get("start_ms", 0) / 1000.0))
                    it_e = float(it.get("end_sec", it.get("end_ms", 0) / 1000.0))
                    if it_s <= c_end and it_e >= c_start:
                        c_items.append(it)
                        cats_present.add(cat)
                        for c_name in it.get("characters", []):
                            chars_present.add(str(c_name))
                        if it.get("character"):
                            chars_present.add(str(it["character"]))

            ch_status = "OK"
            if not c_cues:
                ch_status = "EMPTY_TRANSCRIPT"
            elif len(c_cues) >= 15 and len(c_items) <= 1:
                ch_status = "SPARSE_EVIDENCE"

            chunks.append(
                ChunkCoverageRecord(
                    chunk_index=idx,
                    start_sec=c_start,
                    end_sec=c_end,
                    cues_count=len(c_cues),
                    items_count=len(c_items),
                    categories_present=sorted(list(cats_present)),
                    characters_present=sorted(list(chars_present)),
                    payload_bytes=c_bytes,
                    status=ch_status,
                )
            )

    # 4. Category coverage records
    categories: dict[str, CategoryCoverageRecord] = {}
    minutes = max(0.1, dur / 60.0)
    for cat, items in sorted(evidence_data.items(), key=lambda x: x[0]):
        if not isinstance(items, list):
            continue
        c_count = len(items)
        density = c_count / minutes
        first_sec = min((float(it.get("start_sec", it.get("start_ms", 0) / 1000.0)) for it in items), default=None)
        last_sec = max((float(it.get("end_sec", it.get("end_ms", 0) / 1000.0)) for it in items), default=None)
        is_empty = (c_count == 0)

        # Verification status
        if c_count > 0:
            v_status = "PRESENT"
        elif not has_speech and not visual_ranges:
            v_status = "VERIFIED_EMPTY"
        else:
            targeted_gaps = [g for g in final_gaps if cat in g.target_categories]
            if targeted_gaps and all(g.status == "PERSISTENT_EMPTY" for g in targeted_gaps):
                v_status = "VERIFIED_EMPTY"
            else:
                v_status = "SUSPECT_EMPTY"

        categories[cat] = CategoryCoverageRecord(
            category=cat,
            count=c_count,
            density_per_minute=density,
            first_occurrence_sec=first_sec,
            last_occurrence_sec=last_sec,
            is_empty=is_empty,
            verification_status=v_status,
        )

    # 5. Character coverage records
    speaker_counts: dict[str, int] = {}
    for _, _, text in tuple_cues:
        for spk in _extract_speakers_from_text(text):
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

    characters: dict[str, CharacterCoverageRecord] = {}
    all_names = set(speaker_counts.keys())
    for it in [item for sublist in evidence_data.values() if isinstance(sublist, list) for item in sublist if isinstance(item, dict)]:
        for c in it.get("characters", []):
            all_names.add(str(c))
        if it.get("character"):
            all_names.add(str(it["character"]))

    for name in sorted(all_names):
        if not name.strip():
            continue
        first_seen = dur
        last_seen = 0.0
        seen_in_evidence = 0
        cats_involved: set[str] = set()
        has_action = False

        for s, e, text in tuple_cues:
            if name.lower() in text.lower():
                first_seen = min(first_seen, s)
                last_seen = max(last_seen, e)

        for cat, items in evidence_data.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                c_list = it.get("characters", [])
                c_single = str(it.get("character", ""))
                if name in c_list or name == c_single:
                    seen_in_evidence += 1
                    cats_involved.add(cat)
                    it_s = float(it.get("start_sec", it.get("start_ms", 0) / 1000.0))
                    it_e = float(it.get("end_sec", it.get("end_ms", 0) / 1000.0))
                    first_seen = min(first_seen, it_s)
                    last_seen = max(last_seen, it_e)
                    if cat in ("character_decisions", "failures", "consequences"):
                        has_action = True

        if first_seen > last_seen:
            first_seen = 0.0
            last_seen = 0.0

        characters[name] = CharacterCoverageRecord(
            character_name=name,
            dialogue_mention_count=speaker_counts.get(name, 0),
            evidence_item_count=seen_in_evidence,
            first_seen_sec=first_seen,
            last_seen_sec=last_seen,
            categories_involved=sorted(list(cats_involved)),
            is_supporting=speaker_counts.get(name, 0) < 10,
            has_decisions_or_actions=has_action,
        )

    # 6. Evaluate CoverageStatus
    actionable_gaps = [g for g in final_gaps if g.is_actionable and g.status == "PENDING"]

    if dur <= 0.0 or (not has_speech and not visual_ranges):
        status = CoverageStatus.EMPTY_NO_DATA
    elif actionable_gaps:
        status = CoverageStatus.PARTIAL_WITH_GAPS
    elif visual_ranges and visual_coverage_ratio >= 0.85:
        status = CoverageStatus.COMPLETE
    else:
        status = CoverageStatus.COMPLETE_TRANSCRIPT_ONLY

    return EpisodeCoverage(
        episode_id=episode_id,
        duration_seconds=dur,
        status=status,
        timeline=timeline,
        chunks=chunks,
        categories=categories,
        characters=characters,
        gaps=final_gaps,
        chunks_count=len(chunks),
        cues_count=len(tuple_cues),
        has_speech=has_speech,
        category_counts=category_counts,
    )


@dataclass
class SecondPassRequest:
    """Planned bounded second pass request envelope for an actionable gap."""

    gap_id: str
    gap_type: GapType
    start_sec: float
    end_sec: float
    target_categories: list[str]
    target_characters: list[str]
    cues: list[tuple[float, float, str]]
    user_text: str
    estimated_bytes: int
    gap_ids: list[str] = field(default_factory=list)

    @property
    def start_ms(self) -> int:
        return round(self.start_sec * 1000)

    @property
    def end_ms(self) -> int:
        return round(self.end_sec * 1000)


def plan_second_pass_requests(
    episode_id: str,
    source_video: str,
    duration_seconds: float,
    gaps: list[CoverageGap],
    cues: Sequence[Any],
    existing_evidence: dict[str, list[dict[str, Any]]],
    model: str,
    thinking: str = "auto",
    target_ceiling: int = TARGET_PAYLOAD_CEILING,
    system_prompt: str = "",
    hard_ceiling: int = HARD_PAYLOAD_CEILING,
) -> list[SecondPassRequest]:
    """Deterministically merge and plan bounded second-pass request envelopes.

    Recursive splitting by cue list/index while exact measured.
    Anchors are deterministically compacted when needed to prevent cue dropping.
    Single oversized cues are binary search capped with timing preserved and [capped] suffix.
    Split envelopes share gap_ids.
    All cues are preserved in order.
    """
    actionable_gaps = [g for g in gaps if g.is_actionable and g.status == "PENDING"]
    if not actionable_gaps:
        return []

    dur = max(1.0, float(duration_seconds))
    tuple_cues = [_cue_tuple(c) for c in cues if _cue_tuple(c)[2].strip()]
    tuple_cues.sort(key=lambda c: (c[0], c[1]))

    # Merge overlapping or adjacent actionable gaps (within 15s)
    sorted_gaps = sorted(actionable_gaps, key=lambda g: (g.start_sec, g.end_sec))
    merged_gap_groups: list[list[CoverageGap]] = []

    for g in sorted_gaps:
        if not merged_gap_groups:
            merged_gap_groups.append([g])
            continue
        last_group = merged_gap_groups[-1]
        group_end = max(x.end_sec for x in last_group)

        if g.start_sec <= group_end + 15.0:
            last_group.append(g)
        else:
            merged_gap_groups.append([g])

    requests: list[SecondPassRequest] = []

    for grp in merged_gap_groups:
        grp_start = max(0.0, min(x.start_sec for x in grp))
        grp_end = min(dur, max(x.end_sec for x in grp))

        cats: set[str] = set()
        chars: set[str] = set()
        for x in grp:
            cats.update(x.target_categories)
            chars.update(x.target_characters)

        target_cats_list = sorted(list(cats))
        target_chars_list = sorted(list(chars))
        target_cats_str = ", ".join(target_cats_list) if target_cats_list else "major_scenes, dialogue"
        target_chars_str = ", ".join(target_chars_list) if target_chars_list else "all relevant characters"

        primary_gap_id = grp[0].gap_id
        grp_gap_type = grp[0].gap_type
        grp_gap_ids = [g.gap_id for g in grp]

        # Context padding: +/- 10s around gap
        pad_start = max(0.0, grp_start - 10.0)
        pad_end = min(dur, grp_end + 10.0)

        matching_cues = [
            c for c in tuple_cues
            if (c[0] <= pad_end and c[1] >= pad_start)
        ]
        matching_cues.sort(key=lambda c: (c[0], c[1]))

        def _format_gap_user_text(
            cur_start: float,
            cur_end: float,
            cur_cues: list[tuple[float, float, str]],
            anch_list: list[str],
        ) -> str:
            start_ms = round(cur_start * 1000)
            end_ms = round(cur_end * 1000)
            anch_txt = "\n".join(f"- {a}" for a in anch_list) if anch_list else "None."
            chunk_txt = (
                "\n".join(f"[{s:.1f}s - {e:.1f}s] {t}" for s, e, t in cur_cues)
                if cur_cues
                else "[No spoken dialogue in gap]"
            )
            return (
                f"TARGETED EVIDENCE GAP SECOND PASS\n"
                f"Episode ID: {episode_id}\n"
                f"Source Video: {source_video}\n"
                f"Gap Range: {start_ms} to {end_ms} ms ({cur_start:.1f}s - {cur_end:.1f}s)\n"
                f"Episode Duration: {dur:.1f}s\n"
                f"Target Missing Categories: {target_cats_str}\n"
                f"Target Characters: {target_chars_str}\n\n"
                f"Already Known Events in Vicinity (DO NOT DUPLICATE):\n{anch_txt}\n\n"
                f"Timestamped Transcript Excerpt:\n{chunk_txt}\n"
            )

        def _estimate(text: str) -> int:
            return estimate_request_size(
                model=model,
                system=system_prompt,
                user=text,
                thinking=thinking,
                variant=0,
            )

        def _plan_subchunk(
            cur_start: float,
            cur_end: float,
            cur_cues: list[tuple[float, float, str]],
        ) -> list[SecondPassRequest]:
            # Collect anchors in vicinity
            v_start = max(0.0, cur_start - 10.0)
            v_end = min(dur, cur_end + 10.0)
            anchors: list[str] = []
            for cat_name, items in sorted(existing_evidence.items()):
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    it_s = float(it.get("start_sec", it.get("start_ms", 0) / 1000.0))
                    it_e = float(it.get("end_sec", it.get("end_ms", 0) / 1000.0))
                    if it_s <= v_end and it_e >= v_start:
                        sum_txt = str(it.get("summary", it.get("quote", it.get("detail", ""))))[:80].strip()
                        if sum_txt:
                            anchors.append(f"[{it_s:.1f}s - {it_e:.1f}s]: {sum_txt}")

            # 1. Try full anchors
            u_text = _format_gap_user_text(cur_start, cur_end, cur_cues, anchors)
            est = _estimate(u_text)
            if est <= target_ceiling:
                return [
                    SecondPassRequest(
                        gap_id=primary_gap_id,
                        gap_type=grp_gap_type,
                        start_sec=cur_start,
                        end_sec=cur_end,
                        target_categories=target_cats_list,
                        target_characters=target_chars_list,
                        cues=cur_cues,
                        user_text=u_text,
                        estimated_bytes=est,
                        gap_ids=grp_gap_ids,
                    )
                ]

            # 2. Try compacted anchors (shortened text)
            if anchors:
                compact_anchors = [a[:40] + ("..." if len(a) > 40 else "") for a in anchors]
                u_text = _format_gap_user_text(cur_start, cur_end, cur_cues, compact_anchors)
                est = _estimate(u_text)
                if est <= target_ceiling:
                    return [
                        SecondPassRequest(
                            gap_id=primary_gap_id,
                            gap_type=grp_gap_type,
                            start_sec=cur_start,
                            end_sec=cur_end,
                            target_categories=target_cats_list,
                            target_characters=target_chars_list,
                            cues=cur_cues,
                            user_text=u_text,
                            estimated_bytes=est,
                            gap_ids=grp_gap_ids,
                        )
                    ]

                # 3. Drop anchors from end until empty
                curr_anchors = list(compact_anchors)
                while curr_anchors:
                    curr_anchors.pop()
                    u_text = _format_gap_user_text(cur_start, cur_end, cur_cues, curr_anchors)
                    est = _estimate(u_text)
                    if est <= target_ceiling:
                        return [
                            SecondPassRequest(
                                gap_id=primary_gap_id,
                                gap_type=grp_gap_type,
                                start_sec=cur_start,
                                end_sec=cur_end,
                                target_categories=target_cats_list,
                                target_characters=target_chars_list,
                                cues=cur_cues,
                                user_text=u_text,
                                estimated_bytes=est,
                                gap_ids=grp_gap_ids,
                            )
                        ]

            # Zero anchors check
            u_text = _format_gap_user_text(cur_start, cur_end, cur_cues, [])
            est = _estimate(u_text)
            if est <= target_ceiling:
                return [
                    SecondPassRequest(
                        gap_id=primary_gap_id,
                        gap_type=grp_gap_type,
                        start_sec=cur_start,
                        end_sec=cur_end,
                        target_categories=target_cats_list,
                        target_characters=target_chars_list,
                        cues=cur_cues,
                        user_text=u_text,
                        estimated_bytes=est,
                        gap_ids=grp_gap_ids,
                    )
                ]

            # If cues cannot fit with zero anchors:
            # Case A: len(cur_cues) > 1 -> split by cue index halves
            if len(cur_cues) > 1:
                mid = len(cur_cues) // 2
                left_cues = cur_cues[:mid]
                right_cues = cur_cues[mid:]

                r0_start = right_cues[0][0]
                l_last_end = left_cues[-1][1]

                if cur_start < r0_start < cur_end:
                    split_sec = r0_start
                elif cur_start < l_last_end < cur_end:
                    split_sec = l_last_end
                elif cur_end > cur_start:
                    split_sec = (cur_start + cur_end) / 2.0
                else:
                    split_sec = cur_start

                left_plans = _plan_subchunk(cur_start, split_sec, left_cues)
                right_plans = _plan_subchunk(split_sec, cur_end, right_cues)
                return left_plans + right_plans

            # Case B: len(cur_cues) == 1 -> single cue oversized, binary search cap text
            if len(cur_cues) == 1:
                s, e, text = cur_cues[0]
                suffix = "... [capped]"
                low = 0
                high = len(text)
                best_text = ""
                while low <= high:
                    mid = (low + high) // 2
                    candidate = text[:mid] + (suffix if mid < len(text) else "")
                    cand_text = _format_gap_user_text(cur_start, cur_end, [(s, e, candidate)], [])
                    cand_est = _estimate(cand_text)
                    if cand_est <= target_ceiling:
                        best_text = candidate
                        low = mid + 1
                    else:
                        high = mid - 1

                if not best_text:
                    cand_text = _format_gap_user_text(cur_start, cur_end, [(s, e, suffix)], [])
                    if _estimate(cand_text) <= target_ceiling:
                        best_text = suffix
                    else:
                        best_text = "[capped]"

                capped_cues = [(s, e, best_text)]
                final_text = _format_gap_user_text(cur_start, cur_end, capped_cues, [])
                final_est = _estimate(final_text)
                return [
                    SecondPassRequest(
                        gap_id=primary_gap_id,
                        gap_type=grp_gap_type,
                        start_sec=cur_start,
                        end_sec=cur_end,
                        target_categories=target_cats_list,
                        target_characters=target_chars_list,
                        cues=capped_cues,
                        user_text=final_text,
                        estimated_bytes=final_est,
                        gap_ids=grp_gap_ids,
                    )
                ]

            # Case C: len(cur_cues) == 0 -> empty gap
            final_text = _format_gap_user_text(cur_start, cur_end, [], [])
            final_est = _estimate(final_text)
            return [
                SecondPassRequest(
                    gap_id=primary_gap_id,
                    gap_type=grp_gap_type,
                    start_sec=cur_start,
                    end_sec=cur_end,
                    target_categories=target_cats_list,
                    target_characters=target_chars_list,
                    cues=[],
                    user_text=final_text,
                    estimated_bytes=final_est,
                    gap_ids=grp_gap_ids,
                )
            ]

        planned = _plan_subchunk(grp_start, grp_end, matching_cues)
        requests.extend(planned)

    return requests


def merge_second_pass_results(
    base_evidence_data: dict[str, list[dict[str, Any]]],
    second_pass_results: Sequence[dict[str, Any]],
    episode_id: str,
    duration_sec: float,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Deterministically deduplicate and merge second pass items into evidence data.

    Preserves distinct categories, supporting character actions, and tags source modality.
    """
    from .evidence import _validate_and_normalize_evidence_item

    merged_data = {cat: list(items) for cat, items in base_evidence_data.items()}
    items_added = 0

    for result in second_pass_results:
        if not isinstance(result, dict):
            continue

        for cat, new_items in result.items():
            if not isinstance(new_items, list) or cat in ("episode_id", "range_start_ms", "range_end_ms"):
                continue

            target_cat = cat if cat in merged_data else "major_scenes"
            if target_cat not in merged_data:
                merged_data[target_cat] = []
            for item in new_items:
                if not isinstance(item, dict):
                    continue
                norm = _validate_and_normalize_evidence_item(
                    item,
                    episode_id=episode_id,
                    max_duration=duration_sec,
                    chunk_start_sec=float(item.get("start_sec", 0.0)),
                    chunk_end_sec=float(item.get("end_sec", duration_sec)),
                )
                if norm is None:
                    continue

                norm["source_modality"] = "TRANSCRIPT_GROUNDED"

                # Deduplication against existing items in target_cat
                sum_new = str(norm.get("summary", norm.get("quote", norm.get("detail", "")))).strip()
                tokens_new = _normalize_tokens(sum_new)
                is_duplicate = False

                for ex in merged_data[target_cat]:
                    ex_s = float(ex.get("start_sec", 0.0))
                    ex_e = float(ex.get("end_sec", 0.0))

                    overlap = max(0.0, min(norm["end_sec"], ex_e) - max(norm["start_sec"], ex_s))
                    dur_new = max(0.1, norm["end_sec"] - norm["start_sec"])
                    dur_ex = max(0.1, ex_e - ex_s)
                    overlap_ratio = overlap / min(dur_new, dur_ex)
                    time_close = abs(norm["start_sec"] - ex_s) <= 5.0 and abs(norm["end_sec"] - ex_e) <= 5.0

                    if overlap_ratio >= 0.5 or time_close:
                        sum_ex = str(ex.get("summary", ex.get("quote", ex.get("detail", "")))).strip()
                        tokens_ex = _normalize_tokens(sum_ex)
                        jaccard = (
                            len(tokens_new & tokens_ex) / len(tokens_new | tokens_ex)
                            if (tokens_new | tokens_ex)
                            else 0.0
                        )
                        substr = (
                            len(sum_new) > 8
                            and len(sum_ex) > 8
                            and (sum_new.lower() in sum_ex.lower() or sum_ex.lower() in sum_new.lower())
                        )

                        if jaccard >= 0.35 or substr:
                            is_duplicate = True
                            # Merge characters if new item has additional characters
                            new_chars = norm.get("characters", [])
                            if new_chars:
                                ex_chars = set(ex.get("characters", []))
                                ex_chars.update(new_chars)
                                ex["characters"] = sorted(list(ex_chars))
                            break

                if not is_duplicate:
                    merged_data[target_cat].append(norm)
                    items_added += 1

    return merged_data, items_added


__all__ = [
    "HARD_PAYLOAD_CEILING",
    "TARGET_PAYLOAD_CEILING",
    "STANDARD_EVIDENCE_CATEGORIES",
    "CategoryCoverageRecord",
    "CharacterCoverageRecord",
    "ChunkCoverageRecord",
    "CoverageGap",
    "CoverageStatus",
    "EpisodeCoverage",
    "GapType",
    "SecondPassRequest",
    "TimelineCoverage",
    "TimeRange",
    "compute_episode_coverage",
    "detect_coverage_gaps",
    "merge_intervals",
    "merge_second_pass_results",
    "plan_second_pass_requests",
    "reconcile_coverage_gaps",
    "subtract_intervals",
    "validate_gap_result_schema",
]
