"""Core domain models, dataclasses, serialization, and validation for ToolRecap V2."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

from .enums import AnalysisScope, AudioPolicy, CandidateScope, OutputStatus
from .title import resolve_unique_titles, sanitize_title


class DomainError(Exception):
    """Base domain error."""


class ValidationError(DomainError):
    """Raised when manifest, segment, or source reference validation fails."""


@dataclass
class MediaSelection:
    video_path: str = ""
    audio_track: int = 0
    subtitle_track: int | None = None
    start_seconds: float = 0.0
    end_seconds: float | None = None
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaSelection":
        return cls(
            video_path=str(data.get("video_path", "")),
            audio_track=int(data.get("audio_track", 0)),
            subtitle_track=int(data["subtitle_track"]) if data.get("subtitle_track") is not None else None,
            start_seconds=float(data.get("start_seconds", 0.0)),
            end_seconds=float(data["end_seconds"]) if data.get("end_seconds") is not None else None,
            extra_info=dict(data.get("extra_info", {})),
        )


@dataclass
class SourceEpisode:
    episode_id: str
    source_video: str
    season_number: int | None = None
    episode_number: int | None = None
    title: str = ""
    duration_seconds: float = 0.0
    media_selection: MediaSelection | None = None
    status: str = "WAITING"
    stage: str = "Sẵn sàng"
    progress: int = 0
    error: str | None = None
    current_message: str = "Sẵn sàng"
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.media_selection is not None:
            d["media_selection"] = self.media_selection.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceEpisode":
        raw_media = data.get("media_selection")
        media = MediaSelection.from_dict(raw_media) if isinstance(raw_media, dict) else None
        return cls(
            episode_id=str(data.get("episode_id", "")),
            source_video=str(data.get("source_video", "")),
            season_number=int(data["season_number"]) if data.get("season_number") is not None else None,
            episode_number=int(data["episode_number"]) if data.get("episode_number") is not None else None,
            title=str(data.get("title", "")),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            media_selection=media,
            status=str(data.get("status") or "WAITING"),
            stage=str(data.get("stage") or "Sẵn sàng"),
            progress=int(data.get("progress") or 0),
            error=str(data["error"]) if data.get("error") is not None else None,
            current_message=str(data.get("current_message") or "Sẵn sàng"),
            cached=bool(data.get("cached", False)),
        )


@dataclass
class SourceClip:
    episode_id: str
    source_video: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "source_video": self.source_video,
            "start": float(self.start),
            "end": float(self.end),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceClip":
        return cls(
            episode_id=str(data.get("episode_id", "")),
            source_video=str(data.get("source_video", "")),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
        )


@dataclass
class Segment:
    segment_id: str
    source_clips: list[SourceClip] = field(default_factory=list)
    original_dialogue: str = ""
    narration: str = ""
    audio_policy: str = AudioPolicy.DUCK.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "source_clips": [c.to_dict() for c in self.source_clips],
            "original_dialogue": self.original_dialogue,
            "narration": self.narration,
            "audio_policy": self.audio_policy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        clips_raw = data.get("source_clips", [])
        clips = [
            c if isinstance(c, SourceClip) else SourceClip.from_dict(c)
            for c in clips_raw
            if isinstance(c, (dict, SourceClip))
        ]
        return cls(
            segment_id=str(data.get("segment_id", "")),
            source_clips=clips,
            original_dialogue=str(data.get("original_dialogue", "")),
            narration=str(data.get("narration", "")),
            audio_policy=str(data.get("audio_policy", AudioPolicy.DUCK.value)),
        )


@dataclass
class CommentaryOutput:
    output_id: str
    title: str
    sanitized_title: str = ""
    candidate_scope: str = CandidateScope.SINGLE_EPISODE.value
    segments: list[Segment] = field(default_factory=list)
    status: str = OutputStatus.WAITING.value
    publication_video_path: str | None = None
    publication_original_srt_path: str | None = None
    publication_narration_srt_path: str | None = None
    error: str | None = None
    progress: int = 0

    @property
    def video_path(self) -> str | None:
        return self.publication_video_path

    @video_path.setter
    def video_path(self, val: str | None) -> None:
        self.publication_video_path = val

    @property
    def original_srt_path(self) -> str | None:
        return self.publication_original_srt_path

    @original_srt_path.setter
    def original_srt_path(self, val: str | None) -> None:
        self.publication_original_srt_path = val

    @property
    def narration_srt_path(self) -> str | None:
        return self.publication_narration_srt_path

    @narration_srt_path.setter
    def narration_srt_path(self, val: str | None) -> None:
        self.publication_narration_srt_path = val

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_id": self.output_id,
            "title": self.title,
            "sanitized_title": self.sanitized_title,
            "candidate_scope": self.candidate_scope,
            "segments": [s.to_dict() for s in self.segments],
            "status": self.status,
            "publication_video_path": self.publication_video_path,
            "publication_original_srt_path": self.publication_original_srt_path,
            "publication_narration_srt_path": self.publication_narration_srt_path,
            "error": self.error,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommentaryOutput":
        raw_segments = data.get("segments", [])
        segments = [
            s if isinstance(s, Segment) else Segment.from_dict(s)
            for s in raw_segments
            if isinstance(s, (dict, Segment))
        ]
        title = str(data.get("title", ""))
        sanitized = str(data.get("sanitized_title", "")) or sanitize_title(title)
        return cls(
            output_id=str(data.get("output_id", "")),
            title=title,
            sanitized_title=sanitized,
            candidate_scope=str(data.get("candidate_scope", CandidateScope.SINGLE_EPISODE.value)),
            segments=segments,
            status=str(data.get("status", OutputStatus.WAITING.value)),
            publication_video_path=data.get("publication_video_path") or data.get("video_path"),
            publication_original_srt_path=data.get("publication_original_srt_path") or data.get("original_srt_path"),
            publication_narration_srt_path=data.get("publication_narration_srt_path") or data.get("narration_srt_path"),
            error=data.get("error"),
            progress=int(data.get("progress", 0)),
        )


@dataclass
class EpisodeEvidence:
    episode_id: str
    source_video: str
    duration_seconds: float = 0.0
    coverage: dict[str, Any] = field(default_factory=dict)
    missing_reasons: list[str] = field(default_factory=list)
    source_mtime: float = 0.0
    source_size: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeEvidence":
        return cls(
            episode_id=str(data.get("episode_id", "")),
            source_video=str(data.get("source_video", "")),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            coverage=dict(data.get("coverage", {})),
            missing_reasons=[str(r) for r in data.get("missing_reasons", [])],
            source_mtime=float(data.get("source_mtime", 0.0)),
            source_size=int(data.get("source_size", 0)),
            data=dict(data.get("data", {})),
        )


SUMMARY_SCHEMA_VERSION = "v1"


@dataclass
class CompactSummaryItem:
    refs: list[str] = field(default_factory=list)
    episode_id: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0
    characters: list[str] = field(default_factory=list)
    summary: str = ""
    categories: list[str] = field(default_factory=list)
    item_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "refs": list(self.refs),
            "episode_id": self.episode_id,
            "start_sec": round(float(self.start_sec), 3),
            "end_sec": round(float(self.end_sec), 3),
            "characters": list(self.characters),
            "summary": self.summary,
            "categories": list(self.categories),
            "item_type": self.item_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompactSummaryItem":
        if not isinstance(data, dict):
            raise ValueError("CompactSummaryItem dữ liệu phải là dictionary.")
        summary = str(data.get("summary", "")).strip()
        if not summary:
            raise ValueError("CompactSummaryItem thiếu trường summary.")
        return cls(
            refs=[str(r) for r in data.get("refs", [])],
            episode_id=str(data.get("episode_id", "")),
            start_sec=float(data.get("start_sec", 0.0)),
            end_sec=float(data.get("end_sec", 0.0)),
            characters=[str(c) for c in data.get("characters", [])],
            summary=summary,
            categories=[str(c) for c in data.get("categories", [])],
            item_type=str(data.get("item_type", "")),
        )


@dataclass
class CompactEpisodeSummary:
    episode_id: str
    title: str = ""
    duration_seconds: float = 0.0
    items: list[CompactSummaryItem] = field(default_factory=list)
    schema_version: str = SUMMARY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "title": self.title,
            "duration_seconds": round(float(self.duration_seconds), 3),
            "items": [item.to_dict() for item in self.items],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompactEpisodeSummary":
        if not isinstance(data, dict):
            raise ValueError("CompactEpisodeSummary dữ liệu phải là dictionary.")
        ep_id = str(data.get("episode_id", "")).strip()
        if not ep_id:
            raise ValueError("CompactEpisodeSummary thiếu episode_id.")
        raw_items = data.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("CompactEpisodeSummary trường items phải là một list.")
        items = [
            item if isinstance(item, CompactSummaryItem) else CompactSummaryItem.from_dict(item)
            for item in raw_items
            if isinstance(item, (dict, CompactSummaryItem))
        ]
        return cls(
            episode_id=ep_id,
            title=str(data.get("title", "")),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            items=items,
            schema_version=str(data.get("schema_version", SUMMARY_SCHEMA_VERSION)),
        )

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _normalize_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 1}


def _extract_item_info(category: str, raw: dict[str, Any]) -> tuple[str, list[str], str]:
    """Extract (summary_text, characters, item_type) from raw evidence item."""
    chars: list[str] = []
    summary = ""
    item_type = category

    if "characters" in raw and isinstance(raw["characters"], list):
        chars.extend([str(c).strip() for c in raw["characters"] if str(c).strip()])
    if "character" in raw and str(raw["character"]).strip():
        c = str(raw["character"]).strip()
        if c not in chars:
            chars.append(c)
    if "parties" in raw and isinstance(raw["parties"], list):
        for p in raw["parties"]:
            ps = str(p).strip()
            if ps and ps not in chars:
                chars.append(ps)
    if "speaker" in raw and str(raw["speaker"]).strip():
        s = str(raw["speaker"]).strip()
        if s not in chars:
            chars.append(s)

    raw_summary = str(raw.get("summary", "")).strip()
    raw_dialogue = ""
    if isinstance(raw.get("dialogue_evidence"), list) and raw["dialogue_evidence"]:
        raw_dialogue = " ".join(str(d).strip() for d in raw["dialogue_evidence"] if str(d).strip())
    elif isinstance(raw.get("dialogue_evidence"), str):
        raw_dialogue = str(raw["dialogue_evidence"]).strip()

    fallback_text = raw_summary or raw_dialogue or str(raw.get("description", "")).strip() or str(raw.get("detail", "")).strip()

    if category == "major_scenes":
        summary = raw_summary or fallback_text
        item_type = "scene"
    elif category == "dialogue":
        speaker = str(raw.get("speaker", "")).strip()
        quote = str(raw.get("quote", "")).strip() or fallback_text
        summary = f"{speaker}: \"{quote}\"" if speaker else (f"\"{quote}\"" if quote else "")
        item_type = "dialogue"
    elif category == "character_decisions":
        char = str(raw.get("character", "")).strip()
        dec = str(raw.get("decision", "")).strip() or fallback_text
        motive = str(raw.get("motive", "")).strip()
        if dec:
            prefix = f"{char} decides: " if char else "Decision: "
            summary = f"{prefix}{dec}" + (f" (motive: {motive})" if motive else "")
        item_type = "decision"
    elif category == "supporting_developments":
        char = str(raw.get("character", "")).strip()
        dev = str(raw.get("development", "")).strip() or fallback_text
        if dev:
            summary = f"{char}: {dev}" if char else dev
        item_type = "supporting_development"
    elif category == "relationships":
        dyn = str(raw.get("dynamic", "")).strip() or fallback_text
        if dyn:
            prefix = f"Relationship ({', '.join(chars)}): " if chars else "Relationship: "
            summary = f"{prefix}{dyn}"
        item_type = "relationship"
    elif category == "reveals":
        rev = str(raw.get("reveal", "")).strip() or fallback_text
        if rev:
            summary = f"Reveal: {rev}"
        item_type = "reveal"
    elif category == "reversals":
        rev = str(raw.get("reversal", "")).strip() or fallback_text
        if rev:
            summary = f"Reversal: {rev}"
        item_type = "reversal"
    elif category == "failures":
        char = str(raw.get("character", "")).strip()
        fail = str(raw.get("failure", "")).strip() or fallback_text
        if fail:
            summary = f"{char} failure: {fail}" if char else f"Failure: {fail}"
        item_type = "failure"
    elif category == "consequences":
        cq = str(raw.get("consequence", "")).strip() or fallback_text
        if cq:
            summary = f"Consequence: {cq}"
        item_type = "consequence"
    elif category == "performance_moments":
        pm = str(raw.get("description", "")).strip() or fallback_text
        if pm:
            summary = f"Key moment: {pm}"
        item_type = "performance_moment"
    elif category == "setup_payoff":
        stype = str(raw.get("type", "setup")).strip()
        detail = str(raw.get("detail", "")).strip() or fallback_text
        if detail:
            summary = f"{stype.title()}: {detail}"
        item_type = stype.lower() if stype else "setup"
    elif category == "unresolved":
        q = str(raw.get("question", "")).strip() or fallback_text
        if q:
            summary = f"Unresolved: {q}"
        item_type = "unresolved"
    elif category == "conflicts":
        conf = str(raw.get("conflict", "")).strip() or fallback_text
        if conf:
            prefix = f"Conflict ({', '.join(chars)}): " if chars else "Conflict: "
            summary = f"{prefix}{conf}"
        item_type = "conflict"
    elif category == "subplots":
        name = str(raw.get("name", "")).strip()
        prog = str(raw.get("progress", "")).strip() or fallback_text
        if prog:
            summary = f"Subplot '{name}': {prog}" if name else f"Subplot: {prog}"
        item_type = "subplot"
    elif category == "strengths_weaknesses":
        char = str(raw.get("character", "")).strip()
        s = str(raw.get("strength", "")).strip()
        w = str(raw.get("weakness", "")).strip()
        parts = []
        if s:
            parts.append(f"strength: {s}")
        if w:
            parts.append(f"weakness: {w}")
        if not parts and fallback_text:
            parts.append(fallback_text)
        if parts:
            summary = f"{char} ({', '.join(parts)})" if char else ", ".join(parts)
        item_type = "trait"
    else:
        # Fallback for any other category
        text_vals = [str(v).strip() for k, v in raw.items() if isinstance(v, str) and str(v).strip() and k not in ("episode_id", "item_type")]
        summary = "; ".join(text_vals) if text_vals else (fallback_text or f"{category} event")

    if not summary.strip():
        summary = fallback_text

    return summary.strip(), chars, item_type


def build_compact_summary(
    episode: SourceEpisode,
    evidence: EpisodeEvidence,
    *,
    schema_version: str = SUMMARY_SCHEMA_VERSION,
) -> CompactEpisodeSummary:
    """Deterministically build a grounded CompactEpisodeSummary from EpisodeEvidence.

    Extracts narrative items from all evidence categories, preserving episode identity,
    timestamps, characters, and short summaries. Deduplicates overlapping evidence
    across categories using normalized token similarity and time overlap, merging
    categories, refs, and characters without overdropping supporting content.
    """
    raw_items: list[CompactSummaryItem] = []
    ep_id = episode.episode_id
    max_duration = float(episode.duration_seconds) if episode.duration_seconds > 0.0 else float(evidence.duration_seconds)

    for cat_name, cat_list in sorted(evidence.data.items(), key=lambda x: x[0]):
        if not isinstance(cat_list, list):
            continue
        for idx, entry in enumerate(cat_list):
            if not isinstance(entry, dict):
                continue

            ref = f"{cat_name}:{idx}"
            if "start_sec" in entry:
                s_sec = max(0.0, float(entry["start_sec"]))
            elif "start_ms" in entry:
                s_sec = max(0.0, float(entry["start_ms"]) / 1000.0)
            else:
                s_sec = 0.0

            if "end_sec" in entry:
                e_sec = max(s_sec, float(entry["end_sec"]))
            elif "end_ms" in entry:
                e_sec = max(s_sec, float(entry["end_ms"]) / 1000.0)
            else:
                e_sec = s_sec + 5.0

            if max_duration > 0.0:
                e_sec = min(e_sec, max_duration)

            text_summary, chars, itype = _extract_item_info(cat_name, entry)
            if not text_summary:
                continue

            raw_items.append(
                CompactSummaryItem(
                    refs=[ref],
                    episode_id=ep_id,
                    start_sec=s_sec,
                    end_sec=e_sec,
                    characters=chars,
                    summary=text_summary,
                    categories=[cat_name],
                    item_type=itype,
                )
            )

    # Sort raw items deterministically by start time, end time, category
    raw_items.sort(key=lambda x: (x.start_sec, x.end_sec, x.categories[0] if x.categories else ""))

    # Deduplicate overlapping items across categories
    merged: list[CompactSummaryItem] = []
    for item in raw_items:
        tokens_curr = _normalize_tokens(item.summary)
        matched_idx = -1

        for i, existing in enumerate(merged):
            overlap = max(0.0, min(item.end_sec, existing.end_sec) - max(item.start_sec, existing.start_sec))
            dur_item = max(0.1, item.end_sec - item.start_sec)
            dur_exist = max(0.1, existing.end_sec - existing.start_sec)
            overlap_ratio = overlap / min(dur_item, dur_exist)
            time_close = abs(item.start_sec - existing.start_sec) <= 5.0 and abs(item.end_sec - existing.end_sec) <= 5.0

            if not (overlap_ratio >= 0.5 or time_close):
                continue

            tokens_exist = _normalize_tokens(existing.summary)
            jaccard = (
                len(tokens_curr & tokens_exist) / len(tokens_curr | tokens_exist)
                if (tokens_curr | tokens_exist)
                else 0.0
            )
            substr = (
                len(item.summary) > 8
                and len(existing.summary) > 8
                and (item.summary.lower() in existing.summary.lower() or existing.summary.lower() in item.summary.lower())
            )

            if jaccard >= 0.35 or substr:
                matched_idx = i
                break

        if matched_idx >= 0:
            ex = merged[matched_idx]
            # Merge refs, categories, characters
            ex.refs = sorted(set(ex.refs + item.refs))
            ex.categories = sorted(set(ex.categories + item.categories))
            ex.characters = sorted(set(ex.characters + item.characters))
            ex.start_sec = min(ex.start_sec, item.start_sec)
            ex.end_sec = max(ex.end_sec, item.end_sec)

            # Preserve setups, payoffs, supporting developments as primary item_type
            priority_types = ("setup", "payoff", "unresolved", "supporting_development", "reveal", "reversal")
            if item.item_type in priority_types and ex.item_type not in priority_types:
                ex.item_type = item.item_type

            # Merge summary text if one is not substring of the other
            if item.summary.lower() not in ex.summary.lower():
                if ex.summary.lower() in item.summary.lower():
                    ex.summary = item.summary
                else:
                    combined = f"{ex.summary} | {item.summary}"
                    ex.summary = combined[:300]
        else:
            merged.append(item)

    return CompactEpisodeSummary(
        episode_id=ep_id,
        title=episode.title,
        duration_seconds=episode.duration_seconds if episode.duration_seconds > 0.0 else evidence.duration_seconds,
        items=merged,
        schema_version=schema_version,
    )


@dataclass
class AnalysisManifest:
    project_id: str
    analysis_scope: str = AnalysisScope.SINGLE_EPISODE.value
    source_episodes: list[SourceEpisode] = field(default_factory=list)
    outputs: list[CommentaryOutput] = field(default_factory=list)
    created_at: str = ""
    recap_language: str = "en-US"
    recap_mode: str = "MAIN_STORIES"
    content_type: str = "US_TV_SHOW"
    source_rights_status: str = "UNVERIFIED"

    def validate(self) -> None:
        """Strict validation of manifest integrity:
        - Source episodes must be present and non-empty.
        - Episode IDs must be unique.
        - Source clips must reference valid episodes and valid source files (no hallucinations).
        - Source clips must have valid non-negative timings bounded by episode duration.
        - Output IDs must be unique.
        - Deterministically sanitize and resolve output titles with collision suffixes.
        - Empty outputs (0 candidates) is valid.
        - Multiple outputs (0, 1, N) are unlimited without artificial slicing.
        """
        if not self.source_episodes:
            raise ValidationError("AnalysisManifest must contain at least one source episode.")

        # Check unique episode IDs and build lookup
        episode_map: dict[str, SourceEpisode] = {}
        for ep in self.source_episodes:
            if not ep.episode_id:
                raise ValidationError("SourceEpisode missing episode_id.")
            if ep.episode_id in episode_map:
                raise ValidationError(f"Duplicate episode_id '{ep.episode_id}' in source episodes.")
            episode_map[ep.episode_id] = ep

        # Empty outputs is valid (0 candidates)
        if not self.outputs:
            return

        # Unique output IDs
        seen_output_ids: set[str] = set()
        for out in self.outputs:
            if not out.output_id:
                raise ValidationError("CommentaryOutput missing output_id.")
            if out.output_id in seen_output_ids:
                raise ValidationError(f"Duplicate output_id '{out.output_id}' in outputs.")
            seen_output_ids.add(out.output_id)

        # Sanitize and resolve unique titles deterministically <= 120 chars
        raw_titles = [out.title or out.output_id for out in self.outputs]
        sanitized_unique_titles = resolve_unique_titles(raw_titles, max_length=120)
        for out, san_title in zip(self.outputs, sanitized_unique_titles):
            out.sanitized_title = san_title

        # Validate segments and source clips
        for out in self.outputs:
            for seg in out.segments:
                for clip in seg.source_clips:
                    # 1. Reject hallucinated episode IDs
                    if clip.episode_id not in episode_map:
                        raise ValidationError(
                            f"Invalid episode_id '{clip.episode_id}' in source clip: not found in source episodes."
                        )

                    ep = episode_map[clip.episode_id]

                    # 2. Reject hallucinated source files
                    if clip.source_video and ep.source_video:
                        c_path = Path(clip.source_video).resolve()
                        e_path = Path(ep.source_video).resolve()
                        if c_path != e_path and clip.source_video != ep.source_video:
                            raise ValidationError(
                                f"Source clip file '{clip.source_video}' does not match episode '{ep.episode_id}' file '{ep.source_video}'."
                            )

                    # 3. Reject invalid timings
                    if clip.start < 0.0:
                        raise ValidationError(f"Invalid clip start timestamp {clip.start}: must be >= 0.")
                    if clip.end <= clip.start:
                        raise ValidationError(
                            f"Invalid clip end timestamp {clip.end}: must be greater than start ({clip.start})."
                        )
                    if ep.duration_seconds > 0.0 and clip.end > ep.duration_seconds + 0.5:
                        raise ValidationError(
                            f"Clip end {clip.end}s exceeds episode '{ep.episode_id}' duration {ep.duration_seconds}s."
                        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "analysis_scope": self.analysis_scope,
            "source_episodes": [ep.to_dict() for ep in self.source_episodes],
            "outputs": [out.to_dict() for out in self.outputs],
            "created_at": self.created_at,
            "recap_language": self.recap_language,
            "recap_mode": self.recap_mode,
            "content_type": self.content_type,
            "source_rights_status": self.source_rights_status,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any], validate: bool = True) -> "AnalysisManifest":
        raw_eps = data.get("source_episodes", [])
        episodes = [
            ep if isinstance(ep, SourceEpisode) else SourceEpisode.from_dict(ep)
            for ep in raw_eps
            if isinstance(ep, (dict, SourceEpisode))
        ]

        raw_outs = data.get("outputs", [])
        outputs = [
            out if isinstance(out, CommentaryOutput) else CommentaryOutput.from_dict(out)
            for out in raw_outs
            if isinstance(out, (dict, CommentaryOutput))
        ]

        manifest = cls(
            project_id=str(data.get("project_id", "")),
            analysis_scope=str(data.get("analysis_scope", AnalysisScope.SINGLE_EPISODE.value)),
            source_episodes=episodes,
            outputs=outputs,
            created_at=str(data.get("created_at", "")),
            recap_language=str(data.get("recap_language", "en-US")),
            recap_mode=str(data.get("recap_mode", "MAIN_STORIES")),
            content_type=str(data.get("content_type", "US_TV_SHOW")),
            source_rights_status=str(data.get("source_rights_status", data.get("rights", "UNVERIFIED"))),
        )

        if validate:
            manifest.validate()

        return manifest

    @classmethod
    def from_json(cls, json_str: str, validate: bool = True) -> "AnalysisManifest":
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValidationError("JSON root must be an object.")
        return cls.from_dict(data, validate=validate)
