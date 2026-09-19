"""Core domain models, dataclasses, serialization, and validation for ToolRecap V2."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
