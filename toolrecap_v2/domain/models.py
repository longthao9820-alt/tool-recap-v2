"""Core domain models, dataclasses, serialization, and validation for ToolRecap V2."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

from .enums import (
    AnalysisScope,
    AudioPolicy,
    CandidateScope,
    CandidateStatus,
    CompactionLevel,
    ConsolidationAction,
    ConsolidationReason,
    OutputStatus,
    ZeroOutputReason,
)
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
    segment_type: str = ""
    purpose: str = ""
    subtitle_policy: str = ""
    recommended_visual_speed: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "segment_id": self.segment_id,
            "source_clips": [c.to_dict() for c in self.source_clips],
            "original_dialogue": self.original_dialogue,
            "narration": self.narration,
            "audio_policy": self.audio_policy,
        }
        if self.segment_type:
            data["segment_type"] = self.segment_type
        if self.purpose:
            data["purpose"] = self.purpose
        if self.subtitle_policy:
            data["subtitle_policy"] = self.subtitle_policy
        if self.recommended_visual_speed != 1.0:
            data["recommended_visual_speed"] = float(self.recommended_visual_speed)
        return data

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
            segment_type=str(data.get("segment_type", "")),
            purpose=str(data.get("purpose", "")),
            subtitle_policy=str(data.get("subtitle_policy", "")),
            recommended_visual_speed=float(data.get("recommended_visual_speed", 1.0) or 1.0),
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
    candidate_id: str = ""
    source_candidate_id: str = ""
    file_name: str = ""
    output_type: str = ""
    language: str = ""

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
        d: dict[str, Any] = {
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
        cid = getattr(self, "source_candidate_id", "") or getattr(self, "candidate_id", "")
        if cid:
            d["source_candidate_id"] = cid
            d["candidate_id"] = cid
        if self.file_name:
            d["file_name"] = self.file_name
        if self.output_type:
            d["output_type"] = self.output_type
        if self.language:
            d["language"] = self.language
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

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
        cid = str(data.get("source_candidate_id") or data.get("candidate_id") or "").strip()
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
            candidate_id=cid,
            source_candidate_id=cid,
            file_name=str(data.get("file_name", "")),
            output_type=str(data.get("output_type", "")),
            language=str(data.get("language", "")),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CommentaryOutput":
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValidationError("JSON root must be an object.")
        return cls.from_dict(data)


@dataclass
class CandidateSourceRange:
    episode_id: str = ""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    evidence_ref: str = ""

    @property
    def ref(self) -> str:
        return self.evidence_ref

    @property
    def start(self) -> float:
        return self.start_seconds

    @property
    def end(self) -> float:
        return self.end_seconds

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end_seconds) - float(self.start_seconds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "start_seconds": round(float(self.start_seconds), 3),
            "end_seconds": round(float(self.end_seconds), 3),
            "evidence_ref": self.evidence_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateSourceRange":
        if not isinstance(data, dict):
            return cls()
        ep_id = str(data.get("episode_id", ""))
        start = float(data.get("start_seconds", data.get("start", 0.0)))
        end = float(data.get("end_seconds", data.get("end", 0.0)))
        ref = str(data.get("evidence_ref", data.get("ref", "")))
        return cls(
            episode_id=ep_id,
            start_seconds=start,
            end_seconds=end,
            evidence_ref=ref,
        )


@dataclass
class CandidateProposal:
    proposal_id: str = ""
    title: str = ""
    candidate_scope: str = CandidateScope.CROSS_EPISODE.value
    episodes: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    description: str = ""
    editorial_reason: str = ""
    status: str = "keep"  # keep, reject, merged
    subject: str = ""
    central_thesis: str = ""
    primary_character: str = ""
    supporting_characters: list[str] = field(default_factory=list)
    source_ranges: list[CandidateSourceRange] = field(default_factory=list)
    setup: str = ""
    development: str = ""
    turning: str = ""
    payoff: str = ""
    consequence: str = ""
    observed_facts: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)
    praise: str = ""
    criticism: str = ""
    alternative: str = ""
    why: str = ""
    hooks: list[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    overlap_tags: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "candidate_scope": self.candidate_scope,
            "episodes": list(self.episodes),
            "characters": list(self.characters),
            "description": self.description,
            "editorial_reason": self.editorial_reason,
            "status": self.status,
            "subject": self.subject,
            "central_thesis": self.central_thesis,
            "primary_character": self.primary_character,
            "supporting_characters": list(self.supporting_characters),
            "source_ranges": [r.to_dict() for r in self.source_ranges],
            "setup": self.setup,
            "development": self.development,
            "turning": self.turning,
            "payoff": self.payoff,
            "consequence": self.consequence,
            "observed_facts": list(self.observed_facts),
            "supporting_evidence": list(self.supporting_evidence),
            "counter_evidence": list(self.counter_evidence),
            "praise": self.praise,
            "criticism": self.criticism,
            "alternative": self.alternative,
            "why": self.why,
            "hooks": list(self.hooks),
            "estimated_duration": float(self.estimated_duration),
            "overlap_tags": list(self.overlap_tags),
            "confidence": float(self.confidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateProposal":
        if not isinstance(data, dict):
            return cls()

        raw_ranges = data.get("source_ranges", [])
        ranges = [
            r if isinstance(r, CandidateSourceRange) else CandidateSourceRange.from_dict(r)
            for r in raw_ranges
            if isinstance(r, (dict, CandidateSourceRange))
        ]

        raw_hooks = data.get("hooks", [])
        if isinstance(raw_hooks, str):
            hooks = [raw_hooks] if raw_hooks.strip() else []
        elif isinstance(raw_hooks, list):
            hooks = [str(h) for h in raw_hooks]
        else:
            hooks = []

        supp = data.get("supporting_evidence", data.get("supporting_evidence_refs", []))
        if isinstance(supp, list):
            supp_evidence = [str(x) for x in supp]
        elif isinstance(supp, str):
            supp_evidence = [supp] if supp.strip() else []
        else:
            supp_evidence = []

        counter = data.get("counter_evidence", data.get("counter_evidence_refs", []))
        if isinstance(counter, list):
            counter_evidence = [str(x) for x in counter]
        elif isinstance(counter, str):
            counter_evidence = [counter] if counter.strip() else []
        else:
            counter_evidence = []

        obs = data.get("observed_facts", [])
        if isinstance(obs, list):
            obs_facts = [str(x) for x in obs]
        elif isinstance(obs, str):
            obs_facts = [obs] if obs.strip() else []
        else:
            obs_facts = []

        tags = data.get("overlap_tags", [])
        if isinstance(tags, list):
            overlap_tags = [str(x) for x in tags]
        elif isinstance(tags, str):
            overlap_tags = [tags] if tags.strip() else []
        else:
            overlap_tags = []

        turning = str(data.get("turning", data.get("turning_point", "")))

        raw_dur = data.get("estimated_duration", data.get("estimated_duration_seconds", data.get("duration", 0.0)))
        try:
            duration = float(raw_dur or 0.0)
        except (ValueError, TypeError):
            duration = 0.0

        def _to_str(val: Any) -> str:
            if isinstance(val, list):
                return "\n".join(str(v) for v in val)
            return str(val or "")

        raw_conf = data.get("confidence", 1.0)
        try:
            confidence = float(raw_conf if raw_conf is not None else 1.0)
        except (ValueError, TypeError):
            confidence = 1.0

        return cls(
            proposal_id=str(data.get("proposal_id", "")),
            title=str(data.get("title", "")),
            candidate_scope=str(data.get("candidate_scope", CandidateScope.CROSS_EPISODE.value)),
            episodes=[str(e) for e in data.get("episodes", [])],
            characters=[str(c) for c in data.get("characters", [])],
            description=str(data.get("description", "")),
            editorial_reason=str(data.get("editorial_reason", "")),
            status=str(data.get("status", "keep")),
            subject=str(data.get("subject", "")),
            central_thesis=str(data.get("central_thesis", "")),
            primary_character=str(data.get("primary_character", "")),
            supporting_characters=[str(c) for c in data.get("supporting_characters", [])],
            source_ranges=ranges,
            setup=str(data.get("setup", "")),
            development=str(data.get("development", "")),
            turning=turning,
            payoff=str(data.get("payoff", "")),
            consequence=str(data.get("consequence", "")),
            observed_facts=obs_facts,
            supporting_evidence=supp_evidence,
            counter_evidence=counter_evidence,
            praise=_to_str(data.get("praise")),
            criticism=_to_str(data.get("criticism")),
            alternative=_to_str(data.get("alternative")),
            why=str(data.get("why", "")),
            hooks=hooks,
            estimated_duration=duration,
            overlap_tags=overlap_tags,
            confidence=confidence,
        )


CONSOLIDATION_SCHEMA_VERSION: str = "v1"


@dataclass
class ConsolidationDecision:
    candidate_id: str = ""
    action: str = ConsolidationAction.KEEP.value
    reason_code: str = ""
    reason: str = ""
    target_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action": self.action,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "target_id": self.target_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsolidationDecision:
        if not isinstance(data, dict):
            return cls()
        cid = str(data.get("candidate_id") or data.get("proposal_id") or data.get("id", "")).strip()
        raw_action = str(data.get("action", ConsolidationAction.KEEP.value)).strip().upper()
        if raw_action not in (
            ConsolidationAction.KEEP.value,
            ConsolidationAction.MERGE.value,
            ConsolidationAction.REJECT.value,
        ):
            raw_action = ConsolidationAction.KEEP.value
        rcode = str(data.get("reason_code", "")).strip()
        reason = str(data.get("reason") or data.get("explanation") or "").strip()
        target = str(data.get("target_id") or data.get("into_id") or data.get("merged_into") or "").strip()
        return cls(
            candidate_id=cid,
            action=raw_action,
            reason_code=rcode,
            reason=reason,
            target_id=target,
        )


@dataclass
class ConsolidatedCandidateSet:
    candidates: list[CandidateProposal] = field(default_factory=list)
    decisions: list[ConsolidationDecision] = field(default_factory=list)
    schema_version: str = CONSOLIDATION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def merged_count(self) -> int:
        return sum(1 for d in self.decisions if d.action == ConsolidationAction.MERGE.value)

    @property
    def rejected_count(self) -> int:
        return sum(1 for d in self.decisions if d.action == ConsolidationAction.REJECT.value)

    @property
    def kept_count(self) -> int:
        return sum(1 for d in self.decisions if d.action == ConsolidationAction.KEEP.value)

    @property
    def eligible_count(self) -> int:
        return len(self.candidates)

    def get_decision(self, candidate_id: str) -> ConsolidationDecision | None:
        cid = str(candidate_id).strip()
        for d in self.decisions:
            if d.candidate_id == cid:
                return d
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "consolidated_candidates": [c.to_dict() for c in self.candidates],
            "decisions": [d.to_dict() for d in self.decisions],
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsolidatedCandidateSet:
        if not isinstance(data, dict):
            return cls()
        raw_cands = data.get("consolidated_candidates", data.get("candidates", []))
        cands = [
            c if isinstance(c, CandidateProposal) else CandidateProposal.from_dict(c)
            for c in raw_cands
            if isinstance(c, (dict, CandidateProposal))
        ]
        raw_decs = data.get("decisions", [])
        decs = [
            d if isinstance(d, ConsolidationDecision) else ConsolidationDecision.from_dict(d)
            for d in raw_decs
            if isinstance(d, (dict, ConsolidationDecision))
        ]
        meta = data.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        sver = str(data.get("schema_version", data.get("schema", CONSOLIDATION_SCHEMA_VERSION)))
        return cls(
            candidates=cands,
            decisions=decs,
            schema_version=sver,
            metadata=meta,
        )

    @classmethod
    def from_json(cls, json_str: str) -> ConsolidatedCandidateSet:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValidationError("JSON root must be an object.")
        return cls.from_dict(data)


@dataclass
class PipelineHealth:
    coverage_ledgers: list[Any] | dict[str, Any] = field(default_factory=list)
    discovery_completed: bool = True
    discovery_error: str | None = None
    discovery_schema_valid: bool = True
    discovered_count: int = 0
    consolidation_completed: bool = True
    consolidation_error: str | None = None
    consolidation_schema_valid: bool = True
    consolidation_decisions: list[ConsolidationDecision] | list[dict[str, Any]] = field(default_factory=list)
    consolidated_candidates: list[CandidateProposal] | list[dict[str, Any]] = field(default_factory=list)
    finalizer_attempted: bool = False
    finalizer_completed: bool = False
    finalizer_results: list[Any] = field(default_factory=list)
    finalizer_error: str | None = None
    parser_failed: bool = False
    parser_error: str | None = None
    schema_rejected: bool = False
    schema_error: str | None = None
    total_evidence_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cands_serialized = []
        for c in self.consolidated_candidates:
            if hasattr(c, "to_dict"):
                cands_serialized.append(c.to_dict())
            elif isinstance(c, dict):
                cands_serialized.append(dict(c))
            else:
                cands_serialized.append(str(c))

        decs_serialized = []
        for d in self.consolidation_decisions:
            if hasattr(d, "to_dict"):
                decs_serialized.append(d.to_dict())
            elif isinstance(d, dict):
                decs_serialized.append(dict(d))
            else:
                decs_serialized.append(str(d))

        ledgers_serialized = []
        if isinstance(self.coverage_ledgers, dict):
            ledgers_serialized = {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.coverage_ledgers.items()
            }
        elif isinstance(self.coverage_ledgers, list):
            ledgers_serialized = [
                x.to_dict() if hasattr(x, "to_dict") else x
                for x in self.coverage_ledgers
            ]

        return {
            "coverage_ledgers": ledgers_serialized,
            "discovery_completed": self.discovery_completed,
            "discovery_error": self.discovery_error,
            "discovery_schema_valid": self.discovery_schema_valid,
            "discovered_count": self.discovered_count,
            "consolidation_completed": self.consolidation_completed,
            "consolidation_error": self.consolidation_error,
            "consolidation_schema_valid": self.consolidation_schema_valid,
            "consolidation_decisions": decs_serialized,
            "consolidated_candidates": cands_serialized,
            "finalizer_attempted": self.finalizer_attempted,
            "finalizer_completed": self.finalizer_completed,
            "finalizer_results": list(self.finalizer_results),
            "finalizer_error": self.finalizer_error,
            "parser_failed": self.parser_failed,
            "parser_error": self.parser_error,
            "schema_rejected": self.schema_rejected,
            "schema_error": self.schema_error,
            "total_evidence_count": self.total_evidence_count,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineHealth:
        if not isinstance(data, dict):
            return cls()
        cands_raw = data.get("consolidated_candidates", [])
        cands = [
            c if isinstance(c, CandidateProposal) else CandidateProposal.from_dict(c)
            for c in cands_raw
            if isinstance(c, (dict, CandidateProposal))
        ]
        decs_raw = data.get("consolidation_decisions", [])
        decs = [
            d if isinstance(d, ConsolidationDecision) else ConsolidationDecision.from_dict(d)
            for d in decs_raw
            if isinstance(d, (dict, ConsolidationDecision))
        ]
        return cls(
            coverage_ledgers=data.get("coverage_ledgers", []),
            discovery_completed=bool(data.get("discovery_completed", True)),
            discovery_error=data.get("discovery_error"),
            discovery_schema_valid=bool(data.get("discovery_schema_valid", True)),
            discovered_count=int(data.get("discovered_count", 0)),
            consolidation_completed=bool(data.get("consolidation_completed", True)),
            consolidation_error=data.get("consolidation_error"),
            consolidation_schema_valid=bool(data.get("consolidation_schema_valid", True)),
            consolidation_decisions=decs,
            consolidated_candidates=cands,
            finalizer_attempted=bool(data.get("finalizer_attempted", False)),
            finalizer_completed=bool(data.get("finalizer_completed", False)),
            finalizer_results=list(data.get("finalizer_results", [])),
            finalizer_error=data.get("finalizer_error"),
            parser_failed=bool(data.get("parser_failed", False)),
            parser_error=data.get("parser_error"),
            schema_rejected=bool(data.get("schema_rejected", False)),
            schema_error=data.get("schema_error"),
            total_evidence_count=int(data.get("total_evidence_count", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class VerificationResult:
    reason: ZeroOutputReason | str = ZeroOutputReason.NO_ELIGIBLE_CANDIDATES.value
    is_valid_zero: bool = False
    recovered_candidates: list[CandidateProposal] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value if isinstance(self.reason, ZeroOutputReason) else str(self.reason),
            "is_valid_zero": self.is_valid_zero,
            "recovered_candidates": [c.to_dict() for c in self.recovered_candidates],
            "diagnostics": dict(self.diagnostics),
            "completed": self.completed,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationResult:
        if not isinstance(data, dict):
            return cls()
        raw_cands = data.get("recovered_candidates", data.get("candidates", []))
        cands = [
            c if isinstance(c, CandidateProposal) else CandidateProposal.from_dict(c)
            for c in raw_cands
            if isinstance(c, (dict, CandidateProposal))
        ]
        reason_val = data.get("reason", ZeroOutputReason.NO_ELIGIBLE_CANDIDATES.value)
        if isinstance(reason_val, str):
            try:
                reason_obj = ZeroOutputReason(reason_val)
            except ValueError:
                reason_obj = reason_val
        else:
            reason_obj = reason_val
        return cls(
            reason=reason_obj,
            is_valid_zero=bool(data.get("is_valid_zero", False)),
            recovered_candidates=cands,
            diagnostics=dict(data.get("diagnostics", {})),
            completed=bool(data.get("completed", False)),
            rationale=str(data.get("rationale", "")),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> VerificationResult:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValidationError("JSON root must be an object.")
        return cls.from_dict(data)



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

    def __init__(
        self,
        episode_id: str = "",
        source_video: str = "",
        duration_seconds: float = 0.0,
        coverage: dict[str, Any] | None = None,
        missing_reasons: list[str] | None = None,
        source_mtime: float = 0.0,
        source_size: int = 0,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.episode_id = episode_id
        self.source_video = source_video
        self.duration_seconds = duration_seconds
        self.coverage = coverage or {}
        self.missing_reasons = missing_reasons or []
        self.source_mtime = source_mtime
        self.source_size = source_size
        self.data = dict(data or {})
        if kwargs:
            self.data.update(kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in ("data", "__dataclass_fields__"):
            raise AttributeError(name)
        if "data" in self.__dict__ and name in self.__dict__["data"]:
            return self.__dict__["data"][name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
    fragment_id: str = ""
    fragment_index: int = 0
    total_fragments: int = 1

    def to_dict(self) -> dict[str, Any]:
        d = {
            "episode_id": self.episode_id,
            "title": self.title,
            "duration_seconds": round(float(self.duration_seconds), 3),
            "items": [item.to_dict() for item in self.items],
            "schema_version": self.schema_version,
        }
        if self.fragment_id:
            d["fragment_id"] = self.fragment_id
            d["fragment_index"] = self.fragment_index
            d["total_fragments"] = self.total_fragments
        return d

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
            fragment_id=str(data.get("fragment_id", "")),
            fragment_index=int(data.get("fragment_index", 0)),
            total_fragments=int(data.get("total_fragments", 1)),
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
    elif category == "contradictions":
        c_text = str(raw.get("contradiction", "")).strip() or fallback_text
        if c_text:
            summary = f"Contradiction: {c_text}"
        item_type = "contradiction"
    elif category == "dilemmas":
        d_text = str(raw.get("dilemma", "")).strip() or fallback_text
        if d_text:
            summary = f"Dilemma: {d_text}"
        item_type = "dilemma"
    elif category == "visual_storytelling":
        v_text = str(raw.get("visual", raw.get("description", ""))).strip() or fallback_text
        if v_text:
            summary = f"Visual: {v_text}"
        item_type = "visual_storytelling"
    elif category == "recurring_behavior":
        b_text = str(raw.get("behavior", raw.get("recurring_behavior", ""))).strip() or fallback_text
        if b_text:
            summary = f"Behavior: {b_text}"
        item_type = "recurring_behavior"
    elif category == "power_shifts":
        ps_text = str(raw.get("power_shift", raw.get("shift", ""))).strip() or fallback_text
        if ps_text:
            summary = f"Power shift: {ps_text}"
        item_type = "power_shift"
    elif category == "reactions":
        r_text = str(raw.get("reaction", "")).strip() or fallback_text
        if r_text:
            summary = f"Reaction: {r_text}"
        item_type = "reaction"
    elif category == "counter_evidence":
        ce_text = str(raw.get("counter_evidence", raw.get("claim", ""))).strip() or fallback_text
        if ce_text:
            summary = f"Counter evidence: {ce_text}"
        item_type = "counter_evidence"
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
    coverage_ledger: Any = None,
) -> CompactEpisodeSummary:
    """Deterministically build a grounded CompactEpisodeSummary from EpisodeEvidence and coverage ledger.

    Extracts narrative items from all evidence categories, preserving episode identity,
    timestamps, characters, and short summaries. Deduplicates overlapping evidence
    across categories using normalized token similarity and time overlap, merging
    categories, refs, and characters without overdropping supporting content.
    """
    raw_items: list[CompactSummaryItem] = []
    ep_id = episode.episode_id
    max_duration = float(episode.duration_seconds) if episode.duration_seconds > 0.0 else float(evidence.duration_seconds)
    eff_ledger = coverage_ledger if coverage_ledger is not None else evidence.coverage

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


SKELETON_TEXT_CAP = 80


def compact_summary(
    summary: CompactEpisodeSummary,
    level: CompactionLevel | str = CompactionLevel.FULL,
    *,
    skeleton_cap: int = SKELETON_TEXT_CAP,
) -> CompactEpisodeSummary:
    """Deterministically compact a CompactEpisodeSummary to FULL, TRIMMED, PRIORITY, or SKELETON.

    Preserves original episode_id, fragment metadata, timestamps, categories, characters,
    and ref identity. Supporting character evidence explicitly prioritized to survive.
    """
    lvl = CompactionLevel(level) if isinstance(level, str) else level
    if lvl == CompactionLevel.FULL:
        return CompactEpisodeSummary(
            episode_id=summary.episode_id,
            title=summary.title,
            duration_seconds=summary.duration_seconds,
            items=[
                CompactSummaryItem(
                    refs=list(it.refs),
                    episode_id=it.episode_id or summary.episode_id,
                    start_sec=it.start_sec,
                    end_sec=it.end_sec,
                    characters=list(it.characters),
                    summary=it.summary,
                    categories=list(it.categories),
                    item_type=it.item_type,
                )
                for it in summary.items
            ],
            schema_version=summary.schema_version,
            fragment_id=summary.fragment_id,
            fragment_index=summary.fragment_index,
            total_fragments=summary.total_fragments,
        )

    if lvl == CompactionLevel.TRIMMED:
        items: list[CompactSummaryItem] = []
        for it in summary.items:
            t_sum = it.summary[:140] if len(it.summary) > 140 else it.summary
            items.append(
                CompactSummaryItem(
                    refs=list(it.refs[:3]),
                    episode_id=it.episode_id or summary.episode_id,
                    start_sec=it.start_sec,
                    end_sec=it.end_sec,
                    characters=list(it.characters),
                    summary=t_sum,
                    categories=list(it.categories),
                    item_type=it.item_type,
                )
            )
        return CompactEpisodeSummary(
            episode_id=summary.episode_id,
            title=summary.title,
            duration_seconds=summary.duration_seconds,
            items=items,
            schema_version=summary.schema_version,
            fragment_id=summary.fragment_id,
            fragment_index=summary.fragment_index,
            total_fragments=summary.total_fragments,
        )

    if lvl == CompactionLevel.PRIORITY:
        priority_items: list[CompactSummaryItem] = []
        fallback_items: list[CompactSummaryItem] = []
        for it in summary.items:
            is_supporting = (
                it.item_type in ("supporting_development", "trait")
                or any("supporting" in c.lower() or "strengths" in c.lower() for c in it.categories)
                or bool(it.characters)
            )
            is_turning_point = (
                it.item_type in ("setup", "payoff", "reveal", "reversal", "decision", "consequence", "failure", "conflict", "unresolved")
                or any(c in ("setup_payoff", "reveals", "reversals", "character_decisions", "consequences", "failures", "conflicts", "unresolved") for c in it.categories)
            )
            is_policy_priority = (
                it.item_type in (
                    "supporting_development", "relationship", "reveal", "reversal", "setup", "payoff",
                    "consequence", "conflict", "subplot", "recurring_behavior", "contradiction",
                    "counter_evidence", "dilemma", "power_shift", "reaction", "visual_storytelling",
                )
                or any(
                    c in (
                        "supporting_developments", "relationships", "reveals", "reversals", "setup_payoff",
                        "consequences", "conflicts", "subplots", "recurring_behavior", "contradictions",
                        "counter_evidence", "dilemmas", "power_shifts", "reactions", "visual_storytelling",
                    )
                    for c in it.categories
                )
            )
            t_sum = it.summary[:100] if len(it.summary) > 100 else it.summary
            compacted_it = CompactSummaryItem(
                refs=list(it.refs[:2]),
                episode_id=it.episode_id or summary.episode_id,
                start_sec=it.start_sec,
                end_sec=it.end_sec,
                characters=list(it.characters),
                summary=t_sum,
                categories=list(it.categories),
                item_type=it.item_type,
            )
            if is_supporting or is_turning_point or is_policy_priority:
                priority_items.append(compacted_it)
            else:
                fallback_items.append(compacted_it)

        selected_items = priority_items if priority_items else fallback_items
        return CompactEpisodeSummary(
            episode_id=summary.episode_id,
            title=summary.title,
            duration_seconds=summary.duration_seconds,
            items=selected_items,
            schema_version=summary.schema_version,
            fragment_id=summary.fragment_id,
            fragment_index=summary.fragment_index,
            total_fragments=summary.total_fragments,
        )

    # SKELETON
    skeleton_items: list[CompactSummaryItem] = []
    fallback_skeleton: list[CompactSummaryItem] = []
    for it in summary.items:
        is_supporting = (
            it.item_type in ("supporting_development", "trait")
            or any("supporting" in c.lower() or "strengths" in c.lower() for c in it.categories)
            or bool(it.characters)
        )
        is_turning_point = (
            it.item_type in ("setup", "payoff", "reveal", "reversal", "decision", "consequence", "failure", "conflict", "unresolved")
            or any(c in ("setup_payoff", "reveals", "reversals", "character_decisions", "consequences", "failures", "conflicts", "unresolved") for c in it.categories)
        )
        is_policy_priority = (
            it.item_type in (
                "supporting_development", "relationship", "reveal", "reversal", "setup", "payoff",
                "consequence", "conflict", "subplot", "recurring_behavior", "contradiction",
                "counter_evidence", "dilemma", "power_shift", "reaction", "visual_storytelling",
            )
            or any(
                c in (
                    "supporting_developments", "relationships", "reveals", "reversals", "setup_payoff",
                    "consequences", "conflicts", "subplots", "recurring_behavior", "contradictions",
                    "counter_evidence", "dilemmas", "power_shifts", "reactions", "visual_storytelling",
                )
                for c in it.categories
            )
        )
        t_sum = it.summary[:skeleton_cap] if len(it.summary) > skeleton_cap else it.summary
        compacted_it = CompactSummaryItem(
            refs=[str(r)[:80] for r in (it.refs[:2] if it.refs else [f"{it.item_type or 'ev'}:0"])],
            episode_id=it.episode_id or summary.episode_id,
            start_sec=it.start_sec,
            end_sec=it.end_sec,
            characters=[str(c)[:80] for c in it.characters[:5]],
            summary=t_sum[:80],
            categories=[str(c)[:50] for c in it.categories[:5]],
            item_type=it.item_type,
        )
        if is_supporting or is_turning_point or is_policy_priority:
            skeleton_items.append(compacted_it)
        else:
            fallback_skeleton.append(compacted_it)

    final_skeleton_items = skeleton_items if skeleton_items else fallback_skeleton
    return CompactEpisodeSummary(
        episode_id=summary.episode_id,
        title=summary.title[:120] if summary.title else "",
        duration_seconds=summary.duration_seconds,
        items=final_skeleton_items,
        schema_version=summary.schema_version,
        fragment_id=summary.fragment_id,
        fragment_index=summary.fragment_index,
        total_fragments=summary.total_fragments,
    )


def split_summary_by_timeline(summary: CompactEpisodeSummary) -> list[CompactEpisodeSummary]:
    """Split a summary into two timeline fragments preserving original episode_id and grounding."""
    if len(summary.items) <= 1:
        return [summary]

    mid = len(summary.items) // 2
    base_ep_id = summary.episode_id
    base_prefix = summary.fragment_id or base_ep_id

    frag1 = CompactEpisodeSummary(
        episode_id=base_ep_id,
        title=summary.title,
        duration_seconds=summary.duration_seconds,
        items=list(summary.items[:mid]),
        schema_version=summary.schema_version,
        fragment_id=f"{base_prefix}_f1",
        fragment_index=0,
        total_fragments=2,
    )
    frag2 = CompactEpisodeSummary(
        episode_id=base_ep_id,
        title=summary.title,
        duration_seconds=summary.duration_seconds,
        items=list(summary.items[mid:]),
        schema_version=summary.schema_version,
        fragment_id=f"{base_prefix}_f2",
        fragment_index=1,
        total_fragments=2,
    )
    return [frag1, frag2]



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
    zero_output_reason: str | None = None
    zero_output_status: str | None = None
    verification: VerificationResult | dict[str, Any] | None = None

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
        raw_titles = [
            Path(out.file_name).stem if out.file_name else (out.title or out.output_id)
            for out in self.outputs
        ]
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
        d: dict[str, Any] = {
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
        if self.zero_output_reason is not None:
            d["zero_output_reason"] = self.zero_output_reason
        if self.zero_output_status is not None:
            d["zero_output_status"] = self.zero_output_status
        if self.verification is not None:
            d["verification"] = self.verification.to_dict() if hasattr(self.verification, "to_dict") else self.verification
        return d

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

        ver_raw = data.get("verification")
        ver_obj = None
        if ver_raw is not None:
            ver_obj = ver_raw if isinstance(ver_raw, VerificationResult) else VerificationResult.from_dict(ver_raw)

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
            zero_output_reason=data.get("zero_output_reason"),
            zero_output_status=data.get("zero_output_status"),
            verification=ver_obj,
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
