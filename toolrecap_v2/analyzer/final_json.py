"""Canonical AI Finalizer boundary and deterministic Final JSON validation.

This module deliberately contains no editorial ranking, candidate filtering, season
compaction, or story selection.  The configured Finalizer owns those decisions.
ToolRecap only transports source observations, validates the returned render plan,
and asks the same Finalizer to repair technical contract violations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any, Callable

from ..api_client import OpenAICompatibleClient
from ..domain.enums import AudioPolicy, CandidateScope, OutputStatus
from ..domain.models import (
    AnalysisManifest,
    CommentaryOutput,
    EpisodeEvidence,
    Segment,
    SourceClip,
    SourceEpisode,
)
from ..domain.title import sanitize_title
from ..settings import AppSettings
from .errors import AnalysisCancelledError, AnalysisError
from .phases import AnalysisPhase, PhaseCallback


FINAL_JSON_SCHEMA_VERSION = "1.0"
SUPPORTED_AUDIO_POLICIES = {policy.value for policy in AudioPolicy}


FINAL_JSON_SYSTEM_PROMPT = """You are the final editor for ToolRecap.
You receive source identities and technical Scanner observations for one complete
project, plus the user's raw Recap Prompt. You alone make the editorial decisions:
what outputs to create, what stories and characters to cover, titles, narration,
dialogue, source footage, timestamps, ordering, and output separation.

Return exactly one JSON object conforming to the supplied Final JSON contract.
An explicit \"outputs\": [] is a valid editorial decision. Never omit \"outputs\".
Use only supplied episode IDs, source files, facts, and timestamps. Return JSON only.
""".strip()


@dataclass(frozen=True)
class FinalJsonIssue:
    """One deterministic, machine-readable technical contract violation."""

    path: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


class FinalJsonValidationError(AnalysisError):
    """Raised when a Finalizer response is JSON but cannot be rendered safely."""

    def __init__(self, issues: list[FinalJsonIssue]) -> None:
        self.issues = list(issues)
        summary = "; ".join(f"{i.path}: {i.code} — {i.message}" for i in self.issues)
        super().__init__(summary or "FINAL_JSON_VALIDATION_ERROR")


def _source_contract(episodes: list[SourceEpisode]) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": ep.episode_id,
            "source_file": ep.source_video,
            "duration_ms": int(round(max(0.0, ep.duration_seconds) * 1000.0)),
            "title": ep.title,
            "season_number": ep.season_number,
            "episode_number": ep.episode_number,
        }
        for ep in episodes
    ]


def final_json_contract() -> dict[str, Any]:
    """Return the renderer-facing contract sent to the configured Finalizer."""
    return {
        "schema_version": FINAL_JSON_SCHEMA_VERSION,
        "required_root_fields": ["outputs"],
        "outputs": {
            "type": "array",
            "empty_is_valid": True,
            "item": {
                "required": ["output_id", "title", "file_name", "output_type", "language", "segments"],
                "optional": ["candidate_scope"],
                "segments": {
                    "type": "array",
                    "min_items": 1,
                    "item": {
                        "required": ["segment_id", "segment_type", "audio_policy", "source_clips"],
                        "optional": [
                            "purpose",
                            "narration_text",
                            "original_dialogue_text",
                            "subtitle_policy",
                            "recommended_visual_speed",
                        ],
                        "source_clips": {
                            "type": "array",
                            "min_items": 1,
                            "item_required": ["episode_id", "source_file", "start_ms", "end_ms"],
                        },
                    },
                },
            },
        },
        "supported_audio_policies": sorted(SUPPORTED_AUDIO_POLICIES),
        "rules": [
            "narration segments require non-empty narration_text",
            "original_only segments require original_dialogue_text or source_clips",
            "0 <= start_ms < end_ms <= source duration",
            "all IDs must be unique within their scope",
            "source references must resolve exactly to the supplied project sources",
        ],
    }


def build_finalizer_project_prompt(
    project_id: str,
    episodes: list[SourceEpisode],
    evidence_map: dict[str, EpisodeEvidence],
    settings: AppSettings,
    scope: str,
) -> str:
    """Build one project-level request while preserving the raw Recap Prompt verbatim."""
    observations = {
        episode_id: evidence.to_dict()
        for episode_id, evidence in sorted(evidence_map.items())
    }
    payload = {
        "project_id": project_id,
        "analysis_scope": scope,
        "recap_language": settings.recap_language,
        "recap_mode": settings.recap_mode,
        "content_type": settings.content_type,
        "source_rights_status": settings.source_rights_status,
        "sources": _source_contract(episodes),
        "scanner_observations": observations,
        "final_json_contract": final_json_contract(),
    }
    raw_prompt = settings.recap_prompt or ""
    return (
        "Complete this ToolRecap project and return its canonical Final JSON.\n\n"
        "BEGIN RAW RECAP PROMPT (authoritative; preserve its meaning exactly)\n"
        f"{raw_prompt}\n"
        "END RAW RECAP PROMPT\n\n"
        "PROJECT TECHNICAL DATA\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _issue(
    issues: list[FinalJsonIssue],
    path: str,
    code: str,
    message: str,
    **details: Any,
) -> None:
    issues.append(FinalJsonIssue(path=path, code=code, message=message, details=details))


def validate_final_json(
    raw: Any,
    *,
    project_id: str,
    episodes: list[SourceEpisode],
    scope: str,
    settings: AppSettings,
    require_source_files: bool = False,
) -> AnalysisManifest:
    """Validate Final JSON technically and convert it to a render manifest.

    This routine never judges editorial merit and never silently clamps, substitutes,
    drops, or rewrites an AI-selected output, segment, source, or timestamp.
    """
    issues: list[FinalJsonIssue] = []
    if not isinstance(raw, dict):
        raise FinalJsonValidationError(
            [FinalJsonIssue("$", "SCHEMA_ERROR", "Final JSON root must be an object.")]
        )
    if "outputs" not in raw:
        raise FinalJsonValidationError(
            [FinalJsonIssue("$.outputs", "SCHEMA_ERROR", "Required field 'outputs' is missing.")]
        )
    raw_outputs = raw.get("outputs")
    if not isinstance(raw_outputs, list):
        raise FinalJsonValidationError(
            [FinalJsonIssue("$.outputs", "SCHEMA_ERROR", "Field 'outputs' must be an array.")]
        )

    episode_map = {ep.episode_id: ep for ep in episodes}
    source_by_name: dict[str, list[SourceEpisode]] = {}
    for ep in episodes:
        source_by_name.setdefault(Path(ep.source_video).name.casefold(), []).append(ep)

    outputs: list[CommentaryOutput] = []
    output_ids: set[str] = set()
    file_names: set[str] = set()
    for oi, out_raw in enumerate(raw_outputs):
        opath = f"$.outputs[{oi}]"
        if not isinstance(out_raw, dict):
            _issue(issues, opath, "SCHEMA_ERROR", "Output must be an object.")
            continue
        output_id = str(out_raw.get("output_id") or "").strip()
        title = str(out_raw.get("title") or "").strip()
        file_name = str(out_raw.get("file_name") or "").strip()
        output_type = str(out_raw.get("output_type") or "").strip()
        language = str(out_raw.get("language") or "").strip()
        if not output_id:
            _issue(issues, f"{opath}.output_id", "REQUIRED_FIELD", "output_id is required.")
        elif output_id in output_ids:
            _issue(issues, f"{opath}.output_id", "DUPLICATE_ID", "output_id must be unique.", value=output_id)
        else:
            output_ids.add(output_id)
        if not title:
            _issue(issues, f"{opath}.title", "REQUIRED_FIELD", "title is required.")
        if not file_name:
            _issue(issues, f"{opath}.file_name", "REQUIRED_FIELD", "file_name is required.")
        else:
            file_path = Path(file_name)
            safe_stem = sanitize_title(file_path.stem)
            if file_path.name != file_name or file_path.suffix.casefold() != ".mp4" or safe_stem != file_path.stem:
                _issue(
                    issues,
                    f"{opath}.file_name",
                    "INVALID_FILE_NAME",
                    "file_name must be a filesystem-safe .mp4 basename.",
                    requested=file_name,
                )
            folded_name = file_name.casefold()
            if folded_name in file_names:
                _issue(issues, f"{opath}.file_name", "DUPLICATE_FILE_NAME", "file_name must be unique.")
            file_names.add(folded_name)
        if not output_type:
            _issue(issues, f"{opath}.output_type", "REQUIRED_FIELD", "output_type is required.")
        if not language:
            _issue(issues, f"{opath}.language", "REQUIRED_FIELD", "language is required.")

        segments_raw = out_raw.get("segments")
        if not isinstance(segments_raw, list) or not segments_raw:
            _issue(issues, f"{opath}.segments", "SCHEMA_ERROR", "segments must be a non-empty array.")
            segments_raw = []

        segments: list[Segment] = []
        segment_ids: set[str] = set()
        for si, seg_raw in enumerate(segments_raw):
            spath = f"{opath}.segments[{si}]"
            if not isinstance(seg_raw, dict):
                _issue(issues, spath, "SCHEMA_ERROR", "Segment must be an object.")
                continue
            segment_id = str(seg_raw.get("segment_id") or "").strip()
            segment_type = str(seg_raw.get("segment_type") or "").strip().lower()
            narration = str(seg_raw.get("narration_text", seg_raw.get("narration", "")) or "").strip()
            dialogue = str(
                seg_raw.get("original_dialogue_text", seg_raw.get("original_dialogue", "")) or ""
            ).strip()
            audio_policy = str(seg_raw.get("audio_policy") or "").strip().lower()
            if not segment_id:
                _issue(issues, f"{spath}.segment_id", "REQUIRED_FIELD", "segment_id is required.")
            elif segment_id in segment_ids:
                _issue(issues, f"{spath}.segment_id", "DUPLICATE_ID", "segment_id must be unique within an output.")
            else:
                segment_ids.add(segment_id)
            if not segment_type:
                _issue(issues, f"{spath}.segment_type", "REQUIRED_FIELD", "segment_type is required.")
            if audio_policy not in SUPPORTED_AUDIO_POLICIES:
                _issue(
                    issues,
                    f"{spath}.audio_policy",
                    "UNSUPPORTED_AUDIO_POLICY",
                    "audio_policy is not supported by the renderer.",
                    requested=audio_policy,
                    supported=sorted(SUPPORTED_AUDIO_POLICIES),
                )
            if audio_policy != AudioPolicy.ORIGINAL_ONLY.value and not narration:
                _issue(
                    issues,
                    f"{spath}.narration_text",
                    "REQUIRED_FIELD",
                    "A narration segment requires non-empty narration_text.",
                )

            clips_raw = seg_raw.get("source_clips")
            if not isinstance(clips_raw, list) or not clips_raw:
                _issue(issues, f"{spath}.source_clips", "SCHEMA_ERROR", "source_clips must be a non-empty array.")
                clips_raw = []
            clips: list[SourceClip] = []
            for ci, clip_raw in enumerate(clips_raw):
                cpath = f"{spath}.source_clips[{ci}]"
                if not isinstance(clip_raw, dict):
                    _issue(issues, cpath, "SCHEMA_ERROR", "Source clip must be an object.")
                    continue
                episode_id = str(clip_raw.get("episode_id") or "").strip()
                source_file = str(
                    clip_raw.get("source_file", clip_raw.get("source_video", "")) or ""
                ).strip()
                ep = episode_map.get(episode_id)
                if ep is None:
                    _issue(
                        issues,
                        f"{cpath}.episode_id",
                        "SOURCE_RESOLUTION_ERROR",
                        "episode_id does not exist in this project.",
                        requested=episode_id,
                    )
                    continue
                if scope == "SINGLE_EPISODE" and episode_id != episodes[0].episode_id:
                    _issue(issues, f"{cpath}.episode_id", "SOURCE_RESOLUTION_ERROR", "Single-video project references another episode.")
                candidates = source_by_name.get(Path(source_file).name.casefold(), []) if source_file else []
                source_matches = source_file in {ep.source_video, Path(ep.source_video).name} or ep in candidates
                if not source_matches:
                    _issue(
                        issues,
                        f"{cpath}.source_file",
                        "SOURCE_RESOLUTION_ERROR",
                        "source_file does not resolve to the requested episode.",
                        episode_id=episode_id,
                        requested=source_file,
                        expected=ep.source_video,
                    )
                if require_source_files and not Path(ep.source_video).is_file():
                    _issue(
                        issues,
                        f"{cpath}.source_file",
                        "SOURCE_RESOLUTION_ERROR",
                        "Resolved source file does not exist on disk.",
                        resolved=ep.source_video,
                    )

                try:
                    start_ms = int(clip_raw["start_ms"]) if "start_ms" in clip_raw else round(float(clip_raw["start"]) * 1000)
                    end_ms = int(clip_raw["end_ms"]) if "end_ms" in clip_raw else round(float(clip_raw["end"]) * 1000)
                except (KeyError, TypeError, ValueError):
                    _issue(issues, cpath, "TIMESTAMP_ERROR", "start_ms and end_ms are required integers.")
                    continue
                duration_ms = int(round(max(0.0, ep.duration_seconds) * 1000.0))
                if start_ms < 0 or end_ms <= start_ms:
                    _issue(
                        issues,
                        cpath,
                        "TIMESTAMP_ERROR",
                        "Clip range must satisfy 0 <= start_ms < end_ms.",
                        requested=[start_ms, end_ms],
                    )
                elif duration_ms > 0 and end_ms > duration_ms:
                    _issue(
                        issues,
                        cpath,
                        "TIMESTAMP_OUT_OF_RANGE",
                        "Clip end exceeds source duration.",
                        source_file=ep.source_video,
                        requested=[start_ms, end_ms],
                        source_duration_ms=duration_ms,
                    )
                clips.append(
                    SourceClip(
                        episode_id=episode_id,
                        source_video=ep.source_video,
                        start=start_ms / 1000.0,
                        end=end_ms / 1000.0,
                    )
                )

            try:
                visual_speed = float(seg_raw.get("recommended_visual_speed", 1.0) or 1.0)
            except (TypeError, ValueError):
                visual_speed = 1.0
                _issue(issues, f"{spath}.recommended_visual_speed", "SCHEMA_ERROR", "recommended_visual_speed must be numeric.")
            if abs(visual_speed - 1.0) > 0.0001:
                _issue(
                    issues,
                    f"{spath}.recommended_visual_speed",
                    "UNSUPPORTED_VISUAL_SPEED",
                    "The current renderer supports deterministic 1.0x source timing only.",
                    requested=visual_speed,
                )
            subtitle_policy = str(seg_raw.get("subtitle_policy") or "").strip().lower()
            if subtitle_policy != "both":
                _issue(
                    issues,
                    f"{spath}.subtitle_policy",
                    "UNSUPPORTED_SUBTITLE_POLICY",
                    "The current publication contract requires both narration and original-dialogue SRT files.",
                    requested=subtitle_policy,
                    supported=["both"],
                )
            segment = Segment(
                segment_id=segment_id,
                source_clips=clips,
                original_dialogue=dialogue,
                narration=narration,
                audio_policy=audio_policy,
                segment_type=segment_type,
                purpose=str(seg_raw.get("purpose") or ""),
                subtitle_policy=subtitle_policy,
                recommended_visual_speed=visual_speed,
            )
            segments.append(segment)

        scope_value = str(out_raw.get("candidate_scope") or (
            CandidateScope.SINGLE_EPISODE.value if scope == "SINGLE_EPISODE" else CandidateScope.SEASON_ARC.value
        ))
        outputs.append(
            CommentaryOutput(
                output_id=output_id,
                title=title,
                candidate_scope=scope_value,
                segments=segments,
                status=OutputStatus.WAITING.value,
                file_name=file_name,
                output_type=output_type,
                language=language,
            )
        )

    if issues:
        raise FinalJsonValidationError(issues)

    for out in outputs:
        out.sanitized_title = Path(out.file_name).stem

    manifest = AnalysisManifest(
        project_id=project_id,
        analysis_scope=scope,
        source_episodes=episodes,
        outputs=outputs,
        recap_language=settings.recap_language,
        recap_mode=settings.recap_mode,
        content_type=settings.content_type,
        source_rights_status=settings.source_rights_status,
        zero_output_reason="VALID_EMPTY_OUTPUT" if not outputs else None,
        zero_output_status="VALID_EMPTY_OUTPUT" if not outputs else None,
        verification={
            "kind": "TECHNICAL_JSON_VALIDATION",
            "schema_version": FINAL_JSON_SCHEMA_VERSION,
            "valid": True,
            "output_count": len(outputs),
        },
    )
    manifest.validate()
    return manifest


def build_repair_prompt(raw: dict[str, Any], issues: list[FinalJsonIssue]) -> str:
    """Return a repair request containing the exact validator errors and prior JSON."""
    return (
        "Repair the Final JSON below. Preserve all editorial choices unless a listed technical error requires a change. "
        "Do not remove an output merely to avoid fixing it. Return the complete repaired JSON object only.\n\n"
        f"VALIDATION_ERRORS\n{json.dumps([i.to_dict() for i in issues], ensure_ascii=False, indent=2)}\n\n"
        f"INVALID_FINAL_JSON\n{json.dumps(raw, ensure_ascii=False, indent=2)}"
    )


class CanonicalProjectFinalizer:
    """One configured Finalizer role producing one canonical project Final JSON."""

    def __init__(self, settings: AppSettings, client: OpenAICompatibleClient) -> None:
        self.settings = settings
        self.client = client

    def finalize(
        self,
        *,
        project_id: str,
        episodes: list[SourceEpisode],
        evidence_map: dict[str, EpisodeEvidence],
        scope: str,
        cancel_event: threading.Event | None = None,
        on_phase: PhaseCallback | None = None,
        log: Callable[[str], None] | None = None,
        require_source_files: bool = False,
    ) -> AnalysisManifest:
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("Finalizer cancelled before request.")
        if on_phase:
            on_phase(AnalysisPhase.FINALIZER, project_id, {"status": "requesting_final_json"})
        raw = self.client.chat_json(
            model=self.settings.finalizer_model,
            thinking=self.settings.finalizer_thinking,
            system=FINAL_JSON_SYSTEM_PROMPT,
            user_text=build_finalizer_project_prompt(project_id, episodes, evidence_map, self.settings, scope),
            max_tokens=32_000,
            cancel_event=cancel_event,
            phase="finalizer",
            log=log,
        )

        max_repairs = max(0, int(getattr(self.settings, "final_json_repair_attempts", 2)))
        for repair_index in range(max_repairs + 1):
            if cancel_event and cancel_event.is_set():
                raise AnalysisCancelledError("Finalizer validation/repair cancelled.")
            try:
                if on_phase:
                    on_phase(AnalysisPhase.JSON_VALIDATION, project_id, {"attempt": repair_index + 1})
                return validate_final_json(
                    raw,
                    project_id=project_id,
                    episodes=episodes,
                    scope=scope,
                    settings=self.settings,
                    require_source_files=require_source_files,
                )
            except FinalJsonValidationError as exc:
                if repair_index >= max_repairs:
                    raise
                if log:
                    log(
                        f"Final JSON failed technical validation; requesting repair "
                        f"{repair_index + 1}/{max_repairs}: {exc}"
                    )
                if on_phase:
                    on_phase(
                        AnalysisPhase.JSON_REPAIR,
                        project_id,
                        {"attempt": repair_index + 1, "errors": [i.to_dict() for i in exc.issues]},
                    )
                raw = self.client.chat_json(
                    model=self.settings.finalizer_model,
                    thinking=self.settings.finalizer_thinking,
                    system=(
                        FINAL_JSON_SYSTEM_PROMPT
                        + "\nYou are repairing technical contract errors reported by ToolRecap's deterministic validator."
                    ),
                    user_text=build_repair_prompt(raw, exc.issues),
                    max_tokens=32_000,
                    cancel_event=cancel_event,
                    phase="finalizer",
                    log=log,
                )
        raise AnalysisError("Final JSON repair policy exhausted.")


__all__ = [
    "CanonicalProjectFinalizer",
    "FINAL_JSON_SCHEMA_VERSION",
    "FINAL_JSON_SYSTEM_PROMPT",
    "FinalJsonIssue",
    "FinalJsonValidationError",
    "build_finalizer_project_prompt",
    "build_repair_prompt",
    "final_json_contract",
    "validate_final_json",
]
