"""Candidate mining and script finalization for single episode and season commentary outputs."""
from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Callable

from ..api_client import APIError, OpenAICompatibleClient
from ..domain.enums import AudioPolicy, CandidateScope, OutputStatus
from ..domain.models import CommentaryOutput, EpisodeEvidence, Segment, SourceClip, SourceEpisode, ValidationError
from ..domain.title import resolve_unique_titles, sanitize_title
from ..settings import AppSettings
from .connection import SeasonConnectionResult
from .errors import AnalysisCancelledError, AnalysisError
from .phases import AnalysisPhase, PhaseCallback
from .prompts import FINALIZER_SYSTEM_PROMPT


class CandidateFinalizer:
    """Mines candidate proposals and finalizes CommentaryOutput plans."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.client = client

    def finalize_single(
        self,
        episode: SourceEpisode,
        evidence: EpisodeEvidence,
        *,
        cancel_event: threading.Event | None = None,
        on_phase: PhaseCallback | None = None,
        log: Callable[[str], None] | None = None,
        legacy_wrapper: bool = False,
    ) -> list[CommentaryOutput]:
        """Finalize commentary outputs for a single episode."""
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("Finalizer đã bị hủy.")

        if on_phase:
            on_phase(AnalysisPhase.SEASON_MINING, episode.episode_id, {"status": "finalizing_single"})

        is_gateway_enabled = (
            self.settings.gateway_enabled
            and bool(self.settings.api_endpoint.strip())
            and self.settings.api_endpoint.strip().lower() != "offline"
            and self.client is not None
        )

        if is_gateway_enabled and self.client is not None:
            user_text = (
                f"Episode ID: {episode.episode_id}\n"
                f"Source Video: {episode.source_video}\n"
                f"Duration: {episode.duration_seconds:.1f}s\n"
                f"Recap instructions:\n{self.settings.recap_prompt or 'Standard video recap'}\n\n"
                f"Recap Language: {self.settings.recap_language}\n"
                f"Recap Mode: {self.settings.recap_mode}\n"
                f"Content Type: {self.settings.content_type}\n"
                f"Rights Status: {self.settings.source_rights_status}\n\n"
                f"Episode Evidence:\n{json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2)}\n"
            )

            try:
                raw_result = self.client.chat_json(
                    model=self.settings.finalizer_model,
                    thinking=self.settings.finalizer_thinking,
                    system=FINALIZER_SYSTEM_PROMPT,
                    user_text=user_text,
                    cancel_event=cancel_event,
                )
            except APIError as exc:
                raise AnalysisError(
                    f"Không thể kết nối đến AI Gateway ({self.settings.api_endpoint}): Finalizer lỗi: {exc}"
                ) from exc

            return self.parse_outputs(raw_result, [episode])
        else:
            if legacy_wrapper:
                return self._finalize_single_offline(episode, evidence)
            return []

    def finalize_season(
        self,
        episodes: list[SourceEpisode],
        evidence_map: dict[str, EpisodeEvidence],
        connection_result: SeasonConnectionResult,
        *,
        cancel_event: threading.Event | None = None,
        on_phase: PhaseCallback | None = None,
        log: Callable[[str], None] | None = None,
        legacy_wrapper: bool = False,
    ) -> list[CommentaryOutput]:
        """Finalize commentary outputs for a full season with cross-episode candidates."""
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("Finalizer đã bị hủy.")

        if on_phase:
            on_phase(AnalysisPhase.SEASON_MINING, "season", {"status": "finalizing_season"})

        is_gateway_enabled = (
            self.settings.gateway_enabled
            and bool(self.settings.api_endpoint.strip())
            and self.settings.api_endpoint.strip().lower() != "offline"
            and self.client is not None
        )

        if is_gateway_enabled and self.client is not None:
            episodes_summary = [
                {
                    "episode_id": ep.episode_id,
                    "source_video": ep.source_video,
                    "duration_seconds": ep.duration_seconds,
                    "title": ep.title,
                }
                for ep in episodes
            ]

            coverage_notice = ""
            if not connection_result.is_complete:
                coverage_notice = (
                    f"\nCRITICAL COVERAGE NOTICE:\n"
                    f"Missing episodes: {connection_result.missing_episodes}.\n"
                    f"This is a PARTIAL season analysis. All outputs and clips MUST ONLY reference available episodes.\n"
                )

            user_text = (
                f"Finalize commentary outputs for the season.\n"
                f"{coverage_notice}\n"
                f"Recap instructions:\n{self.settings.recap_prompt or 'Standard video recap'}\n"
                f"Recap Language: {self.settings.recap_language}\n"
                f"Recap Mode: {self.settings.recap_mode}\n"
                f"Content Type: {self.settings.content_type}\n"
                f"Rights Status: {self.settings.source_rights_status}\n\n"
                f"Source Episodes:\n{json.dumps(episodes_summary, ensure_ascii=False, indent=2)}\n\n"
                f"Season Connection Analysis & Candidate Proposals:\n"
                f"{json.dumps(connection_result.to_dict(), ensure_ascii=False, indent=2)}\n"
            )

            try:
                raw_result = self.client.chat_json(
                    model=self.settings.finalizer_model,
                    thinking=self.settings.finalizer_thinking,
                    system=FINALIZER_SYSTEM_PROMPT,
                    user_text=user_text,
                    cancel_event=cancel_event,
                )
            except APIError as exc:
                raise AnalysisError(
                    f"Không thể kết nối đến AI Gateway ({self.settings.api_endpoint}): Finalizer lỗi: {exc}"
                ) from exc

            return self.parse_outputs(raw_result, episodes)
        else:
            if legacy_wrapper:
                return self._finalize_season_offline(episodes, evidence_map, connection_result)
            return []

    def parse_outputs(
        self,
        raw: dict[str, Any],
        episodes: list[SourceEpisode],
    ) -> list[CommentaryOutput]:
        """Parse AI response into CommentaryOutput objects with strict validation.

        - Parses ALL items in outputs[] (no artificial slicing or quota).
        - Preserves titles and deterministically sanitizes them.
        - Validates source clips against episode sources and durations.
        - Rejects hallucinated episode IDs or source file mismatches.
        - Supports cross-episode clipping (e.g. E01, E03, E05).
        - Empty outputs [] is valid.
        """
        episode_map: dict[str, SourceEpisode] = {ep.episode_id: ep for ep in episodes}

        # Check for outputs list
        raw_outputs = raw.get("outputs")
        if raw_outputs is None:
            # Fallback if AI returned single output format with 'segments'
            if "segments" in raw:
                raw_outputs = [raw]
            else:
                raw_outputs = []

        if not isinstance(raw_outputs, list):
            raw_outputs = []

        # Empty outputs is valid
        if not raw_outputs:
            return []

        parsed_outputs: list[CommentaryOutput] = []

        for i, out_dict in enumerate(raw_outputs, start=1):
            if not isinstance(out_dict, dict):
                continue

            output_id = str(out_dict.get("output_id", f"output_{i:02d}"))
            title = str(out_dict.get("title", f"Recap Output {i}"))
            scope_val = str(out_dict.get("candidate_scope", CandidateScope.SINGLE_EPISODE.value))
            if scope_val not in {s.value for s in CandidateScope}:
                scope_val = CandidateScope.SINGLE_EPISODE.value

            segments_raw = out_dict.get("segments", [])
            if not isinstance(segments_raw, list):
                segments_raw = []

            segments: list[Segment] = []
            for j, s_dict in enumerate(segments_raw, start=1):
                if not isinstance(s_dict, dict):
                    continue

                segment_id = str(s_dict.get("segment_id", f"seg_{j:02d}"))
                narration = str(s_dict.get("narration", "") or s_dict.get("narration_text", "")).strip()
                audio_policy = str(s_dict.get("audio_policy", AudioPolicy.DUCK.value)).strip().lower()
                if audio_policy not in {p.value for p in AudioPolicy}:
                    audio_policy = AudioPolicy.DUCK.value

                orig_dialogue = str(
                    s_dict.get("original_dialogue", "") or s_dict.get("original_dialogue_text", "")
                ).strip()

                # Parse source clips
                clips_raw = s_dict.get("source_clips", [])
                if not isinstance(clips_raw, list):
                    clips_raw = []

                # Fallback if segment defined start/end directly
                if not clips_raw:
                    start_s = float(s_dict.get("start", s_dict.get("start_ms", 0) / 1000.0))
                    end_s = float(s_dict.get("end", s_dict.get("end_ms", start_s + 5.0) / 1000.0))
                    ep_id = str(s_dict.get("episode_id", episodes[0].episode_id))
                    clips_raw = [
                        {
                            "episode_id": ep_id,
                            "source_video": episode_map.get(ep_id, episodes[0]).source_video,
                            "start": start_s,
                            "end": end_s,
                        }
                    ]

                clips: list[SourceClip] = []
                for c_dict in clips_raw:
                    if not isinstance(c_dict, dict):
                        continue

                    ep_id = str(c_dict.get("episode_id", "")).strip()
                    if ep_id not in episode_map:
                        raise ValidationError(
                            f"Invalid episode_id '{ep_id}' in source clip: not found in source episodes."
                        )

                    ep = episode_map[ep_id]
                    src_vid = str(c_dict.get("source_video", "") or ep.source_video).strip()

                    # Validate source file matches
                    if src_vid and ep.source_video:
                        if src_vid != ep.source_video and Path(src_vid).name != Path(ep.source_video).name:
                            raise ValidationError(
                                f"Source clip file '{src_vid}' does not match episode '{ep.episode_id}' file '{ep.source_video}'."
                            )

                    # Timestamps (seconds)
                    start_sec = float(c_dict.get("start", c_dict.get("start_ms", 0) / 1000.0))
                    end_sec = float(c_dict.get("end", c_dict.get("end_ms", start_sec + 5.0) / 1000.0))

                    if start_sec < 0.0:
                        raise ValidationError(
                            f"Invalid clip start timestamp {start_sec}: must be >= 0."
                        )
                    if end_sec <= start_sec:
                        raise ValidationError(
                            f"Invalid clip end timestamp {end_sec}: must be greater than start ({start_sec})."
                        )
                    if ep.duration_seconds > 0.0 and end_sec > ep.duration_seconds + 0.5:
                        raise ValidationError(
                            f"Clip end {end_sec}s exceeds episode '{ep.episode_id}' duration {ep.duration_seconds}s."
                        )

                    clips.append(
                        SourceClip(
                            episode_id=ep_id,
                            source_video=src_vid or ep.source_video,
                            start=start_sec,
                            end=end_sec,
                        )
                    )

                if audio_policy in (AudioPolicy.MUTE.value, "mute"):
                    # Normalize "mute" policy from AI to mix/duck behavior so original audio is preserved
                    audio_policy = AudioPolicy.DUCK.value

                if audio_policy != AudioPolicy.ORIGINAL_ONLY.value and not narration:
                    raise ValidationError(
                        f"Phân đoạn '{segment_id}' là phân đoạn thuyết minh nhưng thiếu nội dung narration."
                    )

                segments.append(
                    Segment(
                        segment_id=segment_id,
                        source_clips=clips,
                        original_dialogue=orig_dialogue,
                        narration=narration,
                        audio_policy=audio_policy,
                    )
                )

            parsed_outputs.append(
                CommentaryOutput(
                    output_id=output_id,
                    title=title,
                    sanitized_title="",
                    candidate_scope=scope_val,
                    segments=segments,
                    status=OutputStatus.WAITING.value,
                )
            )

        # Deterministically sanitize and resolve unique titles
        raw_titles = [out.title or out.output_id for out in parsed_outputs]
        unique_titles = resolve_unique_titles(raw_titles, max_length=120)
        for out, san_title in zip(parsed_outputs, unique_titles):
            out.sanitized_title = san_title

        return parsed_outputs

    def _finalize_single_offline(
        self,
        episode: SourceEpisode,
        evidence: EpisodeEvidence,
    ) -> list[CommentaryOutput]:
        """Deterministic offline generation for single episode."""
        dur = max(3.0, episode.duration_seconds)
        clips = [
            SourceClip(
                episode_id=episode.episode_id,
                source_video=episode.source_video,
                start=0.0,
                end=dur,
            )
        ]
        segment = Segment(
            segment_id="seg_01",
            source_clips=clips,
            original_dialogue="",
            narration=f"A complete recap of {episode.title or episode.episode_id}.",
            audio_policy=AudioPolicy.MUTE.value,
        )
        title = episode.title or f"Recap {episode.episode_id}"
        return [
            CommentaryOutput(
                output_id="out_01",
                title=title,
                sanitized_title=sanitize_title(title),
                candidate_scope=CandidateScope.SINGLE_EPISODE.value,
                segments=[segment],
                status=OutputStatus.WAITING.value,
            )
        ]

    def _finalize_season_offline(
        self,
        episodes: list[SourceEpisode],
        evidence_map: dict[str, EpisodeEvidence],
        connection_result: SeasonConnectionResult,
    ) -> list[CommentaryOutput]:
        """Deterministic offline generation for season with multiple candidate outputs."""
        outputs: list[CommentaryOutput] = []

        # 1. Full season arc output
        season_segments: list[Segment] = []
        for i, ep in enumerate(episodes, start=1):
            if ep.episode_id in evidence_map:
                dur = max(1.0, ep.duration_seconds)
                clip_dur = min(30.0, dur)
                season_segments.append(
                    Segment(
                        segment_id=f"seg_{i:02d}",
                        source_clips=[
                            SourceClip(
                                episode_id=ep.episode_id,
                                source_video=ep.source_video,
                                start=0.0,
                                end=clip_dur,
                            )
                        ],
                        narration=f"Key events unfolding in {ep.title or ep.episode_id}.",
                        audio_policy=AudioPolicy.MUTE.value,
                    )
                )

        if season_segments:
            outputs.append(
                CommentaryOutput(
                    output_id="out_season_arc",
                    title="Season Comprehensive Arc",
                    sanitized_title=sanitize_title("Season Comprehensive Arc"),
                    candidate_scope=CandidateScope.SEASON_ARC.value,
                    segments=season_segments,
                    status=OutputStatus.WAITING.value,
                )
            )

        # 2. Supporting character arc output
        if len(episodes) >= 2:
            supporting_segments: list[Segment] = []
            for j, ep in enumerate(episodes[:3], start=1):
                dur = max(1.0, ep.duration_seconds)
                supporting_segments.append(
                    Segment(
                        segment_id=f"seg_supp_{j:02d}",
                        source_clips=[
                            SourceClip(
                                episode_id=ep.episode_id,
                                source_video=ep.source_video,
                                start=0.0,
                                end=min(15.0, dur),
                            )
                        ],
                        narration=f"Supporting character developments in {ep.episode_id}.",
                        audio_policy=AudioPolicy.MUTE.value,
                    )
                )
            outputs.append(
                CommentaryOutput(
                    output_id="out_supporting_arc",
                    title="Supporting Character Arc",
                    sanitized_title=sanitize_title("Supporting Character Arc"),
                    candidate_scope=CandidateScope.CROSS_EPISODE.value,
                    segments=supporting_segments,
                    status=OutputStatus.WAITING.value,
                )
            )

        return outputs
