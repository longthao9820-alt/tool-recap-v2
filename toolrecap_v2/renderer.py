"""Canonical publication renderer for multi-source commentary and video recaps.

Orchestrates sequential output processing, stream extraction, video concatenation,
real audio mixing with ducking, SRT remap and narration alignment, subtitle burning,
and strict publication validation.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

from .audio_mix import (
    AudioMixSettings,
    plan_audio_mix,
    validate_mixed_audio,
)
from .domain.enums import AudioPolicy, OutputStatus
from .domain.models import AnalysisManifest, CommentaryOutput, SourceClip, ValidationError
from .domain.title import resolve_unique_titles, sanitize_title
from .media import (
    CutClipError,
    MediaError,
    MediaProbeResult,
    RenderCancelled,
    RenderStageError,
    concat_media_clips,
    cut_clip,
    find_binary,
    probe_duration,
    probe_typed_media,
    render_final_video,
    run_command,
    write_srt_file,
)
from .output_validation import validate_publication_folder
from .paths import default_data_directory
from .settings import AppSettings
from .subtitles.models import SubtitleCue
from .subtitles.remap import cues_to_srt_rows, remap_subtitles
from .voice.catalog import DEFAULT_VOICE_ID
from .voice.manager import get_voice_manager, validate_wav_audio
from .voice.runtime import VOICE_MODEL_REVISION, runtime_fingerprint


def compute_output_render_signature(
    out: CommentaryOutput,
    settings: AppSettings | dict[str, Any] | None = None,
    voice_id: str | None = None,
    analysis_signature: str = "",
) -> str:
    """Hash output editorial content and render-only settings."""
    segments_payload: list[dict[str, Any]] = []
    for s in getattr(out, "segments", []):
        clips_payload: list[dict[str, Any]] = []
        for c in getattr(s, "source_clips", []):
            clips_payload.append({
                "episode_id": str(getattr(c, "episode_id", "")),
                "source_video": str(getattr(c, "source_video", "")),
                "start": float(getattr(c, "start", 0.0)),
                "end": float(getattr(c, "end", 0.0)),
            })
        segments_payload.append({
            "segment_id": str(getattr(s, "segment_id", "")),
            "audio_policy": str(getattr(s, "audio_policy", "")),
            "narration": str(getattr(s, "narration", "")).strip(),
            "source_clips": clips_payload,
        })

    def _get_setting(key: str, default: Any = None) -> Any:
        if settings is None:
            return default
        if isinstance(settings, dict):
            return settings.get(key, default)
        return getattr(settings, key, default)

    eff_voice_id = voice_id or _get_setting("voice_id", "") or ""

    payload = {
        "version": "out-render-v1",
        "analysis_signature": analysis_signature,
        "output_id": str(getattr(out, "output_id", "")),
        "title": str(getattr(out, "title", "")),
        "sanitized_title": str(getattr(out, "sanitized_title", "")),
        "segments": segments_payload,
        "voice_id": eff_voice_id,
        "voice_style": _get_setting("voice_style", "film_recap"),
        "voice_runtime_fingerprint": runtime_fingerprint(),
        "voice_model_revision": VOICE_MODEL_REVISION,
        "quality": _get_setting("quality", "1080p"),
        "use_gpu": bool(_get_setting("use_gpu", True)),
        "generate_srt": bool(_get_setting("generate_srt", True)),
        "burn_subtitles": bool(_get_setting("burn_subtitles", False)),
        "original_audio_gain_db": float(_get_setting("original_audio_gain_db", 0.0)),
        "commentary_gain_db": float(_get_setting("commentary_gain_db", 0.0)),
        "auto_duck": bool(_get_setting("auto_duck", True)),
        "ducking_amount_db": float(_get_setting("ducking_amount_db", -14.0)),
        "target_loudness_lufs": float(_get_setting("target_loudness_lufs", -16.0)),
        "true_peak_dbtp": float(_get_setting("true_peak_dbtp", -1.5)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _dispatch_callback(
    callbacks: Any,
    out_idx: int,
    total_out: int,
    out: CommentaryOutput,
    progress: int,
    message: str,
) -> None:
    if not callbacks:
        return
    if callable(callbacks):
        try:
            callbacks(out_idx, total_out, out, progress, message)
        except TypeError:
            try:
                callbacks(progress, message)
            except Exception:
                pass
    elif isinstance(callbacks, dict):
        cb = callbacks.get("on_progress")
        if callable(cb):
            try:
                cb(out_idx, total_out, out, progress, message)
            except TypeError:
                cb(progress, message)


def _wrap_stage_error(
    exc: Exception,
    out: CommentaryOutput,
    stage: str,
    default_category: str,
) -> RenderStageError:
    if isinstance(exc, RenderStageError):
        if not exc.output_id:
            exc.output_id = out.output_id
        if not exc.output_title:
            exc.output_title = out.title
        if not exc.stage or exc.stage == "render":
            exc.stage = stage
        if not exc.category or exc.category == "RENDER_STAGE_ERROR":
            exc.category = default_category
        return exc
    category = getattr(exc, "category", "") or default_category
    msg = str(exc)
    prefix = f"[{category}] "
    if msg.startswith(prefix):
        msg = msg[len(prefix):]
    return RenderStageError(
        message=msg,
        output_id=out.output_id,
        output_title=out.title,
        stage=stage,
        category=category,
        cause=exc,
    )


class PublicationRenderer:
    """Renders all CommentaryOutputs from an AnalysisManifest sequentially to final publication format."""

    def __init__(self, voice_manager: Any = None) -> None:
        self.voice_manager = voice_manager

    def render_manifest(
        self,
        manifest: AnalysisManifest,
        settings: AppSettings,
        voice_id: str | None = None,
        transcript_cues_by_episode: dict[str, Sequence[SubtitleCue]] | None = None,
        output_root: Path | str | None = None,
        callbacks: Any = None,
        cancel: threading.Event | None = None,
        *,
        cancel_event: threading.Event | None = None,
        voice_style: str | None = None,
        allow_mock_synth: bool = False,
        resume_completed: bool = False,
        analysis_signature: str = "",
        target_signatures: dict[str, str] | None = None,
        on_output_complete: Callable[[CommentaryOutput, int, int], None] | None = None,
    ) -> list[CommentaryOutput]:
        """Loop through all outputs sequentially and produce clean publication folders.

        Each output produces:
        - {output_root}/{safe_title}/{safe_title}.mp4
        - {output_root}/{safe_title}/{safe_title}.original.srt
        - {output_root}/{safe_title}/{safe_title}.narration.srt
        """
        cancel_evt = cancel or cancel_event
        if cancel_evt and cancel_evt.is_set():
            raise RenderCancelled("Quá trình kết xuất đã bị hủy.")

        out_root = Path(output_root or settings.output_dir or (default_data_directory() / "outputs")).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        eff_voice_id = voice_id or settings.voice_id or DEFAULT_VOICE_ID
        eff_voice_style = voice_style or getattr(settings, "voice_style", "film_recap")
        voice_mgr = self.voice_manager or get_voice_manager()

        rendered_outputs: list[CommentaryOutput] = []
        total_outputs = len(manifest.outputs)

        # Ensure unique safe titles across all outputs to prevent publication folder collisions
        raw_titles = [out.sanitized_title or out.title or out.output_id for out in manifest.outputs]
        unique_safe_titles = resolve_unique_titles(raw_titles, max_length=120)

        for out_idx, (out, safe_title) in enumerate(zip(manifest.outputs, unique_safe_titles), start=1):
            if cancel_evt and cancel_evt.is_set():
                raise RenderCancelled("Quá trình kết xuất đã bị hủy.")

            out.sanitized_title = safe_title
            pub_dir = out_root / safe_title
            pub_dir.mkdir(parents=True, exist_ok=True)

            target_sig = (
                target_signatures.get(out.output_id)
                if target_signatures and out.output_id in target_signatures
                else compute_output_render_signature(
                    out=out,
                    settings=settings,
                    voice_id=eff_voice_id,
                    analysis_signature=analysis_signature,
                )
            )

            # Prior signature match check
            prior_sig = getattr(out, "render_signature", "")
            sig_matches = (prior_sig == target_sig) if prior_sig else not bool(analysis_signature or target_signatures)

            # Resume is strictly render-layer state. A previously completed output
            # is reused only when the caller confirmed the render dependency
            # signature and the exact publication folder still validates.
            if resume_completed and out.status == OutputStatus.COMPLETED.value and sig_matches:
                try:
                    has_commentary = any(
                        seg.audio_policy != AudioPolicy.ORIGINAL_ONLY.value
                        and bool(seg.narration.strip())
                        for seg in out.segments
                    )
                    validated = validate_publication_folder(
                        pub_dir=pub_dir,
                        safe_title=safe_title,
                        check_original_srt=True,
                        check_narration_srt=has_commentary,
                        require_audio=True,
                    )
                    out.publication_video_path = validated["video_path"]
                    out.publication_original_srt_path = validated["original_srt_path"]
                    out.publication_narration_srt_path = validated["narration_srt_path"]
                    out.progress = 100
                    out.render_signature = target_sig
                    rendered_outputs.append(out)
                    if on_output_complete:
                        try:
                            on_output_complete(out, out_idx, total_outputs)
                        except Exception:
                            pass
                    _dispatch_callback(
                        callbacks,
                        out_idx,
                        total_outputs,
                        out,
                        100,
                        "Đã xác minh output hoàn tất; tiếp tục từ điểm render kế tiếp.",
                    )
                    continue
                except Exception:
                    out.status = OutputStatus.WAITING.value
                    out.progress = 0
                    out.error = None
                    out.render_signature = ""
            else:
                if out.status == OutputStatus.COMPLETED.value and not sig_matches:
                    out.status = OutputStatus.WAITING.value
                    out.progress = 0
                    out.error = None
                    out.render_signature = ""

            # If not reusing completed output, clean existing publication directory to prevent stale extras
            if pub_dir.exists():
                shutil.rmtree(pub_dir, ignore_errors=True)
            pub_dir.mkdir(parents=True, exist_ok=True)

            # Temp folder outside publication directory
            temp_dir = out_root / ".temp_render" / f"{safe_title}_{uuid.uuid4().hex[:8]}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            out.status = OutputStatus.RUNNING.value
            out.progress = 5
            _dispatch_callback(callbacks, out_idx, total_outputs, out, 5, f"Bắt đầu kết xuất: {safe_title}")

            try:
                self._render_single_output(
                    out=out,
                    safe_title=safe_title,
                    pub_dir=pub_dir,
                    temp_dir=temp_dir,
                    settings=settings,
                    voice_id=eff_voice_id,
                    voice_style=eff_voice_style,
                    voice_mgr=voice_mgr,
                    transcript_cues_by_episode=transcript_cues_by_episode or {},
                    callbacks=callbacks,
                    out_idx=out_idx,
                    total_outputs=total_outputs,
                    cancel_evt=cancel_evt,
                    allow_mock_synth=allow_mock_synth,
                )
                out.render_signature = target_sig
                rendered_outputs.append(out)
                if on_output_complete:
                    try:
                        on_output_complete(out, out_idx, total_outputs)
                    except Exception:
                        pass
            except RenderCancelled:
                out.status = OutputStatus.CANCELLED.value
                out.error = "Đã dừng bởi người dùng"
                (pub_dir / f"{safe_title}.mp4").unlink(missing_ok=True)
                raise
            except Exception as exc:
                if cancel_evt and cancel_evt.is_set():
                    out.status = OutputStatus.CANCELLED.value
                    out.error = "Đã dừng bởi người dùng"
                    (pub_dir / f"{safe_title}.mp4").unlink(missing_ok=True)
                    raise RenderCancelled("Quá trình kết xuất đã bị hủy.") from exc

                (pub_dir / f"{safe_title}.mp4").unlink(missing_ok=True)

                if isinstance(exc, RenderStageError):
                    stage_err = exc
                    if not stage_err.output_id:
                        stage_err.output_id = out.output_id
                    if not stage_err.output_title:
                        stage_err.output_title = out.title
                else:
                    category = getattr(exc, "category", "") or "RENDER_STAGE_ERROR"
                    stage = getattr(exc, "stage", "render")
                    msg = str(exc)
                    if category and msg.startswith(f"[{category}] "):
                        msg_clean = msg[len(f"[{category}] "):]
                    else:
                        msg_clean = msg
                    stage_err = RenderStageError(
                        message=msg_clean,
                        output_id=out.output_id,
                        output_title=out.title,
                        stage=stage,
                        category=category,
                        cause=exc,
                    )

                out.status = OutputStatus.ERROR.value
                out.error = str(stage_err)

                # Subsequent outputs WAITING
                for rem in manifest.outputs[out_idx:]:
                    rem.status = OutputStatus.WAITING.value
                    rem.progress = 0

                raise stage_err
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return rendered_outputs

    def _render_single_output(
        self,
        out: CommentaryOutput,
        safe_title: str,
        pub_dir: Path,
        temp_dir: Path,
        settings: AppSettings,
        voice_id: str,
        voice_style: str,
        voice_mgr: Any,
        transcript_cues_by_episode: dict[str, Sequence[SubtitleCue]],
        callbacks: Any,
        out_idx: int,
        total_outputs: int,
        cancel_evt: threading.Event | None,
        allow_mock_synth: bool,
    ) -> None:
        """Render one CommentaryOutput to its publication folder."""
        # 1. Collect all clips across segments
        all_clips: list[SourceClip] = []
        for seg in out.segments:
            all_clips.extend(seg.source_clips)

        if not all_clips:
            raise RenderStageError(
                f"Output '{out.title}' không chứa phân đoạn video nào (source_clips).",
                output_id=out.output_id,
                output_title=out.title,
                stage="source_resolution",
                category="CUT_CLIP_ERROR",
            )
        is_all_orig_only = all(
            seg.audio_policy == AudioPolicy.ORIGINAL_ONLY.value for seg in out.segments
        )
        if not is_all_orig_only and not any(seg.narration and seg.narration.strip() for seg in out.segments):
            raise RenderStageError(
                f"Output '{out.title}' không có commentary narration; không thể tạo narration SRT hợp lệ.",
                output_id=out.output_id,
                output_title=out.title,
                stage="source_resolution",
                category="VOICE_ERROR",
            )

        # 2. Cut clips using selected audio stream index
        cut_clips: list[Path] = []
        for c_idx, clip in enumerate(all_clips, start=1):
            if cancel_evt and cancel_evt.is_set():
                raise RenderCancelled()

            progress_pct = 10 + int((c_idx / len(all_clips)) * 25)
            _dispatch_callback(
                callbacks,
                out_idx,
                total_outputs,
                out,
                progress_pct,
                f"Cắt phân đoạn {c_idx}/{len(all_clips)}...",
            )

            clip_path = Path(clip.source_video).resolve()
            if not clip_path.is_file():
                raise MediaError(f"Không tìm thấy video nguồn cho clip: {clip_path}")

            # Determine selected audio stream index
            probe = probe_typed_media(clip_path)
            audio_idx: int | None = None
            if probe.has_audio:
                if probe.selected_audio and probe.selected_audio.selected_stream:
                    audio_idx = probe.selected_audio.selected_stream.audio_index
                else:
                    audio_idx = 0

            clip_out = temp_dir / f"clip_{c_idx:03d}.mp4"
            try:
                cut_clip(
                    source_path=clip_path,
                    output_path=clip_out,
                    start_sec=clip.start,
                    end_sec=clip.end,
                    audio_stream_index=audio_idx,
                    use_gpu=settings.use_gpu,
                    cancel_event=cancel_evt,
                )
            except RenderCancelled:
                raise
            except Exception as exc:
                raise CutClipError(
                    f"Lỗi khi cắt phân đoạn {c_idx} từ '{clip_path}': {exc}",
                    clip_index=c_idx,
                    source_path=str(clip_path),
                    cause=exc,
                ) from exc
            cut_clips.append(clip_out)

        # 3. Concatenate video clips into assembled_raw.mp4
        if cancel_evt and cancel_evt.is_set():
            raise RenderCancelled()

        _dispatch_callback(callbacks, out_idx, total_outputs, out, 38, "Ghép nối các phân đoạn video...")
        assembled_raw = temp_dir / "assembled_raw.mp4"
        expected_durations = [clip.duration for clip in all_clips]

        log_cb = (
            (lambda msg: _dispatch_callback(callbacks, out_idx, total_outputs, out, 38, msg))
            if callbacks
            else None
        )

        assembled_result = concat_media_clips(
            clip_paths=cut_clips,
            output=assembled_raw,
            cancel=cancel_evt,
            log=log_cb,
            require_audio=True,
            expected_durations=expected_durations,
        )
        if isinstance(assembled_result, MediaProbeResult):
            assembled_probe = assembled_result
        else:
            assembled_probe = probe_typed_media(assembled_raw)

        total_duration = assembled_probe.duration
        has_original = assembled_probe.has_audio

        # 4. Extract Original audio stream from assembled video
        orig_audio_wav: Path | None = None
        if has_original:
            if cancel_evt and cancel_evt.is_set():
                raise RenderCancelled()
            orig_audio_wav = temp_dir / "original_audio.wav"
            try:
                extract_cmd = [
                    find_binary("ffmpeg"),
                    "-y",
                    "-i", str(assembled_raw),
                    "-vn",
                    "-c:a", "pcm_s16le",
                    "-ar", "48000",
                    "-ac", "2",
                    str(orig_audio_wav),
                ]
                run_command(extract_cmd, cancel_event=cancel_evt)
                if cancel_evt and cancel_evt.is_set():
                    raise RenderCancelled()
                validate_wav_audio(orig_audio_wav)
            except RenderCancelled:
                raise
            except Exception as exc:
                raise _wrap_stage_error(
                    exc=exc,
                    out=out,
                    stage="extract_audio",
                    default_category="AUDIO_LAYOUT_ERROR",
                ) from exc

        # 5. Narration Speech Synthesis & Commentary Timeline WAV with silence placement
        if cancel_evt and cancel_evt.is_set():
            raise RenderCancelled()

        current_timeline_time = 0.0
        narration_parts: list[tuple[float, Path]] = []
        narration_cues: list[tuple[float, float, str]] = []

        try:
            for s_idx, seg in enumerate(out.segments, start=1):
                if cancel_evt and cancel_evt.is_set():
                    raise RenderCancelled()

                seg_dur = sum(c.duration for c in seg.source_clips)
                seg_start = current_timeline_time

                # Normalize / check audio policy and narration presence
                is_orig_only = (seg.audio_policy == AudioPolicy.ORIGINAL_ONLY.value)
                has_narr_text = bool(seg.narration and seg.narration.strip())

                if not is_orig_only:
                    if not has_narr_text:
                        raise ValidationError(
                            f"Phân đoạn '{seg.segment_id}' là phân đoạn thuyết minh nhưng thiếu nội dung narration."
                        )

                    progress_pct = 40 + int((s_idx / len(out.segments)) * 25)
                    _dispatch_callback(
                        callbacks,
                        out_idx,
                        total_outputs,
                        out,
                        progress_pct,
                        f"Tổng hợp giọng nói phân đoạn {s_idx}/{len(out.segments)}...",
                    )

                    seg_wav = temp_dir / f"narr_{s_idx:03d}.wav"
                    voice_mgr.synthesize(
                        voice_id=voice_id,
                        text=seg.narration.strip(),
                        output_path=seg_wav,
                        style=voice_style,
                        cancel_event=cancel_evt,
                        allow_mock_synth=allow_mock_synth,
                    )
                    if cancel_evt and cancel_evt.is_set():
                        raise RenderCancelled()
                    validate_wav_audio(seg_wav)
                    speech_dur = probe_duration(seg_wav)
                    if speech_dur > seg_dur + 0.10:
                        raise ValidationError(
                            f"Narration của phân đoạn '{seg.segment_id}' dài {speech_dur:.2f}s nhưng footage chỉ có {seg_dur:.2f}s. "
                            "Finalizer phải chọn footage đủ dài; ứng dụng không cắt mất lời thuyết minh."
                        )

                    narration_parts.append((seg_start, seg_wav))
                    cue_end = min(seg_start + speech_dur, total_duration)
                    narration_cues.append((seg_start, cue_end, seg.narration.strip()))

                current_timeline_time += seg_dur
        except RenderCancelled:
            raise
        except Exception as exc:
            raise _wrap_stage_error(
                exc=exc,
                out=out,
                stage="voice",
                default_category="VOICE_ERROR",
            ) from exc

        has_commentary = len(narration_parts) > 0
        is_original_dialogue_only = not has_commentary
        commentary_audio_path: Path | None = None

        if has_commentary:
            if cancel_evt and cancel_evt.is_set():
                raise RenderCancelled()
            _dispatch_callback(callbacks, out_idx, total_outputs, out, 68, "Tạo dòng âm thanh thuyết minh (timeline)...")
            commentary_audio_path = temp_dir / "commentary_timeline.wav"
            try:
                timeline_cmd = [find_binary("ffmpeg"), "-y"]
                filter_inputs: list[str] = []
                for p_idx, (st_sec, w_file) in enumerate(narration_parts):
                    timeline_cmd.extend(["-i", str(w_file)])
                    st_ms = int(round(st_sec * 1000.0))
                    filter_inputs.append(
                        f"[{p_idx}:a]aformat=channel_layouts=stereo:sample_rates=48000,adelay={st_ms}|{st_ms}[a{p_idx}]"
                    )

                if len(narration_parts) == 1:
                    fg = f"{filter_inputs[0]}; [a0]apad[out]"
                else:
                    labels = "".join(f"[a{i}]" for i in range(len(narration_parts)))
                    fg = (
                        f"{'; '.join(filter_inputs)}; "
                        f"{labels}amix=inputs={len(narration_parts)}:duration=longest:dropout_transition=0:normalize=0,apad[out]"
                    )

                timeline_cmd.extend([
                    "-filter_complex", fg,
                    "-map", "[out]",
                    "-t", f"{total_duration:.3f}",
                    "-c:a", "pcm_s16le",
                    "-ar", "48000",
                    "-ac", "2",
                    str(commentary_audio_path),
                ])
                run_command(timeline_cmd, cancel_event=cancel_evt)
                if cancel_evt and cancel_evt.is_set():
                    raise RenderCancelled()
                validate_wav_audio(commentary_audio_path)
            except RenderCancelled:
                raise
            except Exception as exc:
                raise _wrap_stage_error(
                    exc=exc,
                    out=out,
                    stage="timeline",
                    default_category="AUDIO_LAYOUT_ERROR",
                ) from exc

        # 6. Real Audio Mix with smooth sidechain compression and loudness normalization
        if cancel_evt and cancel_evt.is_set():
            raise RenderCancelled()

        _dispatch_callback(callbacks, out_idx, total_outputs, out, 75, "Trộn âm thanh tự động (Real Audio Mix)...")
        mixed_audio_wav = temp_dir / "mixed_final.wav"
        try:
            mix_settings = AudioMixSettings(
                original_gain_db=settings.original_audio_gain_db,
                commentary_gain_db=settings.commentary_gain_db,
                auto_duck=settings.auto_duck,
                amount=settings.ducking_amount_db,
                target_lufs=settings.target_loudness_lufs,
                true_peak=settings.true_peak_dbtp,
                attack_ms=20.0,
                release_ms=250.0,
                sample_rate=48000,
                channels=2,
                duration_mode="first",
            )

            plan = plan_audio_mix(
                output_path=mixed_audio_wav,
                original_audio_path=orig_audio_wav if has_original else None,
                commentary_audio_path=commentary_audio_path if has_commentary else None,
                settings=mix_settings,
                is_original_dialogue_only=is_original_dialogue_only,
            )
            mix_cmd = plan.build_command(ffmpeg_bin=find_binary("ffmpeg"))
            run_command(mix_cmd, cancel_event=cancel_evt)
            if cancel_evt and cancel_evt.is_set():
                raise RenderCancelled()
            validate_mixed_audio(mixed_audio_wav, settings=mix_settings)
        except RenderCancelled:
            raise
        except Exception as exc:
            raise _wrap_stage_error(
                exc=exc,
                out=out,
                stage="mix",
                default_category="AUDIO_LAYOUT_ERROR",
            ) from exc

        # 7. Subtitle Processing & Sidecar SRT Generation
        if cancel_evt and cancel_evt.is_set():
            raise RenderCancelled()

        _dispatch_callback(callbacks, out_idx, total_outputs, out, 85, "Xử lý phụ đề SRT...")

        try:
            # Original SRT from unified transcript cues
            remapped_orig_cues = remap_subtitles(transcript_cues_by_episode, all_clips)
            orig_srt_rows = cues_to_srt_rows(remapped_orig_cues)
            if not orig_srt_rows:
                raise MediaError(
                    f"Tạo phụ đề gốc (Original subtitle) thất bại: không tìm thấy phụ đề thoại gốc nào cho '{out.title}'."
                )

            orig_srt_path = pub_dir / f"{safe_title}.original.srt"
            write_srt_file(orig_srt_rows, orig_srt_path)

            # Narration SRT (only commentary narration)
            narr_srt_path = pub_dir / f"{safe_title}.narration.srt"
            write_srt_file(narration_cues, narr_srt_path)
        except RenderCancelled:
            raise
        except Exception as exc:
            raise _wrap_stage_error(
                exc=exc,
                out=out,
                stage="subtitle",
                default_category="SUBTITLE_ERROR",
            ) from exc

        # 8. Mux Video and Mixed Audio, then Final Video Render
        if cancel_evt and cancel_evt.is_set():
            raise RenderCancelled()

        _dispatch_callback(callbacks, out_idx, total_outputs, out, 90, "Mã hóa video đầu ra cuối cùng...")
        final_video_path = pub_dir / f"{safe_title}.mp4"

        try:
            if settings.burn_subtitles:
                srt_to_burn = narr_srt_path if has_commentary else orig_srt_path
                assembled_with_audio = temp_dir / "assembled_with_audio.mp4"
                mux_cmd = [
                    find_binary("ffmpeg"),
                    "-y",
                    "-i", str(assembled_raw),
                    "-i", str(mixed_audio_wav),
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    str(assembled_with_audio),
                ]
                run_command(mux_cmd, cancel_event=cancel_evt)
                if cancel_evt and cancel_evt.is_set():
                    raise RenderCancelled()

                render_final_video(
                    raw_video=assembled_with_audio,
                    output_path=final_video_path,
                    srt_path=srt_to_burn,
                    quality=settings.quality,
                    use_gpu=settings.use_gpu,
                    cancel_event=cancel_evt,
                    audio_codec="copy",
                )
            else:
                mux_cmd = [
                    find_binary("ffmpeg"),
                    "-y",
                    "-i", str(assembled_raw),
                    "-i", str(mixed_audio_wav),
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "copy",
                    "-shortest",
                    str(final_video_path),
                ]
                try:
                    run_command(mux_cmd, cancel_event=cancel_evt)
                except RenderCancelled:
                    final_video_path.unlink(missing_ok=True)
                    raise
                except MediaError:
                    # Fallback to -c:a aac if container rejects direct audio stream copy
                    fallback_cmd = [
                        find_binary("ffmpeg"),
                        "-y",
                        "-i", str(assembled_raw),
                        "-i", str(mixed_audio_wav),
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        "-shortest",
                        str(final_video_path),
                    ]
                    run_command(fallback_cmd, cancel_event=cancel_evt)

            if cancel_evt and cancel_evt.is_set():
                final_video_path.unlink(missing_ok=True)
                raise RenderCancelled()
        except RenderCancelled:
            final_video_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            final_video_path.unlink(missing_ok=True)
            raise _wrap_stage_error(
                exc=exc,
                out=out,
                stage="final_encode",
                default_category="FINAL_ENCODE_ERROR",
            ) from exc

        # 9. Validate Publication Folder
        _dispatch_callback(callbacks, out_idx, total_outputs, out, 97, "Kiểm tra chất lượng tệp xuất bản...")
        try:
            validate_publication_folder(
                pub_dir=pub_dir,
                safe_title=safe_title,
                expected_duration=total_duration,
                check_original_srt=True,
                check_narration_srt=has_commentary,
                require_audio=has_original or has_commentary,
            )
        except RenderCancelled:
            raise
        except Exception as exc:
            raise _wrap_stage_error(
                exc=exc,
                out=out,
                stage="publication_validate",
                default_category="INTERMEDIATE_VALIDATION_ERROR",
            ) from exc

        # 10. Update CommentaryOutput paths
        out.publication_video_path = str(final_video_path)
        out.publication_original_srt_path = str(orig_srt_path)
        out.publication_narration_srt_path = str(narr_srt_path)
        out.status = OutputStatus.COMPLETED.value
        out.progress = 100
        _dispatch_callback(callbacks, out_idx, total_outputs, out, 100, "Hoàn tất kết xuất output thành công!")
