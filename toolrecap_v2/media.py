"""Media processing with FFmpeg: probing, clipping, audio mixing, subtitle burning, and safe cancellation."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from fractions import Fraction
from typing import Any, Callable, Sequence

from .gpu import (
    build_encoder_args,
    bundled_binary,
    get_acceleration_plan,
    get_fallback_candidates,
    video_encode_args,
)
from .subtitles.models import (
    AudioSelectionResult,
    AudioStreamInfo,
    FastConcatDecision,
    MediaProbeResult,
    NormalizationProfile,
    StreamSignature,
    SubtitleStreamInfo,
    VideoStreamInfo,
    normalize_language_code,
)



LogCallback = Callable[[str], None]


class MediaError(RuntimeError):
    pass


class RenderCancelled(MediaError):
    pass


class MediaErrorCategory(str, Enum):
    CUT_CLIP_ERROR = "CUT_CLIP_ERROR"
    FAST_CONCAT_ERROR = "FAST_CONCAT_ERROR"
    NORMALIZED_CONCAT_ERROR = "NORMALIZED_CONCAT_ERROR"
    INTERMEDIATE_VALIDATION_ERROR = "INTERMEDIATE_VALIDATION_ERROR"
    CONCAT_PIPELINE_ERROR = "CONCAT_PIPELINE_ERROR"
    AUDIO_LAYOUT_ERROR = "AUDIO_LAYOUT_ERROR"
    VOICE_ERROR = "VOICE_ERROR"
    FINAL_ENCODE_ERROR = "FINAL_ENCODE_ERROR"
    SUBTITLE_ERROR = "SUBTITLE_ERROR"
    PUBLICATION_VALIDATION_ERROR = "PUBLICATION_VALIDATION_ERROR"


class MediaPipelineError(MediaError):
    """Base error for media pipeline operations with category classification."""
    category: str = "MEDIA_PIPELINE_ERROR"

    def __init__(self, message: str, category: str | None = None):
        if category:
            self.category = category
        prefix = f"[{self.category}] "
        msg_str = str(message)
        if not msg_str.startswith(prefix):
            formatted_msg = f"{prefix}{msg_str}"
        else:
            formatted_msg = msg_str
        super().__init__(formatted_msg)


class CutClipError(MediaPipelineError):
    """Error during cutting of a source clip."""
    category: str = "CUT_CLIP_ERROR"

    def __init__(
        self,
        message: str,
        clip_index: int = -1,
        source_path: str = "",
        cause: Exception | None = None,
    ):
        self.clip_index = clip_index
        self.source_path = source_path
        super().__init__(message, category="CUT_CLIP_ERROR")
        if cause is not None:
            self.__cause__ = cause


class FastConcatError(MediaPipelineError):
    """Error during fast demux copy concatenation or its intermediate validation."""
    category: str = "FAST_CONCAT_ERROR"

    def __init__(self, message: str):
        super().__init__(message, category="FAST_CONCAT_ERROR")


class NormalizedConcatError(MediaPipelineError):
    """Error during normalized filter concatenation encoding."""
    category: str = "NORMALIZED_CONCAT_ERROR"

    def __init__(self, message: str):
        super().__init__(message, category="NORMALIZED_CONCAT_ERROR")


class IntermediateValidationError(MediaPipelineError):
    """Error validating assembled intermediate video/audio properties."""
    category: str = "INTERMEDIATE_VALIDATION_ERROR"

    def __init__(self, message: str):
        super().__init__(message, category="INTERMEDIATE_VALIDATION_ERROR")


class ConcatPipelineError(MediaPipelineError):
    """Error when both fast concat and normalized concat fail, preserving both diagnostics."""
    category: str = "CONCAT_PIPELINE_ERROR"

    def __init__(
        self,
        message: str,
        fast_error: Exception | None = None,
        normalized_error: Exception | None = None,
    ):
        self.fast_error = fast_error
        self.normalized_error = normalized_error
        super().__init__(message, category="CONCAT_PIPELINE_ERROR")


class AudioLayoutError(MediaPipelineError):
    """Error during audio layout normalization, extraction, or mixing."""
    category: str = "AUDIO_LAYOUT_ERROR"

    def __init__(self, message: str):
        super().__init__(message, category="AUDIO_LAYOUT_ERROR")


class RenderStageError(MediaPipelineError):
    """Error occurring during a specific rendering stage of an output."""
    category: str = "RENDER_STAGE_ERROR"
    stage: str = "render"
    output_id: str = ""
    output_title: str = ""

    def __init__(
        self,
        message: str,
        output_id: str = "",
        output_title: str = "",
        stage: str = "",
        category: str = "",
        cause: Exception | None = None,
    ):
        self.output_id = output_id
        self.output_title = output_title
        if stage:
            self.stage = stage
        eff_cat = category or (getattr(cause, "category", None) if cause else None) or self.category
        self.category = eff_cat
        if cause is not None:
            self.__cause__ = cause
            self.cause = cause
        else:
            self.cause = None
        self.message = message
        super().__init__(message, category=eff_cat)


def find_binary(name: str) -> str:
    binary = bundled_binary(name)
    if not binary:
        raise MediaError(f"Không tìm thấy công cụ {name}. Hãy đảm bảo FFmpeg đã được cài đặt.")
    return str(binary)


def _kill_process_tree(pid: int) -> None:
    """Kill process and all its children cleanly."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=5,
            )
        except Exception:
            pass


def run_command(
    args: list[str],
    *,
    cancel_event: threading.Event | None = None,
    log: LogCallback | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute FFmpeg command with cancellation checking and process tree cleanup."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=flags,
    )
    lines: list[str] = []

    def _read_output() -> None:
        try:
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                cleaned = line.rstrip()
                lines.append(cleaned)
                if log and ("error" in cleaned.lower() or "warning" in cleaned.lower() or "frame=" in cleaned):
                    log(cleaned)
        except Exception:
            pass

    reader = threading.Thread(target=_read_output, daemon=True)
    reader.start()

    while True:
        if cancel_event and cancel_event.is_set():
            _kill_process_tree(process.pid)
            try:
                process.kill()
                process.wait(timeout=2)
            except Exception:
                pass
            raise RenderCancelled("Xử lý đã bị dừng theo yêu cầu của người dùng.")

        ret = process.poll()
        if ret is not None:
            break

        if cancel_event:
            cancel_event.wait(timeout=0.08)
        else:
            time.sleep(0.08)

    reader.join(timeout=2)
    if cancel_event and cancel_event.is_set():
        _kill_process_tree(process.pid)
        raise RenderCancelled("Xử lý đã bị dừng theo yêu cầu của người dùng.")

    if process.returncode != 0:
        tail = "\n".join(lines[-15:])
        raise MediaError(f"FFmpeg thất bại (mã lỗi {process.returncode}):\n{tail}")

    return subprocess.CompletedProcess(args, process.returncode, "\n".join(lines), "")


COMMENTARY_KEYWORDS = (
    "commentary", "director", "comment", "cast", "crew", "reaction", "trivia", "riff", "spoilers"
)
DESCRIPTIVE_KEYWORDS = (
    "audio description", "visual description", "dvs", "descriptive", "visually impaired", "hearing impaired"
)


def select_english_audio_stream(audio_streams: list[AudioStreamInfo]) -> AudioSelectionResult:
    """Deterministically select English program audio stream.

    Avoids director/commentary and visually descriptive tracks.
    If no clean English program audio track exists, falls back with an explicit non-silent warning.
    """
    if not audio_streams:
        return AudioSelectionResult(
            selected_stream=None,
            has_warning=True,
            warning="Cảnh báo: Không tìm thấy luồng âm thanh nào trong container video.",
            reason="no_audio",
        )

    # Clean English program audio candidates
    eng_clean = [
        a for a in audio_streams
        if a.language == "eng" and not a.is_commentary and not a.is_descriptive
    ]
    if eng_clean:
        # Deterministic scoring: prefer default track, higher channel count, higher bitrate, lower index
        best = max(
            eng_clean,
            key=lambda a: (100 if a.default else 0, a.channels * 10, a.bitrate, -a.index),
        )
        return AudioSelectionResult(
            selected_stream=best,
            has_warning=False,
            warning=None,
            reason=f"Chọn luồng âm thanh chính tiếng Anh #{best.index} ({best.codec}, channels={best.channels})",
        )

    # Fallback path: No clean English program audio
    # 1. Try clean non-commentary track in other language (or und)
    clean_other = [a for a in audio_streams if not a.is_commentary and not a.is_descriptive]
    if clean_other:
        best = max(
            clean_other,
            key=lambda a: (100 if a.default else 0, a.channels * 10, a.bitrate, -a.index),
        )
        warn = (
            f"Cảnh báo: Không tìm thấy luồng âm thanh tiếng Anh chính thức không chứa bình luận; "
            f"tự động chuyển sang luồng #{best.index} (codec={best.codec}, ngôn ngữ={best.language}, tiêu đề='{best.title}')."
        )
        return AudioSelectionResult(
            selected_stream=best,
            has_warning=True,
            warning=warn,
            reason="fallback_non_english",
        )

    # 2. Only commentary/descriptive tracks exist
    best = min(audio_streams, key=lambda a: a.index)
    warn = (
        f"Cảnh báo: Tất cả luồng âm thanh đều là bình luận hoặc mô tả hình ảnh; "
        f"tự động chọn luồng #{best.index} (codec={best.codec}, tiêu đề='{best.title}')."
    )
    return AudioSelectionResult(
        selected_stream=best,
        has_warning=True,
        warning=warn,
        reason="fallback_commentary",
    )


def parse_rational(val: Any) -> tuple[float, str]:
    """Parse rational frame rate or timebase string, returning (float_val, normalized_rational_str)."""
    if val is None:
        return 0.0, ""
    s = str(val).strip()
    if not s or s in ("0/0", "0", "0.0", "N/A", "unknown", "none"):
        return 0.0, ""
    try:
        if "/" in s:
            parts = s.split("/", 1)
            num_f, den_f = float(parts[0]), float(parts[1])
            if den_f == 0 or num_f <= 0:
                return 0.0, ""
            if num_f.is_integer() and den_f.is_integer():
                num_i, den_i = int(num_f), int(den_f)
                frac = Fraction(num_i, den_i)
                return float(frac), f"{frac.numerator}/{frac.denominator}"
            val_float = num_f / den_f
            return val_float, f"{round(num_f)}/{round(den_f)}"
        num = float(s)
        if num <= 0:
            return 0.0, ""
        frac = Fraction(s).limit_denominator(1001)
        return float(frac), f"{frac.numerator}/{frac.denominator}"
    except Exception:
        return 0.0, ""


def normalize_sar(raw_sar: Any) -> str:
    """Normalize Sample Aspect Ratio (SAR) to 'N:D' string. Returns empty string if unknown or invalid."""
    if raw_sar is None:
        return ""
    s = str(raw_sar).strip()
    if not s or s in ("0:1", "0/1", "0:0", "0/0", "0", "N/A", "unknown", "none"):
        return ""
    s = s.replace("/", ":")
    if ":" in s:
        parts = s.split(":", 1)
        try:
            num, den = int(parts[0]), int(parts[1])
            if num <= 0 or den <= 0:
                return ""
            frac = Fraction(num, den)
            return f"{frac.numerator}:{frac.denominator}"
        except Exception:
            return ""
    try:
        val = float(s)
        if val <= 0:
            return ""
        if val == 1.0:
            return "1:1"
        frac = Fraction(s).limit_denominator(100)
        return f"{frac.numerator}:{frac.denominator}"
    except Exception:
        return ""


def normalize_time_base(tb: Any) -> str:
    """Normalize timebase string like '1/1000' or '1/48000'. Returns empty string if invalid/unknown."""
    if tb is None:
        return ""
    s = str(tb).strip()
    if not s or s in ("0/0", "0", "0.0", "N/A", "unknown", "none"):
        return ""
    try:
        if "/" in s:
            parts = s.split("/", 1)
            num, den = int(parts[0]), int(parts[1])
            if num <= 0 or den <= 0:
                return ""
            frac = Fraction(num, den)
            return f"{frac.numerator}/{frac.denominator}"
    except Exception:
        pass
    return s


def parse_stream_metadata(
    raw_streams: list[dict],
) -> tuple[list[VideoStreamInfo], list[AudioStreamInfo], list[SubtitleStreamInfo], AudioSelectionResult]:
    """Parse raw ffprobe stream dicts into typed metadata and perform audio selection."""
    video_streams: list[VideoStreamInfo] = []
    audio_streams: list[AudioStreamInfo] = []
    subtitle_streams: list[SubtitleStreamInfo] = []

    v_counter = 0
    a_counter = 0
    s_counter = 0

    for s in raw_streams:
        codec_type = s.get("codec_type")
        idx = int(s.get("index", 0))
        codec_name = str(s.get("codec_name", ""))
        tags = s.get("tags", {}) if isinstance(s.get("tags"), dict) else {}
        disposition = s.get("disposition", {}) if isinstance(s.get("disposition"), dict) else {}
        title = str(tags.get("title", ""))
        raw_lang = tags.get("language")
        norm_lang = normalize_language_code(raw_lang)
        is_default = bool(disposition.get("default", 0))
        is_forced = bool(disposition.get("forced", 0))

        if codec_type == "video":
            avg_fps, avg_rat = parse_rational(s.get("avg_frame_rate"))
            r_fps, r_rat = parse_rational(s.get("r_frame_rate"))
            if avg_fps > 0:
                fps = avg_fps
                fps_rational = avg_rat
            elif r_fps > 0:
                fps = r_fps
                fps_rational = r_rat
            else:
                fps = 0.0
                fps_rational = ""

            duration = float(s.get("duration") or 0.0)
            bitrate = int(s.get("bit_rate") or 0)
            width = int(s.get("width") or 0)
            height = int(s.get("height") or 0)
            pix_fmt = str(s.get("pix_fmt") or "")
            profile = str(s.get("profile") or "")
            time_base = normalize_time_base(s.get("time_base"))
            sar = normalize_sar(s.get("sample_aspect_ratio") or s.get("sar"))
            try:
                start_time = float(s.get("start_time") or 0.0)
            except (ValueError, TypeError):
                start_time = 0.0

            video_streams.append(
                VideoStreamInfo(
                    index=idx,
                    video_index=v_counter,
                    codec=codec_name,
                    width=width,
                    height=height,
                    fps=fps,
                    duration=duration,
                    bitrate=bitrate,
                    title=title,
                    default=is_default,
                    forced=is_forced,
                    fps_rational=fps_rational,
                    pix_fmt=pix_fmt,
                    profile=profile,
                    time_base=time_base,
                    sar=sar,
                    start_time=start_time,
                )
            )
            v_counter += 1

        elif codec_type == "audio":
            try:
                channels = int(s.get("channels") or 0)
            except (ValueError, TypeError):
                channels = 0

            channel_layout = str(s.get("channel_layout") or "")
            bitrate = int(s.get("bit_rate") or 0)
            try:
                sample_rate = int(s.get("sample_rate") or 0)
            except (ValueError, TypeError):
                sample_rate = 0

            sample_fmt = str(s.get("sample_fmt") or "")
            profile = str(s.get("profile") or "")
            time_base = normalize_time_base(s.get("time_base"))
            try:
                start_time = float(s.get("start_time") or 0.0)
            except (ValueError, TypeError):
                start_time = 0.0

            title_lower = title.lower()

            is_commentary = bool(disposition.get("commentary", 0)) or any(
                k in title_lower for k in COMMENTARY_KEYWORDS
            )
            is_descriptive = (
                bool(disposition.get("descriptions", 0))
                or bool(disposition.get("visual_impaired", 0))
                or bool(disposition.get("hearing_impaired", 0))
                or any(k in title_lower for k in DESCRIPTIVE_KEYWORDS)
            )

            audio_streams.append(
                AudioStreamInfo(
                    index=idx,
                    audio_index=a_counter,
                    codec=codec_name,
                    language=norm_lang,
                    title=title,
                    channels=channels,
                    channel_layout=channel_layout,
                    bitrate=bitrate,
                    default=is_default,
                    forced=is_forced,
                    is_commentary=is_commentary,
                    is_descriptive=is_descriptive,
                    sample_rate=sample_rate,
                    sample_fmt=sample_fmt,
                    profile=profile,
                    time_base=time_base,
                    start_time=start_time,
                )
            )
            a_counter += 1

        elif codec_type == "subtitle":
            is_bitmap = codec_name.lower() in ("hdmv_pgs_subtitle", "dvd_subtitle", "dvdsub", "pgs")
            subtitle_forced = is_forced or ("forced" in title.lower())

            subtitle_streams.append(
                SubtitleStreamInfo(
                    index=idx,
                    subtitle_index=s_counter,
                    codec=codec_name,
                    language=norm_lang,
                    title=title,
                    default=is_default,
                    forced=subtitle_forced,
                    is_bitmap=is_bitmap,
                )
            )
            s_counter += 1

    selected_audio = select_english_audio_stream(audio_streams)
    return video_streams, audio_streams, subtitle_streams, selected_audio


def probe_typed_media(path: str | Path) -> MediaProbeResult:
    """Extract comprehensive typed stream and format information using ffprobe."""
    media_path = Path(path).resolve()
    if not media_path.is_file():
        raise MediaError(f"Không tìm thấy file video: {media_path}")

    command = [
        find_binary("ffprobe"),
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(media_path),
    ]
    res = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if res.returncode != 0:
        raise MediaError(f"ffprobe không thể đọc metadata của file: {res.stderr.strip()}")

    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe trả về JSON không hợp lệ: {exc}") from exc

    streams = data.get("streams", [])
    fmt = data.get("format", {})

    v_streams, a_streams, s_streams, sel_audio = parse_stream_metadata(streams)

    primary_v = v_streams[0] if v_streams else None
    primary_a = a_streams[0] if a_streams else None

    duration = float(fmt.get("duration") or 0.0)
    if duration <= 0.0 and primary_v:
        duration = float(primary_v.duration or 0.0)

    width = primary_v.width if primary_v else 1920
    height = primary_v.height if primary_v else 1080

    fmt_start_time = 0.0
    try:
        fmt_start_time = float(fmt.get("start_time") or 0.0)
    except (ValueError, TypeError):
        fmt_start_time = 0.0

    return MediaProbeResult(
        path=str(media_path),
        duration=duration,
        width=width,
        height=height,
        has_video=primary_v is not None,
        has_audio=primary_a is not None,
        video_codec=primary_v.codec if primary_v else None,
        audio_codec=primary_a.codec if primary_a else None,
        video_streams=v_streams,
        audio_streams=a_streams,
        subtitle_streams=s_streams,
        selected_audio=sel_audio,
        start_time=fmt_start_time,
    )


def probe_media(path: str | Path) -> dict:
    """Extract stream and format information using ffprobe, preserving legacy dict format with typed enrichments."""
    typed_info = probe_typed_media(path)
    res = typed_info.to_dict()
    # Add convenience top-level fields for audio warning
    res["audio_warning"] = typed_info.selected_audio.warning if typed_info.selected_audio and typed_info.selected_audio.has_warning else None
    return res


def signature_from_probe(probe: MediaProbeResult | dict[str, Any]) -> StreamSignature:
    """Extract StreamSignature from primary video stream and selected audio stream.

    Uses primary video + selected audio (not primary_a), because cut maps selected track.
    """
    v: VideoStreamInfo | dict[str, Any] | None = None
    a: AudioStreamInfo | dict[str, Any] | None = None
    has_v = False
    has_a = False

    if isinstance(probe, MediaProbeResult):
        has_v = probe.has_video and len(probe.video_streams) > 0
        if probe.video_streams:
            v = probe.video_streams[0]

        has_a = probe.has_audio
        # Selected audio takes precedence over primary audio
        if probe.selected_audio and probe.selected_audio.selected_stream:
            a = probe.selected_audio.selected_stream
            has_a = True
        elif probe.audio_streams:
            a = probe.audio_streams[0]
            has_a = True
        else:
            has_a = False
    elif isinstance(probe, dict):
        raw_v = probe.get("video_streams", [])
        if raw_v:
            v = raw_v[0]
            has_v = bool(probe.get("has_video", True))
        else:
            has_v = False

        sel_audio = probe.get("selected_audio")
        if isinstance(sel_audio, dict) and sel_audio.get("selected_stream"):
            a = sel_audio["selected_stream"]
            has_a = True
        elif isinstance(sel_audio, AudioSelectionResult) and sel_audio.selected_stream:
            a = sel_audio.selected_stream
            has_a = True
        elif probe.get("audio_streams"):
            a = probe["audio_streams"][0]
            has_a = bool(probe.get("has_audio", True))
        else:
            has_a = False
    else:
        raise TypeError(f"Expected MediaProbeResult or dict, got {type(probe)}")

    sig = StreamSignature()

    if has_v and v is not None:
        if isinstance(v, VideoStreamInfo):
            sig.video_codec = v.codec
            sig.video_profile = v.profile
            sig.width = v.width
            sig.height = v.height
            sig.pix_fmt = v.pix_fmt
            sig.fps_rational = v.fps_rational
            sig.fps = v.fps
            sig.video_time_base = v.time_base
            sig.sar = v.sar
        elif isinstance(v, dict):
            sig.video_codec = str(v.get("codec") or v.get("codec_name") or "")
            sig.video_profile = str(v.get("profile") or "")
            sig.width = int(v.get("width") or 0)
            sig.height = int(v.get("height") or 0)
            sig.pix_fmt = str(v.get("pix_fmt") or "")
            fps_val, fps_rat = parse_rational(
                v.get("fps_rational") or v.get("avg_frame_rate") or v.get("r_frame_rate") or v.get("fps")
            )
            sig.fps_rational = str(v.get("fps_rational") or fps_rat)
            sig.fps = float(v.get("fps") or fps_val)
            sig.video_time_base = normalize_time_base(v.get("time_base"))
            sig.sar = normalize_sar(v.get("sar") or v.get("sample_aspect_ratio") or v.get("SAR"))

    if has_a and a is not None:
        sig.has_audio = True
        if isinstance(a, AudioStreamInfo):
            sig.audio_codec = a.codec
            sig.audio_profile = a.profile
            sig.sample_rate = a.sample_rate
            sig.sample_fmt = a.sample_fmt
            sig.channels = a.channels
            sig.channel_layout = a.channel_layout
            sig.audio_time_base = a.time_base
        elif isinstance(a, dict):
            sig.audio_codec = str(a.get("codec") or a.get("codec_name") or "")
            sig.audio_profile = str(a.get("profile") or "")
            sig.sample_rate = int(a.get("sample_rate") or 0)
            sig.sample_fmt = str(a.get("sample_fmt") or "")
            sig.channels = int(a.get("channels") or 0)
            sig.channel_layout = str(a.get("channel_layout") or "")
            sig.audio_time_base = normalize_time_base(a.get("time_base"))
    else:
        sig.has_audio = False

    return sig


def evaluate_fast_concat(
    probes: Sequence[MediaProbeResult | dict[str, Any]],
    *,
    require_audio: bool = False,
) -> FastConcatDecision:
    """Strictly evaluate if clips can be concatenated via stream-copy fast concat.

    Invariants:
    - Empty probe list returns compatible=False.
    - All clips must have valid video.
    - Every compared property must be known and non-zero/non-empty.
    - Video: codec, profile, width, height, pix_fmt, fps, timebase, SAR must be known and identical.
    - Audio presence must be identical across all clips.
    - If audio present: channel_layout must be known (not empty or 'unknown');
      codec, sample_rate, sample_fmt, channels, channel_layout, timebase must be known and identical.
    - If require_audio is True, clips with no audio are ineligible for fast concat.
    """
    if not probes:
        return FastConcatDecision(
            compatible=False,
            reason="Không có clip nào để ghép (empty probes list).",
            differences=["no_clips"],
        )

    signatures = [signature_from_probe(p) for p in probes]
    differences: list[str] = []

    if require_audio and all(not sig.has_audio for sig in signatures):
        differences.append("require_audio_no_audio")

    # 1. Per-clip knownness checks
    for idx, sig in enumerate(signatures):
        if not sig.video_codec or sig.video_codec.lower() in ("unknown", "none"):
            differences.append(f"clip_{idx}_video_codec_unknown")
        if not sig.video_profile or sig.video_profile.lower() in ("unknown", "none", "und"):
            differences.append(f"clip_{idx}_video_profile_unknown")
        if sig.width <= 0 or sig.height <= 0:
            differences.append(f"clip_{idx}_invalid_dimensions:{sig.width}x{sig.height}")
        if not sig.pix_fmt or sig.pix_fmt.lower() in ("unknown", "none"):
            differences.append(f"clip_{idx}_pix_fmt_unknown")
        if sig.fps <= 0 or not sig.fps_rational or sig.fps_rational in ("0/0", "0/1", "0"):
            differences.append(f"clip_{idx}_fps_unknown")
        if not sig.video_time_base or sig.video_time_base.lower() in ("unknown", "none", "0/0", "0"):
            differences.append(f"clip_{idx}_video_timebase_unknown")
        if not sig.sar or sig.sar.lower() in ("unknown", "none", "0:1", "0/1", "0:0", "0/0"):
            differences.append(f"clip_{idx}_sar_unknown")

        if sig.has_audio:
            if not sig.channel_layout or sig.channel_layout.lower() in ("unknown", "none"):
                differences.append(f"clip_{idx}_audio_channel_layout_unknown")
            if not sig.audio_codec or sig.audio_codec.lower() in ("unknown", "none"):
                differences.append(f"clip_{idx}_audio_codec_unknown")
            if sig.sample_rate <= 0:
                differences.append(f"clip_{idx}_sample_rate_unknown")
            if not sig.sample_fmt or sig.sample_fmt.lower() in ("unknown", "none"):
                differences.append(f"clip_{idx}_sample_fmt_unknown")
            if sig.channels <= 0:
                differences.append(f"clip_{idx}_channels_unknown")
            if not sig.audio_time_base or sig.audio_time_base.lower() in ("unknown", "none", "0/0", "0"):
                differences.append(f"clip_{idx}_audio_timebase_unknown")

    # 2. Audio presence consistency across all clips
    first_has_audio = signatures[0].has_audio
    for idx, sig in enumerate(signatures[1:], start=1):
        if sig.has_audio != first_has_audio:
            differences.append(f"audio_presence_mismatch:clip_0={first_has_audio}_vs_clip_{idx}={sig.has_audio}")

    # 3. Cross-clip equality checks against base clip (clip 0)
    base = signatures[0]
    for idx, sig in enumerate(signatures[1:], start=1):
        if sig.video_codec.lower() != base.video_codec.lower():
            differences.append(f"video_codec_mismatch:{base.video_codec}_vs_{sig.video_codec}")
        if sig.video_profile.lower() != base.video_profile.lower():
            differences.append(f"video_profile_mismatch:{base.video_profile}_vs_{sig.video_profile}")
        if (sig.width, sig.height) != (base.width, base.height):
            differences.append(f"resolution_mismatch:{base.width}x{base.height}_vs_{sig.width}x{sig.height}")
        if sig.pix_fmt.lower() != base.pix_fmt.lower():
            differences.append(f"pix_fmt_mismatch:{base.pix_fmt}_vs_{sig.pix_fmt}")
        if sig.fps_rational != base.fps_rational:
            differences.append(f"fps_mismatch:{base.fps_rational}_vs_{sig.fps_rational}")
        if sig.video_time_base != base.video_time_base:
            differences.append(f"video_timebase_mismatch:{base.video_time_base}_vs_{sig.video_time_base}")
        if sig.sar != base.sar:
            differences.append(f"sar_mismatch:{base.sar}_vs_{sig.sar}")

        if base.has_audio and sig.has_audio:
            if sig.audio_codec.lower() != base.audio_codec.lower():
                differences.append(f"audio_codec_mismatch:{base.audio_codec}_vs_{sig.audio_codec}")
            if sig.sample_rate != base.sample_rate:
                differences.append(f"sample_rate_mismatch:{base.sample_rate}_vs_{sig.sample_rate}")
            if sig.sample_fmt.lower() != base.sample_fmt.lower():
                differences.append(f"sample_fmt_mismatch:{base.sample_fmt}_vs_{sig.sample_fmt}")
            if sig.channels != base.channels:
                differences.append(f"channels_mismatch:{base.channels}_vs_{sig.channels}")
            if sig.channel_layout.lower() != base.channel_layout.lower():
                differences.append(f"channel_layout_mismatch:{base.channel_layout}_vs_{sig.channel_layout}")
            if sig.audio_time_base != base.audio_time_base:
                differences.append(f"audio_timebase_mismatch:{base.audio_time_base}_vs_{sig.audio_time_base}")

    if differences:
        return FastConcatDecision(
            compatible=False,
            reason=f"Không thể fast concat do không tương thích hoặc thuộc tính không hợp lệ ({len(differences)} điểm): {'; '.join(differences)}.",
            differences=differences,
        )

    return FastConcatDecision(
        compatible=True,
        reason="Tất cả các luồng tương thích hoàn toàn cho fast copy concat.",
        differences=[],
    )


def build_normalization_profile(
    probes: list[MediaProbeResult] | list[dict[str, Any]],
    *,
    synthesize_audio: bool = True,
) -> NormalizationProfile:
    """Build standardized normalization profile from input probes.

    Invariants:
    - Canvas: first valid clip canvas with even dimensions (floor, min 2). Defaults to 1920x1080.
    - FPS: first clip with fps > 0, else 24.0 ('24/1').
    - Audio: standard target 48000 Hz, fltp, stereo, aac.
    - requires_audio: True if synthesize_audio is True, or if any clip has audio.
    """
    canvas_w = 1920
    canvas_h = 1080
    fps = 24.0
    fps_rational = "24/1"
    has_any_audio = False

    # Find first valid clip canvas
    for p in probes:
        w, h = 0, 0
        if isinstance(p, MediaProbeResult):
            if p.video_streams:
                w, h = p.video_streams[0].width, p.video_streams[0].height
            else:
                w, h = p.width, p.height
        elif isinstance(p, dict):
            v_list = p.get("video_streams", [])
            if v_list and isinstance(v_list[0], dict):
                w = int(v_list[0].get("width") or 0)
                h = int(v_list[0].get("height") or 0)
            elif v_list and isinstance(v_list[0], VideoStreamInfo):
                w, h = v_list[0].width, v_list[0].height
            else:
                w = int(p.get("width") or 0)
                h = int(p.get("height") or 0)

        if w > 0 and h > 0:
            canvas_w = max(2, (int(w) // 2) * 2)
            canvas_h = max(2, (int(h) // 2) * 2)
            break

    # Find first clip with fps > 0
    for p in probes:
        p_fps = 0.0
        p_rat = ""
        if isinstance(p, MediaProbeResult):
            if p.video_streams:
                p_fps = p.video_streams[0].fps
                p_rat = p.video_streams[0].fps_rational
        elif isinstance(p, dict):
            v_list = p.get("video_streams", [])
            if v_list and isinstance(v_list[0], dict):
                p_fps = float(v_list[0].get("fps") or 0.0)
                p_rat = str(v_list[0].get("fps_rational") or "")
            elif v_list and isinstance(v_list[0], VideoStreamInfo):
                p_fps = v_list[0].fps
                p_rat = v_list[0].fps_rational
            else:
                p_fps = float(p.get("fps") or 0.0)
                p_rat = str(p.get("fps_rational") or "")

        if p_fps > 0:
            fps = p_fps
            fps_rational = p_rat or f"{round(p_fps)}/1"
            break

    # Check audio presence across probes
    for p in probes:
        if isinstance(p, MediaProbeResult):
            if p.has_audio:
                has_any_audio = True
                break
        elif isinstance(p, dict):
            if bool(p.get("has_audio", False)) or bool(p.get("audio_streams")):
                has_any_audio = True
                break

    requires_audio = True if (has_any_audio or synthesize_audio) else False

    return NormalizationProfile(
        width=canvas_w,
        height=canvas_h,
        fps=fps,
        fps_rational=fps_rational,
        video_codec="h264",
        pix_fmt="yuv420p",
        video_profile="high",
        video_time_base="1/1000",
        sar="1:1",
        audio_codec="aac",
        sample_rate=48000,
        sample_fmt="fltp",
        channels=2,
        channel_layout="stereo",
        audio_bitrate="192k",
        requires_audio=requires_audio,
        synthesize_audio=synthesize_audio,
    )


def probe_intermediate_details(path: str | Path) -> dict[str, Any]:
    """Inspect intermediate media details including packet counts via ffprobe -count_packets."""
    media_path = Path(path).resolve()
    if not media_path.is_file():
        raise IntermediateValidationError(f"Không tìm thấy tệp intermediate: {media_path}")

    cmd = [
        find_binary("ffprobe"),
        "-v", "error",
        "-count_packets",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(media_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            timeout=30,
        )
    except Exception as exc:
        raise IntermediateValidationError(f"Không thể chạy ffprobe để kiểm tra intermediate: {exc}") from exc

    if proc.returncode != 0:
        raise IntermediateValidationError(
            f"FFprobe kiểm tra intermediate thất bại (code {proc.returncode}): {proc.stderr}"
        )

    try:
        data = json.loads(proc.stdout)
        return data
    except Exception as exc:
        raise IntermediateValidationError(f"Dữ liệu JSON từ ffprobe không hợp lệ: {exc}") from exc


def validate_intermediate_output(
    path: str | Path,
    expected_duration: float | None = None,
    require_audio: bool = True,
    *,
    details: dict[str, Any] | None = None,
) -> MediaProbeResult:
    """Validate technical properties of assembled intermediate media file.

    Checks:
    - File exists and size > 0
    - Video stream exists with valid dimensions and packets (>0 where available, else frames/duration)
    - Audio stream exists if required, with known channels, valid sample rate, known channel layout,
      and packets (>0 where available, else frames/duration)
    - Format start_time and stream start_times >= -0.05
    - Duration is valid (> 0) and within expected tolerance max(0.75, 2% of expected_duration)
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise IntermediateValidationError(f"Tệp intermediate không tồn tại: {p}")
    size = p.stat().st_size
    if size <= 0:
        raise IntermediateValidationError(f"Tệp intermediate rỗng (kích thước {size} bytes): {p}")

    if details is None:
        details = probe_intermediate_details(p)

    streams = details.get("streams", [])
    fmt = details.get("format", {})

    v_streams = [s for s in streams if s.get("codec_type") == "video"]
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]

    # 1. Video validation
    if not v_streams:
        raise IntermediateValidationError(f"Tệp intermediate không chứa luồng video: {p}")
    primary_v = v_streams[0]
    w = int(primary_v.get("width") or 0)
    h = int(primary_v.get("height") or 0)
    if w <= 0 or h <= 0:
        raise IntermediateValidationError(f"Kích thước video intermediate không hợp lệ: {w}x{h}")

    v_packets_raw = primary_v.get("nb_read_packets")
    v_frames_raw = primary_v.get("nb_frames")
    v_dur_raw = float(primary_v.get("duration") or fmt.get("duration") or 0.0)

    if v_packets_raw is not None and str(v_packets_raw).strip() not in ("N/A", ""):
        try:
            if int(v_packets_raw) <= 0:
                raise IntermediateValidationError(
                    f"Luồng video không có packet nào (nb_read_packets={v_packets_raw})"
                )
        except ValueError:
            pass
    else:
        frames_ok = False
        if v_frames_raw is not None and str(v_frames_raw).strip() not in ("N/A", ""):
            try:
                if int(v_frames_raw) > 0:
                    frames_ok = True
            except ValueError:
                pass
        if not frames_ok and v_dur_raw <= 0:
            raise IntermediateValidationError("Luồng video không có frame và duration <= 0")

    # 2. Audio validation
    if require_audio:
        if not a_streams:
            raise IntermediateValidationError(f"Tệp intermediate yêu cầu có âm thanh nhưng không có luồng audio: {p}")
        primary_a = a_streams[0]
        ch = int(primary_a.get("channels") or 0)
        if ch <= 0:
            raise IntermediateValidationError(f"Luồng audio có số kênh không hợp lệ (channels={ch})")
        layout = str(primary_a.get("channel_layout") or "").strip().lower()
        if not layout or layout in ("unknown", "none", "0"):
            raise IntermediateValidationError(f"Luồng audio có channel layout không xác định: '{layout}'")
        sr = int(primary_a.get("sample_rate") or 0)
        if sr <= 0:
            raise IntermediateValidationError(f"Luồng audio có sample rate không hợp lệ (sample_rate={sr})")

        a_packets_raw = primary_a.get("nb_read_packets")
        a_frames_raw = primary_a.get("nb_frames")
        a_dur_raw = float(primary_a.get("duration") or fmt.get("duration") or 0.0)
        if a_packets_raw is not None and str(a_packets_raw).strip() not in ("N/A", ""):
            try:
                if int(a_packets_raw) <= 0:
                    raise IntermediateValidationError(
                        f"Luồng audio không có packet nào (nb_read_packets={a_packets_raw})"
                    )
            except ValueError:
                pass
        else:
            frames_ok = False
            if a_frames_raw is not None and str(a_frames_raw).strip() not in ("N/A", ""):
                try:
                    if int(a_frames_raw) > 0:
                        frames_ok = True
                except ValueError:
                    pass
            if not frames_ok and a_dur_raw <= 0:
                raise IntermediateValidationError("Luồng audio không có frame và duration <= 0")

    # 3. Start time validation (>= -0.05)
    fmt_st = 0.0
    try:
        fmt_st = float(fmt.get("start_time") or 0.0)
    except (ValueError, TypeError):
        fmt_st = 0.0
    if fmt_st < -0.05:
        raise IntermediateValidationError(f"Start time của format bị âm: {fmt_st:.4f} < -0.05")

    for idx, s in enumerate(streams):
        try:
            s_st = float(s.get("start_time") or 0.0)
            if s_st < -0.05:
                c_type = s.get("codec_type", "stream")
                raise IntermediateValidationError(f"Start time của {c_type} (stream {idx}) bị âm: {s_st:.4f} < -0.05")
        except (ValueError, TypeError):
            pass

    # 4. Duration validation and tolerance
    actual_duration = float(fmt.get("duration") or 0.0)
    if actual_duration <= 0 and v_streams:
        actual_duration = float(v_streams[0].get("duration") or 0.0)
    if actual_duration <= 0:
        raise IntermediateValidationError(f"Duration intermediate không hợp lệ: {actual_duration}")

    if expected_duration is not None and expected_duration > 0:
        tol = max(0.75, 0.02 * expected_duration)
        diff = abs(actual_duration - expected_duration)
        if diff > tol:
            raise IntermediateValidationError(
                f"Thời lượng intermediate lệch quá mức: thực tế={actual_duration:.3f}s, "
                f"kỳ vọng={expected_duration:.3f}s, chênh lệch={diff:.3f}s > dung sai={tol:.3f}s"
            )

    parsed_v, parsed_a, parsed_s, sel_audio = parse_stream_metadata(streams)
    return MediaProbeResult(
        path=str(p),
        duration=actual_duration,
        width=parsed_v[0].width if parsed_v else 0,
        height=parsed_v[0].height if parsed_v else 0,
        has_video=len(parsed_v) > 0,
        has_audio=len(parsed_a) > 0,
        video_codec=parsed_v[0].codec if parsed_v else None,
        audio_codec=parsed_a[0].codec if parsed_a else None,
        video_streams=parsed_v,
        audio_streams=parsed_a,
        subtitle_streams=parsed_s,
        selected_audio=sel_audio,
        start_time=fmt_st,
    )


def concat_media_clips(
    clip_paths: Sequence[str | Path],
    output: str | Path,
    cancel: threading.Event | None = None,
    log: LogCallback | None = None,
    require_audio: bool = True,
    expected_durations: Sequence[float] | None = None,
    probes: Sequence[MediaProbeResult | dict[str, Any]] | None = None,
) -> MediaProbeResult:
    """Concatenate media clips using fast demux copy when compatible, falling back to normalized encoding.

    Invariants:
    - Probes each cut clip exactly once if probes argument is None.
    - Evaluates strict fast concat compatibility.
    - If compatible: attempts stream-copy concat demuxer and validates output.
    - If fast concat fails: saves fast error and proceeds to normalized concat.
    - If incompatible: skips fast concat, logs reason, runs normalized only.
    - In normalized mode: normalizes video scale/pad (even), setsar=1, fps, format=yuv420p, settb=AVTB, setpts=PTS-STARTPTS.
      Audio existing: aresample=48000:async=0:first_pts=0, aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo, asetpts=PTS-STARTPTS.
      Audio missing: anullsrc=channel_layout=stereo:sample_rate=48000, atrim=duration=D, asetpts=PTS-STARTPTS.
    - Validates assembled output before returning.
    - On cancel: terminates ffmpeg process tree and cleans up partials and assembled output.
    - Dual diagnostics: if both fast and normalized fail, raises ConcatPipelineError preserving both errors.
    """
    if cancel and cancel.is_set():
        raise RenderCancelled("Xử lý đã bị dừng theo yêu cầu của người dùng.")

    if not clip_paths:
        raise IntermediateValidationError("Không có clip nào để ghép.")

    out_path = Path(output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list_file = out_path.with_suffix(".concat_list.txt")

    # 1. Probe cut clips exactly once if not provided
    if probes is not None:
        clip_probes = list(probes)
    else:
        clip_probes = []
        for cp in clip_paths:
            if cancel and cancel.is_set():
                raise RenderCancelled("Xử lý đã bị dừng theo yêu cầu của người dùng.")
            clip_probes.append(probe_typed_media(cp))

    # Calculate expected total duration
    exp_total_dur = 0.0
    if expected_durations:
        exp_total_dur = sum(float(d) for d in expected_durations if d > 0)
    if exp_total_dur <= 0:
        for p in clip_probes:
            d = p.duration if isinstance(p, MediaProbeResult) else float(p.get("duration") or 0.0)
            if d > 0:
                exp_total_dur += d

    # 2. Evaluate fast concat compatibility
    decision = evaluate_fast_concat(clip_probes, require_audio=require_audio)
    fast_error: Exception | None = None

    if decision.compatible:
        if log:
            log(f"Fast concat tương thích: {decision.reason}")
        try:
            lines = [f"file '{Path(p).resolve().as_posix()}'" for p in clip_paths]
            concat_list_file.write_text("\n".join(lines), encoding="utf-8")

            cmd = [
                find_binary("ffmpeg"),
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(out_path),
            ]
            run_command(cmd, cancel_event=cancel, log=log)

            if cancel and cancel.is_set():
                raise RenderCancelled("Xử lý đã bị dừng theo yêu cầu của người dùng.")

            validated = validate_intermediate_output(
                out_path,
                expected_duration=exp_total_dur if exp_total_dur > 0 else None,
                require_audio=require_audio,
            )
            return validated

        except RenderCancelled:
            if out_path.is_file():
                out_path.unlink(missing_ok=True)
            if concat_list_file.is_file():
                concat_list_file.unlink(missing_ok=True)
            raise
        except Exception as exc:
            if out_path.is_file():
                out_path.unlink(missing_ok=True)
            fast_error = exc if isinstance(exc, FastConcatError) else FastConcatError(str(exc))
            if log:
                log(f"Fast concat thất bại, chuyển sang normalized concat: {fast_error}")
        finally:
            if concat_list_file.is_file():
                concat_list_file.unlink(missing_ok=True)
    else:
        if log:
            log(f"Fast concat không được áp dụng ({decision.reason}), thực hiện normalized concat.")

    # 3. Normalized concat
    if cancel and cancel.is_set():
        if out_path.is_file():
            out_path.unlink(missing_ok=True)
        raise RenderCancelled("Xử lý đã bị dừng theo yêu cầu của người dùng.")

    try:
        profile = build_normalization_profile(clip_probes, synthesize_audio=require_audio)
        W = profile.width
        H = profile.height
        fps_val = profile.fps

        n_clips = len(clip_paths)
        filter_parts: list[str] = []

        # Construct video filters
        for i in range(n_clips):
            vf = (
                f"[{i}:v:0]"
                f"scale='trunc(iw*min({W}/iw,{H}/ih)/2)*2':'trunc(ih*min({W}/iw,{H}/ih)/2)*2',"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,"
                f"fps={fps_val:.6f},"
                f"format=yuv420p,"
                f"settb=AVTB,"
                f"setpts=PTS-STARTPTS"
                f"[v{i}]"
            )
            filter_parts.append(vf)

        # Check whether any clip has audio or if require_audio is True
        any_has_audio = False
        for p in clip_probes:
            ha = p.has_audio if isinstance(p, MediaProbeResult) else bool(p.get("has_audio"))
            if ha:
                any_has_audio = True
                break
        create_audio = require_audio or any_has_audio

        if create_audio:
            # Construct audio filters
            for i in range(n_clips):
                probe_i = clip_probes[i]
                has_a = probe_i.has_audio if isinstance(probe_i, MediaProbeResult) else bool(probe_i.get("has_audio"))
                if has_a:
                    af = (
                        f"[{i}:a:0]"
                        f"aresample=48000:async=0:first_pts=0,"
                        f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                        f"asetpts=PTS-STARTPTS"
                        f"[a{i}]"
                    )
                else:
                    dur_i = 0.0
                    if isinstance(probe_i, MediaProbeResult):
                        dur_i = probe_i.duration
                    elif isinstance(probe_i, dict):
                        dur_i = float(probe_i.get("duration") or 0.0)
                    if dur_i <= 0 and expected_durations and i < len(expected_durations):
                        dur_i = float(expected_durations[i])
                    if dur_i <= 0:
                        dur_i = 0.1

                    af = (
                        f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                        f"atrim=duration={dur_i:.6f},"
                        f"asetpts=PTS-STARTPTS"
                        f"[a{i}]"
                    )
                filter_parts.append(af)

            interleaved = "".join(f"[v{i}][a{i}]" for i in range(n_clips))
            filter_parts.append(f"{interleaved}concat=n={n_clips}:v=1:a=1[v][a]")
        else:
            v_only = "".join(f"[v{i}]" for i in range(n_clips))
            filter_parts.append(f"{v_only}concat=n={n_clips}:v=1:a=0[v]")

        fc_string = "; ".join(filter_parts)

        norm_cmd = [find_binary("ffmpeg"), "-y"]
        for cp in clip_paths:
            norm_cmd.extend(["-i", str(Path(cp).resolve())])

        norm_cmd.extend([
            "-filter_complex", fc_string,
            "-map", "[v]",
        ])
        if create_audio:
            norm_cmd.extend(["-map", "[a]"])

        norm_cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
        ])
        if create_audio:
            norm_cmd.extend([
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",
            ])
        norm_cmd.extend([
            "-movflags", "+faststart",
            str(out_path),
        ])

        try:
            run_command(norm_cmd, cancel_event=cancel, log=log)
        except MediaError as me:
            raise NormalizedConcatError(f"Normalized concat FFmpeg thất bại: {me}") from me

        if cancel and cancel.is_set():
            raise RenderCancelled("Xử lý đã bị dừng theo yêu cầu của người dùng.")

        validated = validate_intermediate_output(
            out_path,
            expected_duration=exp_total_dur if exp_total_dur > 0 else None,
            require_audio=require_audio,
        )
        return validated

    except RenderCancelled:
        if out_path.is_file():
            out_path.unlink(missing_ok=True)
        raise
    except Exception as norm_exc:
        if out_path.is_file():
            out_path.unlink(missing_ok=True)
        if isinstance(norm_exc, (NormalizedConcatError, IntermediateValidationError)):
            norm_err = norm_exc
        else:
            norm_err = NormalizedConcatError(str(norm_exc))

        if fast_error is not None:
            raise ConcatPipelineError(
                f"Cả fast concat và normalized concat đều thất bại.\nFast: {fast_error}\nNormalized: {norm_err}",
                fast_error=fast_error,
                normalized_error=norm_err,
            ) from norm_err
        else:
            raise norm_err from norm_exc


def extract_embedded_subtitle(
    video_path: str | Path,
    stream_index: int,
    output_path: str | Path,
    *,
    output_format: str = "srt",
) -> Path:
    """Extract an embedded subtitle track to file using FFmpeg."""
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_binary("ffmpeg"),
        "-y",
        "-i", str(video_path),
        "-map", f"0:{stream_index}",
        "-c:s", "srt" if output_format == "srt" else "copy",
        str(out_p),
    ]
    run_command(cmd)
    return out_p


def demux_embedded_pgs(
    video_path: str | Path,
    stream_index: int,
    output_path: str | Path,
) -> Path:
    """Demux an embedded Blu-ray PGS SUP track using FFmpeg."""
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_binary("ffmpeg"),
        "-y",
        "-i", str(video_path),
        "-map", f"0:{stream_index}",
        "-c:s", "copy",
        "-f", "data",
        str(out_p),
    ]
    run_command(cmd)
    return out_p



def probe_duration(path: str | Path) -> float:
    """Return media duration in seconds."""
    info = probe_media(path)
    return float(info.get("duration") or 0.0)


def cut_clip(
    source_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    *,
    audio_stream_index: int | None = None,
    use_gpu: bool = True,
    cancel_event: threading.Event | None = None,
    log: LogCallback | None = None,
) -> Path:
    """Cut a clip from source video accurately using bounded hybrid/hardware acceleration fallback."""
    duration = max(0.1, end_sec - start_sec)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_codec: str | None = None
    source_has_audio: bool = False
    source_audio_count: int = 0
    try:
        info = probe_media(source_path)
        video_codec = info.get("video_codec")
        source_has_audio = bool(info.get("has_audio"))
        source_audio_count = len(info.get("audio_streams", []))
    except Exception:
        pass

    has_valid_audio = (
        audio_stream_index is not None
        and source_has_audio
        and (source_audio_count == 0 or audio_stream_index < source_audio_count)
    )

    plan = get_acceleration_plan(use_gpu=use_gpu, video_codec=video_codec)
    candidates = get_fallback_candidates(plan, use_gpu=use_gpu)

    last_error: MediaError | None = None
    for decode_method, encoder_name in candidates:
        if cancel_event and cancel_event.is_set():
            raise RenderCancelled("Xử lý đã bị dừng theo yêu cầu của người dùng.")

        cmd = [find_binary("ffmpeg"), "-y"]
        if decode_method == "qsv":
            cmd.extend(["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"])
        cmd.extend(["-ss", f"{start_sec:.3f}", "-i", str(source_path), "-t", f"{duration:.3f}"])

        if decode_method == "qsv" and encoder_name != "h264_qsv":
            cmd.extend(["-vf", "hwdownload,format=nv12"])

        if has_valid_audio:
            cmd.extend(["-map", "0:v:0", "-map", f"0:a:{audio_stream_index}"])
        else:
            cmd.extend(["-map", "0:v:0", "-an"])

        enc_args = build_encoder_args(encoder_name, quality="standard", fast=True)
        cmd.extend(enc_args)
        if has_valid_audio:
            cmd.extend([
                "-c:a", "aac",
                "-b:a", "192k",
            ])
        cmd.extend([
            "-avoid_negative_ts", "1",
            str(output_path),
        ])

        try:
            run_command(cmd, cancel_event=cancel_event, log=log)
            return output_path
        except RenderCancelled:
            if output_path.is_file():
                output_path.unlink(missing_ok=True)
            raise
        except MediaError as exc:
            last_error = exc
            if output_path.is_file():
                output_path.unlink(missing_ok=True)
            if log:
                log(f"Cắt phân đoạn với {decode_method}+{encoder_name} thất bại, thử phương án tiếp theo...")

    if last_error:
        raise last_error
    raise MediaError(f"Không thể cắt phân đoạn video từ {source_path}")


def render_final_video(
    raw_video: Path,
    output_path: Path,
    *,
    srt_path: Path | None = None,
    quality: str = "high",
    use_gpu: bool = True,
    cancel_event: threading.Event | None = None,
    log: LogCallback | None = None,
    audio_codec: str = "copy",
) -> Path:
    """Render final recap video with optional SRT subtitles using bounded hybrid fallback."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_codec: str | None = None
    try:
        info = probe_media(raw_video)
        video_codec = info.get("video_codec")
    except Exception:
        video_codec = "h264"

    plan = get_acceleration_plan(use_gpu=use_gpu, video_codec=video_codec)
    candidates = get_fallback_candidates(plan, use_gpu=use_gpu)

    has_subtitles = srt_path is not None and Path(srt_path).is_file()
    sub_style = "force_style='FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2'"

    last_error: MediaError | None = None
    for decode_method, encoder_name in candidates:
        if cancel_event and cancel_event.is_set():
            raise RenderCancelled("Xử lý đã bị dừng theo yêu cầu của người dùng.")

        cmd = [find_binary("ffmpeg"), "-y"]
        if decode_method == "qsv":
            cmd.extend(["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"])
        cmd.extend(["-i", str(raw_video)])

        vf_parts: list[str] = []
        if decode_method == "qsv":
            if has_subtitles or encoder_name != "h264_qsv":
                vf_parts.append("hwdownload,format=nv12")

        if has_subtitles:
            escaped_srt = str(srt_path).replace("\\", "/").replace(":", "\\:")
            vf_parts.append(f"subtitles='{escaped_srt}':{sub_style}")

        if vf_parts:
            cmd.extend(["-vf", ",".join(vf_parts)])

        enc_args = build_encoder_args(encoder_name, quality=quality, fast=False)
        cmd.extend(enc_args)
        cmd.extend([
            "-c:a", audio_codec,
        ])
        if audio_codec != "copy":
            cmd.extend(["-b:a", "192k"])
        cmd.append(str(output_path))

        try:
            run_command(cmd, cancel_event=cancel_event, log=log)
            return output_path
        except RenderCancelled:
            if output_path.is_file():
                output_path.unlink(missing_ok=True)
            raise
        except MediaError as exc:
            last_error = exc
            if output_path.is_file():
                output_path.unlink(missing_ok=True)
            if log:
                log(f"Mã hóa video cuối với {decode_method}+{encoder_name} thất bại, thử phương án tiếp theo...")

    if last_error:
        raise last_error
    raise MediaError(f"Không thể mã hóa video đầu ra {output_path}")


def format_srt_time(seconds: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm."""
    millis = int(round(seconds * 1000))
    hours = millis // 3600000
    millis %= 3600000
    minutes = millis // 60000
    millis %= 60000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt_file(rows: list[tuple[float, float, str]], output_path: Path) -> Path:
    """Write subtitle rows (start_sec, end_sec, text) to SRT file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for idx, (start, end, text) in enumerate(rows, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text.strip())
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
