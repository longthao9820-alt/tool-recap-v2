"""Media processing with FFmpeg: probing, clipping, audio mixing, subtitle burning, and safe cancellation."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

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
    MediaProbeResult,
    SubtitleStreamInfo,
    VideoStreamInfo,
    normalize_language_code,
)



LogCallback = Callable[[str], None]


class MediaError(RuntimeError):
    pass


class RenderCancelled(MediaError):
    pass


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
            fps = 0.0
            r_fps = s.get("r_frame_rate", "")
            if "/" in r_fps:
                parts = r_fps.split("/")
                try:
                    num, den = float(parts[0]), float(parts[1])
                    fps = num / den if den != 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    fps = 0.0
            duration = float(s.get("duration") or 0.0)
            bitrate = int(s.get("bit_rate") or 0)
            width = int(s.get("width") or 1920)
            height = int(s.get("height") or 1080)

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
                )
            )
            v_counter += 1

        elif codec_type == "audio":
            channels = int(s.get("channels") or 2)
            channel_layout = str(s.get("channel_layout") or "stereo")
            bitrate = int(s.get("bit_rate") or 0)
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
    )


def probe_media(path: str | Path) -> dict:
    """Extract stream and format information using ffprobe, preserving legacy dict format with typed enrichments."""
    typed_info = probe_typed_media(path)
    res = typed_info.to_dict()
    # Add convenience top-level fields for audio warning
    res["audio_warning"] = typed_info.selected_audio.warning if typed_info.selected_audio and typed_info.selected_audio.has_warning else None
    return res


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
    try:
        info = probe_media(source_path)
        video_codec = info.get("video_codec")
    except Exception:
        pass

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

        if audio_stream_index is not None:
            cmd.extend(["-map", "0:v:0", "-map", f"0:a:{audio_stream_index}"])

        enc_args = build_encoder_args(encoder_name, quality="standard", fast=True)
        cmd.extend(enc_args)
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "192k",
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
