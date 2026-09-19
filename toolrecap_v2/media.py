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


def probe_media(path: str | Path) -> dict:
    """Extract stream and format information using ffprobe."""
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
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = float(fmt.get("duration") or 0.0)
    if duration <= 0.0 and video_stream:
        duration = float(video_stream.get("duration") or 0.0)

    width = int(video_stream.get("width") or 1920) if video_stream else 1920
    height = int(video_stream.get("height") or 1080) if video_stream else 1080

    return {
        "path": str(media_path),
        "duration": duration,
        "width": width,
        "height": height,
        "has_video": video_stream is not None,
        "has_audio": audio_stream is not None,
        "video_codec": video_stream.get("codec_name") if video_stream else None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
    }


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
            "-c:a", "aac",
            "-b:a", "192k",
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
