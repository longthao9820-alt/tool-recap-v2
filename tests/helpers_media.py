"""Media test helpers for robust concat r9 testing.

Provides fast, tiny synthetic clip generation and pure mock probe generation
while avoiding FFmpeg layout command pitfalls.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from toolrecap_v2.media import (
    AudioSelectionResult,
    AudioStreamInfo,
    MediaProbeResult,
    VideoStreamInfo,
    find_binary,
)


def create_synthetic_test_clip(
    output_path: Path | str,
    *,
    duration: float = 0.4,
    width: int = 160,
    height: int = 120,
    fps: int | float = 25,
    has_audio: bool = True,
    channels: int = 2,
    channel_layout: str = "stereo",
    sample_rate: int = 48000,
    audio_codec: str = "aac",
    video_codec: str = "libx264",
    tone_freq: float = 1000.0,
    pix_fmt: str = "yuv420p",
) -> Path:
    """Create a minimal real MP4 video clip for fast testing.

    Avoids unsupported channel layout pitfalls across FFmpeg builds by mapping
    known layouts cleanly to corresponding channel counts and parameters.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_binary("ffmpeg")

    dur_str = f"{duration:.3f}"
    cmd = [
        ffmpeg,
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={dur_str}:size={width}x{height}:rate={fps}",
    ]

    if has_audio:
        cmd.extend([
            "-f", "lavfi",
            "-i", f"sine=frequency={tone_freq}:duration={dur_str}",
            "-c:v", video_codec,
            "-preset", "ultrafast",
            "-pix_fmt", pix_fmt,
            "-c:a", audio_codec,
            "-ar", str(sample_rate),
        ])
        norm_layout = channel_layout.strip().lower()
        if norm_layout in ("stereo", "2") or channels == 2:
            cmd.extend(["-channel_layout", "stereo", "-ac", "2"])
        elif norm_layout in ("mono", "1") or channels == 1:
            cmd.extend(["-channel_layout", "mono", "-ac", "1"])
        elif norm_layout in ("5.1", "5.1(side)") or channels == 6:
            cmd.extend(["-channel_layout", channel_layout, "-ac", "6"])
        elif channel_layout:
            cmd.extend(["-channel_layout", channel_layout, "-ac", str(channels)])
        else:
            cmd.extend(["-ac", str(channels)])
    else:
        cmd.extend([
            "-c:v", video_codec,
            "-preset", "ultrafast",
            "-pix_fmt", pix_fmt,
            "-an",
        ])

    cmd.extend([
        "-avoid_negative_ts", "1",
        str(out_p),
    ])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg synthetic clip generation failed ({proc.returncode}): {proc.stderr}")

    return out_p


def create_mock_probe(
    *,
    path: str = "mock.mp4",
    duration: float = 1.0,
    width: int = 1920,
    height: int = 1080,
    fps: float = 24.0,
    fps_rational: str = "24/1",
    video_codec: str = "h264",
    pix_fmt: str = "yuv420p",
    profile: str = "High",
    time_base: str = "1/1000",
    sar: str = "1:1",
    has_audio: bool = True,
    audio_codec: str = "aac",
    sample_rate: int = 48000,
    sample_fmt: str = "fltp",
    channels: int = 2,
    channel_layout: str = "stereo",
    audio_time_base: str = "1/48000",
    start_time: float = 0.0,
) -> MediaProbeResult:
    """Generate a typed MediaProbeResult for pure matrix testing without disk I/O."""
    v_streams = [
        VideoStreamInfo(
            index=0,
            video_index=0,
            codec=video_codec,
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            bitrate=2000000,
            fps_rational=fps_rational,
            pix_fmt=pix_fmt,
            profile=profile,
            time_base=time_base,
            sar=sar,
            start_time=start_time,
        )
    ]
    a_streams = []
    sel_audio = None
    if has_audio:
        a_info = AudioStreamInfo(
            index=1,
            audio_index=0,
            codec=audio_codec,
            channels=channels,
            channel_layout=channel_layout,
            sample_rate=sample_rate,
            sample_fmt=sample_fmt,
            language="eng",
            bitrate=192000,
            profile="LC",
            time_base=audio_time_base,
            start_time=start_time,
        )
        a_streams.append(a_info)
        sel_audio = AudioSelectionResult(
            selected_stream=a_info,
            reason="mock_default",
        )

    return MediaProbeResult(
        path=path,
        duration=duration,
        width=width,
        height=height,
        has_video=True,
        has_audio=has_audio,
        video_codec=video_codec,
        audio_codec=audio_codec if has_audio else None,
        video_streams=v_streams,
        audio_streams=a_streams,
        subtitle_streams=[],
        selected_audio=sel_audio,
        start_time=start_time,
    )
