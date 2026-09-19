"""Tests for GPU encoder detection, video encode args, hybrid acceleration, and SRT subtitle formatting."""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from toolrecap_v2.gpu import (
    AccelerationPlan,
    DecoderStatus,
    EncoderStatus,
    HardwareCapabilities,
    build_encoder_args,
    detect_gpu_encoder,
    get_acceleration_plan,
    get_fallback_candidates,
    select_decoder,
    select_encoder,
    video_encode_args,
)
from toolrecap_v2.media import (
    MediaError,
    RenderCancelled,
    cut_clip,
    format_srt_time,
    render_final_video,
    write_srt_file,
)


def test_detect_gpu_encoder() -> None:
    status = detect_gpu_encoder()
    assert status.encoder in {"h264_nvenc", "h264_amf", "h264_qsv", "libx264"}
    assert status.label


def test_video_encode_args_presets() -> None:
    # High quality
    args_high, status_high = video_encode_args("high", use_gpu=False)
    assert "-c:v" in args_high
    assert "libx264" in args_high
    assert "-crf" in args_high

    # Standard quality
    args_std, _ = video_encode_args("standard", use_gpu=False)
    assert "-c:v" in args_std
    assert "libx264" in args_std

    # Source quality
    args_src, _ = video_encode_args("source", use_gpu=False)
    assert "-c:v" in args_src
    assert "libx264" in args_src


def test_format_srt_time() -> None:
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(1.5) == "00:00:01,500"
    assert format_srt_time(65.25) == "00:01:05,250"
    assert format_srt_time(3661.123) == "01:01:01,123"


def test_write_srt_file(tmp_path: Path) -> None:
    rows = [
        (0.0, 3.5, "First narration line."),
        (4.0, 8.2, "Second narration line."),
    ]
    srt_path = tmp_path / "test.srt"
    write_srt_file(rows, srt_path)

    assert srt_path.is_file()
    content = srt_path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:03,500" in content
    assert "First narration line." in content
    assert "00:00:04,000 --> 00:00:08,200" in content
    assert "Second narration line." in content


def test_mock_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    caps = HardwareCapabilities(
        qsv_decode=True,
        nvenc_encode=True,
        amf_encode=False,
        qsv_encode=True,
        gpu_names=("Intel UHD Graphics 770", "NVIDIA GeForce RTX 3060"),
        primary_gpu="NVIDIA GeForce RTX 3060",
    )
    monkeypatch.setattr("toolrecap_v2.gpu.detect_hardware_capabilities", lambda **kw: caps)

    plan = get_acceleration_plan(use_gpu=True)
    assert plan.is_hybrid is True
    assert plan.decoder.method == "qsv"
    assert plan.decoder.available is True
    assert plan.encoder.encoder == "h264_nvenc"
    assert plan.encoder.available is True

    candidates = get_fallback_candidates(plan, use_gpu=True)
    assert candidates == [
        ("qsv", "h264_nvenc"),
        ("cpu", "h264_nvenc"),
        ("cpu", "h264_qsv"),
        ("cpu", "libx264"),
    ]


def test_mock_intel_only(monkeypatch: pytest.MonkeyPatch) -> None:
    caps = HardwareCapabilities(
        qsv_decode=True,
        nvenc_encode=False,
        amf_encode=False,
        qsv_encode=True,
        gpu_names=("Intel Iris Xe Graphics",),
        primary_gpu="Intel Iris Xe Graphics",
    )
    monkeypatch.setattr("toolrecap_v2.gpu.detect_hardware_capabilities", lambda **kw: caps)

    plan = get_acceleration_plan(use_gpu=True)
    assert plan.is_hybrid is False
    assert plan.decoder.method == "qsv"
    assert plan.encoder.encoder == "h264_qsv"

    candidates = get_fallback_candidates(plan, use_gpu=True)
    assert candidates == [
        ("qsv", "h264_qsv"),
        ("cpu", "h264_qsv"),
        ("cpu", "libx264"),
    ]


def test_mock_nvidia_only(monkeypatch: pytest.MonkeyPatch) -> None:
    caps = HardwareCapabilities(
        qsv_decode=False,
        nvenc_encode=True,
        amf_encode=False,
        qsv_encode=False,
        gpu_names=("NVIDIA GeForce RTX 4070",),
        primary_gpu="NVIDIA GeForce RTX 4070",
    )
    monkeypatch.setattr("toolrecap_v2.gpu.detect_hardware_capabilities", lambda **kw: caps)

    plan = get_acceleration_plan(use_gpu=True)
    assert plan.is_hybrid is False
    assert plan.decoder.method == "cpu"
    assert plan.encoder.encoder == "h264_nvenc"

    candidates = get_fallback_candidates(plan, use_gpu=True)
    assert candidates == [
        ("cpu", "h264_nvenc"),
        ("cpu", "libx264"),
    ]


def test_mock_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    caps = HardwareCapabilities(
        qsv_decode=False,
        nvenc_encode=False,
        amf_encode=False,
        qsv_encode=False,
        gpu_names=(),
        primary_gpu="CPU",
    )
    monkeypatch.setattr("toolrecap_v2.gpu.detect_hardware_capabilities", lambda **kw: caps)

    plan = get_acceleration_plan(use_gpu=True)
    assert plan.decoder.method == "cpu"
    assert plan.encoder.encoder == "libx264"
    candidates = get_fallback_candidates(plan, use_gpu=True)
    assert candidates == [("cpu", "libx264")]

    plan_disabled = get_acceleration_plan(use_gpu=False)
    assert plan_disabled.decoder.method == "cpu"
    assert plan_disabled.encoder.encoder == "libx264"
    candidates_disabled = get_fallback_candidates(plan_disabled, use_gpu=False)
    assert candidates_disabled == [("cpu", "libx264")]


def test_video_codec_incompatible_qsv(monkeypatch: pytest.MonkeyPatch) -> None:
    caps = HardwareCapabilities(
        qsv_decode=True,
        nvenc_encode=True,
        amf_encode=False,
        qsv_encode=True,
        gpu_names=("Intel UHD Graphics", "NVIDIA RTX 3060"),
        primary_gpu="NVIDIA RTX 3060",
    )
    monkeypatch.setattr("toolrecap_v2.gpu.detect_hardware_capabilities", lambda **kw: caps)

    plan_prores = get_acceleration_plan(use_gpu=True, video_codec="prores")
    assert plan_prores.decoder.method == "cpu"
    assert "không hỗ trợ" in plan_prores.decoder.reason

    plan_h264 = get_acceleration_plan(use_gpu=True, video_codec="h264")
    assert plan_h264.decoder.method == "qsv"


def test_qsv_decode_failure_preserving_nvenc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    caps = HardwareCapabilities(
        qsv_decode=True,
        nvenc_encode=True,
        amf_encode=False,
        qsv_encode=True,
        gpu_names=("Intel UHD Graphics", "NVIDIA RTX 3060"),
        primary_gpu="NVIDIA RTX 3060",
    )
    monkeypatch.setattr("toolrecap_v2.gpu.detect_hardware_capabilities", lambda **kw: caps)

    commands_run: list[list[str]] = []

    def mock_run_command(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands_run.append(cmd)
        if "-hwaccel" in cmd:
            raise MediaError("QSV decode hardware initialization failed")
        # Software decode with NVENC succeeds
        out_file = Path(cmd[-1])
        out_file.write_text("dummy video content", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("toolrecap_v2.media.run_command", mock_run_command)
    monkeypatch.setattr("toolrecap_v2.media.find_binary", lambda name: "ffmpeg")

    raw_video = tmp_path / "raw.mp4"
    raw_video.write_text("raw", encoding="utf-8")
    out_video = tmp_path / "out.mp4"

    res = render_final_video(raw_video, out_video, use_gpu=True)
    assert res == out_video
    assert out_video.is_file()

    assert len(commands_run) == 2
    # Attempt 1 had QSV decode + NVENC encode
    cmd1 = commands_run[0]
    assert "-hwaccel" in cmd1 and "qsv" in cmd1
    assert "-c:v" in cmd1 and "h264_nvenc" in cmd1

    # Attempt 2 had CPU decode + NVENC encode preserved
    cmd2 = commands_run[1]
    assert "-hwaccel" not in cmd2
    assert "-c:v" in cmd2 and "h264_nvenc" in cmd2


def test_encoder_failure_cpu_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    caps = HardwareCapabilities(
        qsv_decode=False,
        nvenc_encode=True,
        amf_encode=False,
        qsv_encode=False,
        gpu_names=("NVIDIA RTX 3060",),
        primary_gpu="NVIDIA RTX 3060",
    )
    monkeypatch.setattr("toolrecap_v2.gpu.detect_hardware_capabilities", lambda **kw: caps)

    commands_run: list[list[str]] = []

    def mock_run_command(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands_run.append(cmd)
        if "h264_nvenc" in cmd:
            raise MediaError("NVENC device out of memory")
        out_file = Path(cmd[-1])
        out_file.write_text("cpu encoded", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("toolrecap_v2.media.run_command", mock_run_command)
    monkeypatch.setattr("toolrecap_v2.media.find_binary", lambda name: "ffmpeg")

    raw_video = tmp_path / "raw.mp4"
    raw_video.write_text("raw", encoding="utf-8")
    out_video = tmp_path / "out.mp4"

    res = render_final_video(raw_video, out_video, use_gpu=True)
    assert res == out_video
    assert out_video.is_file()

    assert len(commands_run) == 2
    # Attempt 1: NVENC
    assert "h264_nvenc" in commands_run[0]
    # Attempt 2: libx264
    assert "libx264" in commands_run[1]


def test_argument_ordering(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    caps = HardwareCapabilities(
        qsv_decode=True,
        nvenc_encode=True,
        amf_encode=False,
        qsv_encode=True,
        gpu_names=("Intel UHD", "NVIDIA RTX"),
        primary_gpu="NVIDIA RTX",
    )
    monkeypatch.setattr("toolrecap_v2.gpu.detect_hardware_capabilities", lambda **kw: caps)

    captured_cmds: list[list[str]] = []

    def mock_run_command(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        out_file = Path(cmd[-1])
        out_file.write_text("done", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("toolrecap_v2.media.run_command", mock_run_command)
    monkeypatch.setattr("toolrecap_v2.media.find_binary", lambda name: "ffmpeg")

    raw_video = tmp_path / "raw.mp4"
    raw_video.write_text("raw", encoding="utf-8")
    srt_file = tmp_path / "subs.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    out_video = tmp_path / "out.mp4"

    # Test render_final_video with subtitles
    render_final_video(raw_video, out_video, srt_path=srt_file, use_gpu=True)
    cmd = captured_cmds[0]

    # -hwaccel must be before -i
    hwaccel_idx = cmd.index("-hwaccel")
    i_idx = cmd.index("-i")
    assert hwaccel_idx < i_idx

    # -i must be before -c:v
    cv_idx = cmd.index("-c:v")
    assert i_idx < cv_idx

    # Subtitles filter must include hwdownload when qsv decode is active
    vf_idx = cmd.index("-vf")
    vf_val = cmd[vf_idx + 1]
    assert "hwdownload,format=nv12" in vf_val
    assert "subtitles=" in vf_val

    # Test cut_clip
    captured_cmds.clear()
    cut_out = tmp_path / "cut.mp4"
    cut_clip(raw_video, cut_out, 0.0, 5.0, use_gpu=True)
    cut_cmd = captured_cmds[0]
    cut_hw_idx = cut_cmd.index("-hwaccel")
    cut_i_idx = cut_cmd.index("-i")
    cut_cv_idx = cut_cmd.index("-c:v")
    assert cut_hw_idx < cut_i_idx
    assert cut_i_idx < cut_cv_idx
    assert "hwdownload,format=nv12" in cut_cmd[cut_cmd.index("-vf") + 1]


def test_finite_attempts_and_cancellation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    caps = HardwareCapabilities(
        qsv_decode=True,
        nvenc_encode=True,
        amf_encode=False,
        qsv_encode=True,
        gpu_names=("Intel UHD", "NVIDIA RTX"),
        primary_gpu="NVIDIA RTX",
    )
    monkeypatch.setattr("toolrecap_v2.gpu.detect_hardware_capabilities", lambda **kw: caps)

    attempts = 0

    def mock_run_always_fail(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        raise MediaError(f"Attempt {attempts} failed")

    monkeypatch.setattr("toolrecap_v2.media.run_command", mock_run_always_fail)
    monkeypatch.setattr("toolrecap_v2.media.find_binary", lambda name: "ffmpeg")

    raw_video = tmp_path / "raw.mp4"
    raw_video.write_text("raw", encoding="utf-8")
    out_video = tmp_path / "out.mp4"

    # All attempts fail: must be exactly 4 attempts (bounded) and raise MediaError
    with pytest.raises(MediaError) as exc_info:
        render_final_video(raw_video, out_video, use_gpu=True)
    assert attempts == 4
    assert "Attempt 4 failed" in str(exc_info.value)

    # Cancellation test: must raise RenderCancelled on first attempt and NOT retry
    cancel_attempts = 0
    cancel_evt = threading.Event()

    def mock_run_cancelled(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal cancel_attempts
        cancel_attempts += 1
        raise RenderCancelled("User cancelled render")

    monkeypatch.setattr("toolrecap_v2.media.run_command", mock_run_cancelled)

    with pytest.raises(RenderCancelled):
        render_final_video(raw_video, out_video, use_gpu=True, cancel_event=cancel_evt)
    assert cancel_attempts == 1

