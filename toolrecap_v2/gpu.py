"""GPU and FFmpeg encoder detection and configuration for ToolRecap V2."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .paths import application_root


def bundled_binary(name: str) -> Path | None:
    """Find binary in application runtime or system PATH."""
    suffix = ".exe" if sys.platform == "win32" else ""
    root = application_root()

    candidates = [
        root / "runtime" / "ffmpeg" / "bin" / f"{name}{suffix}",
        root / "runtime" / "ffmpeg" / f"{name}{suffix}",
        root / f"{name}{suffix}",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand

    external = shutil.which(name)
    if external:
        return Path(external).resolve()
    return None


@dataclass(frozen=True)
class DecoderStatus:
    available: bool
    method: str  # "qsv" or "cpu"
    label: str
    supported_codecs: tuple[str, ...] = ("h264", "hevc", "h265", "vp9", "av1", "mpeg2video")
    reason: str = ""


@dataclass(frozen=True)
class EncoderStatus:
    available: bool
    gpu_name: str
    encoder: str  # "h264_nvenc", "h264_amf", "h264_qsv", "libx264"
    label: str
    reason: str = ""


@dataclass(frozen=True)
class HardwareCapabilities:
    qsv_decode: bool
    nvenc_encode: bool
    amf_encode: bool
    qsv_encode: bool
    gpu_names: tuple[str, ...]
    primary_gpu: str


@dataclass(frozen=True)
class AccelerationPlan:
    decoder: DecoderStatus
    encoder: EncoderStatus
    capabilities: HardwareCapabilities

    @property
    def is_hybrid(self) -> bool:
        return (
            self.decoder.method == "qsv"
            and self.encoder.encoder != "h264_qsv"
            and self.encoder.encoder != "libx264"
        )

    @property
    def summary_label(self) -> str:
        return f"Giải mã: {self.decoder.label} | Mã hóa: {self.encoder.label}"


_cached_status: EncoderStatus | None = None
_cached_capabilities: HardwareCapabilities | None = None


def _run_text(command: list[str]) -> str:
    try:
        res = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return res.stdout + res.stderr
    except (OSError, subprocess.SubprocessError):
        return ""


def get_all_gpu_names() -> list[str]:
    names: list[str] = []
    # 1. Windows PowerShell check for all display adapters
    if sys.platform == "win32":
        ps = shutil.which("powershell")
        if ps:
            cmd = [
                ps,
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ]
            out = _run_text(cmd)
            for line in out.splitlines():
                clean = line.strip()
                if clean and "virtual" not in clean.lower() and clean not in names:
                    names.append(clean)

    # 2. nvidia-smi check
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        out = _run_text([nvidia, "--query-gpu=name", "--format=csv,noheader"])
        for line in out.splitlines():
            clean = line.strip()
            if clean and clean not in names:
                names.append(clean)

    if not names:
        names.append("GPU chung")
    return names


def _gpu_name() -> str:
    names = get_all_gpu_names()
    return names[0] if names else "GPU chung"


def _test_encoder(ffmpeg: Path, encoder: str) -> bool:
    devnull = "NUL" if sys.platform == "win32" else "/dev/null"
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=640x360:d=0.15:r=25",
        "-frames:v",
        "2",
        "-c:v",
        encoder,
        "-f",
        "null",
        devnull,
    ]
    try:
        res = subprocess.run(
            command,
            capture_output=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _test_qsv_decode(ffmpeg: Path) -> bool:
    devnull = "NUL" if sys.platform == "win32" else "/dev/null"
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-init_hw_device",
        "qsv",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=0.1:r=25",
        "-frames:v",
        "1",
        "-f",
        "null",
        devnull,
    ]
    try:
        res = subprocess.run(
            command,
            capture_output=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def detect_hardware_capabilities(*, refresh: bool = False) -> HardwareCapabilities:
    """Probe FFmpeg runtime capabilities for QSV decode, NVENC/AMF/QSV encode, and CPU."""
    global _cached_capabilities
    if _cached_capabilities is not None and not refresh:
        return _cached_capabilities

    ffmpeg = bundled_binary("ffmpeg")
    if not ffmpeg:
        _cached_capabilities = HardwareCapabilities(
            qsv_decode=False,
            nvenc_encode=False,
            amf_encode=False,
            qsv_encode=False,
            gpu_names=(),
            primary_gpu="Không có FFmpeg",
        )
        return _cached_capabilities

    encoders_output = _run_text([str(ffmpeg), "-hide_banner", "-encoders"])
    hwaccels_output = _run_text([str(ffmpeg), "-hide_banner", "-hwaccels"])

    nvenc_ok = ("h264_nvenc" in encoders_output) and _test_encoder(ffmpeg, "h264_nvenc")
    amf_ok = ("h264_amf" in encoders_output) and _test_encoder(ffmpeg, "h264_amf")
    qsv_enc_ok = ("h264_qsv" in encoders_output) and _test_encoder(ffmpeg, "h264_qsv")
    qsv_dec_ok = ("qsv" in hwaccels_output) and _test_qsv_decode(ffmpeg)

    names = get_all_gpu_names()
    primary = names[0] if names else "CPU"

    _cached_capabilities = HardwareCapabilities(
        qsv_decode=qsv_dec_ok,
        nvenc_encode=nvenc_ok,
        amf_encode=amf_ok,
        qsv_encode=qsv_enc_ok,
        gpu_names=tuple(names),
        primary_gpu=primary,
    )
    return _cached_capabilities


def select_decoder(
    caps: HardwareCapabilities,
    *,
    use_gpu: bool = True,
    video_codec: str | None = None,
) -> DecoderStatus:
    if not use_gpu:
        return DecoderStatus(
            available=False,
            method="cpu",
            label="CPU (Phần mềm)",
            reason="Tăng tốc GPU bị tắt trong cài đặt.",
        )
    if not caps.qsv_decode:
        return DecoderStatus(
            available=False,
            method="cpu",
            label="CPU (Phần mềm)",
            reason="Intel QSV decode không khả dụng trên phần cứng này.",
        )

    supported = ("h264", "hevc", "h265", "vp9", "av1", "mpeg2video")
    if video_codec and video_codec.lower() not in supported:
        return DecoderStatus(
            available=False,
            method="cpu",
            label="CPU (Phần mềm)",
            reason=f"Codec {video_codec} không hỗ trợ giải mã phần cứng QSV.",
        )

    return DecoderStatus(
        available=True,
        method="qsv",
        label="Intel QSV",
        reason="",
    )


def select_encoder(
    caps: HardwareCapabilities,
    *,
    use_gpu: bool = True,
) -> EncoderStatus:
    if not use_gpu:
        return EncoderStatus(
            available=False,
            gpu_name="CPU",
            encoder="libx264",
            label="CPU (x264)",
            reason="Tăng tốc GPU bị tắt trong cài đặt.",
        )

    if caps.nvenc_encode:
        return EncoderStatus(
            available=True,
            gpu_name=caps.primary_gpu,
            encoder="h264_nvenc",
            label="NVIDIA NVENC",
            reason="",
        )
    if caps.amf_encode:
        return EncoderStatus(
            available=True,
            gpu_name=caps.primary_gpu,
            encoder="h264_amf",
            label="AMD AMF",
            reason="",
        )
    if caps.qsv_encode:
        return EncoderStatus(
            available=True,
            gpu_name=caps.primary_gpu,
            encoder="h264_qsv",
            label="Intel Quick Sync",
            reason="",
        )

    return EncoderStatus(
        available=False,
        gpu_name=caps.primary_gpu,
        encoder="libx264",
        label="CPU (x264)",
        reason="Không tìm thấy phần cứng tương thích; dùng mã hóa CPU chất lượng cao.",
    )


def get_acceleration_plan(
    *,
    use_gpu: bool = True,
    video_codec: str | None = None,
    refresh: bool = False,
) -> AccelerationPlan:
    caps = detect_hardware_capabilities(refresh=refresh)
    decoder = select_decoder(caps, use_gpu=use_gpu, video_codec=video_codec)
    encoder = select_encoder(caps, use_gpu=use_gpu)
    return AccelerationPlan(decoder=decoder, encoder=encoder, capabilities=caps)


def get_fallback_candidates(
    plan: AccelerationPlan,
    *,
    use_gpu: bool = True,
) -> list[tuple[str, str]]:
    """Return ordered list of (decode_method, encoder_name) candidates for bounded fallback.

    Order:
    1. (selected_decode, selected_encode) (e.g. qsv + nvenc in hybrid)
    2. If hw decode was used: (cpu, selected_encode) (decode failed, retry software decode preserving encoder)
    3. If hw encoder was used: (cpu, next_hw_encoder) for each alternative hw encoder
    4. (cpu, libx264) (eventual CPU fallback)
    """
    if not use_gpu:
        return [("cpu", "libx264")]

    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(dec: str, enc: str) -> None:
        pair = (dec, enc)
        if pair not in seen:
            seen.add(pair)
            candidates.append(pair)

    # 1. Primary selected attempt
    add(plan.decoder.method, plan.encoder.encoder)

    # 2. If primary decode was hardware, retry with software decode + same encoder
    if plan.decoder.method != "cpu":
        add("cpu", plan.encoder.encoder)

    # 3. If primary encoder was hardware, prepare next candidate hardware encoders with cpu decode
    caps = plan.capabilities
    hw_order: list[str] = []
    if caps.nvenc_encode:
        hw_order.append("h264_nvenc")
    if caps.amf_encode:
        hw_order.append("h264_amf")
    if caps.qsv_encode:
        hw_order.append("h264_qsv")

    for enc in hw_order:
        if enc != plan.encoder.encoder:
            add("cpu", enc)

    # 4. Final CPU fallback
    add("cpu", "libx264")

    return candidates


def build_encoder_args(encoder: str, quality: str = "high", *, fast: bool = False) -> list[str]:
    """Return FFmpeg video encoding arguments for a specific encoder and quality."""
    if encoder == "h264_nvenc":
        if fast:
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull", "-cq", "20"]
        cq, preset = {
            "standard": (23, "p4"),
            "high": (19, "p5"),
            "source": (17, "p6"),
        }.get(quality, (19, "p5"))
        return [
            "-c:v", "h264_nvenc",
            "-preset", preset,
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(cq),
            "-b:v", "0",
        ]

    if encoder == "h264_amf":
        if fast:
            return ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp_i", "20", "-qp_p", "20"]
        qp = {"standard": "23", "high": "19", "source": "17"}.get(quality, "19")
        return [
            "-c:v", "h264_amf",
            "-quality", "quality",
            "-rc", "cqp",
            "-qp_i", qp,
            "-qp_p", qp,
        ]

    if encoder == "h264_qsv":
        if fast:
            return ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "20"]
        q = {"standard": "23", "high": "19", "source": "17"}.get(quality, "19")
        return [
            "-c:v", "h264_qsv",
            "-preset", "medium",
            "-global_quality", q,
        ]

    # CPU libx264 fallback
    if fast:
        return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p"]
    crf, preset = {
        "standard": (23, "veryfast"),
        "high": (19, "medium"),
        "source": (17, "slow"),
    }.get(quality, (19, "medium"))
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]


def detect_gpu_encoder(*, refresh: bool = False) -> EncoderStatus:
    """Detect available hardware video encoder (NVENC, AMF, QSV) or fall back to CPU libx264."""
    global _cached_status
    if _cached_status is not None and not refresh:
        return _cached_status
    plan = get_acceleration_plan(use_gpu=True, refresh=refresh)
    _cached_status = plan.encoder
    return _cached_status


def video_encode_args(quality: str, *, use_gpu: bool = True) -> tuple[list[str], EncoderStatus]:
    """Return FFmpeg video encoding arguments and active encoder status."""
    caps = detect_hardware_capabilities()
    status = select_encoder(caps, use_gpu=use_gpu)
    args = build_encoder_args(status.encoder, quality=quality, fast=False)
    return args, status
