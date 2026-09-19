"""Audio mix filter-plan module for ToolRecap V2.

Builds deterministic FFmpeg filter graphs and commands to mix Original audio
with commentary narration, featuring smooth sidechain compression (auto-ducking),
loudness normalization (EBU R128 loudnorm), and peak limiting.

Designed as an independent module with explicit dataclass settings so subsequent
renderers and media runners can invoke it directly without tight coupling.

Strict invariant:
All comments, identifiers, and documentation strictly refer to Original audio.
"""
from __future__ import annotations

import array
import math
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class AudioMixError(RuntimeError):
    """Base error for audio mixing operations."""
    pass


class AudioValidationError(AudioMixError):
    """Raised when mixed audio fails quality, silence, or peak constraints."""
    pass


@dataclass(frozen=True)
class AudioMixSettings:
    """Explicit configuration settings for mixing Original audio and commentary narration.

    Attributes:
        original_gain_db: Gain adjustment for Original audio in dB (0.0 = unity).
        commentary_gain_db: Gain adjustment for commentary narration in dB (0.0 = unity).
        auto_duck: Whether to automatically duck Original audio when commentary speaks.
        amount: Ducking attenuation depth in dB (e.g. -12.0 dB).
        target_lufs: EBU R128 integrated loudness target in LUFS (e.g. -16.0 LUFS).
        true_peak: Maximum true peak limit in dBFS (e.g. -1.5 dBFS).
        attack_ms: Sidechain compression attack time in milliseconds (e.g. 20.0 ms).
        release_ms: Sidechain compression release time in milliseconds (e.g. 250.0 ms).
        sample_rate: Output audio sample rate in Hz (default 48000 Hz).
        channels: Output audio channel count (default 2 for stereo).
        duration_mode: Duration handling for mixed stream: 'longest', 'first', or 'shortest'.
    """
    original_gain_db: float = 0.0
    commentary_gain_db: float = 0.0
    auto_duck: bool = True
    amount: float = -12.0
    target_lufs: float = -16.0
    true_peak: float = -1.5
    attack_ms: float = 20.0
    release_ms: float = 250.0
    sample_rate: int = 48000
    channels: int = 2
    duration_mode: str = "longest"

    @property
    def duck_amount(self) -> float:
        return self.amount

    @property
    def duck_amount_db(self) -> float:
        return self.amount

    @property
    def true_peak_db(self) -> float:
        return self.true_peak

    @property
    def true_peak_linear(self) -> float:
        """Linear amplitude corresponding to true_peak in dBFS."""
        val = 10.0 ** (self.true_peak / 20.0)
        return min(1.0, max(0.0625, val))


@dataclass(frozen=True)
class AudioMixPlan:
    """Generated audio mix plan containing filter graph, inputs, and command representation."""
    settings: AudioMixSettings
    has_original: bool
    has_commentary: bool
    is_original_dialogue_only: bool
    filter_graph: str
    input_files: tuple[Path, ...]
    output_file: Path

    def build_command(self, ffmpeg_bin: str = "ffmpeg") -> list[str]:
        """Construct the complete FFmpeg CLI command for this mix plan."""
        cmd: list[str] = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"]
        for inp in self.input_files:
            cmd.extend(["-i", str(inp)])

        cmd.extend(["-filter_complex", self.filter_graph, "-map", "[out]"])

        out_str = str(self.output_file).lower()
        if out_str.endswith((".aac", ".m4a", ".mp4")):
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-c:a", "pcm_s16le"])

        cmd.extend([
            "-ar", str(self.settings.sample_rate),
            "-ac", str(self.settings.channels),
            str(self.output_file),
        ])
        return cmd


def build_audio_mix_filter_graph(
    settings: AudioMixSettings,
    *,
    has_original: bool = True,
    has_commentary: bool = True,
    is_original_dialogue_only: bool = False,
) -> str:
    """Generate a real FFmpeg audio filter graph based on settings and available audio streams.

    Scenarios:
    1. Both Original audio and Commentary audio present:
       - Smooth sidechain compression (auto-ducking) on Original audio driven by commentary narration envelope.
       - If auto_duck is False: directly amix Original audio + commentary voice.
       - Followed by EBU R128 loudness normalization and true peak limiter (no clipping).
    2. Original dialogue / no narration:
       - Preserves Original audio without ducking or amix.
    3. Missing Original audio:
       - Accounts for missing Original audio; processes commentary narration directly through normalization and limiter.
    """
    if not has_original and not has_commentary:
        raise AudioMixError("Cả âm thanh gốc (Original audio) và âm thanh thuyết minh (Commentary) đều không có.")

    limiter_val = settings.true_peak_linear
    sr = settings.sample_rate
    layout = "stereo" if settings.channels == 2 else ("mono" if settings.channels == 1 else f"{settings.channels}c")

    # Scenario 2: Original dialogue only or no commentary narration -> Preserve Original audio
    if not has_commentary or is_original_dialogue_only:
        orig_gain_filter = f",volume={settings.original_gain_db:.2f}dB" if abs(settings.original_gain_db) > 1e-4 else ""
        return (
            f"[0:a]aformat=channel_layouts={layout}:sample_rates={sr}"
            f"{orig_gain_filter},"
            f"alimiter=limit={limiter_val:.4f}:attack=5:release=50:level=disabled[out]"
        )

    # Scenario 3: Missing Original audio -> Process commentary narration only
    if not has_original:
        comm_gain_filter = f",volume={settings.commentary_gain_db:.2f}dB" if abs(settings.commentary_gain_db) > 1e-4 else ""
        return (
            f"[0:a]aformat=channel_layouts={layout}:sample_rates={sr}"
            f"{comm_gain_filter},"
            f"loudnorm=I={settings.target_lufs:.1f}:TP={settings.true_peak:.1f}:LRA=11,"
            f"alimiter=limit={limiter_val:.4f}:attack=5:release=50:level=disabled[out]"
        )

    # Scenario 1: Both Original audio (input 0) and Commentary narration (input 1) present
    orig_gain_part = f",volume={settings.original_gain_db:.2f}dB" if abs(settings.original_gain_db) > 1e-4 else ""
    comm_gain_part = f",volume={settings.commentary_gain_db:.2f}dB" if abs(settings.commentary_gain_db) > 1e-4 else ""

    orig_prep = f"[0:a]aformat=channel_layouts={layout}:sample_rates={sr}{orig_gain_part}[orig_ready]"
    comm_prep = f"[1:a]aformat=channel_layouts={layout}:sample_rates={sr}{comm_gain_part}[comm_ready]"

    if settings.auto_duck:
        # Calculate compression ratio and threshold from ducking amount
        duck_depth = abs(settings.amount)
        ratio = max(2.0, min(20.0, duck_depth / 2.0))
        threshold = 0.05
        attack = max(1.0, settings.attack_ms)
        release = max(10.0, settings.release_ms)

        # Split commentary into control sidechain and mixing stream
        split_comm = "[comm_ready]asplit=2[comm_sc][comm_mix]"
        # apad on sidechain control stream ensures sidechaincompress processes the full Original audio duration
        pad_sc = "[comm_sc]apad[comm_sc_pad]"
        duck_filter = (
            f"[orig_ready][comm_sc_pad]sidechaincompress="
            f"threshold={threshold:.3f}:ratio={ratio:.1f}:attack={attack:.1f}:release={release:.1f}[ducked_orig]"
        )
        mix_filter = f"[ducked_orig][comm_mix]amix=inputs=2:duration={settings.duration_mode}:dropout_transition=2:normalize=0[mixed]"
    else:
        # Auto-duck off: still amix Original audio + commentary voice
        split_comm = ""
        pad_sc = ""
        duck_filter = ""
        mix_filter = f"[orig_ready][comm_ready]amix=inputs=2:duration={settings.duration_mode}:dropout_transition=2:normalize=0[mixed]"

    post_filter = (
        f"[mixed]loudnorm=I={settings.target_lufs:.1f}:TP={settings.true_peak:.1f}:LRA=11[normed];"
        f"[normed]alimiter=limit={limiter_val:.4f}:attack=5:release=50:level=disabled[out]"
    )

    parts = [orig_prep, comm_prep]
    if split_comm:
        parts.append(split_comm)
    if pad_sc:
        parts.append(pad_sc)
    if duck_filter:
        parts.append(duck_filter)
    parts.append(mix_filter)
    parts.append(post_filter)

    return ";".join(parts)


def plan_audio_mix(
    output_path: Path | str,
    original_audio_path: Path | str | None = None,
    commentary_audio_path: Path | str | None = None,
    settings: AudioMixSettings | None = None,
    *,
    is_original_dialogue_only: bool = False,
) -> AudioMixPlan:
    """Create an AudioMixPlan based on provided audio source paths and mix settings."""
    cfg = settings or AudioMixSettings()
    out_file = Path(output_path).resolve()

    has_orig = False
    orig_path: Path | None = None
    if original_audio_path is not None:
        p = Path(original_audio_path).resolve()
        if p.is_file():
            has_orig = True
            orig_path = p

    has_comm = False
    comm_path: Path | None = None
    if commentary_audio_path is not None:
        p = Path(commentary_audio_path).resolve()
        if p.is_file():
            has_comm = True
            comm_path = p

    if not has_orig and not has_comm:
        raise AudioMixError("Không tìm thấy tệp âm thanh hợp lệ nào (cả Original audio và Commentary đều thiếu).")

    input_files: list[Path] = []
    if has_orig and has_comm and not is_original_dialogue_only:
        assert orig_path is not None and comm_path is not None
        input_files = [orig_path, comm_path]
    elif has_orig:
        assert orig_path is not None
        input_files = [orig_path]
    elif has_comm:
        assert comm_path is not None
        input_files = [comm_path]

    fg = build_audio_mix_filter_graph(
        cfg,
        has_original=has_orig,
        has_commentary=has_comm,
        is_original_dialogue_only=is_original_dialogue_only,
    )

    return AudioMixPlan(
        settings=cfg,
        has_original=has_orig,
        has_commentary=has_comm,
        is_original_dialogue_only=is_original_dialogue_only,
        filter_graph=fg,
        input_files=tuple(input_files),
        output_file=out_file,
    )


def build_audio_mix_command(
    output_path: Path | str,
    original_audio_path: Path | str | None = None,
    commentary_audio_path: Path | str | None = None,
    settings: AudioMixSettings | None = None,
    *,
    is_original_dialogue_only: bool = False,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build the executable FFmpeg command list for audio mixing."""
    plan = plan_audio_mix(
        output_path=output_path,
        original_audio_path=original_audio_path,
        commentary_audio_path=commentary_audio_path,
        settings=settings,
        is_original_dialogue_only=is_original_dialogue_only,
    )
    return plan.build_command(ffmpeg_bin=ffmpeg_bin)


def execute_audio_mix(
    output_path: Path | str,
    original_audio_path: Path | str | None = None,
    commentary_audio_path: Path | str | None = None,
    settings: AudioMixSettings | None = None,
    *,
    is_original_dialogue_only: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    timeout: float = 120.0,
) -> Path:
    """Execute FFmpeg audio mix and validate resulting mixed audio file."""
    plan = plan_audio_mix(
        output_path=output_path,
        original_audio_path=original_audio_path,
        commentary_audio_path=commentary_audio_path,
        settings=settings,
        is_original_dialogue_only=is_original_dialogue_only,
    )
    out_path = plan.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = plan.build_command(ffmpeg_bin=ffmpeg_bin)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0:
            raise AudioMixError(f"Lệnh FFmpeg trộn âm thanh thất bại (mã {res.returncode}): {res.stderr.strip()}")
    except FileNotFoundError as exc:
        raise AudioMixError(f"Không tìm thấy thực thi FFmpeg tại '{ffmpeg_bin}': {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioMixError(f"Quá thời gian thực thi FFmpeg trộn âm thanh ({timeout}s).") from exc

    validate_mixed_audio(out_path, settings=plan.settings)
    return out_path


def validate_mixed_audio(
    audio_path: Path | str,
    *,
    settings: AudioMixSettings | None = None,
    max_peak_db: float | None = None,
    min_duration: float = 0.1,
) -> dict[str, Any]:
    """Validate mixed audio file: checks PCM format, non-silence, duration, true peak, and clipping freedom."""
    path = Path(audio_path).resolve()
    if not path.is_file():
        raise AudioValidationError(f"Không tìm thấy tệp âm thanh đã trộn: {path}")

    if path.stat().st_size == 0:
        raise AudioValidationError("Tệp âm thanh rỗng.")

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()

            if wav_file.getcomptype() != "NONE" or sample_width != 2:
                raise AudioValidationError("Định dạng âm thanh đầu ra phải là PCM16 không nén.")
            if channels <= 0 or frame_rate <= 0 or frame_count <= 0:
                raise AudioValidationError("Thông số âm thanh WAV không hợp lệ.")

            duration = frame_count / frame_rate
            if duration < min_duration:
                raise AudioValidationError(f"Thời lượng âm thanh quá ngắn: {duration:.2f}s < {min_duration}s.")

            raw = wav_file.readframes(frame_count)
            samples = array.array("h")
            samples.frombytes(raw)
            if sys.byteorder != "little":
                samples.byteswap()

            if not samples:
                raise AudioValidationError("Tệp âm thanh không chứa mẫu dữ liệu.")

            # RMS check for silence
            sum_sq = sum(s * s for s in samples)
            rms = (sum_sq / len(samples)) ** 0.5 / 32768.0
            if rms < 0.0005:
                raise AudioValidationError(f"Âm thanh bị im lặng hoàn toàn (RMS={rms:.6f}).")

            # Peak amplitude and clipping check
            max_abs = max(abs(s) for s in samples)
            peak_dbfs = 20.0 * math.log10(max_abs / 32768.0) if max_abs > 0 else -100.0

            # Absolute clipping threshold check (32767 for signed 16-bit)
            if max_abs >= 32767 or peak_dbfs >= 0.0:
                raise AudioValidationError(f"Phát hiện clipping âm thanh: peak={peak_dbfs:.2f} dBFS.")

            # True peak limit constraint (allow 0.15 dB safety tolerance for float/int rounding)
            allowed_peak = max_peak_db if max_peak_db is not None else (settings.true_peak if settings else -1.5)
            if peak_dbfs > (allowed_peak + 0.15):
                raise AudioValidationError(
                    f"Mức đỉnh âm thanh ({peak_dbfs:.2f} dBFS) vượt quá ngưỡng True Peak cho phép ({allowed_peak:.2f} dBFS)."
                )

            return {
                "sample_rate": frame_rate,
                "channels": channels,
                "duration": duration,
                "rms": rms,
                "peak_dbfs": peak_dbfs,
                "max_sample": max_abs,
                "is_clipping": False,
                "passes_true_peak": True,
            }
    except (wave.Error, OSError, ValueError) as exc:
        raise AudioValidationError(f"Lỗi đọc/phân tích tệp âm thanh: {exc}") from exc
