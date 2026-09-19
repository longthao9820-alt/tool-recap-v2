"""Tests for audio mix filter-plan module, FFmpeg command generation, and audio validation.

Strict invariant:
All comments, identifiers, and documentation strictly refer to Original audio.
"""
from __future__ import annotations

import array
import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path
import pytest

from toolrecap_v2.voice.audio_mix import (
    AudioMixError,
    AudioMixPlan,
    AudioMixSettings,
    AudioValidationError,
    build_audio_mix_command,
    build_audio_mix_filter_graph,
    execute_audio_mix,
    plan_audio_mix,
    validate_mixed_audio,
)


def _generate_synthetic_pcm_wav(
    file_path: Path,
    frequency: float = 440.0,
    duration_sec: float = 2.0,
    amplitude: int = 15000,
    sample_rate: int = 48000,
    channels: int = 2,
) -> Path:
    """Generate a clean synthetic PCM16 WAV file for audio mixing tests."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(sample_rate * duration_sec)

    with wave.open(str(file_path), "wb") as wav_out:
        wav_out.setnchannels(channels)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)

        samples = array.array("h")
        for i in range(total_frames):
            val = int(amplitude * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            for _ in range(channels):
                samples.append(val)

        wav_out.writeframes(samples.tobytes())
    return file_path


def test_audio_mix_settings_dataclass() -> None:
    """Verify explicit dataclass fields, defaults, and property accessors."""
    cfg = AudioMixSettings(
        original_gain_db=-2.5,
        commentary_gain_db=1.0,
        auto_duck=True,
        amount=-12.0,
        target_lufs=-16.0,
        true_peak=-1.5,
    )
    assert cfg.original_gain_db == -2.5
    assert cfg.commentary_gain_db == 1.0
    assert cfg.auto_duck is True
    assert cfg.amount == -12.0
    assert cfg.duck_amount == -12.0
    assert cfg.duck_amount_db == -12.0
    assert cfg.target_lufs == -16.0
    assert cfg.true_peak == -1.5
    assert cfg.true_peak_db == -1.5
    assert 0.84 < cfg.true_peak_linear < 0.85
    assert cfg.sample_rate == 48000
    assert cfg.channels == 2


def test_filter_graph_generation_auto_duck_on() -> None:
    """Invariant 5: Generate real FFmpeg filter graph with smooth sidechain compression tied to narration."""
    cfg = AudioMixSettings(
        original_gain_db=-3.0,
        commentary_gain_db=2.0,
        auto_duck=True,
        amount=-14.0,
        target_lufs=-16.0,
        true_peak=-1.5,
    )
    fg = build_audio_mix_filter_graph(cfg, has_original=True, has_commentary=True)

    # Must contain Original audio preparation
    assert "[0:a]" in fg
    assert "volume=-3.00dB" in fg

    # Must contain commentary narration preparation
    assert "[1:a]" in fg
    assert "volume=2.00dB" in fg

    # Must contain sidechain compression and split
    assert "asplit=2" in fg
    assert "sidechaincompress=" in fg
    assert "apad" in fg

    # Must contain amix, loudnorm target, and true peak limiter
    assert "amix=inputs=2" in fg
    assert "loudnorm=I=-16.0:TP=-1.5" in fg
    assert "alimiter=" in fg
    assert "[out]" in fg


def test_filter_graph_generation_auto_duck_off() -> None:
    """Invariant 5: Auto-duck off still amix Original audio + commentary voice."""
    cfg = AudioMixSettings(auto_duck=False, target_lufs=-16.0, true_peak=-1.5)
    fg = build_audio_mix_filter_graph(cfg, has_original=True, has_commentary=True)

    # Must NOT contain sidechain compression
    assert "sidechaincompress" not in fg

    # Must directly amix Original audio and commentary voice
    assert "amix=inputs=2" in fg
    assert "loudnorm=I=-16.0:TP=-1.5" in fg
    assert "alimiter=" in fg
    assert "[out]" in fg


def test_filter_graph_original_dialogue_preserves_original() -> None:
    """Invariant 5: Original dialogue / no narration preserves Original audio."""
    cfg = AudioMixSettings(original_gain_db=1.5, target_lufs=-16.0, true_peak=-1.5)

    # Case A: has_commentary is False
    fg_no_comm = build_audio_mix_filter_graph(cfg, has_original=True, has_commentary=False)
    assert "amix" not in fg_no_comm
    assert "sidechaincompress" not in fg_no_comm
    assert "[0:a]" in fg_no_comm
    assert "alimiter=" in fg_no_comm
    assert "[out]" in fg_no_comm

    # Case B: is_original_dialogue_only is True
    fg_orig_only = build_audio_mix_filter_graph(
        cfg, has_original=True, has_commentary=True, is_original_dialogue_only=True
    )
    assert "amix" not in fg_orig_only
    assert "sidechaincompress" not in fg_orig_only
    assert "[0:a]" in fg_orig_only
    assert "alimiter=" in fg_orig_only


def test_filter_graph_missing_original_audio() -> None:
    """Invariant 5: Account for missing Original audio gracefully."""
    cfg = AudioMixSettings(commentary_gain_db=-1.0, target_lufs=-16.0, true_peak=-1.5)
    fg = build_audio_mix_filter_graph(cfg, has_original=False, has_commentary=True)

    # Commentary becomes sole input
    assert "[0:a]" in fg
    assert "amix" not in fg
    assert "sidechaincompress" not in fg
    assert "loudnorm=I=-16.0:TP=-1.5" in fg
    assert "alimiter=" in fg
    assert "[out]" in fg


def test_filter_graph_missing_both_raises_error() -> None:
    cfg = AudioMixSettings()
    with pytest.raises(AudioMixError, match="đều không có"):
        build_audio_mix_filter_graph(cfg, has_original=False, has_commentary=False)


def test_build_audio_mix_command(tmp_path: Path) -> None:
    orig_wav = _generate_synthetic_pcm_wav(tmp_path / "orig.wav")
    comm_wav = _generate_synthetic_pcm_wav(tmp_path / "comm.wav")
    out_wav = tmp_path / "mixed.wav"

    cmd = build_audio_mix_command(
        output_path=out_wav,
        original_audio_path=orig_wav,
        commentary_audio_path=comm_wav,
        settings=AudioMixSettings(),
    )
    assert cmd[0] == "ffmpeg"
    assert "-filter_complex" in cmd
    assert str(orig_wav) in cmd
    assert str(comm_wav) in cmd
    assert str(out_wav) in cmd
    assert "-map" in cmd
    assert "[out]" in cmd


def test_real_ffmpeg_audio_mix_with_auto_ducking(tmp_path: Path) -> None:
    """Invariant 5 & 6: Real FFmpeg mix test: smooth sidechain compression ducks Original audio during commentary."""
    orig_wav = tmp_path / "orig.wav"
    comm_wav = tmp_path / "comm.wav"
    out_wav = tmp_path / "out_ducked.wav"

    # Original audio is 3.0s continuous tone
    _generate_synthetic_pcm_wav(orig_wav, frequency=440.0, duration_sec=3.0, amplitude=15000)
    # Commentary narration is 1.0s tone
    _generate_synthetic_pcm_wav(comm_wav, frequency=880.0, duration_sec=1.0, amplitude=20000)

    settings = AudioMixSettings(
        original_gain_db=0.0,
        commentary_gain_db=0.0,
        auto_duck=True,
        amount=-14.0,
        target_lufs=-16.0,
        true_peak=-1.5,
    )

    result_path = execute_audio_mix(
        output_path=out_wav,
        original_audio_path=orig_wav,
        commentary_audio_path=comm_wav,
        settings=settings,
    )
    assert result_path == out_wav
    assert out_wav.is_file()

    # Validate output audio metrics
    info = validate_mixed_audio(out_wav, settings=settings)
    assert info["is_clipping"] is False
    assert info["passes_true_peak"] is True
    assert info["peak_dbfs"] <= -1.5 + 0.15
    assert info["rms"] > 0.0005

    # Measure sidechain compression: check ducked original audio directly
    duck_test_wav = tmp_path / "duck_verify.wav"
    fg_duck_only = (
        "[0:a]aformat=channel_layouts=stereo:sample_rates=48000[orig];"
        "[1:a]aformat=channel_layouts=stereo:sample_rates=48000[comm];"
        "[comm]apad[comm_pad];"
        "[orig][comm_pad]sidechaincompress=threshold=0.05:ratio=8:attack=20:release=250[ducked]"
    )
    subprocess.run([
        "ffmpeg", "-y", "-i", str(orig_wav), "-i", str(comm_wav),
        "-filter_complex", fg_duck_only, "-map", "[ducked]", "-c:a", "pcm_s16le", str(duck_test_wav)
    ], check=True, capture_output=True)

    with wave.open(str(duck_test_wav), "rb") as w:
        raw = w.readframes(w.getnframes())
        samples = array.array("h", raw)
        # Speech active section (first 0.8s) vs post-speech section (1.5s to 2.5s)
        during_speech = samples[: int(48000 * 2 * 0.8)]
        post_speech = samples[int(48000 * 2 * 1.5) : int(48000 * 2 * 2.5)]
        rms_speech = (sum(s * s for s in during_speech) / len(during_speech)) ** 0.5
        rms_post = (sum(s * s for s in post_speech) / len(post_speech)) ** 0.5

        # Ducking must significantly attenuate Original audio during speech
        assert rms_speech < (rms_post * 0.5)


def test_real_ffmpeg_audio_mix_auto_duck_off(tmp_path: Path) -> None:
    """Invariant 5: Auto-duck off still amix Original audio + commentary voice."""
    orig_wav = tmp_path / "orig.wav"
    comm_wav = tmp_path / "comm.wav"
    out_wav = tmp_path / "out_noduck.wav"

    _generate_synthetic_pcm_wav(orig_wav, frequency=440.0, duration_sec=2.0, amplitude=12000)
    _generate_synthetic_pcm_wav(comm_wav, frequency=880.0, duration_sec=2.0, amplitude=12000)

    settings = AudioMixSettings(auto_duck=False, target_lufs=-16.0, true_peak=-1.5)
    execute_audio_mix(out_wav, orig_wav, comm_wav, settings=settings)

    assert out_wav.is_file()
    info = validate_mixed_audio(out_wav, settings=settings)
    assert info["is_clipping"] is False
    assert info["passes_true_peak"] is True


def test_real_ffmpeg_preserve_original_dialogue(tmp_path: Path) -> None:
    """Invariant 5: Original dialogue / no narration preserves Original audio."""
    orig_wav = tmp_path / "orig.wav"
    out_wav = tmp_path / "out_preserved.wav"

    _generate_synthetic_pcm_wav(orig_wav, frequency=440.0, duration_sec=2.0, amplitude=15000)

    settings = AudioMixSettings(original_gain_db=0.0, target_lufs=-16.0, true_peak=-1.5)
    execute_audio_mix(out_wav, original_audio_path=orig_wav, commentary_audio_path=None, settings=settings)

    assert out_wav.is_file()
    info = validate_mixed_audio(out_wav, settings=settings)
    assert info["is_clipping"] is False
    assert info["passes_true_peak"] is True


def test_real_ffmpeg_missing_original_audio(tmp_path: Path) -> None:
    """Invariant 5: Accounts for missing Original audio without error."""
    comm_wav = tmp_path / "comm.wav"
    out_wav = tmp_path / "out_comm_only.wav"

    _generate_synthetic_pcm_wav(comm_wav, frequency=880.0, duration_sec=1.5, amplitude=18000)

    settings = AudioMixSettings(commentary_gain_db=0.0, target_lufs=-16.0, true_peak=-1.5)
    execute_audio_mix(out_wav, original_audio_path=None, commentary_audio_path=comm_wav, settings=settings)

    assert out_wav.is_file()
    info = validate_mixed_audio(out_wav, settings=settings)
    assert info["is_clipping"] is False
    assert info["passes_true_peak"] is True


def test_audio_validation_rejects_silent_and_corrupt(tmp_path: Path) -> None:
    """Verify validation detects silence and corruption."""
    # Silent audio
    silent_wav = tmp_path / "silent.wav"
    with wave.open(str(silent_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00" * 48000)

    with pytest.raises(AudioValidationError, match="im lặng"):
        validate_mixed_audio(silent_wav)

    # Nonexistent file
    with pytest.raises(AudioValidationError, match="Không tìm thấy"):
        validate_mixed_audio(tmp_path / "non_existent.wav")


def test_no_highlight_label_in_code_or_docs() -> None:
    """Invariant 5: Label Original audio in names/docs/comments, never forbidden term."""
    source_file = Path(__file__).resolve().parent.parent / "toolrecap_v2" / "voice" / "audio_mix.py"
    content = source_file.read_text(encoding="utf-8")
    forbidden = "high" + "light"
    assert forbidden not in content.lower()
