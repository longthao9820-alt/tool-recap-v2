"""Tests for voice catalog, model manager caching, and audio synthesis."""
from __future__ import annotations

from pathlib import Path
import pytest

from toolrecap_v2.voice.catalog import (
    BUILTIN_VOICES,
    DEFAULT_VOICE_ID,
    get_voice_spec,
)
from toolrecap_v2.voice.manager import (
    VoiceError,
    VoiceModelManager,
    validate_wav_audio,
)
from toolrecap_v2.voice.audio_preview import AudioPreviewPlayer


def test_builtin_voice_catalog() -> None:
    assert DEFAULT_VOICE_ID in BUILTIN_VOICES
    assert "piper.en_US-lessac-medium" in BUILTIN_VOICES
    assert "piper.en_US-ryan-medium" in BUILTIN_VOICES
    assert "piper.en_GB-alba-medium" in BUILTIN_VOICES
    assert "piper.en_GB-alan-medium" in BUILTIN_VOICES
    assert "omnivoice.en-storyteller" not in BUILTIN_VOICES

    for vid, spec in BUILTIN_VOICES.items():
        assert spec.voice_id == vid
        assert spec.display_name
        assert spec.preview_text
        assert spec.repo_id


def test_voice_manager_cache_lookup(tmp_path: Path) -> None:
    cache_dir = tmp_path / "models"
    mgr = VoiceModelManager(cache_dir=cache_dir)
    spec = get_voice_spec(DEFAULT_VOICE_ID)

    # Not installed initially
    assert mgr.is_voice_installed(spec.voice_id) is False

    # Simulate installed files in cache
    v_dir = cache_dir / spec.voice_id
    v_dir.mkdir(parents=True, exist_ok=True)
    for req_f in spec.required_files:
        (v_dir / req_f).write_bytes(b"dummy_model_weights")

    assert mgr.is_voice_installed(spec.voice_id) is True
    # Ensure returns cached dir without downloading
    res_dir = mgr.ensure_voice_model(spec.voice_id)
    assert res_dir == v_dir


def test_synthesize_and_validate_audio(tmp_path: Path) -> None:
    mgr = VoiceModelManager(cache_dir=tmp_path / "models")
    out_wav = tmp_path / "test_synth.wav"

    mgr.synthesize(
        DEFAULT_VOICE_ID,
        "This is a test of the audio synthesis pipeline for ToolRecap V2.",
        out_wav,
    )

    assert out_wav.is_file()
    assert out_wav.stat().st_size > 500
    # Must pass strict audio validation
    validate_wav_audio(out_wav)


def test_validate_wav_audio_rejects_corrupted(tmp_path: Path) -> None:
    bad_wav = tmp_path / "bad.wav"
    bad_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00corrupt")
    with pytest.raises(VoiceError):
        validate_wav_audio(bad_wav)

    nonexistent = tmp_path / "missing.wav"
    with pytest.raises(VoiceError, match="Không tìm thấy"):
        validate_wav_audio(nonexistent)


def test_audio_preview_player(tmp_path: Path) -> None:
    mgr = VoiceModelManager(cache_dir=tmp_path / "models")
    out_wav = tmp_path / "preview.wav"
    mgr.synthesize(DEFAULT_VOICE_ID, "Testing preview audio playback.", out_wav)

    player = AudioPreviewPlayer()
    assert player.is_playing is False
    player.play(out_wav)
    player.stop()
    assert player.is_playing is False
