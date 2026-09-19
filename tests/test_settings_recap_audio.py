"""Tests for recap and audio mix settings defaults, persistence, clamping, and migration."""
from __future__ import annotations

import json
from pathlib import Path

from toolrecap_v2.settings import AppSettings, SettingsStore


def test_recap_and_audio_defaults() -> None:
    settings = AppSettings()
    assert settings.recap_language == "en-US"
    assert settings.recap_mode == "MAIN_STORIES"
    assert settings.content_type == "US_TV_SHOW"
    assert settings.source_rights_status == "UNVERIFIED"
    assert settings.voice_style == "film_recap"

    assert settings.original_audio_gain_db == 0.0
    assert settings.commentary_gain_db == 0.0
    assert settings.auto_duck is False
    assert settings.ducking_amount_db == -12.0
    assert settings.target_loudness_lufs == -14.0
    assert settings.true_peak_dbtp == -1.0


def test_settings_store_recap_audio_persistence(tmp_path: Path) -> None:
    file_path = tmp_path / "settings.json"
    store = SettingsStore(file_path)

    settings = AppSettings(
        recap_language="vi-VN",
        recap_mode="ALL_SCENES",
        content_type="ANIME",
        source_rights_status="FAIR_USE",
        voice_style="enthusiastic",
        original_audio_gain_db=-6.0,
        commentary_gain_db=2.5,
        auto_duck=True,
        ducking_amount_db=-18.0,
        target_loudness_lufs=-16.0,
        true_peak_dbtp=-1.5,
    )
    store.save(settings)
    assert file_path.is_file()

    loaded = store.load()
    assert loaded.recap_language == "vi-VN"
    assert loaded.recap_mode == "ALL_SCENES"
    assert loaded.content_type == "ANIME"
    assert loaded.source_rights_status == "FAIR_USE"
    assert loaded.voice_style == "enthusiastic"
    assert loaded.original_audio_gain_db == -6.0
    assert loaded.commentary_gain_db == 2.5
    assert loaded.auto_duck is True
    assert loaded.ducking_amount_db == -18.0
    assert loaded.target_loudness_lufs == -16.0
    assert loaded.true_peak_dbtp == -1.5


def test_settings_audio_clamping(tmp_path: Path) -> None:
    file_path = tmp_path / "settings.json"
    store = SettingsStore(file_path)

    # Set extreme values
    extreme_data = {
        "original_audio_gain_db": 100.0,
        "commentary_gain_db": -200.0,
        "ducking_amount_db": 50.0,  # ducking should be <= 0
        "target_loudness_lufs": 10.0,
        "true_peak_dbtp": 15.0,
    }
    file_path.write_text(json.dumps(extreme_data), encoding="utf-8")

    loaded = store.load()
    assert loaded.original_audio_gain_db == 24.0
    assert loaded.commentary_gain_db == -60.0
    assert loaded.ducking_amount_db == 0.0
    assert loaded.target_loudness_lufs == -5.0
    assert loaded.true_peak_dbtp == 0.0


def test_settings_migration_preserves_gateway_and_stt_config(tmp_path: Path) -> None:
    file_path = tmp_path / "settings.json"
    store = SettingsStore(file_path)

    # Legacy config with gateway and STT
    legacy_data = {
        "api_endpoint": "https://api.custom.com/v1",
        "api_key": "gw-secret-123",
        "scanner_model": "custom-sub",
        "transcription_provider": "openai",
        "transcription_api_key": "sk-stt-secret",
        "transcription_base_url": "https://stt.custom.com/v1",
    }
    file_path.write_text(json.dumps(legacy_data), encoding="utf-8")

    loaded = store.load()
    # Preserved gateway and STT
    assert loaded.api_endpoint == "https://api.custom.com/v1"
    assert loaded.api_key == "gw-secret-123"
    assert loaded.scanner_model == "custom-sub"
    assert loaded.transcription_provider == "openai"
    assert loaded.transcription_api_key == "sk-stt-secret"
    assert loaded.transcription_base_url == "https://stt.custom.com/v1"

    # New fields take defaults
    assert loaded.recap_language == "en-US"
    assert loaded.recap_mode == "MAIN_STORIES"
    assert loaded.original_audio_gain_db == 0.0
    assert loaded.ducking_amount_db == -12.0
