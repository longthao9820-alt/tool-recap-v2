"""User settings and persistent storage outside application root."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import default_data_directory
from .voice.catalog import DEFAULT_VOICE_ID


@dataclass
class AppSettings:
    # Voice and video render settings
    voice_id: str = DEFAULT_VOICE_ID
    quality: str = "high"
    use_gpu: bool = True
    generate_srt: bool = True
    burn_subtitles: bool = True
    output_dir: str = ""

    # AI Gateway settings (V1 parity)
    api_endpoint: str = "http://127.0.0.1:20128/v1"
    api_key: str = ""
    scanner_model: str = "sub"
    scanner_thinking: str = "max"
    finalizer_model: str = "prime"
    finalizer_thinking: str = "high"
    scanner_parallelism: int = 2
    api_chunk_seconds: int = 300
    gateway_enabled: bool = True
    scanner_supports_vision: bool = False
    recap_prompt: str = ""

    # Multi-episode analysis & recap settings (V1 parity)
    recap_language: str = "en-US"
    recap_mode: str = "MAIN_STORIES"
    content_type: str = "US_TV_SHOW"
    source_rights_status: str = "UNVERIFIED"
    voice_style: str = "film_recap"

    # Audio mix settings
    original_audio_gain_db: float = 0.0
    commentary_gain_db: float = 0.0
    auto_duck: bool = False
    ducking_amount_db: float = -12.0
    target_loudness_lufs: float = -14.0
    true_peak_dbtp: float = -1.0

    # Speech-to-text / Transcription settings (distinct from gateway)
    transcription_provider: str = "local"  # "local" (faster-whisper/subtitles) or "openai"
    transcription_api_key: str = ""
    transcription_base_url: str = ""

    # Legacy fields / aliases kept for compatibility
    api_base_url: str = ""


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (default_data_directory() / "settings.json")
        self._lock = threading.RLock()

    def load(self) -> AppSettings:
        with self._lock:
            if not self.path.is_file():
                return AppSettings()
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    return AppSettings()

                # Migration of legacy STT vs Gateway fields:
                has_api_endpoint = "api_endpoint" in raw
                transcription_provider = str(raw.get("transcription_provider", "local")).lower()

                if "transcription_api_key" not in raw and "api_key" in raw:
                    old_key = str(raw.get("api_key", "")).strip()
                    if not has_api_endpoint:
                        # Old V2 format without api_endpoint: api_key belonged to STT
                        if transcription_provider != "local" or old_key.startswith("sk-") or old_key:
                            raw["transcription_api_key"] = old_key
                            raw["api_key"] = ""
                    else:
                        # Has api_endpoint: V1/modern format, api_key belongs to gateway
                        if transcription_provider != "local" and old_key.startswith("sk-") and not raw.get("transcription_api_key"):
                            raw["transcription_api_key"] = old_key

                if "transcription_base_url" not in raw and "api_base_url" in raw:
                    raw["transcription_base_url"] = raw.get("api_base_url", "")

                if "api_endpoint" not in raw:
                    raw["api_endpoint"] = "http://127.0.0.1:20128/v1"

                if not raw.get("api_base_url") and raw.get("transcription_base_url"):
                    raw["api_base_url"] = raw["transcription_base_url"]

                fields = AppSettings.__dataclass_fields__
                valid_data = {k: v for k, v in raw.items() if k in fields}
                settings = AppSettings(**valid_data)

                # Clamp bounded numerical fields
                settings.scanner_parallelism = max(1, min(4, int(settings.scanner_parallelism)))
                settings.api_chunk_seconds = max(60, min(900, int(settings.api_chunk_seconds)))
                settings.original_audio_gain_db = max(-60.0, min(24.0, float(settings.original_audio_gain_db)))
                settings.commentary_gain_db = max(-60.0, min(24.0, float(settings.commentary_gain_db)))
                settings.auto_duck = bool(settings.auto_duck)
                settings.ducking_amount_db = max(-60.0, min(0.0, float(settings.ducking_amount_db)))
                settings.target_loudness_lufs = max(-70.0, min(-5.0, float(settings.target_loudness_lufs)))
                settings.true_peak_dbtp = max(-9.0, min(0.0, float(settings.true_peak_dbtp)))

                return settings
            except Exception:
                return AppSettings()

    def save(self, settings: AppSettings) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(asdict(settings), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
