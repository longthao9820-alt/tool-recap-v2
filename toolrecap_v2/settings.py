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
    voice_id: str = DEFAULT_VOICE_ID
    quality: str = "high"
    use_gpu: bool = True
    generate_srt: bool = True
    burn_subtitles: bool = True
    output_dir: str = ""
    transcription_provider: str = "local"  # "local" (faster-whisper/subtitles) or "openai" / "gemini"
    api_key: str = ""
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
                fields = AppSettings.__dataclass_fields__
                valid_data = {k: v for k, v in raw.items() if k in fields}
                return AppSettings(**valid_data)
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
