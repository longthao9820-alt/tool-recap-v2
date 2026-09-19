"""Independent per-episode subtitle caching with atomic persistence and source fingerprinting."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..paths import default_data_directory
from .models import SubtitleCue


def default_subtitles_cache_dir() -> Path:
    cache_dir = default_data_directory() / "cache" / "subtitles"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_subtitle_cache_key(
    episode_id: str,
    source_video: str | Path,
    subtitle_source_id: str = "",
    ocr_engine_version: str = "rapidocr-v4:1.0",
    source_size: int = 0,
    source_mtime: float = 0.0,
    *,
    ocr_engine: str = "rapidocr",
    model_version: str = "PP-OCRv4",
    track_codec: str = "",
    track_index: int | None = None,
    track_source: str = "",
) -> str:
    """Deterministic cache key based on episode ID, source fingerprint, subtitle track, OCR engine, and model."""
    v_path_str = str(Path(source_video).resolve()) if source_video else ""
    key_payload = {
        "episode_id": episode_id,
        "source_video": v_path_str,
        "subtitle_source_id": subtitle_source_id,
        "ocr_engine_version": ocr_engine_version,
        "ocr_engine": ocr_engine,
        "model_version": model_version,
        "track_codec": track_codec,
        "track_index": track_index,
        "track_source": track_source or subtitle_source_id,
        "source_size": source_size,
        "source_mtime": round(source_mtime, 3),
    }
    raw = json.dumps(key_payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SubtitleCacheManager:
    """Manages independent atomic caching of SubtitleCue lists per episode."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or default_subtitles_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_source_stats(source_video: str | Path) -> tuple[int, float]:
        p = Path(source_video)
        if p.is_file():
            stat = p.stat()
            return stat.st_size, stat.st_mtime
        return 0, 0.0

    def _cache_file_path(self, episode_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in episode_id) or "default"
        return self.cache_dir / f"{safe_id}.subtitles.json"

    def save_cues(
        self,
        episode_id: str,
        source_video: str | Path,
        subtitle_source_id: str,
        cues: list[SubtitleCue],
        ocr_engine_version: str = "rapidocr-v4:1.0",
        *,
        ocr_engine: str = "rapidocr",
        model_version: str = "PP-OCRv4",
        track_codec: str = "",
        track_index: int | None = None,
        track_source: str = "",
    ) -> Path:
        """Atomically persist parsed/OCR'd cues for an episode."""
        size, mtime = self._get_source_stats(source_video)
        key = compute_subtitle_cache_key(
            episode_id=episode_id,
            source_video=source_video,
            subtitle_source_id=subtitle_source_id,
            ocr_engine_version=ocr_engine_version,
            source_size=size,
            source_mtime=mtime,
            ocr_engine=ocr_engine,
            model_version=model_version,
            track_codec=track_codec,
            track_index=track_index,
            track_source=track_source,
        )

        target_path = self._cache_file_path(episode_id)
        tmp_path = target_path.with_suffix(".tmp")

        payload: dict[str, Any] = {
            "cache_key": key,
            "episode_id": episode_id,
            "source_video": str(Path(source_video).resolve()) if source_video else "",
            "subtitle_source_id": subtitle_source_id,
            "ocr_engine_version": ocr_engine_version,
            "ocr_engine": ocr_engine,
            "model_version": model_version,
            "track_codec": track_codec,
            "track_index": track_index,
            "track_source": track_source or subtitle_source_id,
            "source_size": size,
            "source_mtime": mtime,
            "cues": [c.to_dict() for c in cues],
        }

        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, target_path)
        return target_path

    def load_cues(
        self,
        episode_id: str,
        source_video: str | Path,
        subtitle_source_id: str,
        ocr_engine_version: str = "rapidocr-v4:1.0",
        *,
        ocr_engine: str = "rapidocr",
        model_version: str = "PP-OCRv4",
        track_codec: str = "",
        track_index: int | None = None,
        track_source: str = "",
    ) -> list[SubtitleCue] | None:
        """Load cached cues for an episode. Returns None on cache miss or fingerprint invalidation."""
        target_path = self._cache_file_path(episode_id)
        if not target_path.is_file():
            return None

        try:
            raw = json.loads(target_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None

            # Verify episode_id, subtitle source, and ocr engine version
            if raw.get("episode_id") != episode_id:
                return None
            if raw.get("subtitle_source_id") != subtitle_source_id:
                return None
            if raw.get("ocr_engine_version") != ocr_engine_version:
                return None
            if raw.get("ocr_engine", "rapidocr") != ocr_engine:
                return None
            if raw.get("model_version", "PP-OCRv4") != model_version:
                return None
            if track_codec and raw.get("track_codec") != track_codec:
                return None
            if track_index is not None and raw.get("track_index") != track_index:
                return None
            if track_source and raw.get("track_source") != track_source:
                return None

            # Verify source video path
            cached_path = raw.get("source_video", "")
            current_path = str(Path(source_video).resolve()) if source_video else ""
            if cached_path != current_path:
                return None

            # Verify file stats if video file exists on disk
            size, mtime = self._get_source_stats(source_video)
            cached_size = raw.get("source_size", 0)
            cached_mtime = raw.get("source_mtime", 0.0)

            if size != cached_size or abs(mtime - cached_mtime) > 0.001:
                return None

            cues_data = raw.get("cues", [])
            return [SubtitleCue.from_dict(c) for c in cues_data]
        except Exception:
            return None

    def invalidate(self, episode_id: str) -> None:
        """Invalidate cache for an episode."""
        target = self._cache_file_path(episode_id)
        if target.is_file():
            try:
                target.unlink()
            except OSError:
                pass
