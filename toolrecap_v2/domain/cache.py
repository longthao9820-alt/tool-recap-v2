"""Episode evidence cache manager with atomic persistence and independent per-episode keys."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..paths import default_data_directory
from .models import EpisodeEvidence, SourceEpisode


def default_evidence_cache_dir() -> Path:
    cache_dir = default_data_directory() / "cache" / "episode_evidence"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_cache_key(
    episode_id: str,
    source_video: str | Path,
    config_version: str,
    source_size: int = 0,
    source_mtime: float = 0.0,
) -> str:
    """Deterministic cache key based on source identity and analysis config version.

    CRITICAL: API keys, tokens, or credentials are strictly excluded from the key payload.
    """
    path_str = str(Path(source_video).resolve()) if source_video else ""
    key_payload = {
        "episode_id": episode_id,
        "source_video": path_str,
        "config_version": config_version,
        "source_size": source_size,
        "source_mtime": round(source_mtime, 3),
    }
    raw = json.dumps(key_payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EvidenceCacheManager:
    """Manages independent caching of EpisodeEvidence records with atomic disk writes."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or default_evidence_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_source_stats(source_video: str | Path) -> tuple[int, float]:
        path = Path(source_video)
        if path.is_file():
            stat = path.stat()
            return stat.st_size, stat.st_mtime
        return 0, 0.0

    def _cache_file_path(self, episode_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in episode_id)
        return self.cache_dir / f"{safe_id}.evidence.json"

    def save_evidence(
        self,
        episode: SourceEpisode,
        config_version: str,
        evidence: EpisodeEvidence,
    ) -> Path:
        """Atomically save evidence for an episode."""
        size, mtime = self._get_source_stats(episode.source_video)
        key = compute_cache_key(
            episode_id=episode.episode_id,
            source_video=episode.source_video,
            config_version=config_version,
            source_size=size,
            source_mtime=mtime,
        )

        target_path = self._cache_file_path(episode.episode_id)
        tmp_path = target_path.with_suffix(".tmp")

        payload: dict[str, Any] = {
            "cache_key": key,
            "episode_id": episode.episode_id,
            "source_video": str(Path(episode.source_video).resolve()) if episode.source_video else "",
            "config_version": config_version,
            "source_size": size,
            "source_mtime": mtime,
            "evidence": evidence.to_dict(),
        }

        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, target_path)
        return target_path

    def load_evidence(
        self,
        episode: SourceEpisode,
        config_version: str,
    ) -> EpisodeEvidence | None:
        """Load cached evidence for an episode.

        Returns None and invalidates if:
        - Cache does not exist
        - Config version differs
        - Source video path differs
        - Source video size or mtime differs from cached snapshot
        """
        target_path = self._cache_file_path(episode.episode_id)
        if not target_path.is_file():
            return None

        try:
            raw = json.loads(target_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None

            # Verify config version
            if raw.get("config_version") != config_version:
                return None

            # Verify source identity
            cached_path = raw.get("source_video", "")
            current_path = str(Path(episode.source_video).resolve()) if episode.source_video else ""
            if cached_path != current_path:
                return None

            # Verify file stats if file exists on disk
            size, mtime = self._get_source_stats(episode.source_video)
            cached_size = raw.get("source_size", 0)
            cached_mtime = raw.get("source_mtime", 0.0)

            if size != cached_size or abs(mtime - cached_mtime) > 0.001:
                return None

            evidence_data = raw.get("evidence", {})
            return EpisodeEvidence.from_dict(evidence_data)
        except Exception:
            return None

    def invalidate(self, episode_id: str) -> None:
        """Invalidate cache for an episode."""
        target_path = self._cache_file_path(episode_id)
        if target_path.is_file():
            try:
                target_path.unlink()
            except OSError:
                pass
