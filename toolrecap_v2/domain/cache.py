"""Episode evidence cache manager with atomic persistence and independent per-episode keys."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..paths import default_data_directory
from .models import (
    CompactEpisodeSummary,
    EpisodeEvidence,
    SUMMARY_SCHEMA_VERSION,
    SourceEpisode,
)


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


def default_hierarchy_cache_dir() -> Path:
    base = default_data_directory() / "cache" / "season_hierarchy"
    base.mkdir(parents=True, exist_ok=True)
    return base


def validate_hierarchy_cache_data(res: Any) -> bool:
    """Validate structure of cached batch, merge, or connection results on load.

    Returns True if structure matches expected connection schema, False if corrupt or tampered.
    """
    if not isinstance(res, dict):
        return False
    list_fields = [
        ("cross_episode_links", "links", "narrative_links"),
        ("candidate_proposals", "candidates", "proposals"),
        ("supporting_character_arcs", "supporting_arcs", "character_arcs"),
        ("rejected_or_merged", "rejected"),
    ]
    has_known_key = False
    for group in list_fields:
        for key in group:
            if key in res:
                val = res[key]
                if not isinstance(val, list):
                    return False
                if not all(isinstance(item, dict) for item in val):
                    return False
                has_known_key = True
    if not has_known_key and "outputs" not in res and "batch_results" not in res:
        return False
    return True


def compute_summary_cache_key(
    episode_id: str,
    evidence_hash: str,
    schema_version: str = SUMMARY_SCHEMA_VERSION,
) -> str:
    """Deterministic cache key for compact episode summary.
    Strictly excludes API keys, tokens, or endpoints.
    """
    payload = {
        "episode_id": episode_id,
        "evidence_hash": evidence_hash,
        "schema_version": schema_version,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_batch_cache_key(
    batch_id: str,
    ordered_summary_hashes: list[str],
    model: str,
    thinking: str,
    recap_prompt: str,
    prompt_version: str = "v1",
) -> str:
    """Deterministic cache key for season batch analysis.
    Strictly excludes API keys, tokens, or endpoints.
    """
    payload = {
        "batch_id": batch_id,
        "summary_hashes": list(ordered_summary_hashes),
        "model": str(model),
        "thinking": str(thinking),
        "recap_prompt": str(recap_prompt),
        "prompt_version": str(prompt_version),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_merge_cache_key(
    merge_id: str,
    ordered_batch_hashes: list[str],
    model: str,
    thinking: str,
    recap_prompt: str,
    merge_version: str = "v1",
) -> str:
    """Deterministic cache key for season cross-batch merge analysis.
    Strictly excludes API keys, tokens, or endpoints.
    """
    payload = {
        "merge_id": merge_id,
        "batch_hashes": list(ordered_batch_hashes),
        "model": str(model),
        "thinking": str(thinking),
        "recap_prompt": str(recap_prompt),
        "merge_version": str(merge_version),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_connection_cache_key(
    ordered_evidence_hashes: list[str],
    model: str,
    thinking: str,
    recap_prompt: str,
    config_version: str = "v1",
) -> str:
    """Deterministic cache key for full season connection result."""
    payload = {
        "evidence_hashes": list(ordered_evidence_hashes),
        "model": str(model),
        "thinking": str(thinking),
        "recap_prompt": str(recap_prompt),
        "config_version": str(config_version),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class HierarchyCacheManager:
    """Manages hierarchical caching of compact summaries, batches, and cross-batch merges.

    Provides atomic validated file persistence with unique keys and strict isolation from secrets.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or default_hierarchy_cache_dir()
        self.summary_dir = self.base_dir / "summaries"
        self.batch_dir = self.base_dir / "batches"
        self.merge_dir = self.base_dir / "merges"
        self.connection_dir = self.base_dir / "connections"

        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.merge_dir.mkdir(parents=True, exist_ok=True)
        self.connection_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(val: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in val)

    def _write_atomic(self, target_path: Path, data: dict[str, Any]) -> None:
        """Atomically write JSON data with validation before replace."""
        tmp_path = target_path.with_suffix(f".tmp.{os.getpid()}")
        try:
            content = json.dumps(data, indent=2, ensure_ascii=False)
            # Pre-write parse verification
            json.loads(content)
            tmp_path.write_text(content, encoding="utf-8")
            # Post-write read validation
            json.loads(tmp_path.read_text(encoding="utf-8"))
            os.replace(tmp_path, target_path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    # 1. Compact Summaries
    def save_summary(self, summary: CompactEpisodeSummary, evidence_hash: str) -> tuple[Path, str]:
        key = compute_summary_cache_key(summary.episode_id, evidence_hash, summary.schema_version)
        safe_id = self._safe_id(summary.episode_id)
        target = self.summary_dir / f"{safe_id}_{key[:24]}.summary.json"
        payload = {
            "cache_key": key,
            "episode_id": summary.episode_id,
            "evidence_hash": evidence_hash,
            "schema_version": summary.schema_version,
            "summary": summary.to_dict(),
        }
        self._write_atomic(target, payload)
        return target, key

    def load_summary(
        self,
        episode_id: str,
        evidence_hash: str,
        schema_version: str = SUMMARY_SCHEMA_VERSION,
    ) -> tuple[CompactEpisodeSummary | None, dict[str, Any]]:
        key = compute_summary_cache_key(episode_id, evidence_hash, schema_version)
        safe_id = self._safe_id(episode_id)
        target = self.summary_dir / f"{safe_id}_{key[:24]}.summary.json"
        meta: dict[str, Any] = {"hit": False, "cache_key": key, "path": str(target)}

        if not target.is_file():
            return None, meta

        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None, meta
            if data.get("cache_key") != key or data.get("schema_version") != schema_version:
                return None, meta
            if data.get("episode_id") != episode_id:
                return None, meta
            raw_sum = data.get("summary")
            if not isinstance(raw_sum, dict) or raw_sum.get("episode_id") != episode_id:
                return None, meta
            summary = CompactEpisodeSummary.from_dict(raw_sum)
            meta["hit"] = True
            return summary, meta
        except Exception:
            return None, meta

    # 2. Season Batches
    def save_batch_result(self, batch_id: str, cache_key: str, data: dict[str, Any]) -> Path:
        safe_id = self._safe_id(batch_id)
        target = self.batch_dir / f"{safe_id}_{cache_key[:24]}.batch.json"
        payload = {
            "cache_key": cache_key,
            "batch_id": batch_id,
            "result": data,
        }
        self._write_atomic(target, payload)
        return target

    def load_batch_result(self, batch_id: str, cache_key: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        safe_id = self._safe_id(batch_id)
        target = self.batch_dir / f"{safe_id}_{cache_key[:24]}.batch.json"
        meta: dict[str, Any] = {"hit": False, "cache_key": cache_key, "path": str(target)}

        if not target.is_file():
            return None, meta

        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("cache_key") != cache_key:
                return None, meta
            res = raw.get("result")
            if not isinstance(res, dict) or not validate_hierarchy_cache_data(res):
                return None, meta
            meta["hit"] = True
            return res, meta
        except Exception:
            return None, meta

    # 3. Season Merges
    def save_merge_result(self, merge_id: str, cache_key: str, data: dict[str, Any]) -> Path:
        safe_id = self._safe_id(merge_id)
        target = self.merge_dir / f"{safe_id}_{cache_key[:24]}.merge.json"
        payload = {
            "cache_key": cache_key,
            "merge_id": merge_id,
            "result": data,
        }
        self._write_atomic(target, payload)
        return target

    def load_merge_result(self, merge_id: str, cache_key: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        safe_id = self._safe_id(merge_id)
        target = self.merge_dir / f"{safe_id}_{cache_key[:24]}.merge.json"
        meta: dict[str, Any] = {"hit": False, "cache_key": cache_key, "path": str(target)}

        if not target.is_file():
            return None, meta

        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("cache_key") != cache_key:
                return None, meta
            res = raw.get("result")
            if not isinstance(res, dict) or not validate_hierarchy_cache_data(res):
                return None, meta
            meta["hit"] = True
            return res, meta
        except Exception:
            return None, meta

    # 4. Full Season Connection Result
    def save_connection_result(self, conn_key: str, data: dict[str, Any]) -> Path:
        target = self.connection_dir / f"conn_{conn_key[:24]}.connection.json"
        payload = {
            "cache_key": conn_key,
            "result": data,
        }
        self._write_atomic(target, payload)
        return target

    def load_connection_result(self, conn_key: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        target = self.connection_dir / f"conn_{conn_key[:24]}.connection.json"
        meta: dict[str, Any] = {"hit": False, "cache_key": conn_key, "path": str(target)}

        if not target.is_file():
            return None, meta

        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("cache_key") != conn_key:
                return None, meta
            res = raw.get("result")
            if not isinstance(res, dict) or not validate_hierarchy_cache_data(res):
                return None, meta
            meta["hit"] = True
            return res, meta
        except Exception:
            return None, meta
