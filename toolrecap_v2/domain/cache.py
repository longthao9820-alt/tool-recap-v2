"""Episode evidence cache manager with atomic persistence and independent per-episode keys."""
from __future__ import annotations

from datetime import datetime, timezone
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


def compute_source_identity_hash(source_video: str | Path) -> str:
    """Stable hash of resolved source video path for collision-free cache naming."""
    path_str = str(Path(source_video).resolve()) if source_video else ""
    return hashlib.sha256(path_str.encode("utf-8")).hexdigest()[:16]


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

    def _cache_file_path(self, episode_id: str, source_video: str | Path | None = None) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in episode_id) or "ep"
        if source_video:
            src_hash = compute_source_identity_hash(source_video)
            return self.cache_dir / f"{safe_id}_{src_hash}.evidence.json"
        return self.cache_dir / f"{safe_id}.evidence.json"

    def _legacy_cache_file_path(self, episode_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in episode_id) or "ep"
        return self.cache_dir / f"{safe_id}.evidence.json"

    def save_evidence(
        self,
        episode: SourceEpisode | str,
        config_version: str,
        evidence: EpisodeEvidence,
        source_video: str | Path | None = None,
    ) -> Path:
        """Atomically save evidence for an episode."""
        if isinstance(episode, SourceEpisode):
            ep_id = episode.episode_id
            v_path = episode.source_video
        else:
            ep_id = str(episode)
            v_path = source_video or getattr(evidence, "source_video", "")

        size, mtime = self._get_source_stats(v_path)
        key = compute_cache_key(
            episode_id=ep_id,
            source_video=v_path,
            config_version=config_version,
            source_size=size,
            source_mtime=mtime,
        )

        target_path = self._cache_file_path(ep_id, v_path)
        tmp_path = target_path.with_suffix(f".tmp.{os.getpid()}")

        payload: dict[str, Any] = {
            "cache_key": key,
            "episode_id": ep_id,
            "source_video": str(Path(v_path).resolve()) if v_path else "",
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
        episode: SourceEpisode | str,
        config_version: str,
        source_video: str | Path | None = None,
    ) -> EpisodeEvidence | None:
        """Load cached evidence for an episode.

        Returns None and invalidates if:
        - Cache does not exist
        - Config version differs
        - Source video path differs
        - Source video size or mtime differs from cached snapshot
        """
        if isinstance(episode, SourceEpisode):
            ep_id = episode.episode_id
            v_path = episode.source_video
        else:
            ep_id = str(episode)
            v_path = source_video or ""

        target_path = self._cache_file_path(ep_id, v_path)
        target_file = target_path
        is_legacy = False

        if not target_file.is_file():
            legacy_path = self._legacy_cache_file_path(ep_id)
            if legacy_path.is_file():
                target_file = legacy_path
                is_legacy = True
            else:
                return None

        try:
            raw = json.loads(target_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None

            # Verify config version
            if raw.get("config_version") != config_version:
                return None

            # Verify source identity
            cached_path = raw.get("source_video", "")
            current_path = str(Path(v_path).resolve()) if v_path else ""
            if cached_path != current_path:
                return None

            # Verify file stats if file exists on disk
            size, mtime = self._get_source_stats(v_path)
            cached_size = raw.get("source_size", 0)
            cached_mtime = raw.get("source_mtime", 0.0)

            if size != cached_size or abs(mtime - cached_mtime) > 0.001:
                return None

            evidence_data = raw.get("evidence", {})
            evidence = EpisodeEvidence.from_dict(evidence_data)

            # Safely migrate legacy cache file to new hashed path if valid
            if is_legacy and v_path:
                try:
                    tmp_target = target_path.with_suffix(f".tmp.{os.getpid()}")
                    tmp_target.write_text(
                        json.dumps(raw, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    os.replace(tmp_target, target_path)
                    try:
                        legacy_path.unlink()
                    except OSError:
                        pass
                except Exception:
                    pass

            return evidence
        except Exception:
            return None

    def invalidate(
        self,
        episode: SourceEpisode | str,
        source_video: str | Path | None = None,
    ) -> None:
        """Invalidate cache for an episode."""
        if isinstance(episode, SourceEpisode):
            ep_id = episode.episode_id
            v_path = source_video or episode.source_video
        else:
            ep_id = str(episode)
            v_path = source_video

        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in ep_id) or "ep"
        if v_path:
            target = self._cache_file_path(ep_id, v_path)
            if target.is_file():
                try:
                    target.unlink()
                except OSError:
                    pass
            legacy = self._legacy_cache_file_path(ep_id)
            if legacy.is_file():
                try:
                    raw = json.loads(legacy.read_text(encoding="utf-8"))
                    if raw.get("source_video") == str(Path(v_path).resolve()):
                        legacy.unlink()
                except Exception:
                    pass
        else:
            for p in self.cache_dir.glob(f"{safe_id}_*.evidence.json"):
                try:
                    p.unlink()
                except OSError:
                    pass
            legacy = self._legacy_cache_file_path(ep_id)
            if legacy.is_file():
                try:
                    legacy.unlink()
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
    if "outputs" in res:
        val = res["outputs"]
        if not isinstance(val, list) or not all(isinstance(item, dict) for item in val):
            return False
        has_known_key = True
    if not has_known_key and "outputs" not in res and "batch_results" not in res:
        return False
    return True


def validate_finalizer_cache_data(res: Any) -> bool:
    """Validate structure of cached finalizer result.

    Returns True if structure matches expected outputs schema, False if corrupt.
    """
    if not isinstance(res, dict):
        return False
    outputs = res.get("outputs")
    if not isinstance(outputs, list):
        return False
    return all(isinstance(item, dict) for item in outputs)


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
    batch_id: str = "",
    ordered_summary_hashes: list[str] | None = None,
    model: str = "",
    thinking: str = "",
    recap_prompt: str = "",
    prompt_version: str = "v1",
    *,
    algo: str = "v3",
    compaction_level: str = "FULL",
    settings_sig: str = "",
    node_id: str = "",
) -> str:
    """Deterministic cache key for season batch analysis (hierarchy algo v3).
    Strictly excludes API keys, tokens, or endpoints.
    """
    effective_id = node_id or batch_id
    payload = {
        "algo": algo,
        "node_id": effective_id,
        "summary_hashes": list(ordered_summary_hashes or []),
        "model": str(model),
        "thinking": str(thinking),
        "recap_prompt": str(recap_prompt),
        "prompt_version": str(prompt_version),
        "compaction_level": str(compaction_level),
        "settings": str(settings_sig),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_merge_cache_key(
    merge_id: str = "",
    ordered_batch_hashes: list[str] | None = None,
    model: str = "",
    thinking: str = "",
    recap_prompt: str = "",
    merge_version: str = "v1",
    *,
    algo: str = "v3",
    compaction_level: str = "FULL",
    settings_sig: str = "",
    node_id: str = "",
) -> str:
    """Deterministic cache key for season cross-batch merge analysis (hierarchy algo v3).
    Strictly excludes API keys, tokens, or endpoints.
    """
    effective_id = node_id or merge_id
    payload = {
        "algo": algo,
        "node_id": effective_id,
        "batch_hashes": list(ordered_batch_hashes or []),
        "model": str(model),
        "thinking": str(thinking),
        "recap_prompt": str(recap_prompt),
        "merge_version": str(merge_version),
        "compaction_level": str(compaction_level),
        "settings": str(settings_sig),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_connection_cache_key(
    ordered_evidence_hashes: list[str] | None = None,
    model: str = "",
    thinking: str = "",
    recap_prompt: str = "",
    config_version: str = "v1",
    *,
    algo: str = "v3",
    compaction_level: str = "FULL",
    settings_sig: str = "",
) -> str:
    """Deterministic cache key for full season connection result (algo v3).
    Connection key content/order based; source identity remains evidence hash/order.
    """
    payload = {
        "algo": algo,
        "evidence_hashes": list(ordered_evidence_hashes or []),
        "model": str(model),
        "thinking": str(thinking),
        "recap_prompt": str(recap_prompt),
        "config_version": str(config_version),
        "compaction_level": str(compaction_level),
        "settings": str(settings_sig),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_finalizer_cache_key(
    group_payload_hash: str,
    model: str = "",
    thinking: str = "",
    prompt_version: str = "v1",
    scope: str = "",
    algo: str = "v3",
    recap_prompt: str = "",
    recap_language: str = "",
    recap_mode: str = "",
    content_type: str = "",
    source_rights_status: str = "",
    voice_style: str = "",
    recap_settings: dict[str, Any] | None = None,
) -> str:
    """Deterministic cache key for finalizer group analysis.
    Keyed by group payload hash + model + thinking + prompt + scope + algo + recap settings.
    Strictly excludes API keys, tokens, endpoints, or credentials.
    """
    payload: dict[str, Any] = {
        "algo": str(algo),
        "group_payload_hash": str(group_payload_hash),
        "model": str(model),
        "thinking": str(thinking),
        "prompt_version": str(prompt_version),
        "recap_prompt": str(recap_prompt),
        "scope": str(scope),
        "recap_language": str(recap_language),
        "recap_mode": str(recap_mode),
        "content_type": str(content_type),
        "source_rights_status": str(source_rights_status),
        "voice_style": str(voice_style),
    }
    if recap_settings:
        # Strictly exclude secrets and credentials from recap_settings dict
        safe_settings = {
            k: v for k, v in sorted(recap_settings.items())
            if not any(secret_term in k.lower() for secret_term in ("key", "token", "secret", "endpoint", "password", "auth", "credential"))
        }
        payload["recap_settings"] = safe_settings

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
        self.finalizer_dir = self.base_dir / "finalizers"

        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.merge_dir.mkdir(parents=True, exist_ok=True)
        self.connection_dir.mkdir(parents=True, exist_ok=True)
        self.finalizer_dir.mkdir(parents=True, exist_ok=True)

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
    def save_summary(
        self,
        summary: CompactEpisodeSummary,
        evidence_hash: str,
        *,
        algo: str = "v3",
        schema_version: str = SUMMARY_SCHEMA_VERSION,
        node_id: str | None = None,
        created_at: str | None = None,
        compaction_level: str = "FULL",
    ) -> tuple[Path, str]:
        key = compute_summary_cache_key(summary.episode_id, evidence_hash, summary.schema_version)
        safe_id = self._safe_id(summary.episode_id)
        target = self.summary_dir / f"{safe_id}_{key[:24]}.summary.json"
        ts = created_at or datetime.now(timezone.utc).isoformat()
        payload = {
            "algo": algo,
            "schema": schema_version,
            "node_id": node_id or summary.episode_id,
            "created_at": ts,
            "cache_key": key,
            "episode_id": summary.episode_id,
            "evidence_hash": evidence_hash,
            "schema_version": summary.schema_version,
            "compaction_level": compaction_level,
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

        target_file = target
        if not target_file.is_file():
            matches = list(self.summary_dir.glob(f"*{key[:24]}.summary.json"))
            if matches:
                target_file = matches[0]
                meta["path"] = str(target_file)
            else:
                return None, meta

        try:
            data = json.loads(target_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None, meta
            if data.get("cache_key") != key:
                return None, meta
            # Accept either schema or schema_version for v1 compatibility
            file_schema = data.get("schema") or data.get("schema_version")
            if file_schema and file_schema != schema_version:
                return None, meta
            if data.get("episode_id") and data.get("episode_id") != episode_id:
                return None, meta
            raw_sum = data.get("summary")
            if not isinstance(raw_sum, dict) or raw_sum.get("episode_id") != episode_id:
                return None, meta
            summary = CompactEpisodeSummary.from_dict(raw_sum)
            meta["hit"] = True
            if "algo" in data:
                meta["algo"] = data["algo"]
            if "node_id" in data:
                meta["node_id"] = data["node_id"]
            if "created_at" in data:
                meta["created_at"] = data["created_at"]
            return summary, meta
        except Exception:
            return None, meta

    # 2. Season Batches
    def save_batch_result(
        self,
        batch_id: str,
        cache_key: str,
        data: dict[str, Any],
        *,
        algo: str = "v3",
        schema_version: str = "v3",
        compaction_level: str = "FULL",
        created_at: str | None = None,
    ) -> Path:
        safe_id = self._safe_id(batch_id)
        target = self.batch_dir / f"{safe_id}_{cache_key[:24]}.batch.json"
        ts = created_at or datetime.now(timezone.utc).isoformat()
        payload = {
            "algo": algo,
            "schema": schema_version,
            "node_id": batch_id,
            "batch_id": batch_id,
            "created_at": ts,
            "cache_key": cache_key,
            "compaction_level": compaction_level,
            "result": data,
        }
        self._write_atomic(target, payload)
        return target

    def load_batch_result(self, batch_id: str, cache_key: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        safe_id = self._safe_id(batch_id)
        target = self.batch_dir / f"{safe_id}_{cache_key[:24]}.batch.json"
        meta: dict[str, Any] = {"hit": False, "cache_key": cache_key, "path": str(target)}

        target_file = target
        if not target_file.is_file():
            # Backward compatible search for file matching cache_key[:24]
            matches = list(self.batch_dir.glob(f"*{cache_key[:24]}.batch.json"))
            if matches:
                target_file = matches[0]
                meta["path"] = str(target_file)
            else:
                return None, meta

        try:
            raw = json.loads(target_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("cache_key") != cache_key:
                return None, meta
            res = raw.get("result")
            if not isinstance(res, dict) or not validate_hierarchy_cache_data(res):
                return None, meta
            meta["hit"] = True
            if "algo" in raw:
                meta["algo"] = raw["algo"]
            if "node_id" in raw:
                meta["node_id"] = raw["node_id"]
            if "created_at" in raw:
                meta["created_at"] = raw["created_at"]
            return res, meta
        except Exception:
            return None, meta

    # 3. Season Merges
    def save_merge_result(
        self,
        merge_id: str,
        cache_key: str,
        data: dict[str, Any],
        *,
        algo: str = "v3",
        schema_version: str = "v3",
        compaction_level: str = "FULL",
        created_at: str | None = None,
    ) -> Path:
        safe_id = self._safe_id(merge_id)
        target = self.merge_dir / f"{safe_id}_{cache_key[:24]}.merge.json"
        ts = created_at or datetime.now(timezone.utc).isoformat()
        payload = {
            "algo": algo,
            "schema": schema_version,
            "node_id": merge_id,
            "merge_id": merge_id,
            "created_at": ts,
            "cache_key": cache_key,
            "compaction_level": compaction_level,
            "result": data,
        }
        self._write_atomic(target, payload)
        return target

    def load_merge_result(self, merge_id: str, cache_key: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        safe_id = self._safe_id(merge_id)
        target = self.merge_dir / f"{safe_id}_{cache_key[:24]}.merge.json"
        meta: dict[str, Any] = {"hit": False, "cache_key": cache_key, "path": str(target)}

        target_file = target
        if not target_file.is_file():
            matches = list(self.merge_dir.glob(f"*{cache_key[:24]}.merge.json"))
            if matches:
                target_file = matches[0]
                meta["path"] = str(target_file)
            else:
                return None, meta

        try:
            raw = json.loads(target_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("cache_key") != cache_key:
                return None, meta
            res = raw.get("result")
            if not isinstance(res, dict) or not validate_hierarchy_cache_data(res):
                return None, meta
            meta["hit"] = True
            if "algo" in raw:
                meta["algo"] = raw["algo"]
            if "node_id" in raw:
                meta["node_id"] = raw["node_id"]
            if "created_at" in raw:
                meta["created_at"] = raw["created_at"]
            return res, meta
        except Exception:
            return None, meta

    # 4. Full Season Connection Result
    def save_connection_result(
        self,
        conn_key: str,
        data: dict[str, Any],
        *,
        algo: str = "v3",
        schema_version: str = "v3",
        node_id: str = "connection_root",
        created_at: str | None = None,
        compaction_level: str = "FULL",
    ) -> Path:
        safe_id = self._safe_id(node_id)
        target = self.connection_dir / f"{safe_id}_{conn_key[:24]}.connection.json"
        ts = created_at or datetime.now(timezone.utc).isoformat()
        payload = {
            "algo": algo,
            "schema": schema_version,
            "node_id": node_id,
            "created_at": ts,
            "cache_key": conn_key,
            "compaction_level": compaction_level,
            "result": data,
        }
        self._write_atomic(target, payload)
        return target

    def load_connection_result(self, conn_key: str, node_id: str = "connection_root") -> tuple[dict[str, Any] | None, dict[str, Any]]:
        safe_id = self._safe_id(node_id)
        target = self.connection_dir / f"{safe_id}_{conn_key[:24]}.connection.json"
        meta: dict[str, Any] = {"hit": False, "cache_key": conn_key, "path": str(target)}

        target_file = target
        if not target_file.is_file():
            legacy_target = self.connection_dir / f"conn_{conn_key[:24]}.connection.json"
            if legacy_target.is_file():
                target_file = legacy_target
                meta["path"] = str(target_file)
            else:
                matches = list(self.connection_dir.glob(f"*{conn_key[:24]}.connection.json"))
                if matches:
                    target_file = matches[0]
                    meta["path"] = str(target_file)
                else:
                    return None, meta

        try:
            raw = json.loads(target_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("cache_key") != conn_key:
                return None, meta
            res = raw.get("result")
            if not isinstance(res, dict) or not validate_hierarchy_cache_data(res):
                return None, meta
            meta["hit"] = True
            if "algo" in raw:
                meta["algo"] = raw["algo"]
            if "node_id" in raw:
                meta["node_id"] = raw["node_id"]
            if "created_at" in raw:
                meta["created_at"] = raw["created_at"]
            return res, meta
        except Exception:
            return None, meta

    # 5. Finalizer Groups
    def save_finalizer_result(
        self,
        group_id: str,
        cache_key: str,
        data: dict[str, Any],
        *,
        algo: str = "v3",
        schema_version: str = "v3",
        node_id: str = "",
        created_at: str | None = None,
    ) -> Path:
        if not validate_finalizer_cache_data(data):
            raise ValueError("Invalid finalizer cache data: 'outputs' must be a list of dicts.")
        safe_id = self._safe_id(group_id)
        target = self.finalizer_dir / f"{safe_id}_{cache_key[:24]}.finalizer.json"
        ts = created_at or datetime.now(timezone.utc).isoformat()
        payload = {
            "algo": algo,
            "schema": schema_version,
            "node_id": node_id or group_id,
            "group_id": group_id,
            "created_at": ts,
            "cache_key": cache_key,
            "result": data,
        }
        self._write_atomic(target, payload)
        return target

    def load_finalizer_result(
        self,
        group_id: str,
        cache_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        safe_id = self._safe_id(group_id)
        target = self.finalizer_dir / f"{safe_id}_{cache_key[:24]}.finalizer.json"
        meta: dict[str, Any] = {"hit": False, "cache_key": cache_key, "path": str(target)}

        target_file = target
        if not target_file.is_file():
            matches = list(self.finalizer_dir.glob(f"*{cache_key[:24]}.finalizer.json"))
            if matches:
                target_file = matches[0]
                meta["path"] = str(target_file)
            else:
                return None, meta

        try:
            raw = json.loads(target_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("cache_key") != cache_key:
                return None, meta
            res = raw.get("result")
            if not isinstance(res, dict) or not validate_finalizer_cache_data(res):
                return None, meta
            meta["hit"] = True
            if "algo" in raw:
                meta["algo"] = raw["algo"]
            if "node_id" in raw:
                meta["node_id"] = raw["node_id"]
            if "created_at" in raw:
                meta["created_at"] = raw["created_at"]
            return res, meta
        except Exception:
            return None, meta
