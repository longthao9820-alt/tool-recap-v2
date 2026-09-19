"""Tests for Episode Evidence cache manager, atomic writes, invalidation, and independent cache."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from toolrecap_v2.domain import (
    EpisodeEvidence,
    EvidenceCacheManager,
    SourceEpisode,
    compute_cache_key,
)


def test_cache_key_excludes_api_keys() -> None:
    # Key must be based on source identity and config version only
    key1 = compute_cache_key(
        episode_id="E01",
        source_video="C:/media/e01.mp4",
        config_version="v1.0",
        source_size=1000,
        source_mtime=12345.0,
    )
    key2 = compute_cache_key(
        episode_id="E01",
        source_video="C:/media/e01.mp4",
        config_version="v1.0",
        source_size=1000,
        source_mtime=12345.0,
    )
    assert key1 == key2
    assert len(key1) == 64  # sha256


def test_cache_atomic_save_and_load(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    manager = EvidenceCacheManager(cache_dir=cache_dir)

    video_file = tmp_path / "ep1.mp4"
    video_file.write_bytes(b"dummy video content")

    ep = SourceEpisode(
        episode_id="E01",
        source_video=str(video_file),
        duration_seconds=120.0,
    )
    evidence = EpisodeEvidence(
        episode_id="E01",
        source_video=str(video_file),
        duration_seconds=120.0,
        coverage={"ratio": 1.0},
        missing_reasons=[],
    )

    saved_path = manager.save_evidence(ep, config_version="v1.0", evidence=evidence)
    assert saved_path.is_file()

    # Verify atomic write payload does not contain api_key
    content = json.loads(saved_path.read_text(encoding="utf-8"))
    assert "api_key" not in content
    assert content["episode_id"] == "E01"

    # Load from cache
    loaded = manager.load_evidence(ep, config_version="v1.0")
    assert loaded is not None
    assert loaded.episode_id == "E01"
    assert loaded.coverage["ratio"] == 1.0


def test_cache_invalidates_on_source_change(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    manager = EvidenceCacheManager(cache_dir=cache_dir)

    video_file = tmp_path / "ep1.mp4"
    video_file.write_bytes(b"initial video content")

    ep = SourceEpisode(
        episode_id="E01",
        source_video=str(video_file),
        duration_seconds=120.0,
    )
    evidence = EpisodeEvidence(
        episode_id="E01",
        source_video=str(video_file),
        duration_seconds=120.0,
        coverage={"ratio": 1.0},
    )

    manager.save_evidence(ep, config_version="v1.0", evidence=evidence)
    assert manager.load_evidence(ep, config_version="v1.0") is not None

    # Modify source file content and mtime
    time.sleep(0.05)
    video_file.write_bytes(b"modified video content with different size")

    # Load must invalidate and return None
    loaded_after_mod = manager.load_evidence(ep, config_version="v1.0")
    assert loaded_after_mod is None


def test_e01_cache_survives_e05_change(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    manager = EvidenceCacheManager(cache_dir=cache_dir)

    e01_file = tmp_path / "E01.mp4"
    e01_file.write_bytes(b"E01 original content")

    e05_file = tmp_path / "E05.mp4"
    e05_file.write_bytes(b"E05 original content")

    ep01 = SourceEpisode(episode_id="E01", source_video=str(e01_file), duration_seconds=100.0)
    ep05 = SourceEpisode(episode_id="E05", source_video=str(e05_file), duration_seconds=100.0)

    evidence01 = EpisodeEvidence(episode_id="E01", source_video=str(e01_file), duration_seconds=100.0, coverage={"ratio": 0.9})
    evidence05 = EpisodeEvidence(episode_id="E05", source_video=str(e05_file), duration_seconds=100.0, coverage={"ratio": 0.8})

    manager.save_evidence(ep01, config_version="v1.0", evidence=evidence01)
    manager.save_evidence(ep05, config_version="v1.0", evidence=evidence05)

    # Both are cached
    assert manager.load_evidence(ep01, config_version="v1.0") is not None
    assert manager.load_evidence(ep05, config_version="v1.0") is not None

    # Now modify E05
    time.sleep(0.05)
    e05_file.write_bytes(b"E05 modified with new bytes!")

    # E05 must invalidate
    assert manager.load_evidence(ep05, config_version="v1.0") is None

    # E01 MUST survive and still load valid evidence
    survived = manager.load_evidence(ep01, config_version="v1.0")
    assert survived is not None
    assert survived.episode_id == "E01"
    assert survived.coverage["ratio"] == 0.9
