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


def test_two_e01_evidence_collision_isolation_and_selective_invalidation(tmp_path: Path) -> None:
    """Two different video files with identical episode ID E01 do not collide and can be selectively invalidated."""
    cache_dir = tmp_path / "cache"
    manager = EvidenceCacheManager(cache_dir=cache_dir)

    dir_a = tmp_path / "show_a"
    dir_b = tmp_path / "show_b"
    dir_a.mkdir()
    dir_b.mkdir()

    video_a = dir_a / "E01.mp4"
    video_b = dir_b / "E01.mp4"
    video_a.write_bytes(b"Show A Episode 1 video data")
    video_b.write_bytes(b"Show B Episode 1 distinct video data")

    ep_a = SourceEpisode(episode_id="E01", source_video=str(video_a), duration_seconds=120.0)
    ep_b = SourceEpisode(episode_id="E01", source_video=str(video_b), duration_seconds=240.0)

    ev_a = EpisodeEvidence(
        episode_id="E01",
        source_video=str(video_a),
        duration_seconds=120.0,
        coverage={"ratio": 0.95},
        data={"scenes": ["Show A Scene 1"]},
    )
    ev_b = EpisodeEvidence(
        episode_id="E01",
        source_video=str(video_b),
        duration_seconds=240.0,
        coverage={"ratio": 0.85},
        data={"scenes": ["Show B Scene 1"]},
    )

    path_a = manager.save_evidence(ep_a, config_version="v1.0", evidence=ev_a)
    path_b = manager.save_evidence(ep_b, config_version="v1.0", evidence=ev_b)

    # Different cache files created
    assert path_a != path_b
    assert path_a.is_file()
    assert path_b.is_file()

    # Both load their distinct evidence without collision
    loaded_a = manager.load_evidence(ep_a, config_version="v1.0")
    loaded_b = manager.load_evidence(ep_b, config_version="v1.0")
    assert loaded_a is not None
    assert loaded_b is not None
    assert loaded_a.data["scenes"] == ["Show A Scene 1"]
    assert loaded_b.data["scenes"] == ["Show B Scene 1"]

    # Selective invalidation of video_a only
    manager.invalidate("E01", source_video=video_a)
    assert manager.load_evidence(ep_a, config_version="v1.0") is None
    assert manager.load_evidence(ep_b, config_version="v1.0") is not None

    # Full invalidation without source_video removes all remaining E01 caches
    manager.invalidate("E01")
    assert manager.load_evidence(ep_b, config_version="v1.0") is None


def test_evidence_legacy_migration(tmp_path: Path) -> None:
    """Legacy unhashed evidence cache file safely migrates to hashed filename on load."""
    cache_dir = tmp_path / "cache"
    manager = EvidenceCacheManager(cache_dir=cache_dir)

    video_file = tmp_path / "E01.mp4"
    video_file.write_bytes(b"Legacy video content bytes")
    stat = video_file.stat()

    legacy_file = cache_dir / "E01.evidence.json"
    ev = EpisodeEvidence(
        episode_id="E01",
        source_video=str(video_file),
        duration_seconds=100.0,
        coverage={"ratio": 1.0},
        data={"notes": "Legacy note"},
    )
    payload = {
        "cache_key": "legacy_key",
        "episode_id": "E01",
        "source_video": str(video_file.resolve()),
        "config_version": "v1.0",
        "source_size": stat.st_size,
        "source_mtime": stat.st_mtime,
        "evidence": ev.to_dict(),
    }
    legacy_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert legacy_file.is_file()

    # Load triggers automatic safe migration
    loaded = manager.load_evidence("E01", config_version="v1.0", source_video=video_file)
    assert loaded is not None
    assert loaded.data["notes"] == "Legacy note"

    # Legacy file was unlinked and migrated to hashed file
    assert not legacy_file.is_file()
    hashed_files = list(cache_dir.glob("E01_*.evidence.json"))
    assert len(hashed_files) == 1

    # Subsequent load succeeds from new hashed file
    reloaded = manager.load_evidence("E01", config_version="v1.0", source_video=video_file)
    assert reloaded is not None
    assert reloaded.data["notes"] == "Legacy note"
