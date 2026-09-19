"""Tests for project persistence and state recovery on crash/restart."""
from __future__ import annotations

import json
from pathlib import Path

from toolrecap_v2.projects import ProjectRecord, ProjectStore


def test_project_store_atomic_save_and_load(tmp_path: Path) -> None:
    store_file = tmp_path / "projects.json"
    store = ProjectStore(store_file)

    p1 = ProjectRecord(
        id="p1",
        name="Episode 1",
        source_video=str(tmp_path / "ep1.mp4"),
        output_directory=str(tmp_path / "out1"),
        status="COMPLETED",
        progress=100,
    )
    p2 = ProjectRecord(
        id="p2",
        name="Episode 2",
        source_video=str(tmp_path / "ep2.mp4"),
        output_directory=str(tmp_path / "out2"),
        status="WAITING",
        progress=0,
    )

    store.save([p1, p2])
    assert store_file.is_file()

    loaded = store.load()
    assert len(loaded) == 2
    assert loaded[0].id == "p1"
    assert loaded[0].status == "COMPLETED"
    assert loaded[1].id == "p2"
    assert loaded[1].status == "WAITING"


def test_project_store_state_recovery_resets_interrupted(tmp_path: Path) -> None:
    store_file = tmp_path / "projects.json"
    # Write file simulating previous crash while RUNNING
    crash_data = {
        "projects": [
            {
                "id": "crashed_proj",
                "name": "Crashed Episode",
                "source_video": str(tmp_path / "video.mp4"),
                "status": "RUNNING",
                "progress": 45,
                "current_message": "Đang render...",
            }
        ]
    }
    store_file.write_text(json.dumps(crash_data), encoding="utf-8")

    store = ProjectStore(store_file)
    loaded = store.load()
    assert len(loaded) == 1
    # Must recover by resetting status to PAUSED
    assert loaded[0].status == "PAUSED"
    assert "Đã dừng khi ứng dụng đóng trước đó" in loaded[0].current_message


def test_legacy_record_migration_does_not_force_fake_outputs(tmp_path: Path) -> None:
    store_file = tmp_path / "projects.json"
    legacy_data = {
        "projects": [
            {
                "id": "legacy_1",
                "name": "Legacy Ep 1",
                "source_video": str(tmp_path / "legacy.mp4"),
                "status": "COMPLETED",
                "progress": 100,
                "output_video": str(tmp_path / "out.mp4"),
                "output_srt": str(tmp_path / "out.srt"),
            }
        ]
    }
    store_file.write_text(json.dumps(legacy_data), encoding="utf-8")

    store = ProjectStore(store_file)
    loaded = store.load()
    assert len(loaded) == 1
    rec = loaded[0]

    # Source episode migrated from source_video
    assert len(rec.source_episodes) == 1
    assert rec.source_episodes[0].episode_id == "E01"
    assert rec.source_episodes[0].source_video == str(tmp_path / "legacy.mp4")

    # Outputs must remain empty (do NOT force legacy records into completed fake outputs)
    assert rec.outputs == []

    # Subtitle aliases
    assert rec.output_original_srt == str(tmp_path / "out.srt")
    assert rec.output_narration_srt == str(tmp_path / "out.srt")
    assert rec.output_srt == str(tmp_path / "out.srt")
    assert rec.analysis_scope == "SINGLE_EPISODE"
    assert rec.phase == "IDLE"


def test_project_record_multi_episode_persistence(tmp_path: Path) -> None:
    from toolrecap_v2.domain import CommentaryOutput, SourceEpisode

    store_file = tmp_path / "projects.json"
    store = ProjectStore(store_file)

    ep1 = SourceEpisode(episode_id="E01", source_video="C:/media/e1.mp4", title="Ep 1")
    ep2 = SourceEpisode(episode_id="E02", source_video="C:/media/e2.mp4", title="Ep 2")
    out1 = CommentaryOutput(output_id="out_1", title="Recap 1", sanitized_title="Recap 1")

    p = ProjectRecord(
        id="season_proj",
        name="Season 1 Recap",
        source_video="C:/media/e1.mp4",
        analysis_scope="SEASON",
        source_episodes=[ep1, ep2],
        outputs=[out1],
        phase="ANALYZING",
    )

    store.save([p])
    loaded = store.load()
    assert len(loaded) == 1
    rec = loaded[0]

    assert rec.analysis_scope == "SEASON"
    assert rec.phase == "ANALYZING"
    assert len(rec.source_episodes) == 2
    assert rec.source_episodes[0].episode_id == "E01"
    assert rec.source_episodes[1].episode_id == "E02"
    assert len(rec.outputs) == 1
    assert rec.outputs[0].output_id == "out_1"
    assert rec.outputs[0].title == "Recap 1"
