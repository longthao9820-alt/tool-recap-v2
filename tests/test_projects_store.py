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
