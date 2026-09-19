"""Tests for safe cancellation, subprocess termination, and UI state restoration."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
import pytest

from toolrecap_v2.media import RenderCancelled, run_command
from toolrecap_v2.projects import ProjectQueue, ProjectRecord, ProjectStore
from toolrecap_v2.settings import AppSettings


def test_run_command_safe_cancellation() -> None:
    cancel_event = threading.Event()

    # Run a long sleep command
    if sys.platform == "win32":
        cmd = ["powershell", "-NoProfile", "-Command", "Start-Sleep -Seconds 10"]
    else:
        cmd = ["sleep", "10"]

    # Trigger cancel shortly after start
    def _trigger() -> None:
        time.sleep(0.2)
        cancel_event.set()

    t = threading.Thread(target=_trigger)
    t.start()

    start_time = time.time()
    with pytest.raises(RenderCancelled):
        run_command(cmd, cancel_event=cancel_event)
    elapsed = time.time() - start_time
    t.join()

    # Process must have been killed promptly (< 3 seconds, not 10 seconds)
    assert elapsed < 4.0


def test_queue_cancellation_and_ui_restoration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    settings = AppSettings()

    p1 = ProjectRecord(
        id="p1",
        name="Episode 1",
        source_video=str(tmp_path / "ep1.mp4"),
        output_directory=str(tmp_path / "out1"),
    )
    p2 = ProjectRecord(
        id="p2",
        name="Episode 2",
        source_video=str(tmp_path / "ep2.mp4"),
        output_directory=str(tmp_path / "out2"),
    )

    state_changes: list[bool] = []

    def _state_cb(is_running: bool) -> None:
        state_changes.append(is_running)

    queue = ProjectQueue(
        [p1, p2],
        store=store,
        settings=settings,
        on_state_change=_state_cb,
    )

    # Simulate realistic processing duration so cancellation can be tested deterministically
    def _mock_process(record: ProjectRecord) -> None:
        record.status = "RUNNING"
        for _ in range(30):
            if queue._cancel_event.is_set():
                raise RenderCancelled()
            time.sleep(0.05)
        record.status = "COMPLETED"

    monkeypatch.setattr(queue, "_process_single_project", _mock_process)

    queue.start()
    assert queue.is_running is True

    # Cancel while p1 is running
    time.sleep(0.1)
    queue.cancel()

    # Wait for queue thread to terminate
    if queue._thread:
        queue._thread.join(timeout=5)

    assert queue.is_running is False
    # UI state must have received True (when started) and False (when stopped)
    assert True in state_changes
    assert state_changes[-1] is False

    # Records should be in CANCELLED state
    assert p1.status == "CANCELLED"
    assert p2.status == "CANCELLED"


def test_queue_restores_ui_on_error(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    settings = AppSettings()

    # Record pointing to non-existent video
    bad_p = ProjectRecord(
        id="bad",
        name="Bad Episode",
        source_video=str(tmp_path / "nonexistent.mp4"),
        output_directory=str(tmp_path / "out"),
    )

    state_changes: list[bool] = []
    queue = ProjectQueue(
        [bad_p],
        store=store,
        settings=settings,
        on_state_change=lambda is_running: state_changes.append(is_running),
    )

    queue.start()
    if queue._thread:
        queue._thread.join(timeout=5)

    assert queue.is_running is False
    assert state_changes[-1] is False
    assert bad_p.status == "ERROR"
    assert "Không tìm thấy" in (bad_p.error or "")
