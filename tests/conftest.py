"""Pytest fixtures and test environment setup for ToolRecap V2."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

# Ensure Tkinter / Tcl libraries can be found if needed in tests
_py_base = Path(sys.base_prefix)
_tcl_dir = _py_base / "tcl" / "tcl8.6"
_tk_dir = _py_base / "tcl" / "tk8.6"
if _tcl_dir.is_dir():
    os.environ["TCL_LIBRARY"] = _tcl_dir.as_posix()
if _tk_dir.is_dir():
    os.environ["TK_LIBRARY"] = _tk_dir.as_posix()


@pytest.fixture(scope="session")
def tk_root():
    import tkinter as tk
    import gc
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.update_idletasks()
        root.update()
    except Exception:
        pass
    gc.collect()
    try:
        root.destroy()
    except Exception:
        pass
    gc.collect()


@pytest.fixture(scope="session")
def _shared_dummy_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import subprocess
    from toolrecap_v2.gpu import bundled_binary

    base_dir = tmp_path_factory.mktemp("shared_video")
    video_path = base_dir / "base_sample.mp4"
    ffmpeg = bundled_binary("ffmpeg")
    if ffmpeg:
        cmd = [
            str(ffmpeg),
            "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2.0:size=320x240:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2.0",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            str(video_path),
        ]
        subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if not video_path.is_file():
        video_path.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 200)
    return video_path


@pytest.fixture
def dummy_video(tmp_path: Path, _shared_dummy_video: Path) -> Path:
    """Create a minimal real MP4 video file using cached session base for fast tests."""
    import shutil
    video_path = tmp_path / "sample_video.mp4"
    shutil.copyfile(_shared_dummy_video, video_path)
    return video_path
