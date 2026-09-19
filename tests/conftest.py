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
if _tcl_dir.is_dir() and "TCL_LIBRARY" not in os.environ:
    os.environ["TCL_LIBRARY"] = str(_tcl_dir)
if _tk_dir.is_dir() and "TK_LIBRARY" not in os.environ:
    os.environ["TK_LIBRARY"] = str(_tk_dir)


@pytest.fixture(scope="session")
def tk_root():
    import tkinter as tk
    import gc
    root = tk.Tk()
    root.withdraw()
    yield root
    gc.collect()
    try:
        root.update()
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    """Create a minimal real MP4 video file using ffmpeg for testing."""
    import subprocess
    from toolrecap_v2.gpu import bundled_binary

    video_path = tmp_path / "sample_video.mp4"
    ffmpeg = bundled_binary("ffmpeg")
    if ffmpeg:
        # Generate 2 seconds of test video with silence
        cmd = [
            str(ffmpeg),
            "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2:r=24",
            "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            "-shortest",
            str(video_path),
        ]
        subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    if not video_path.is_file():
        # Fallback dummy file
        video_path.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 200)

    return video_path
