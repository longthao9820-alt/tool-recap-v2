"""Video file scanner with non-recursive folder scanning and deterministic natural sorting."""
from __future__ import annotations

import re
from pathlib import Path


SUPPORTED_VIDEO_EXTENSIONS = frozenset({
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".ts",
})


def natural_sort_key(path: Path | str) -> list[int | str]:
    """Generate a natural sort key so that numbers inside strings are sorted numerically.
    Example: ['ep1', 'ep2', 'ep10'] instead of ['ep1', 'ep10', 'ep2'].
    """
    name = Path(path).name.lower()
    parts = re.split(r"(\d+)", name)
    key: list[int | str] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        elif part:
            key.append(part)
    return key


def scan_videos(target_path: Path | str) -> list[Path]:
    """Scan target path for supported video files.
    - If target_path is a single file: returns [target_path] if supported, else [].
    - If target_path is a directory: scans directly (NON-RECURSIVE), filters supported extensions,
      and sorts deterministically using natural sort.
    - If path does not exist or is invalid: returns [].
    """
    if not target_path:
        return []

    path = Path(target_path).expanduser().resolve()
    if not path.exists():
        return []

    if path.is_file():
        if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            return [path]
        return []

    if path.is_dir():
        videos: list[Path] = []
        try:
            for entry in path.iterdir():
                if entry.is_file() and entry.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    videos.append(entry)
        except OSError:
            return []

        videos.sort(key=natural_sort_key)
        return videos

    return []
