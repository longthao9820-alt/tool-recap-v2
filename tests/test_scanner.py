"""Tests for non-recursive folder scanning and deterministic natural sorting."""
from __future__ import annotations

from pathlib import Path
from toolrecap_v2.scanner import natural_sort_key, scan_videos, SUPPORTED_VIDEO_EXTENSIONS


def test_natural_sort_key() -> None:
    files = [
        Path("ep10.mp4"),
        Path("ep1.mp4"),
        Path("ep2.mp4"),
        Path("ep20.mp4"),
        Path("ep3.mp4"),
    ]
    sorted_files = sorted(files, key=natural_sort_key)
    assert [f.name for f in sorted_files] == [
        "ep1.mp4",
        "ep2.mp4",
        "ep3.mp4",
        "ep10.mp4",
        "ep20.mp4",
    ]


def test_scan_videos_non_recursive(tmp_path: Path) -> None:
    # Root level video files
    v1 = tmp_path / "ep01.mp4"
    v2 = tmp_path / "ep02.mkv"
    v1.write_bytes(b"v1")
    v2.write_bytes(b"v2")

    # Root level non-video file
    txt = tmp_path / "notes.txt"
    txt.write_text("not video", encoding="utf-8")

    # Subdirectory with videos
    sub = tmp_path / "subfolder"
    sub.mkdir()
    sub_v = sub / "ep03.mp4"
    sub_v.write_bytes(b"sub_v")

    # Nested subfolder
    nested = sub / "nested"
    nested.mkdir()
    nested_v = nested / "ep04.avi"
    nested_v.write_bytes(b"nested_v")

    # Scan root folder: MUST be non-recursive, only direct root videos
    scanned = scan_videos(tmp_path)
    assert scanned == [v1.resolve(), v2.resolve()]
    assert sub_v.resolve() not in scanned
    assert nested_v.resolve() not in scanned


def test_scan_videos_supported_extensions(tmp_path: Path) -> None:
    expected: list[Path] = []
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        f = tmp_path / f"video{ext}"
        f.write_bytes(b"vid")
        expected.append(f.resolve())

    # Unsupported
    bad = tmp_path / "audio.mp3"
    bad.write_bytes(b"audio")

    scanned = scan_videos(tmp_path)
    expected.sort(key=natural_sort_key)
    assert scanned == expected
    assert bad.resolve() not in scanned


def test_scan_single_file(tmp_path: Path) -> None:
    v = tmp_path / "single.mp4"
    v.write_bytes(b"data")
    assert scan_videos(v) == [v.resolve()]

    bad = tmp_path / "single.jpg"
    bad.write_bytes(b"data")
    assert scan_videos(bad) == []


def test_scan_nonexistent_path(tmp_path: Path) -> None:
    assert scan_videos(tmp_path / "does_not_exist") == []
    assert scan_videos("") == []
