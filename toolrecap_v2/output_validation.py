"""Output and publication validation module for ToolRecap V2.

Strictly validates publication directory cleanliness, MP4 file integrity via ffprobe,
and SRT subtitle format/timings/non-emptiness.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .media import MediaError, MediaProbeResult, probe_typed_media


class OutputValidationError(MediaError):
    """Raised when publication folder or artifacts fail validation."""
    pass


SRT_TIMESTAMP_PATTERN = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def parse_srt_timestamp(h: str, m: str, s: str, ms: str) -> float:
    """Parse SRT timestamp components to total seconds."""
    return int(h) * 3600.0 + int(m) * 60.0 + int(s) + int(ms) / 1000.0


def validate_srt_file(
    srt_path: Path | str,
    *,
    max_duration: float | None = None,
    allow_empty: bool = False,
) -> list[tuple[float, float, str]]:
    """Validate and parse an SRT subtitle file.

    Enforces:
    - File exists and is non-empty.
    - Contains valid sequential cue blocks with standard timestamps.
    - Timestamps are non-negative and monotonically non-decreasing per cue.
    - Timestamps stay within max_duration (+ tolerance).
    - Subtitle content is non-empty (unless allow_empty=True).
    """
    path = Path(srt_path).resolve()
    if not path.is_file():
        raise OutputValidationError(f"Tệp phụ đề SRT không tồn tại: {path}")

    if path.stat().st_size == 0:
        if allow_empty:
            return []
        raise OutputValidationError(f"Tệp phụ đề SRT rỗng (0 bytes): {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise OutputValidationError(f"Tệp SRT không đúng định dạng UTF-8: {exc}") from exc

    blocks = [b.strip() for b in re.split(r"\n\s*\n", content.strip()) if b.strip()]
    if not blocks:
        if allow_empty:
            return []
        raise OutputValidationError(f"Tệp phụ đề SRT không chứa phân đoạn phụ đề hợp lệ: {path}")

    cues: list[tuple[float, float, str]] = []
    for b_idx, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        # Look for timestamp line
        match = None
        text_start_idx = 1
        for idx, line in enumerate(lines[:3]):
            m = SRT_TIMESTAMP_PATTERN.search(line)
            if m:
                match = m
                text_start_idx = idx + 1
                break

        if not match:
            raise OutputValidationError(
                f"Phân đoạn phụ đề #{b_idx} thiếu dòng mốc thời gian hợp lệ trong '{path.name}':\n{block}"
            )

        start_sec = parse_srt_timestamp(match.group(1), match.group(2), match.group(3), match.group(4))
        end_sec = parse_srt_timestamp(match.group(5), match.group(6), match.group(7), match.group(8))

        if start_sec < 0.0:
            raise OutputValidationError(
                f"Mốc bắt đầu phụ đề âm ({start_sec}s) trong phân đoạn #{b_idx} của '{path.name}'."
            )
        if end_sec < start_sec:
            raise OutputValidationError(
                f"Mốc kết thúc ({end_sec}s) nhỏ hơn mốc bắt đầu ({start_sec}s) trong phân đoạn #{b_idx} của '{path.name}'."
            )

        if max_duration is not None and start_sec > max_duration + 1.0:
            raise OutputValidationError(
                f"Mốc phụ đề bắt đầu ({start_sec}s) vượt quá thời lượng video ({max_duration:.2f}s) trong '{path.name}'."
            )

        cue_text = "\n".join(lines[text_start_idx:]).strip()
        cues.append((start_sec, end_sec, cue_text))

    if not cues and not allow_empty:
        raise OutputValidationError(f"Tệp SRT không có nội dung phụ đề hợp lệ: {path}")

    return cues


def validate_mp4_file(
    mp4_path: Path | str,
    *,
    min_duration: float = 0.1,
    require_audio: bool = True,
) -> MediaProbeResult:
    """Validate an MP4 video file using ffprobe.

    Enforces:
    - File exists and has non-zero size.
    - Valid ffprobe output.
    - Has at least one valid video stream.
    - Has audio stream if require_audio is True.
    - Duration is at least min_duration seconds.
    """
    path = Path(mp4_path).resolve()
    if not path.is_file():
        raise OutputValidationError(f"Tệp video MP4 không tồn tại: {path}")

    if path.stat().st_size == 0:
        raise OutputValidationError(f"Tệp video MP4 rỗng (0 bytes): {path}")

    probe = probe_typed_media(path)

    if not probe.has_video:
        raise OutputValidationError(f"Tệp video MP4 không chứa luồng video (has_video=False): {path}")

    if probe.duration < min_duration:
        raise OutputValidationError(
            f"Thời lượng video MP4 quá ngắn ({probe.duration:.2f}s < {min_duration}s): {path}"
        )

    if require_audio and not probe.has_audio:
        raise OutputValidationError(f"Tệp video MP4 thiếu luồng âm thanh (has_audio=False): {path}")

    return probe


def validate_publication_folder(
    pub_dir: Path | str,
    safe_title: str,
    *,
    expected_duration: float | None = None,
    check_original_srt: bool = True,
    check_narration_srt: bool = True,
    require_audio: bool = True,
) -> dict[str, Any]:
    """Validate that the publication folder is clean, complete, and contains exactly the expected 3 files.

    Expected exact files:
    1. {safe_title}.mp4
    2. {safe_title}.original.srt
    3. {safe_title}.narration.srt

    Any extraneous files (temp, logs, partial files) trigger validation failure.
    """
    p_dir = Path(pub_dir).resolve()
    if not p_dir.is_dir():
        raise OutputValidationError(f"Thư mục xuất bản không tồn tại: {p_dir}")

    # Check directory contents
    actual_files = {f.name for f in p_dir.iterdir() if f.is_file()}
    actual_dirs = [d.name for d in p_dir.iterdir() if d.is_dir()]

    if actual_dirs:
        raise OutputValidationError(
            f"Thư mục xuất bản chứa thư mục con không mong muốn ({actual_dirs}) trong '{p_dir}'."
        )

    expected_video_name = f"{safe_title}.mp4"
    expected_orig_srt_name = f"{safe_title}.original.srt"
    expected_narr_srt_name = f"{safe_title}.narration.srt"

    expected_files = {expected_video_name, expected_orig_srt_name, expected_narr_srt_name}

    missing_files = expected_files - actual_files
    if missing_files:
        raise OutputValidationError(
            f"Thư mục xuất bản thiếu các tệp bắt buộc {sorted(missing_files)} trong '{p_dir}'."
        )

    extra_files = actual_files - expected_files
    if extra_files:
        raise OutputValidationError(
            f"Thư mục xuất bản chứa tệp thừa/tạm ({sorted(extra_files)}) trong '{p_dir}'."
        )

    # Validate video
    video_path = p_dir / expected_video_name
    probe = validate_mp4_file(video_path, require_audio=require_audio)

    # Validate original subtitles
    orig_srt_path = p_dir / expected_orig_srt_name
    orig_cues = validate_srt_file(
        orig_srt_path,
        max_duration=probe.duration,
        allow_empty=not check_original_srt,
    )

    # Validate narration subtitles
    narr_srt_path = p_dir / expected_narr_srt_name
    narr_cues = validate_srt_file(
        narr_srt_path,
        max_duration=probe.duration,
        allow_empty=not check_narration_srt,
    )

    return {
        "pub_dir": str(p_dir),
        "video_path": str(video_path),
        "original_srt_path": str(orig_srt_path),
        "narration_srt_path": str(narr_srt_path),
        "duration": probe.duration,
        "width": probe.width,
        "height": probe.height,
        "original_cues_count": len(orig_cues),
        "narration_cues_count": len(narr_cues),
    }
