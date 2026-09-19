"""Direct subtitle parsers for SRT, ASS/SSA, and WebVTT with tag normalization."""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Sequence

from .models import SubtitleCue


def strip_formatting_tags(text: str) -> str:
    """Strip HTML tags, ASS override tags, and WebVTT styling tags from subtitle text."""
    if not text:
        return ""

    # Replace ASS line breaks \N, \n, and hard space \h
    cleaned = text.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")

    # Strip ASS curly brace override tags: {\an8}, {\pos(1,2)}, {\c&H...&}, etc.
    cleaned = re.sub(r"\{[^}]*\}", "", cleaned)

    # Strip HTML / WebVTT angle bracket tags: <i>, <b>, <font...>, <v Speaker>, <c.color>, etc.
    cleaned = re.sub(r"</?[a-zA-Z0-9_\-.:]+(?:\s+[^>]*)?>", "", cleaned)

    # Unescape HTML entities
    cleaned = html.unescape(cleaned)

    # Normalize line breaks and whitespace
    lines = [line.strip() for line in cleaned.splitlines()]
    # Remove consecutive blank lines
    filtered_lines: list[str] = []
    prev_blank = False
    for line in lines:
        if line:
            filtered_lines.append(line)
            prev_blank = False
        elif not prev_blank:
            filtered_lines.append("")
            prev_blank = True

    return "\n".join(filtered_lines).strip()


def normalize_subtitle_text(text: str) -> str:
    """Normalize subtitle text, removing tags and trimming whitespace."""
    return strip_formatting_tags(text)


def parse_timestamp_srt(ts_str: str) -> int:
    """Parse SRT timestamp 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' to milliseconds."""
    ts_str = ts_str.strip().replace(",", ".")
    match = re.match(r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{1,3})", ts_str)
    if not match:
        raise ValueError(f"Invalid SRT timestamp format: {ts_str}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis_str = match.group(4).ljust(3, "0")[:3]
    millis = int(millis_str)
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


def parse_timestamp_vtt(ts_str: str) -> int:
    """Parse WebVTT timestamp 'HH:MM:SS.mmm' or 'MM:SS.mmm' to milliseconds."""
    ts_str = ts_str.strip()
    match = re.match(r"(?:(?:(\d+):)?(\d{2}):)?(\d{2})[.,](\d{1,3})", ts_str)
    if not match:
        raise ValueError(f"Invalid WebVTT timestamp format: {ts_str}")
    h_str, m_str, s_str, ms_str = match.groups()
    hours = int(h_str) if h_str is not None else 0
    minutes = int(m_str) if m_str is not None else 0
    seconds = int(s_str)
    millis = int(ms_str.ljust(3, "0")[:3])
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


def parse_timestamp_ass(ts_str: str) -> int:
    """Parse ASS/SSA timestamp 'H:MM:SS.cc' (centiseconds) to milliseconds."""
    ts_str = ts_str.strip()
    match = re.match(r"(\d+):(\d{2}):(\d{2})[.,](\d{1,3})", ts_str)
    if not match:
        raise ValueError(f"Invalid ASS timestamp format: {ts_str}")
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    cs_str = match.group(4)
    if len(cs_str) == 2:
        millis = int(cs_str) * 10
    elif len(cs_str) == 1:
        millis = int(cs_str) * 100
    else:
        millis = int(cs_str[:3].ljust(3, "0"))
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


def parse_srt(
    content_or_path: str | Path,
    *,
    source_type: str = "sidecar",
    source_format: str = "srt",
    stream_index: int | None = None,
    source_file: str | None = None,
    language: str = "eng",
    episode_id: str = "",
    source_video: str = "",
) -> list[SubtitleCue]:
    """Parse SubRip (.srt) subtitle content or file into normalized SubtitleCue items."""
    if isinstance(content_or_path, Path) or (isinstance(content_or_path, str) and "\n" not in content_or_path and Path(content_or_path).is_file()):
        p = Path(content_or_path)
        source_file = source_file or str(p.resolve())
        content = p.read_text(encoding="utf-8-sig", errors="replace")
    else:
        content = str(content_or_path)

    # Normalize line breaks
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Split into cue blocks
    blocks = re.split(r"\n\s*\n+", content.strip())
    cues: list[SubtitleCue] = []

    time_pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})"
    )

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        time_line_idx = -1
        time_match = None
        for idx, line in enumerate(lines):
            m = time_pattern.search(line)
            if m:
                time_line_idx = idx
                time_match = m
                break

        if not time_match or time_line_idx == -1:
            continue

        start_str, end_str = time_match.group(1), time_match.group(2)
        try:
            start_ms = parse_timestamp_srt(start_str)
            end_ms = parse_timestamp_srt(end_str)
        except ValueError:
            continue

        if end_ms <= start_ms:
            continue

        raw_text = "\n".join(lines[time_line_idx + 1:])
        clean_text = normalize_subtitle_text(raw_text)
        if not clean_text:
            continue

        cues.append(
            SubtitleCue(
                start_ms=start_ms,
                end_ms=end_ms,
                text=clean_text,
                source_type=source_type,
                source_format=source_format,
                stream_index=stream_index,
                source_file=source_file,
                language=language,
                confidence=1.0,
                episode_id=episode_id,
                source_video=source_video,
            )
        )

    return cues


def parse_vtt(
    content_or_path: str | Path,
    *,
    source_type: str = "sidecar",
    source_format: str = "vtt",
    stream_index: int | None = None,
    source_file: str | None = None,
    language: str = "eng",
    episode_id: str = "",
    source_video: str = "",
) -> list[SubtitleCue]:
    """Parse WebVTT (.vtt) subtitle content or file into normalized SubtitleCue items."""
    if isinstance(content_or_path, Path) or (isinstance(content_or_path, str) and "\n" not in content_or_path and Path(content_or_path).is_file()):
        p = Path(content_or_path)
        source_file = source_file or str(p.resolve())
        content = p.read_text(encoding="utf-8-sig", errors="replace")
    else:
        content = str(content_or_path)

    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Remove WEBVTT header and header comments/styles
    if content.startswith("WEBVTT"):
        parts = re.split(r"\n\s*\n+", content, maxsplit=1)
        content = parts[1] if len(parts) > 1 else ""

    blocks = re.split(r"\n\s*\n+", content.strip())
    cues: list[SubtitleCue] = []

    time_pattern = re.compile(
        r"((?:\d{1,2}:)?\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*((?:\d{1,2}:)?\d{2}:\d{2}[,\.]\d{1,3})"
    )

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        # Skip NOTE blocks
        if lines[0].startswith("NOTE"):
            continue

        time_line_idx = -1
        time_match = None
        for idx, line in enumerate(lines):
            m = time_pattern.search(line)
            if m:
                time_line_idx = idx
                time_match = m
                break

        if not time_match or time_line_idx == -1:
            continue

        start_str, end_str = time_match.group(1), time_match.group(2)
        try:
            start_ms = parse_timestamp_vtt(start_str)
            end_ms = parse_timestamp_vtt(end_str)
        except ValueError:
            continue

        if end_ms <= start_ms:
            continue

        raw_text = "\n".join(lines[time_line_idx + 1:])
        clean_text = normalize_subtitle_text(raw_text)
        if not clean_text:
            continue

        cues.append(
            SubtitleCue(
                start_ms=start_ms,
                end_ms=end_ms,
                text=clean_text,
                source_type=source_type,
                source_format=source_format,
                stream_index=stream_index,
                source_file=source_file,
                language=language,
                confidence=1.0,
                episode_id=episode_id,
                source_video=source_video,
            )
        )

    return cues


def parse_ass(
    content_or_path: str | Path,
    *,
    source_type: str = "sidecar",
    source_format: str = "ass",
    stream_index: int | None = None,
    source_file: str | None = None,
    language: str = "eng",
    episode_id: str = "",
    source_video: str = "",
) -> list[SubtitleCue]:
    """Parse Advanced SubStation Alpha (.ass / .ssa) content or file into normalized SubtitleCue items."""
    if isinstance(content_or_path, Path) or (isinstance(content_or_path, str) and "\n" not in content_or_path and Path(content_or_path).is_file()):
        p = Path(content_or_path)
        source_file = source_file or str(p.resolve())
        content = p.read_text(encoding="utf-8-sig", errors="replace")
    else:
        content = str(content_or_path)

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.splitlines()

    in_events = False
    format_fields: list[str] = []
    cues: list[SubtitleCue] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue

        if stripped.lower().startswith("[events]"):
            in_events = True
            continue
        elif stripped.startswith("[") and in_events:
            # Reached next section
            break

        if not in_events:
            continue

        if stripped.lower().startswith("format:"):
            # Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
            header_vals = stripped[7:].strip().split(",")
            format_fields = [h.strip().lower() for h in header_vals]
            continue

        if stripped.lower().startswith("dialogue:"):
            payload = stripped[9:].strip()
            if not format_fields:
                # Default standard format
                format_fields = ["layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"]

            # Split payload by comma up to len(format_fields) - 1 times
            parts = payload.split(",", len(format_fields) - 1)
            if len(parts) < len(format_fields):
                continue

            field_map = dict(zip(format_fields, parts))
            start_str = field_map.get("start", "").strip()
            end_str = field_map.get("end", "").strip()
            text_str = field_map.get("text", "")

            try:
                start_ms = parse_timestamp_ass(start_str)
                end_ms = parse_timestamp_ass(end_str)
            except ValueError:
                continue

            if end_ms <= start_ms:
                continue

            clean_text = normalize_subtitle_text(text_str)
            if not clean_text:
                continue

            cues.append(
                SubtitleCue(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=clean_text,
                    source_type=source_type,
                    source_format=source_format,
                    stream_index=stream_index,
                    source_file=source_file,
                    language=language,
                    confidence=1.0,
                    episode_id=episode_id,
                    source_video=source_video,
                )
            )

    return cues
