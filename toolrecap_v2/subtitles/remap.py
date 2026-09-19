"""Multi-source subtitle remapping: clipping, boundary clamping, timeline offset, and speed adjustment."""
from __future__ import annotations

from typing import Sequence

from ..domain.models import SourceClip
from .models import SubtitleCue


def remap_subtitles(
    cues_by_episode: dict[str, Sequence[SubtitleCue]],
    clips: Sequence[SourceClip],
    *,
    speed_factor: float = 1.0,
) -> list[SubtitleCue]:
    """Remap subtitles across multiple episodes and clips onto a unified timeline.

    Enforces invariants:
    - Clip by episode and source video; strictly reject cues belonging to wrong episodes.
    - Clamp cue boundaries strictly within clip start and end.
    - Offset timestamps to unified timeline.
    - Apply speed factor adjustment if provided.
    - Output deterministic unified SubtitleCue rows sorted by start_ms.
    """
    speed = float(speed_factor) if speed_factor > 0.0 else 1.0
    timeline_offset_ms = 0
    remapped_cues: list[SubtitleCue] = []

    for clip in clips:
        clip_start_ms = int(round(clip.start * 1000.0))
        clip_end_ms = int(round(clip.end * 1000.0))
        clip_duration_ms = max(0, clip_end_ms - clip_start_ms)

        if clip_duration_ms <= 0:
            continue

        raw_cues = cues_by_episode.get(clip.episode_id, [])

        for cue in raw_cues:
            # Strict episode invariant: Reject cues from wrong episode
            if cue.episode_id and cue.episode_id != clip.episode_id:
                continue

            # Check overlap with clip interval [clip_start_ms, clip_end_ms]
            if cue.end_ms <= clip_start_ms or cue.start_ms >= clip_end_ms:
                continue

            # Clamp boundaries
            clamped_start = max(cue.start_ms, clip_start_ms)
            clamped_end = min(cue.end_ms, clip_end_ms)
            if clamped_end <= clamped_start:
                continue

            # Apply speed adjustment and timeline offset
            rel_start = (clamped_start - clip_start_ms) / speed
            rel_end = (clamped_end - clip_start_ms) / speed

            final_start_ms = int(round(timeline_offset_ms + rel_start))
            final_end_ms = int(round(timeline_offset_ms + rel_end))

            remapped = SubtitleCue(
                start_ms=final_start_ms,
                end_ms=final_end_ms,
                text=cue.text,
                source_type=cue.source_type,
                source_format=cue.source_format,
                stream_index=cue.stream_index,
                source_file=cue.source_file,
                language=cue.language,
                confidence=cue.confidence,
                episode_id=clip.episode_id,
                source_video=clip.source_video,
            )
            remapped_cues.append(remapped)

        timeline_offset_ms += int(round(clip_duration_ms / speed))

    remapped_cues.sort(key=lambda c: (c.start_ms, c.end_ms))
    return remapped_cues


def cues_to_srt_rows(cues: Sequence[SubtitleCue]) -> list[tuple[float, float, str]]:
    """Convert SubtitleCue items to (start_sec, end_sec, text) tuples for write_srt_file."""
    return [(c.start_sec, c.end_sec, c.text) for c in cues if c.text.strip()]
