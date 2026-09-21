"""Unified subtitles and media stream selection package for ToolRecap V2."""
from __future__ import annotations

from .cache import (
    SubtitleCacheManager,
    compute_source_identity_hash,
    compute_subtitle_cache_key,
)
from .discovery import (
    discover_sidecars,
    extract_episode_identifiers,
    match_episode,
    select_best_english_subtitles,
)
from .models import (
    AudioSelectionResult,
    AudioStreamInfo,
    FastConcatDecision,
    MediaProbeResult,
    NormalizationProfile,
    PgsSubtitleEvent,
    StreamSignature,
    SubtitleCue,
    SubtitleDiscoveryResult,
    SubtitleStreamInfo,
    SubtitleTrack,
    VideoStreamInfo,
    VobSubEvent,
)
from .ocr import OcrAdapter, OcrModelManager, OcrResult
from .parsers import (
    normalize_subtitle_text,
    parse_ass,
    parse_srt,
    parse_vtt,
    strip_formatting_tags,
)
from .pgs import create_minimal_pgs_sup, parse_pgs_sup
from .pipeline import SubtitlePipeline
from .remap import cues_to_srt_rows, remap_subtitles
from .vobsub import (
    create_synthetic_vobsub,
    extract_spu_events_from_stream,
    extract_vobsub_events,
    parse_vobsub_idx,
)

__all__ = [
    "AudioSelectionResult",
    "AudioStreamInfo",
    "FastConcatDecision",
    "MediaProbeResult",
    "NormalizationProfile",
    "OcrAdapter",
    "OcrModelManager",
    "OcrResult",
    "PgsSubtitleEvent",
    "StreamSignature",
    "SubtitleCacheManager",
    "SubtitleCue",
    "SubtitleDiscoveryResult",
    "SubtitlePipeline",
    "SubtitleStreamInfo",
    "SubtitleTrack",
    "VideoStreamInfo",
    "VobSubEvent",
    "compute_source_identity_hash",
    "compute_subtitle_cache_key",
    "create_minimal_pgs_sup",
    "create_synthetic_vobsub",
    "cues_to_srt_rows",
    "discover_sidecars",
    "extract_episode_identifiers",
    "extract_spu_events_from_stream",
    "extract_vobsub_events",
    "match_episode",
    "normalize_subtitle_text",
    "parse_ass",
    "parse_pgs_sup",
    "parse_srt",
    "parse_vobsub_idx",
    "parse_vtt",
    "remap_subtitles",
    "select_best_english_subtitles",
    "strip_formatting_tags",
]
