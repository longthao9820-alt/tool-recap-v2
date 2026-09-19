"""Domain foundation for ToolRecap V2 multi-episode analysis, commentary, and persistence."""
from __future__ import annotations

from .cache import EvidenceCacheManager, compute_cache_key, default_evidence_cache_dir
from .enums import (
    AnalysisScope,
    AudioPolicy,
    CandidateScope,
    OutputStatus,
    ProjectPhase,
)
from .models import (
    AnalysisManifest,
    CommentaryOutput,
    DomainError,
    EpisodeEvidence,
    MediaSelection,
    Segment,
    SourceClip,
    SourceEpisode,
    ValidationError,
)
from .title import resolve_unique_titles, sanitize_title

__all__ = [
    "AnalysisManifest",
    "AnalysisScope",
    "AudioPolicy",
    "CandidateScope",
    "CommentaryOutput",
    "DomainError",
    "EpisodeEvidence",
    "EvidenceCacheManager",
    "MediaSelection",
    "OutputStatus",
    "ProjectPhase",
    "Segment",
    "SourceClip",
    "SourceEpisode",
    "ValidationError",
    "compute_cache_key",
    "default_evidence_cache_dir",
    "resolve_unique_titles",
    "sanitize_title",
]
