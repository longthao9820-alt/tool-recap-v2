"""Domain enums for ToolRecap V2 multi-episode analysis and commentary."""
from __future__ import annotations

from enum import Enum


class AnalysisScope(str, Enum):
    SINGLE_EPISODE = "SINGLE_EPISODE"
    SEASON = "SEASON"


class CandidateScope(str, Enum):
    SINGLE_SCENE = "SINGLE_SCENE"
    SINGLE_EPISODE = "SINGLE_EPISODE"
    CROSS_EPISODE = "CROSS_EPISODE"
    SEASON_ARC = "SEASON_ARC"


class CandidateStatus(str, Enum):
    KEEP = "keep"
    REJECT = "reject"
    MERGED = "merged"


class AudioPolicy(str, Enum):
    MUTE = "mute"
    DUCK = "duck"
    KEEP = "keep"
    ORIGINAL_ONLY = "original_only"


class ProjectPhase(str, Enum):
    IDLE = "IDLE"
    ANALYZING = "ANALYZING"
    RENDERING = "RENDERING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class OutputStatus(str, Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class CompactionLevel(str, Enum):
    FULL = "FULL"
    TRIMMED = "TRIMMED"
    PRIORITY = "PRIORITY"
    SKELETON = "SKELETON"


class ConsolidationAction(str, Enum):
    KEEP = "KEEP"
    MERGE = "MERGE"
    REJECT = "REJECT"


class ConsolidationReason(str, Enum):
    DUPLICATE_THESIS_EVIDENCE = "duplicate_thesis_evidence"
    WEAK_EVIDENCE = "weak_evidence"
    DISTINCT_THESIS = "distinct_thesis"
    CROSS_EPISODE_MERGE = "cross_episode_merge"
    INVALID_GROUNDING = "invalid_grounding"
    CONSERVATIVE_KEEP = "conservative_keep"
    DISTINCT_SUPPORTING = "distinct_supporting"


class ZeroOutputReason(str, Enum):
    NO_ELIGIBLE_CANDIDATES = "NO_ELIGIBLE_CANDIDATES"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CANDIDATE_DISCOVERY_FAILED = "CANDIDATE_DISCOVERY_FAILED"
    CANDIDATES_REJECTED_BY_VALIDATION = "CANDIDATES_REJECTED_BY_VALIDATION"
    FINALIZER_FAILED = "FINALIZER_FAILED"
    MALFORMED_AI_RESPONSE = "MALFORMED_AI_RESPONSE"
    PARSER_FAILURE = "PARSER_FAILURE"
    COVERAGE_INCOMPLETE = "COVERAGE_INCOMPLETE"
    LOW_COVERAGE_SUSPECT = "LOW_COVERAGE_SUSPECT"
