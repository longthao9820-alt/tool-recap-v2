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
