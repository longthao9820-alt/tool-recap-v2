"""Phase definitions and status callbacks for single and season analysis."""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable


class AnalysisPhase(str, Enum):
    MEDIA_PROBE = "media_probe"
    SUBTITLES = "subtitles"
    SCANNER = "scanner"
    SEASON_BARRIER = "season_barrier"
    SEASON_CONNECTING = "season_connecting"
    SEASON_MINING = "season_mining"
    OUTPUT_PLAN_READY = "output_plan_ready"


PhaseCallback = Callable[[AnalysisPhase, str, dict[str, Any]], None]
