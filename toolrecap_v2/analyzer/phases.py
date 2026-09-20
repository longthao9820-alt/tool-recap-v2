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
    EPISODE_SUMMARIZING = "episode_summarizing"
    BATCH_SUMMARIZING = "episode_summarizing"
    SEASON_BATCH = "season_batch"
    SEASON_MERGING = "season_merging"
    SEASON_MINING = "season_mining"
    OUTPUT_PLAN_READY = "output_plan_ready"


PhaseCallback = Callable[[AnalysisPhase, str, dict[str, Any]], None]
