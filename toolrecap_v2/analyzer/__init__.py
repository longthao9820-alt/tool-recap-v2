"""Analyzer package providing single and season AI analysis, episode evidence, and candidate mining."""
from __future__ import annotations

from .connection import CandidateProposal, SeasonConnectionResult, SeasonConnector
from .engine import AnalysisEngine, compute_final_plan_cache_key, run_analysis
from .errors import AnalysisCancelledError, AnalysisError, CoverageIncompleteError
from .evidence import (
    EVIDENCE_CATEGORIES,
    EvidenceScanner,
    compute_scanner_config_version,
)
from .finalizer import CandidateFinalizer
from .phases import AnalysisPhase, PhaseCallback
from .prompts import (
    FINALIZER_SYSTEM_PROMPT,
    SCANNER_SYSTEM_PROMPT,
    SEASON_CONNECTION_SYSTEM_PROMPT,
)

__all__ = [
    "AnalysisCancelledError",
    "AnalysisEngine",
    "AnalysisError",
    "AnalysisPhase",
    "CandidateFinalizer",
    "CandidateProposal",
    "CoverageIncompleteError",
    "EVIDENCE_CATEGORIES",
    "EvidenceScanner",
    "FINALIZER_SYSTEM_PROMPT",
    "PhaseCallback",
    "SCANNER_SYSTEM_PROMPT",
    "SEASON_CONNECTION_SYSTEM_PROMPT",
    "SeasonConnectionResult",
    "SeasonConnector",
    "compute_final_plan_cache_key",
    "compute_scanner_config_version",
    "run_analysis",
]
