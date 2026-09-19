"""Domain errors for single and season analysis operations."""
from __future__ import annotations

from ..domain.models import DomainError


class AnalysisError(DomainError):
    """Base error for analysis operations."""


class AnalysisCancelledError(AnalysisError):
    """Raised when analysis is cancelled via cancel_event."""


class CoverageIncompleteError(AnalysisError):
    """Raised when season analysis coverage is incomplete and allow_incomplete is False."""

    def __init__(self, message: str, missing_episodes: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_episodes: list[str] = missing_episodes or []
