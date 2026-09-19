"""Season connection pass: cross-episode link analysis, candidate proposal mining, and coverage checking."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from ..api_client import APIError, OpenAICompatibleClient
from ..domain.enums import CandidateScope
from ..domain.models import EpisodeEvidence, SourceEpisode
from ..settings import AppSettings
from .errors import AnalysisCancelledError, AnalysisError, CoverageIncompleteError
from .phases import AnalysisPhase, PhaseCallback
from .prompts import SEASON_CONNECTION_SYSTEM_PROMPT


@dataclass
class CandidateProposal:
    proposal_id: str
    title: str
    candidate_scope: str = CandidateScope.CROSS_EPISODE.value
    episodes: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    editorial_reason: str = ""
    status: str = "keep"  # keep, reject, merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "candidate_scope": self.candidate_scope,
            "episodes": self.episodes,
            "characters": self.characters,
            "editorial_reason": self.editorial_reason,
            "status": self.status,
        }


@dataclass
class SeasonConnectionResult:
    cross_episode_links: list[dict[str, Any]] = field(default_factory=list)
    candidate_proposals: list[CandidateProposal] = field(default_factory=list)
    supporting_character_arcs: list[dict[str, Any]] = field(default_factory=list)
    rejected_or_merged: list[dict[str, Any]] = field(default_factory=list)
    is_complete: bool = True
    missing_episodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cross_episode_links": self.cross_episode_links,
            "candidate_proposals": [p.to_dict() for p in self.candidate_proposals],
            "supporting_character_arcs": self.supporting_character_arcs,
            "rejected_or_merged": self.rejected_or_merged,
            "is_complete": self.is_complete,
            "missing_episodes": self.missing_episodes,
        }


class SeasonConnector:
    """Performs season-wide connection pass across all episode evidence."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.client = client

    def connect_season(
        self,
        episodes: list[SourceEpisode],
        evidence_map: dict[str, EpisodeEvidence],
        *,
        allow_incomplete: bool = False,
        cancel_event: threading.Event | None = None,
        on_phase: PhaseCallback | None = None,
        log: Callable[[str], None] | None = None,
    ) -> SeasonConnectionResult:
        """Analyze connections across all episode evidence records."""
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

        # 1. Coverage check
        missing_episodes: list[str] = []
        for ep in episodes:
            if ep.episode_id not in evidence_map or evidence_map[ep.episode_id] is None:
                missing_episodes.append(ep.episode_id)

        if missing_episodes:
            if not allow_incomplete:
                raise CoverageIncompleteError(
                    f"Season coverage incomplete: missing or failed episodes: {missing_episodes}. "
                    f"Set allow_incomplete=True to proceed with partial season analysis.",
                    missing_episodes=missing_episodes,
                )
            if log:
                log(
                    f"Cảnh báo: Phân tích mùa phim không đầy đủ. Thiếu các tập: {missing_episodes}. "
                    f"AI sẽ chỉ phân tích các tập hiện có."
                )

        if on_phase:
            on_phase(
                AnalysisPhase.SEASON_CONNECTING,
                "season",
                {
                    "total_episodes": len(episodes),
                    "missing_episodes": missing_episodes,
                    "is_complete": len(missing_episodes) == 0,
                },
            )

        # 2. Prepare aggregated evidence payload
        evidence_summary: dict[str, Any] = {}
        for ep in episodes:
            if ep.episode_id in evidence_map and evidence_map[ep.episode_id] is not None:
                ev = evidence_map[ep.episode_id]
                evidence_summary[ep.episode_id] = {
                    "episode_id": ep.episode_id,
                    "title": ep.title,
                    "duration_seconds": ev.duration_seconds,
                    "evidence_categories": {
                        k: v for k, v in ev.data.items() if v
                    },
                }

        coverage_notice = ""
        if missing_episodes:
            coverage_notice = (
                f"\nCRITICAL COVERAGE NOTICE:\n"
                f"The following episodes are MISSING or FAILED: {missing_episodes}.\n"
                f"This is a PARTIAL season analysis. NEVER claim, imply, or hallucinate full season coverage.\n"
                f"Restrict all narrative links and candidate proposals strictly to available episodes: "
                f"{list(evidence_summary.keys())}.\n"
            )

        user_text = (
            f"Analyze season-wide connections across {len(evidence_summary)} available episodes.\n"
            f"{coverage_notice}\n"
            f"Recap instructions:\n{self.settings.recap_prompt or 'Standard video recap'}\n\n"
            f"Aggregated Evidence Map:\n{json.dumps(evidence_summary, ensure_ascii=False, indent=2)}\n"
        )

        is_gateway_enabled = (
            self.settings.gateway_enabled
            and bool(self.settings.api_endpoint.strip())
            and self.settings.api_endpoint.strip().lower() != "offline"
            and self.client is not None
        )

        if is_gateway_enabled and self.client is not None:
            if cancel_event and cancel_event.is_set():
                raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

            try:
                raw_result = self.client.chat_json(
                    model=self.settings.finalizer_model,
                    thinking=self.settings.finalizer_thinking,
                    system=SEASON_CONNECTION_SYSTEM_PROMPT,
                    user_text=user_text,
                    cancel_event=cancel_event,
                )
            except APIError as exc:
                raise AnalysisError(f"AI Gateway connection pass lỗi: {exc}") from exc

            return self._parse_connection_result(
                raw_result,
                is_complete=len(missing_episodes) == 0,
                missing_episodes=missing_episodes,
            )
        else:
            # Deterministic offline season connection
            return self._connect_offline(
                episodes=episodes,
                evidence_map=evidence_map,
                is_complete=len(missing_episodes) == 0,
                missing_episodes=missing_episodes,
            )

    def _parse_connection_result(
        self,
        raw: dict[str, Any],
        is_complete: bool,
        missing_episodes: list[str],
    ) -> SeasonConnectionResult:
        """Parse AI response into SeasonConnectionResult."""
        links = raw.get("cross_episode_links", [])
        if not isinstance(links, list):
            links = []

        proposals_raw = raw.get("candidate_proposals", [])
        if not isinstance(proposals_raw, list):
            proposals_raw = []

        proposals: list[CandidateProposal] = []
        for i, p in enumerate(proposals_raw, start=1):
            if not isinstance(p, dict):
                continue
            scope_val = str(p.get("candidate_scope", CandidateScope.CROSS_EPISODE.value))
            if scope_val not in {s.value for s in CandidateScope}:
                scope_val = CandidateScope.CROSS_EPISODE.value

            proposals.append(
                CandidateProposal(
                    proposal_id=str(p.get("proposal_id", f"prop_{i:02d}")),
                    title=str(p.get("title", f"Candidate {i}")),
                    candidate_scope=scope_val,
                    episodes=[str(e) for e in p.get("episodes", []) if isinstance(e, str)],
                    characters=[str(c) for c in p.get("characters", []) if isinstance(c, str)],
                    editorial_reason=str(p.get("editorial_reason", "")),
                    status=str(p.get("status", "keep")),
                )
            )

        arcs = raw.get("supporting_character_arcs", [])
        if not isinstance(arcs, list):
            arcs = []

        rejected = raw.get("rejected_or_merged", [])
        if not isinstance(rejected, list):
            rejected = []

        return SeasonConnectionResult(
            cross_episode_links=links,
            candidate_proposals=proposals,
            supporting_character_arcs=arcs,
            rejected_or_merged=rejected,
            is_complete=is_complete,
            missing_episodes=missing_episodes,
        )

    def _connect_offline(
        self,
        episodes: list[SourceEpisode],
        evidence_map: dict[str, EpisodeEvidence],
        is_complete: bool,
        missing_episodes: list[str],
    ) -> SeasonConnectionResult:
        """Deterministic offline connection pass."""
        available_ids = [ep.episode_id for ep in episodes if ep.episode_id in evidence_map]

        links = [
            {
                "thread_id": "thread_main_arc",
                "theme": "Overarching Season Narrative",
                "episodes": available_ids,
                "summary": f"Traces causal continuity across {len(available_ids)} episodes.",
            }
        ]

        proposals = [
            CandidateProposal(
                proposal_id="prop_main_arc",
                title="Season Narrative Arc",
                candidate_scope=CandidateScope.SEASON_ARC.value,
                episodes=available_ids,
                characters=["Protagonist"],
                editorial_reason="Primary narrative throughline across available episodes.",
                status="keep",
            ),
            CandidateProposal(
                proposal_id="prop_supporting_arc",
                title="Supporting Character Journey",
                candidate_scope=CandidateScope.CROSS_EPISODE.value,
                episodes=available_ids[: min(3, len(available_ids))],
                characters=["Supporting Character"],
                editorial_reason="Dedicated editorial focus for key supporting character.",
                status="keep",
            ),
        ]

        return SeasonConnectionResult(
            cross_episode_links=links,
            candidate_proposals=proposals,
            supporting_character_arcs=[
                {
                    "character": "Supporting Character",
                    "arc_summary": "Development through secondary storylines.",
                    "episodes": available_ids[: min(3, len(available_ids))],
                    "has_dedicated_candidate": True,
                }
            ],
            rejected_or_merged=[],
            is_complete=is_complete,
            missing_episodes=missing_episodes,
        )
