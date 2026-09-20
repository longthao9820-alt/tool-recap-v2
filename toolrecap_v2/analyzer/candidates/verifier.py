"""Candidate zero-output and low-coverage verification module for ToolRecap V2.

Verifies whether zero-output or suspiciously low candidate coverage is genuine
and grounded, or recovers viable narrative candidates from existing evidence refs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

from ...api_client import (
    APIError,
    OpenAICompatibleClient,
    estimate_request_size,
)
from ...domain.cache import (
    HierarchyCacheManager,
    VERIFICATION_SCHEMA_VERSION,
    compute_verification_cache_key,
    validate_verification_cache_data,
)
from ...domain.enums import (
    CandidateScope,
    CandidateStatus,
    CompactionLevel,
    ConsolidationAction,
    ConsolidationReason,
    ZeroOutputReason,
)
from ...domain.models import (
    CandidateProposal,
    CandidateSourceRange,
    ConsolidationDecision,
    PipelineHealth,
    VerificationResult,
)
from ...domain.policy import (
    CandidateDirective,
    EditorialPolicy,
)
from ...settings import AppSettings
from ..coverage import CoverageStatus
from ..errors import AnalysisCancelledError, AnalysisError
from ..phases import AnalysisPhase, PhaseCallback
from ..prompts import ZERO_OUTPUT_VERIFICATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

HARD_PAYLOAD_CEILING: int = 500_000
TARGET_PAYLOAD_CEILING: int = 480_000
VERIFIER_ALGO_VERSION: str = "v1"
VERIFIER_PROMPT_VERSION: str = "v1"

DEFAULT_MIN_EPISODE_RATIO: float = 0.40
DEFAULT_MIN_EVIDENCE_RATIO: float = 0.10
DEFAULT_MIN_EPISODES_BREADTH: int = 3
DEFAULT_MIN_EVIDENCE_BREADTH: int = 20


def validate_verification_response_schema(
    raw: Any,
    *,
    raise_error: bool = False,
) -> bool:
    """Validate AI response schema for zero/low output verification.

    Requirements:
    - Must be a dictionary.
    - Must contain either:
      - Non-empty 'recovered_candidates' (or 'candidates') as a list of dicts.
      - OR explicit 'confirm_no_eligible' as a boolean.
    """
    if not isinstance(raw, dict):
        if raise_error:
            raise AnalysisError(f"Verification response must be a dict, got {type(raw).__name__}.")
        return False

    has_recovered = False
    cand_list = None
    for key in ("recovered_candidates", "candidates"):
        if key in raw:
            cand_list = raw[key]
            break

    if cand_list is not None:
        if not isinstance(cand_list, list):
            if raise_error:
                raise AnalysisError(f"Field 'recovered_candidates' must be a list, got {type(cand_list).__name__}.")
            return False
        for idx, item in enumerate(cand_list):
            if not isinstance(item, dict):
                if raise_error:
                    raise AnalysisError(f"Candidate {idx} in 'recovered_candidates' must be a dict.")
                return False
        if len(cand_list) > 0:
            has_recovered = True

    has_confirm = False
    if "confirm_no_eligible" in raw:
        val = raw["confirm_no_eligible"]
        if not isinstance(val, bool):
            if raise_error:
                raise AnalysisError(f"Field 'confirm_no_eligible' must be a bool, got {type(val).__name__}.")
            return False
        has_confirm = True

    if not has_recovered and not has_confirm:
        if raise_error:
            raise AnalysisError("Verification response must contain non-empty 'recovered_candidates' or explicit 'confirm_no_eligible' boolean.")
        return False

    return True


def classify_deterministic_reason(
    health: PipelineHealth | dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[ZeroOutputReason | None, dict[str, Any]]:
    """Deterministically classify zero-output reason before any AI call.

    Precedence order:
    1. Parser failure -> PARSER_FAILURE (parser failure can never map NO_ELIGIBLE)
    2. Malformed / schema failure -> MALFORMED_AI_RESPONSE
    3. Discovery failure -> CANDIDATE_DISCOVERY_FAILED
    4. Finalizer failure -> FINALIZER_FAILED
    5. Coverage incomplete -> COVERAGE_INCOMPLETE
    6. Candidates rejected by validation -> CANDIDATES_REJECTED_BY_VALIDATION
    7. Insufficient evidence -> INSUFFICIENT_EVIDENCE
    8. None (healthy state)
    """
    if health is None:
        health_obj = PipelineHealth.from_dict(kwargs)
    elif isinstance(health, dict):
        health_obj = PipelineHealth.from_dict(health)
    else:
        health_obj = health

    diagnostics: dict[str, Any] = {
        "parser_failed": health_obj.parser_failed,
        "parser_error": health_obj.parser_error,
        "schema_rejected": health_obj.schema_rejected,
        "schema_error": health_obj.schema_error,
        "discovery_completed": health_obj.discovery_completed,
        "discovery_error": health_obj.discovery_error,
        "discovery_schema_valid": health_obj.discovery_schema_valid,
        "discovered_count": health_obj.discovered_count,
        "consolidation_completed": health_obj.consolidation_completed,
        "consolidation_error": health_obj.consolidation_error,
        "consolidation_schema_valid": health_obj.consolidation_schema_valid,
        "consolidated_count": len(health_obj.consolidated_candidates),
        "rejected_count": len([d for d in health_obj.consolidation_decisions if getattr(d, "action", "") == ConsolidationAction.REJECT.value or (isinstance(d, dict) and d.get("action") == ConsolidationAction.REJECT.value)]),
        "finalizer_attempted": health_obj.finalizer_attempted,
        "finalizer_completed": health_obj.finalizer_completed,
        "finalizer_error": health_obj.finalizer_error,
        "total_evidence_count": health_obj.total_evidence_count,
    }

    # 1. Parser failure (can NEVER map to NO_ELIGIBLE)
    if health_obj.parser_failed or health_obj.parser_error:
        return ZeroOutputReason.PARSER_FAILURE, diagnostics

    # 2. Malformed / schema rejection
    if (
        health_obj.schema_rejected
        or health_obj.schema_error
        or not health_obj.discovery_schema_valid
        or not health_obj.consolidation_schema_valid
    ):
        return ZeroOutputReason.MALFORMED_AI_RESPONSE, diagnostics

    # 3. Discovery failed
    if health_obj.discovery_error or not health_obj.discovery_completed:
        return ZeroOutputReason.CANDIDATE_DISCOVERY_FAILED, diagnostics

    # 4. Finalizer failed
    if health_obj.finalizer_attempted and (
        health_obj.finalizer_error
        or (not health_obj.finalizer_completed and not health_obj.finalizer_results)
    ):
        return ZeroOutputReason.FINALIZER_FAILED, diagnostics

    # 5. Coverage incomplete
    ledgers = health_obj.coverage_ledgers
    if ledgers:
        ledger_items: list[Any] = []
        if isinstance(ledgers, dict):
            ledger_items = list(ledgers.values())
        elif isinstance(ledgers, list):
            ledger_items = list(ledgers)

        for led in ledger_items:
            status_val = ""
            actionable_gaps = False
            if hasattr(led, "coverage_status"):
                status_val = str(getattr(led, "coverage_status", "")).upper()
            elif hasattr(led, "status"):
                status_val = str(getattr(led, "status", "")).upper()
            elif isinstance(led, dict):
                status_val = str(led.get("coverage_status", led.get("status", ""))).upper()
                gaps = led.get("actionable_gaps", led.get("gaps", []))
                actionable_gaps = bool(gaps)

            if hasattr(led, "actionable_gaps"):
                actionable_gaps = bool(getattr(led, "actionable_gaps", False))
            elif hasattr(led, "gaps"):
                actionable_gaps = bool(getattr(led, "gaps", []))

            # Partial, empty, or failed indicates incomplete coverage
            if any(term in status_val for term in ("PARTIAL", "EMPTY", "FAILED", "INCOMPLETE")):
                return ZeroOutputReason.COVERAGE_INCOMPLETE, diagnostics
            if actionable_gaps and not any(term in status_val for term in ("COMPLETE", "COMPLETE_TRANSCRIPT_ONLY")):
                return ZeroOutputReason.COVERAGE_INCOMPLETE, diagnostics

    # 6. Candidates rejected by validation
    has_rejected_decisions = diagnostics["rejected_count"] > 0
    had_input_candidates = (health_obj.discovered_count > 0) or has_rejected_decisions
    if had_input_candidates and len(health_obj.consolidated_candidates) == 0:
        # Check if all decisions were REJECT
        total_decisions = len(health_obj.consolidation_decisions)
        if total_decisions > 0 and diagnostics["rejected_count"] >= total_decisions:
            return ZeroOutputReason.CANDIDATES_REJECTED_BY_VALIDATION, diagnostics

    # 7. Insufficient evidence
    if health_obj.total_evidence_count == 0 and not ledgers:
        return ZeroOutputReason.INSUFFICIENT_EVIDENCE, diagnostics

    return None, diagnostics


def is_suspicious_low_coverage(
    candidates: Sequence[CandidateProposal],
    coverage_ledgers: Any = None,
    *,
    total_episodes: int = 0,
    total_evidence_items: int = 0,
    min_episode_ratio: float = DEFAULT_MIN_EPISODE_RATIO,
    min_evidence_ratio: float = DEFAULT_MIN_EVIDENCE_RATIO,
    min_episodes_breadth: int = DEFAULT_MIN_EPISODES_BREADTH,
    min_evidence_breadth: int = DEFAULT_MIN_EVIDENCE_BREADTH,
) -> tuple[bool, dict[str, Any]]:
    """Determine whether consolidated candidates represent suspicious low coverage.

    Does NOT use fixed output quotas. Uses relative coverage breadth ratios:
    - Many evidence threads/episodes, but candidates cover tiny fraction.
    - A single strong season arc covering the source arc is valid (no trigger).
    """
    if not candidates:
        return False, {"reason": "zero_candidates", "is_suspicious": False}

    # Check for strong season arc
    for cand in candidates:
        is_season_arc = (
            cand.candidate_scope == CandidateScope.SEASON_ARC.value
            or cand.candidate_scope == CandidateScope.SEASON_ARC
        )
        if is_season_arc:
            # If season arc spans multiple episodes or significant timeline, not suspicious
            num_cand_eps = len(cand.episodes) or len({r.episode_id for r in cand.source_ranges if r.episode_id})
            if total_episodes <= 2 or num_cand_eps >= 2 or (total_episodes > 0 and num_cand_eps / total_episodes >= 0.50):
                return False, {
                    "is_suspicious": False,
                    "reason": "strong_season_arc",
                    "season_arc_candidate": cand.proposal_id,
                }

    # Determine total episodes and evidence
    cand_episodes: set[str] = set()
    cand_refs: set[str] = set()
    for c in candidates:
        cand_episodes.update(c.episodes)
        for r in c.source_ranges:
            if r.episode_id:
                cand_episodes.add(r.episode_id)
            if r.evidence_ref:
                cand_refs.add(r.evidence_ref)
        cand_refs.update(c.supporting_evidence)
        cand_refs.update(c.observed_facts)

    ep_count = total_episodes
    if ep_count <= 0 and coverage_ledgers:
        if isinstance(coverage_ledgers, dict):
            ep_count = len(coverage_ledgers)
        elif isinstance(coverage_ledgers, list):
            ep_count = len(coverage_ledgers)

    if ep_count <= 0:
        ep_count = len(cand_episodes)

    ev_count = total_evidence_items
    if ev_count <= 0 and coverage_ledgers:
        ledger_items: list[Any] = []
        if isinstance(coverage_ledgers, dict):
            ledger_items = list(coverage_ledgers.values())
        elif isinstance(coverage_ledgers, list):
            ledger_items = list(coverage_ledgers)
        for led in ledger_items:
            if hasattr(led, "total_evidence_count"):
                ev_count += int(getattr(led, "total_evidence_count", 0))
            elif isinstance(led, dict):
                ev_count += int(led.get("total_evidence_count", len(led.get("evidence", []))))

    ep_ratio = len(cand_episodes) / max(1, ep_count) if ep_count > 0 else 1.0
    ev_ratio = len(cand_refs) / max(1, ev_count) if ev_count > 0 else 1.0

    diag: dict[str, Any] = {
        "candidate_count": len(candidates),
        "total_episodes": ep_count,
        "candidate_episodes": sorted(cand_episodes),
        "episode_coverage_ratio": round(ep_ratio, 3),
        "total_evidence_items": ev_count,
        "candidate_evidence_refs_count": len(cand_refs),
        "evidence_coverage_ratio": round(ev_ratio, 3),
    }

    has_ep_breadth = ep_count >= min_episodes_breadth
    has_ev_breadth = ev_count >= min_evidence_breadth

    is_suspicious = False
    if has_ep_breadth and ep_ratio < min_episode_ratio:
        is_suspicious = True
        diag["suspicion_trigger"] = f"Episode ratio {ep_ratio:.2f} < {min_episode_ratio:.2f}"
    elif has_ev_breadth and ev_ratio < min_evidence_ratio:
        is_suspicious = True
        diag["suspicion_trigger"] = f"Evidence ratio {ev_ratio:.2f} < {min_evidence_ratio:.2f}"

    diag["is_suspicious"] = is_suspicious
    return is_suspicious, diag


def is_genuine_zero_valid(health: PipelineHealth) -> bool:
    """Verify whether zero output is genuine and valid according to all rules.

    Genuine zero is ONLY valid if:
    - Coverage statuses complete or transcript-only with no actionable gaps
    - Discovery complete
    - Consolidation complete
    - Parser healthy
    - Schema healthy
    - Zero consolidated candidates
    """
    if health.parser_failed or health.parser_error:
        return False
    if health.schema_rejected or health.schema_error:
        return False
    if not health.discovery_completed or health.discovery_error or not health.discovery_schema_valid:
        return False
    if not health.consolidation_completed or health.consolidation_error or not health.consolidation_schema_valid:
        return False
    if len(health.consolidated_candidates) > 0:
        return False
    if health.total_evidence_count == 0 and not health.coverage_ledgers:
        return False

    ledgers = health.coverage_ledgers
    if ledgers:
        ledger_items: list[Any] = []
        if isinstance(ledgers, dict):
            ledger_items = list(ledgers.values())
        elif isinstance(ledgers, list):
            ledger_items = list(ledgers)

        for led in ledger_items:
            status_val = ""
            actionable_gaps = False
            if hasattr(led, "coverage_status"):
                status_val = str(getattr(led, "coverage_status", "")).upper()
            elif hasattr(led, "status"):
                status_val = str(getattr(led, "status", "")).upper()
            elif isinstance(led, dict):
                status_val = str(led.get("coverage_status", led.get("status", ""))).upper()
                gaps = led.get("actionable_gaps", led.get("gaps", []))
                actionable_gaps = bool(gaps)

            if hasattr(led, "actionable_gaps"):
                actionable_gaps = bool(getattr(led, "actionable_gaps", False))
            elif hasattr(led, "gaps"):
                actionable_gaps = bool(getattr(led, "gaps", []))

            # Must be complete or transcript-only with no actionable gaps
            is_comp = any(term in status_val for term in ("COMPLETE", "COMPLETE_TRANSCRIPT_ONLY"))
            if not is_comp or actionable_gaps:
                return False

    return True


def should_trigger_verification(
    health: PipelineHealth | dict[str, Any],
    candidates: Sequence[CandidateProposal] | None = None,
    *,
    total_episodes: int = 0,
    total_evidence_items: int = 0,
) -> tuple[bool, ZeroOutputReason | None, dict[str, Any]]:
    """Determine whether verification should be triggered.

    Returns: (should_trigger, deterministic_reason_if_any, diagnostics).
    If deterministic failure is found, should_trigger is False, and deterministic_reason is set.
    If deterministic state is healthy and candidates == 0 or suspicious: should_trigger is True.
    """
    health_obj = health if isinstance(health, PipelineHealth) else PipelineHealth.from_dict(health)
    reason, diagnostics = classify_deterministic_reason(health_obj)

    # If deterministic failure is present, do NOT trigger AI verification
    if reason in (
        ZeroOutputReason.PARSER_FAILURE,
        ZeroOutputReason.MALFORMED_AI_RESPONSE,
        ZeroOutputReason.CANDIDATE_DISCOVERY_FAILED,
        ZeroOutputReason.FINALIZER_FAILED,
        ZeroOutputReason.COVERAGE_INCOMPLETE,
        ZeroOutputReason.INSUFFICIENT_EVIDENCE,
    ):
        return False, reason, diagnostics

    # Check candidates
    cand_list = candidates if candidates is not None else health_obj.consolidated_candidates
    if len(cand_list) == 0:
        # Zero candidates in healthy pipeline triggers verification
        return True, reason, diagnostics

    # Suspicious low coverage check
    ev_count = total_evidence_items or health_obj.total_evidence_count
    is_susp, susp_diag = is_suspicious_low_coverage(
        cand_list,
        health_obj.coverage_ledgers,
        total_episodes=total_episodes,
        total_evidence_items=ev_count,
    )
    diagnostics.update(susp_diag)
    if is_susp:
        return True, ZeroOutputReason.LOW_COVERAGE_SUSPECT, diagnostics

    return False, None, diagnostics


def format_verification_user_text(
    scope_id: str,
    health: PipelineHealth,
    candidate_directive: CandidateDirective | str | None = None,
    compaction_level: CompactionLevel = CompactionLevel.FULL,
) -> str:
    """Format user prompt text for candidate zero/low output verification."""
    lines: list[str] = [
        f"=== CANDIDATE OUTPUT VERIFICATION: {scope_id} ===",
        f"Consolidated Candidates Count: {len(health.consolidated_candidates)}",
        f"Discovered Count: {health.discovered_count}",
        f"Total Evidence Items: {health.total_evidence_count}",
        "",
    ]

    max_rejected = {
        CompactionLevel.FULL: 30,
        CompactionLevel.TRIMMED: 15,
        CompactionLevel.PRIORITY: 5,
        CompactionLevel.SKELETON: 2,
    }.get(compaction_level, 15)

    max_ledgers = {
        CompactionLevel.FULL: 20,
        CompactionLevel.TRIMMED: 10,
        CompactionLevel.PRIORITY: 5,
        CompactionLevel.SKELETON: 2,
    }.get(compaction_level, 10)

    if candidate_directive:
        cd_str = str(candidate_directive).strip()
        if cd_str:
            if compaction_level == CompactionLevel.SKELETON and len(cd_str) > 200:
                cd_str = cd_str[:200] + "..."
            lines.append("EDITORIAL CANDIDATE DIRECTIVE:")
            lines.append(cd_str)
            lines.append("")

    # Rejected consolidation decisions
    rejected = [
        d for d in health.consolidation_decisions
        if getattr(d, "action", "") == ConsolidationAction.REJECT.value
        or (isinstance(d, dict) and d.get("action") == ConsolidationAction.REJECT.value)
    ]
    if rejected:
        lines.append(f"REJECTED PROPOSALS DURING CONSOLIDATION ({len(rejected)}):")
        for dec in rejected[:max_rejected]:
            if isinstance(dec, ConsolidationDecision):
                lines.append(f"- ID: {dec.candidate_id}, Code: {dec.reason_code}, Reason: {dec.reason}")
            elif isinstance(dec, dict):
                lines.append(f"- ID: {dec.get('candidate_id')}, Code: {dec.get('reason_code')}, Reason: {dec.get('reason')}")
        lines.append("")

    # Coverage summary
    lines.append("COVERAGE & EVIDENCE SUMMARY:")
    ledgers = health.coverage_ledgers
    if ledgers:
        items = list(ledgers.values()) if isinstance(ledgers, dict) else list(ledgers)
        for idx, led in enumerate(items[:max_ledgers]):
            ep_id = getattr(led, "episode_id", None) or (led.get("episode_id") if isinstance(led, dict) else f"ep_{idx}")
            st = getattr(led, "coverage_status", None) or (led.get("coverage_status") if isinstance(led, dict) else "UNKNOWN")
            lines.append(f"- Episode {ep_id}: Status={st}")
    else:
        lines.append(f"- Total Evidence Count: {health.total_evidence_count}")
    lines.append("")

    lines.append("VERIFICATION TASK:")
    lines.append("1. Audit whether any valid narrative candidates were mistakenly rejected or overlooked.")
    lines.append("2. If viable candidates exist, return them in 'recovered_candidates' grounded strictly in the source evidence.")
    lines.append("3. If no viable narrative candidates exist, return 'confirm_no_eligible': true with an explicit 'rationale'.")
    lines.append("4. Never synthesize generic fallback candidates without evidence grounding.")

    return "\n".join(lines)


def estimate_verification_request_size(
    model: str,
    user_text: str,
    thinking: str = "auto",
    max_tokens: int = 32_000,
) -> int:
    """Estimate request size for verification prompt."""
    return estimate_request_size(
        model=model,
        system=ZERO_OUTPUT_VERIFICATION_SYSTEM_PROMPT,
        user_text=user_text,
        thinking=thinking,
        max_tokens=max_tokens,
        variant=0,
    )


def validate_recovered_candidates(
    recovered_raw: list[Any],
    allowed_episode_ids: set[str] | None = None,
    existing_evidence_refs: set[str] | None = None,
    episode_durations: dict[str, float] | None = None,
    log: logging.Logger | None = None,
) -> list[CandidateProposal]:
    """Validate and ground recovered candidates against allowed source rules.

    Rejects:
    - Generic synthesized fallbacks without specific evidence grounding
    - Unknown episodes or invalid source ranges
    - Candidates without any matching evidence refs when existing_evidence_refs provided
    """
    logger_ref = log or logger
    durations = episode_durations or {}
    valid_candidates: list[CandidateProposal] = []

    for idx, item in enumerate(recovered_raw):
        cand = item if isinstance(item, CandidateProposal) else CandidateProposal.from_dict(item)

        # Check proposal ID and title/thesis
        if not cand.proposal_id.strip():
            cand.proposal_id = f"recovered_{idx + 1}"

        t_clean = cand.title.strip().lower()
        th_clean = cand.central_thesis.strip().lower()
        if not t_clean and not th_clean:
            logger_ref.warning(f"Rejecting recovered candidate {cand.proposal_id}: missing title and thesis.")
            continue

        # Reject generic fallback keywords
        generic_markers = (
            "generic fallback",
            "fallback candidate",
            "default recap",
            "placeholder candidate",
            "placeholder",
            "synthetic fallback",
            "fallback",
        )
        if any(m in t_clean or m in th_clean for m in generic_markers):
            logger_ref.warning(f"Rejecting recovered candidate {cand.proposal_id}: generic fallback not permitted.")
            continue

        # Validate episodes
        if allowed_episode_ids is not None:
            cand_eps = [ep for ep in cand.episodes if ep in allowed_episode_ids]
            if not cand_eps and cand.source_ranges:
                cand_eps = [r.episode_id for r in cand.source_ranges if r.episode_id in allowed_episode_ids]
            cand.episodes = list(dict.fromkeys(cand_eps))
            if not cand.episodes:
                logger_ref.warning(f"Rejecting recovered candidate {cand.proposal_id}: no valid allowed episodes.")
                continue

        # Validate source ranges
        valid_ranges: list[CandidateSourceRange] = []
        for r in cand.source_ranges:
            ep_id = r.episode_id
            if allowed_episode_ids is not None and ep_id and ep_id not in allowed_episode_ids:
                continue
            dur = durations.get(ep_id, 0.0)
            start = max(0.0, float(r.start_seconds))
            end = max(start, float(r.end_seconds))
            if dur > 0.0:
                if start > dur:
                    continue
                if end > dur:
                    end = dur
            ev_ref = r.evidence_ref
            if existing_evidence_refs and ev_ref and ev_ref not in existing_evidence_refs:
                ev_ref = None
            valid_ranges.append(CandidateSourceRange(
                episode_id=ep_id,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                evidence_ref=ev_ref,
            ))
        cand.source_ranges = valid_ranges

        # Grounding check against existing evidence refs (if evidence refs are present)
        if existing_evidence_refs:
            cand.supporting_evidence = [ref for ref in cand.supporting_evidence if ref in existing_evidence_refs]
            cand.observed_facts = [ref for ref in cand.observed_facts if ref in existing_evidence_refs]
            matched_refs = (
                set(cand.supporting_evidence)
                | set(cand.observed_facts)
                | {r.evidence_ref for r in valid_ranges if r.evidence_ref}
            )
            if not matched_refs:
                logger_ref.warning(f"Rejecting recovered candidate {cand.proposal_id}: not grounded in existing evidence refs.")
                continue
        else:
            if not valid_ranges and not cand.supporting_evidence and not cand.observed_facts:
                logger_ref.warning(f"Rejecting recovered candidate {cand.proposal_id}: no source ranges or evidence.")
                continue

        # Scope auto-correct
        num_eps = len(cand.episodes)
        if num_eps == 1:
            if cand.candidate_scope in (CandidateScope.CROSS_EPISODE.value, CandidateScope.SEASON_ARC.value):
                cand.candidate_scope = CandidateScope.SINGLE_EPISODE.value
        elif num_eps > 1:
            if cand.candidate_scope in (CandidateScope.SINGLE_EPISODE.value, CandidateScope.SINGLE_SCENE.value):
                cand.candidate_scope = CandidateScope.CROSS_EPISODE.value

        cand.status = CandidateStatus.KEEP.value
        valid_candidates.append(cand)

    return valid_candidates


def compute_health_hash(
    health: PipelineHealth,
    compaction_level: CompactionLevel = CompactionLevel.FULL,
) -> str:
    """Compute deterministic SHA256 hash of pipeline health input."""
    rejected_ids = sorted([
        getattr(d, "candidate_id", "") or d.get("candidate_id", "")
        for d in health.consolidation_decisions
        if getattr(d, "action", "") == ConsolidationAction.REJECT.value
        or (isinstance(d, dict) and d.get("action") == ConsolidationAction.REJECT.value)
    ])
    payload = {
        "compaction_level": compaction_level.value,
        "discovered_count": health.discovered_count,
        "discovery_completed": health.discovery_completed,
        "finalizer_attempted": health.finalizer_attempted,
        "parser_failed": health.parser_failed,
        "rejected_ids": rejected_ids,
        "schema_rejected": health.schema_rejected,
        "total_evidence_count": health.total_evidence_count,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CandidateVerifier:
    """Orchestrates deterministic classification and AI-assisted verification of zero/low candidates."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        client: OpenAICompatibleClient | None = None,
        cache: HierarchyCacheManager | None = None,
        policy: EditorialPolicy | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.policy = policy or EditorialPolicy()
        self.cache = cache or HierarchyCacheManager()
        self.logger = logging.getLogger(f"{__name__}.CandidateVerifier")

        if client is not None:
            self.client = client
        else:
            self.client = OpenAICompatibleClient(
                endpoint=self.settings.api_endpoint,
                api_key=self.settings.api_key,
                timeout=getattr(self.settings, "request_timeout", 120),
            )

    def verify(
        self,
        scope_id: str = "default",
        health: PipelineHealth | dict[str, Any] | None = None,
        candidates: Sequence[CandidateProposal] | None = None,
        *,
        model: str | None = None,
        thinking: str = "auto",
        allowed_episode_ids: set[str] | None = None,
        existing_evidence_refs: set[str] | None = None,
        episode_durations: dict[str, float] | None = None,
        cancellation_token: threading.Event | None = None,
        phase_callback: PhaseCallback | None = None,
    ) -> VerificationResult:
        """Run candidate output verification."""
        health_obj = health if isinstance(health, PipelineHealth) else (PipelineHealth.from_dict(health) if health else PipelineHealth())
        cand_list = list(candidates) if candidates is not None else list(health_obj.consolidated_candidates)

        # 1. Deterministic classification first
        should_trigger, det_reason, diagnostics = should_trigger_verification(
            health_obj,
            cand_list,
        )

        if cancellation_token and cancellation_token.is_set():
            raise AnalysisCancelledError("Candidate verification cancelled (đã bị hủy).")

        # If deterministic state has a failure, return immediately without AI
        if det_reason in (
            ZeroOutputReason.PARSER_FAILURE,
            ZeroOutputReason.MALFORMED_AI_RESPONSE,
            ZeroOutputReason.CANDIDATE_DISCOVERY_FAILED,
            ZeroOutputReason.FINALIZER_FAILED,
            ZeroOutputReason.COVERAGE_INCOMPLETE,
            ZeroOutputReason.INSUFFICIENT_EVIDENCE,
        ):
            self.logger.info(f"Deterministic classification identified failure reason: {det_reason.value}")
            return VerificationResult(
                reason=det_reason,
                is_valid_zero=False,
                recovered_candidates=[],
                diagnostics=diagnostics,
                completed=False,
                rationale=f"Pipeline failure identified deterministically: {det_reason.value}",
            )

        # If not triggered (healthy candidate set, not zero and not suspicious)
        if not should_trigger:
            return VerificationResult(
                reason=ZeroOutputReason.NO_ELIGIBLE_CANDIDATES,
                is_valid_zero=False,
                recovered_candidates=[],
                diagnostics=diagnostics,
                completed=True,
                rationale="Candidate set is healthy and sufficient; verification not triggered.",
            )

        # Triggered: Zero output or suspicious low coverage
        if phase_callback:
            phase_callback(
                AnalysisPhase.ZERO_OUTPUT_VERIFICATION,
                f"Bắt đầu xác minh kết quả ứng viên ({scope_id})...",
                {"trigger_reason": det_reason.value if det_reason else "ZERO_OUTPUT", **diagnostics},
            )

        chosen_model = model or getattr(self.settings, "model", "default-model")
        cand_directive = self.policy.candidate_directive if self.policy else CandidateDirective()
        dir_hash = (
            cand_directive.hash()
            if hasattr(cand_directive, "hash")
            else hashlib.sha256(str(cand_directive).encode("utf-8")).hexdigest()
        )

        # Bounded / adaptive compaction
        selected_compaction = CompactionLevel.FULL
        selected_text = ""
        target_ceiling = getattr(self.settings, "target_payload_ceiling", TARGET_PAYLOAD_CEILING)

        for comp_level in (CompactionLevel.FULL, CompactionLevel.TRIMMED, CompactionLevel.PRIORITY, CompactionLevel.SKELETON):
            text = format_verification_user_text(scope_id, health_obj, cand_directive, comp_level)
            est = estimate_verification_request_size(chosen_model, text, thinking)
            if est <= target_ceiling:
                selected_compaction = comp_level
                selected_text = text
                break

        if not selected_text:
            selected_compaction = CompactionLevel.SKELETON
            selected_text = format_verification_user_text(scope_id, health_obj, cand_directive, CompactionLevel.SKELETON)

        # Cache key
        health_hash = compute_health_hash(health_obj, selected_compaction)
        cache_key = compute_verification_cache_key(
            scope_id=scope_id,
            health_hash=health_hash,
            candidate_directive_hash=dir_hash,
            model=chosen_model,
            thinking=thinking,
            prompt_version=VERIFIER_PROMPT_VERSION,
            algo=VERIFIER_ALGO_VERSION,
            compaction_level=selected_compaction.value,
        )

        # Cancellation check before cache / AI
        if cancellation_token and cancellation_token.is_set():
            raise AnalysisCancelledError("Candidate verification cancelled.")

        # Cache lookup
        cached_data, meta = self.cache.load_verification_result(scope_id, cache_key)
        if meta.get("hit") and cached_data is not None:
            self.logger.info(f"Loaded verification result for {scope_id} from cache.")
            if phase_callback:
                phase_callback(
                    AnalysisPhase.ZERO_OUTPUT_VERIFICATION,
                    f"Tải kết quả xác minh ứng viên {scope_id} từ cache",
                    {"cache": True, **diagnostics},
                )

            if cached_data.get("confirm_no_eligible"):
                valid_zero = is_genuine_zero_valid(health_obj)
                res_reason = (
                    det_reason
                    if det_reason in (ZeroOutputReason.CANDIDATES_REJECTED_BY_VALIDATION, ZeroOutputReason.LOW_COVERAGE_SUSPECT)
                    else ZeroOutputReason.NO_ELIGIBLE_CANDIDATES
                )
                return VerificationResult(
                    reason=res_reason,
                    is_valid_zero=valid_zero,
                    recovered_candidates=[],
                    diagnostics=diagnostics,
                    completed=True,
                    rationale=str(cached_data.get("rationale", "")),
                )
            else:
                raw_cands = cached_data.get("recovered_candidates", cached_data.get("candidates", []))
                validated = validate_recovered_candidates(
                    raw_cands,
                    allowed_episode_ids=allowed_episode_ids,
                    existing_evidence_refs=existing_evidence_refs,
                    episode_durations=episode_durations,
                    log=self.logger,
                )
                res_reason = (
                    ZeroOutputReason.LOW_COVERAGE_SUSPECT
                    if det_reason == ZeroOutputReason.LOW_COVERAGE_SUSPECT
                    else ZeroOutputReason.NO_ELIGIBLE_CANDIDATES
                )
                return VerificationResult(
                    reason=res_reason,
                    is_valid_zero=False,
                    recovered_candidates=validated,
                    diagnostics=diagnostics,
                    completed=True,
                    rationale=str(cached_data.get("rationale", "")),
                )

        # Call AI
        try:
            call_kwargs = {
                "model": chosen_model,
                "system": ZERO_OUTPUT_VERIFICATION_SYSTEM_PROMPT,
                "user_text": selected_text,
                "thinking": thinking,
                "cancel_event": cancellation_token,
                "phase": AnalysisPhase.ZERO_OUTPUT_VERIFICATION,
            }
            try:
                raw_resp = self.client.chat_json(**call_kwargs)
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                call_kwargs.pop("phase", None)
                raw_resp = self.client.chat_json(**call_kwargs)
            validate_verification_response_schema(raw_resp, raise_error=True)
        except AnalysisCancelledError:
            raise
        except Exception as e:
            self.logger.error(f"Verification AI call failed or malformed: {e}")
            return VerificationResult(
                reason=ZeroOutputReason.MALFORMED_AI_RESPONSE,
                is_valid_zero=False,
                recovered_candidates=[],
                diagnostics=diagnostics,
                completed=False,
                rationale=f"Verification response malformed: {e}",
            )

        # Save valid before cancel
        self.cache.save_verification_result(
            scope_id=scope_id,
            cache_key=cache_key,
            data=raw_resp,
            compaction_level=selected_compaction.value,
        )

        # Check cancellation right after save
        if cancellation_token and cancellation_token.is_set():
            raise AnalysisCancelledError("Candidate verification cancelled after save.")

        # Process valid response
        if raw_resp.get("confirm_no_eligible"):
            valid_zero = is_genuine_zero_valid(health_obj)
            res_reason = (
                det_reason
                if det_reason in (ZeroOutputReason.CANDIDATES_REJECTED_BY_VALIDATION, ZeroOutputReason.LOW_COVERAGE_SUSPECT)
                else ZeroOutputReason.NO_ELIGIBLE_CANDIDATES
            )
            res = VerificationResult(
                reason=res_reason,
                is_valid_zero=valid_zero,
                recovered_candidates=[],
                diagnostics=diagnostics,
                completed=True,
                rationale=str(raw_resp.get("rationale", "")),
            )
        else:
            raw_cands = raw_resp.get("recovered_candidates", raw_resp.get("candidates", []))
            validated = validate_recovered_candidates(
                raw_cands,
                allowed_episode_ids=allowed_episode_ids,
                existing_evidence_refs=existing_evidence_refs,
                episode_durations=episode_durations,
                log=self.logger,
            )
            res_reason = (
                ZeroOutputReason.LOW_COVERAGE_SUSPECT
                if det_reason == ZeroOutputReason.LOW_COVERAGE_SUSPECT
                else ZeroOutputReason.NO_ELIGIBLE_CANDIDATES
            )
            res = VerificationResult(
                reason=res_reason,
                is_valid_zero=False,
                recovered_candidates=validated,
                diagnostics=diagnostics,
                completed=True,
                rationale=str(raw_resp.get("rationale", "")),
            )

        if phase_callback:
            phase_callback(
                AnalysisPhase.ZERO_OUTPUT_VERIFICATION,
                f"Hoàn thành xác minh ứng viên ({scope_id})",
                {
                    "is_valid_zero": res.is_valid_zero,
                    "recovered": len(res.recovered_candidates),
                    "reason": res.reason.value if isinstance(res.reason, ZeroOutputReason) else str(res.reason),
                },
            )

        return res


def verify_zero_or_low_output(
    scope_id: str = "default",
    health: PipelineHealth | dict[str, Any] | None = None,
    candidates: Sequence[CandidateProposal] | None = None,
    *,
    settings: AppSettings | None = None,
    client: OpenAICompatibleClient | None = None,
    cache: HierarchyCacheManager | None = None,
    policy: EditorialPolicy | None = None,
    model: str | None = None,
    thinking: str = "auto",
    allowed_episode_ids: set[str] | None = None,
    existing_evidence_refs: set[str] | None = None,
    episode_durations: dict[str, float] | None = None,
    cancellation_token: threading.Event | None = None,
    phase_callback: PhaseCallback | None = None,
) -> VerificationResult:
    """Convenience functional interface for candidate zero/low output verification."""
    verifier = CandidateVerifier(settings=settings, client=client, cache=cache, policy=policy)
    return verifier.verify(
        scope_id=scope_id,
        health=health,
        candidates=candidates,
        model=model,
        thinking=thinking,
        allowed_episode_ids=allowed_episode_ids,
        existing_evidence_refs=existing_evidence_refs,
        episode_durations=episode_durations,
        cancellation_token=cancellation_token,
        phase_callback=phase_callback,
    )
