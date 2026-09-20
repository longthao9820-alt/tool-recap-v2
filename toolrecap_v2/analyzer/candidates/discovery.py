"""Narrative candidate discovery module for single-episode and full-season analysis."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any, Callable, Sequence

from ...api_client import (
    APIError,
    OpenAICompatibleClient,
    estimate_request_size,
)
from ...domain.cache import (
    DISCOVERY_SCHEMA_VERSION,
    HierarchyCacheManager,
    compute_discovery_cache_key,
    validate_discovery_cache_data,
)
from ...domain.enums import CandidateScope, CandidateStatus, CompactionLevel
from ...domain.models import (
    CandidateProposal,
    CandidateSourceRange,
    CompactEpisodeSummary,
    CompactSummaryItem,
    SourceEpisode,
    compact_summary,
    split_summary_by_timeline,
)
from ...domain.policy import (
    CandidateDirective,
    EditorialPolicy,
)
from ...settings import AppSettings
from ..errors import AnalysisCancelledError, AnalysisError
from ..phases import AnalysisPhase, PhaseCallback
from ..prompts import CANDIDATE_DISCOVERY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

HARD_PAYLOAD_CEILING: int = 500_000
TARGET_PAYLOAD_CEILING: int = 480_000
DISCOVERY_ALGO_VERSION: str = "v1"
DISCOVERY_PROMPT_VERSION: str = "v1"


def validate_discovery_response_schema(
    raw: Any,
    *,
    raise_error: bool = False,
) -> bool:
    """Validate AI response schema for candidate discovery.

    Requirements:
    - Must be a dictionary.
    - Must contain 'discovered_candidates' (or 'candidate_proposals' or 'candidates').
    - That value must be a list of dicts.
    - If malformed, returns False (or raises AnalysisError if raise_error is True).
    """
    if not isinstance(raw, dict):
        if raise_error:
            raise AnalysisError(f"Phản hồi candidate discovery phải là dict, nhận được {type(raw).__name__}.")
        return False

    cand_list = None
    for key in ("discovered_candidates", "candidate_proposals", "candidates"):
        if key in raw:
            cand_list = raw[key]
            break

    if cand_list is None:
        if raise_error:
            raise AnalysisError("Phản hồi candidate discovery thiếu trường 'discovered_candidates'.")
        return False

    if not isinstance(cand_list, list):
        if raise_error:
            raise AnalysisError(f"Trường 'discovered_candidates' phải là list, nhận được {type(cand_list).__name__}.")
        return False

    for idx, item in enumerate(cand_list):
        if not isinstance(item, dict):
            if raise_error:
                raise AnalysisError(f"Phần tử {idx} trong 'discovered_candidates' phải là dict, nhận được {type(item).__name__}.")
            return False

    return True


def estimate_discovery_request_size(
    model: str,
    user_text: str,
    thinking: str = "auto",
    max_tokens: int = 32_000,
) -> int:
    """Estimate token/byte request size for candidate discovery prompt."""
    return estimate_request_size(
        model=model,
        system=CANDIDATE_DISCOVERY_SYSTEM_PROMPT,
        user_text=user_text,
        thinking=thinking,
        max_tokens=max_tokens,
        variant=0,
    )


def format_discovery_node_user_text(
    node_id: str,
    summaries: Sequence[CompactEpisodeSummary],
    candidate_directive: CandidateDirective | str | None = None,
    connection_threads: list[dict[str, Any]] | None = None,
    supporting_arcs: list[dict[str, Any]] | None = None,
    coverage_ledger_summary: str = "",
) -> str:
    """Format prompt for a candidate discovery node.

    Strictly uses compact CandidateDirective and grounded summaries/connections.
    Excludes scanner directives or irrelevant scanner categories.
    """
    if isinstance(candidate_directive, CandidateDirective):
        dir_text = candidate_directive.format_directive()
    elif candidate_directive is not None:
        dir_text = str(candidate_directive).strip()
    else:
        dir_text = ""

    parts: list[str] = [
        f"Discover narrative recap candidates for node {node_id}.",
    ]

    ep_ids = [s.episode_id for s in summaries if s.episode_id]
    if ep_ids:
        parts.append(f"Covered episodes: {', '.join(ep_ids)}")

    if dir_text:
        parts.append(f"Candidate Directive:\n{dir_text}")

    if coverage_ledger_summary:
        parts.append(f"Coverage Information:\n{coverage_ledger_summary}")

    if connection_threads:
        parts.append(f"Cross-Episode Connection Threads:\n{json.dumps(connection_threads, ensure_ascii=False, indent=2)}")

    if supporting_arcs:
        parts.append(f"Supporting Character Arcs:\n{json.dumps(supporting_arcs, ensure_ascii=False, indent=2)}")

    parts.append(f"Episode Summaries:\n{json.dumps([s.to_dict() for s in summaries], ensure_ascii=False, indent=2)}")

    return "\n\n".join(parts)


def compute_coverage_hash(coverage_ledger: Any) -> str:
    """Deterministically compute sha256 hash of coverage ledger for cache key."""
    if coverage_ledger is None:
        return "none"
    if hasattr(coverage_ledger, "canonical_hash"):
        try:
            return str(coverage_ledger.canonical_hash())
        except Exception:
            pass
    if hasattr(coverage_ledger, "to_dict"):
        try:
            raw = json.dumps(coverage_ledger.to_dict(), sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()
        except Exception:
            pass
    if isinstance(coverage_ledger, dict):
        raw = json.dumps(coverage_ledger, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if isinstance(coverage_ledger, (list, tuple)):
        items_serialized = []
        for it in coverage_ledger:
            if hasattr(it, "to_dict"):
                items_serialized.append(it.to_dict())
            elif isinstance(it, dict):
                items_serialized.append(it)
            else:
                items_serialized.append(str(it))
        raw = json.dumps(items_serialized, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return hashlib.sha256(str(coverage_ledger).encode("utf-8")).hexdigest()


@dataclass
class DiscoveryNodePlanItem:
    """Planning item for a single discovery AI request."""

    node_id: str
    span_start: int
    span_end: int
    summaries: list[CompactEpisodeSummary]
    compaction_level: CompactionLevel
    user_text: str
    cache_key: str
    child_hashes: list[str]
    estimated_bytes: int


def _highest_compaction_level(levels: Sequence[CompactionLevel]) -> CompactionLevel:
    order = [
        CompactionLevel.FULL,
        CompactionLevel.TRIMMED,
        CompactionLevel.PRIORITY,
        CompactionLevel.SKELETON,
    ]
    max_idx = 0
    for lvl in levels:
        if lvl in order:
            max_idx = max(max_idx, order.index(lvl))
    return order[max_idx]


def _recursively_split_summary_to_fit(
    summary: CompactEpisodeSummary,
    fits_fn: Callable[[CompactEpisodeSummary], bool],
    depth: int = 0,
    max_depth: int = 6,
) -> list[CompactEpisodeSummary]:
    """Recursively split summary items by timeline until each part fits."""
    if fits_fn(summary) or depth >= max_depth or len(summary.items) <= 1:
        return [summary]

    left_summ, right_summ = split_summary_by_timeline(summary)
    if not left_summ.items or not right_summ.items:
        return [summary]

    result: list[CompactEpisodeSummary] = []
    result.extend(_recursively_split_summary_to_fit(left_summ, fits_fn, depth + 1, max_depth))
    result.extend(_recursively_split_summary_to_fit(right_summ, fits_fn, depth + 1, max_depth))
    return result


def plan_discovery_nodes(
    summaries: Sequence[CompactEpisodeSummary],
    connection: Any = None,
    candidate_directive: CandidateDirective | None = None,
    coverage_hash: str = "none",
    model: str = "gemini-2.5-flash",
    thinking: str = "auto",
    target_ceiling: int = TARGET_PAYLOAD_CEILING,
    hard_ceiling: int = HARD_PAYLOAD_CEILING,
    coverage_ledger_summary: str = "",
) -> list[DiscoveryNodePlanItem]:
    """Partition episode summaries and connection threads into bounded discovery nodes.

    Uses adaptive compaction and recursive splitting to keep requests within target_ceiling.
    """
    if not summaries:
        return []

    # Extract connection elements
    cross_links: list[dict[str, Any]] = []
    char_arcs: list[dict[str, Any]] = []
    if connection is not None:
        if hasattr(connection, "cross_episode_links"):
            cross_links = [dict(x) for x in getattr(connection, "cross_episode_links", [])]
        elif isinstance(connection, dict):
            cross_links = [dict(x) for x in connection.get("cross_episode_links", [])]

        if hasattr(connection, "supporting_character_arcs"):
            char_arcs = [dict(x) for x in getattr(connection, "supporting_character_arcs", [])]
        elif isinstance(connection, dict):
            char_arcs = [dict(x) for x in connection.get("supporting_character_arcs", [])]

    cand_directive = candidate_directive or CandidateDirective()
    cand_dir_hash = cand_directive.compute_hash()

    # Step 1: create units for each summary, compacting or splitting if needed
    units: list[tuple[int, int, CompactEpisodeSummary, CompactionLevel]] = []

    for ord_idx, summ in enumerate(summaries):
        def _fits(s: CompactEpisodeSummary) -> bool:
            txt = format_discovery_node_user_text(
                "test_fit",
                [s],
                cand_directive,
                cross_links,
                char_arcs,
                coverage_ledger_summary,
            )
            return estimate_discovery_request_size(model, txt, thinking) <= target_ceiling

        if _fits(summ):
            units.append((ord_idx, ord_idx, summ, CompactionLevel.FULL))
            continue

        fit_level: CompactionLevel | None = None
        compacted_unit: CompactEpisodeSummary | None = None
        for lvl in (CompactionLevel.TRIMMED, CompactionLevel.PRIORITY, CompactionLevel.SKELETON):
            c_s = compact_summary(summ, lvl)
            if _fits(c_s):
                fit_level = lvl
                compacted_unit = c_s
                break

        if fit_level is not None and compacted_unit is not None:
            units.append((ord_idx, ord_idx, compacted_unit, fit_level))
        else:
            skeleton_s = compact_summary(summ, CompactionLevel.SKELETON)
            frags = _recursively_split_summary_to_fit(skeleton_s, _fits)
            for frag in frags:
                units.append((ord_idx, ord_idx, frag, CompactionLevel.SKELETON))

    # Step 2: Greedy packing of units into discovery nodes
    node_plans: list[DiscoveryNodePlanItem] = []
    current_group: list[tuple[int, int, CompactEpisodeSummary, CompactionLevel]] = []

    def _build_plan(group: list[tuple[int, int, CompactEpisodeSummary, CompactionLevel]]) -> DiscoveryNodePlanItem:
        span_start = group[0][0]
        span_end = group[-1][1]
        group_summs = [u[2] for u in group]
        compaction_level = _highest_compaction_level([u[3] for u in group])

        content_raw = json.dumps([s.to_dict() for s in group_summs], sort_keys=True)
        content_hash = hashlib.sha256(content_raw.encode("utf-8")).hexdigest()

        node_id = f"cand_node_{span_start}_{span_end}_{content_hash[:8]}"

        # Filter relevant links/arcs for this node's episodes
        node_ep_set = {s.episode_id for s in group_summs if s.episode_id}
        rel_links = [
            lnk for lnk in cross_links
            if not lnk.get("episodes") or any(ep in node_ep_set for ep in lnk.get("episodes", []))
        ] if cross_links else None
        rel_arcs = [
            arc for arc in char_arcs
            if not arc.get("episodes") or any(ep in node_ep_set for ep in arc.get("episodes", []))
        ] if char_arcs else None

        user_text = format_discovery_node_user_text(
            node_id=node_id,
            summaries=group_summs,
            candidate_directive=cand_directive,
            connection_threads=rel_links,
            supporting_arcs=rel_arcs,
            coverage_ledger_summary=coverage_ledger_summary,
        )
        est_size = estimate_discovery_request_size(model, user_text, thinking)

        if est_size > hard_ceiling:
            raise AnalysisError(
                f"Không thể tạo request candidate discovery an toàn cho node {node_id}: "
                f"{est_size} bytes vượt giới hạn {hard_ceiling} bytes sau mọi bước compact/split."
            )

        child_hashes = [s.canonical_hash() for s in group_summs]
        cache_key = compute_discovery_cache_key(
            node_id=node_id,
            child_hashes=child_hashes,
            candidate_directive_hash=cand_dir_hash,
            coverage_hash=coverage_hash,
            model=model,
            thinking=thinking,
            prompt_version=DISCOVERY_PROMPT_VERSION,
            algo=DISCOVERY_ALGO_VERSION,
            compaction_level=compaction_level.value,
        )

        return DiscoveryNodePlanItem(
            node_id=node_id,
            span_start=span_start,
            span_end=span_end,
            summaries=group_summs,
            compaction_level=compaction_level,
            user_text=user_text,
            cache_key=cache_key,
            child_hashes=child_hashes,
            estimated_bytes=est_size,
        )

    for item in units:
        test_group = current_group + [item]
        test_summs = [u[2] for u in test_group]
        test_text = format_discovery_node_user_text(
            "test_node",
            test_summs,
            cand_directive,
            cross_links,
            char_arcs,
            coverage_ledger_summary,
        )
        est = estimate_discovery_request_size(model, test_text, thinking)

        if est <= target_ceiling or not current_group:
            current_group = test_group
        else:
            node_plans.append(_build_plan(current_group))
            current_group = [item]

    if current_group:
        node_plans.append(_build_plan(current_group))

    return node_plans


def ground_validate_candidate(
    candidate: CandidateProposal,
    allowed_episode_ids: set[str],
    episode_durations: dict[str, float] | None = None,
    log: logging.Logger | None = None,
) -> CandidateProposal | None:
    """Validate and ground candidate proposal against allowed episodes and durations.

    1. Unknown episodes stripped.
    2. Empty candidate rejected with logged reason.
    3. Source ranges timestamps validated within source summary duration.
    4. Scope auto-corrected.
    5. Preserves AI title and central_thesis intact.
    6. No synthetic candidate fabricated.
    """
    durations = episode_durations or {}
    logger_ref = log or logger

    # Strip unknown episodes
    valid_eps = [ep for ep in candidate.episodes if ep in allowed_episode_ids]

    # Validate source ranges
    valid_ranges: list[CandidateSourceRange] = []
    for r in candidate.source_ranges:
        if not r.episode_id or r.episode_id in allowed_episode_ids:
            ep_id = r.episode_id
            dur = durations.get(ep_id, 0.0)
            start = max(0.0, float(r.start_seconds))
            end = max(start, float(r.end_seconds))
            if dur > 0.0:
                if start > dur:
                    # Timestamp past source summary duration: drop range
                    continue
                if end > dur:
                    end = dur
            valid_ranges.append(CandidateSourceRange(
                episode_id=ep_id,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                evidence_ref=r.evidence_ref,
            ))

    # Infer episodes from valid ranges if episodes were empty
    if not valid_eps and valid_ranges:
        valid_eps = list(dict.fromkeys(r.episode_id for r in valid_ranges if r.episode_id))

    # Reject empty candidate
    if not valid_eps and not valid_ranges:
        logger_ref.warning(
            f"Rejecting candidate '{candidate.title}' (id={candidate.proposal_id}): "
            f"no valid episodes in allowed set {allowed_episode_ids}."
        )
        return None

    if not candidate.title.strip() and not candidate.proposal_id.strip():
        logger_ref.warning("Rejecting candidate: empty title and proposal_id.")
        return None

    candidate.episodes = valid_eps
    candidate.source_ranges = valid_ranges

    # Scope auto-correct
    num_eps = len(valid_eps)
    if num_eps == 1:
        if candidate.candidate_scope in (CandidateScope.CROSS_EPISODE.value, CandidateScope.SEASON_ARC.value):
            candidate.candidate_scope = CandidateScope.SINGLE_EPISODE.value
    elif num_eps > 1:
        if candidate.candidate_scope in (CandidateScope.SINGLE_EPISODE.value, CandidateScope.SINGLE_SCENE.value):
            if len(allowed_episode_ids) > 1 and num_eps == len(allowed_episode_ids):
                candidate.candidate_scope = CandidateScope.SEASON_ARC.value
            else:
                candidate.candidate_scope = CandidateScope.CROSS_EPISODE.value

    return candidate


def migrate_legacy_connection_candidates(
    connection: Any,
) -> list[CandidateProposal]:
    """Extract embedded candidate proposals from legacy connection result without calling AI."""
    if connection is None:
        return []

    data: dict[str, Any] = {}
    if isinstance(connection, (str, Path)):
        p = Path(connection)
        if not p.is_file():
            return []
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded.get("result", loaded) if isinstance(loaded.get("result"), dict) else loaded
    elif hasattr(connection, "candidate_proposals"):
        raw_list = getattr(connection, "candidate_proposals", [])
        return [
            p if isinstance(p, CandidateProposal) else CandidateProposal.from_dict(p)
            for p in raw_list
            if isinstance(p, (CandidateProposal, dict))
        ]
    elif isinstance(connection, dict):
        data = connection.get("result", connection) if isinstance(connection.get("result"), dict) else connection
    else:
        return []

    raw_props: list[Any] = []
    for k in ("candidate_proposals", "candidates", "discovered_candidates"):
        if k in data and isinstance(data[k], list):
            raw_props = data[k]
            break

    proposals: list[CandidateProposal] = []
    for item in raw_props:
        if isinstance(item, CandidateProposal):
            proposals.append(item)
        elif isinstance(item, dict):
            proposals.append(CandidateProposal.from_dict(item))

    return proposals


class CandidateDiscoverer:
    """Distinct AI stage for discovering grounded narrative recap candidates.

    Takes episode summaries, connection threads, and coverage ledger.
    Produces validated CandidateProposal records without quota truncation.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        client: Any = None,
        policy: EditorialPolicy | None = None,
        cache: HierarchyCacheManager | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.client = client or OpenAICompatibleClient(
            endpoint=self.settings.api_endpoint,
            api_key=self.settings.api_key,
        )
        self.policy = policy or EditorialPolicy()
        self.cache = cache or HierarchyCacheManager()
        self.logger = logging.getLogger(__name__)

    def discover_single(
        self,
        summary: CompactEpisodeSummary | dict[str, Any],
        connection: Any = None,
        coverage_ledger: Any = None,
        *,
        episode: SourceEpisode | None = None,
        phase_callback: PhaseCallback | None = None,
        cancellation_token: threading.Event | None = None,
        model: str | None = None,
        thinking: str | None = None,
    ) -> list[CandidateProposal]:
        """Discover recap candidates for a single episode."""
        summ_obj = summary if isinstance(summary, CompactEpisodeSummary) else CompactEpisodeSummary.from_dict(summary)
        ep_list = [episode] if episode is not None else None
        return self.discover_season(
            summaries=[summ_obj],
            connection=connection,
            coverage_ledger=coverage_ledger,
            episodes=ep_list,
            phase_callback=phase_callback,
            cancellation_token=cancellation_token,
            model=model,
            thinking=thinking,
        )

    def discover_candidates_single(self, *args: Any, **kwargs: Any) -> list[CandidateProposal]:
        """Alias for discover_single."""
        return self.discover_single(*args, **kwargs)

    def discover_season(
        self,
        summaries: Sequence[CompactEpisodeSummary | dict[str, Any]],
        connection: Any = None,
        coverage_ledger: Any = None,
        *,
        episodes: Sequence[SourceEpisode] | None = None,
        phase_callback: PhaseCallback | None = None,
        cancellation_token: threading.Event | None = None,
        model: str | None = None,
        thinking: str | None = None,
    ) -> list[CandidateProposal]:
        """Discover recap candidates across multiple episodes for a season."""
        effective_model = model or getattr(self.settings, "season_connection_model", None) or getattr(self.settings, "evidence_scanner_model", "gemini-2.5-flash")
        effective_thinking = thinking or getattr(self.settings, "season_connection_thinking", "auto")

        # Convert summaries
        summs: list[CompactEpisodeSummary] = []
        for s in summaries:
            if isinstance(s, CompactEpisodeSummary):
                summs.append(s)
            elif isinstance(s, dict):
                summs.append(CompactEpisodeSummary.from_dict(s))

        # Allowed episodes and durations
        allowed_episode_ids: set[str] = {s.episode_id for s in summs if s.episode_id}
        episode_durations: dict[str, float] = {s.episode_id: s.duration_seconds for s in summs if s.duration_seconds > 0}
        if episodes:
            for ep in episodes:
                if ep.episode_id:
                    allowed_episode_ids.add(ep.episode_id)
                if ep.duration_seconds > 0:
                    episode_durations[ep.episode_id] = ep.duration_seconds

        # Consume legacy connection proposals as discovery seed (Requirement 6)
        seed_proposals: list[CandidateProposal] = []
        if connection is not None:
            seeds = migrate_legacy_connection_candidates(connection)
            for s in seeds:
                validated_s = ground_validate_candidate(s, allowed_episode_ids, episode_durations, self.logger)
                if validated_s:
                    seed_proposals.append(validated_s)

        # Plan discovery nodes
        cand_directive = self.policy.candidate_directive if self.policy else CandidateDirective()
        cov_hash = compute_coverage_hash(coverage_ledger)

        nodes = plan_discovery_nodes(
            summaries=summs,
            connection=connection,
            candidate_directive=cand_directive,
            coverage_hash=cov_hash,
            model=effective_model,
            thinking=effective_thinking,
            target_ceiling=getattr(self.settings, "target_payload_ceiling", TARGET_PAYLOAD_CEILING),
            hard_ceiling=getattr(self.settings, "hard_payload_ceiling", HARD_PAYLOAD_CEILING),
        )

        total_nodes = len(nodes)
        new_proposals: list[CandidateProposal] = []

        for idx, node in enumerate(nodes):
            # Cancellation check
            if cancellation_token and cancellation_token.is_set():
                raise AnalysisCancelledError("Candidate discovery cancelled.")

            # Cache lookup
            cached_data, meta = self.cache.load_discovery_result(node.node_id, node.cache_key)
            if meta.get("hit") and cached_data is not None:
                raw_cands = cached_data.get("discovered_candidates", cached_data.get("candidate_proposals", []))
                self.logger.info(f"Loaded discovery node {node.node_id} from cache: {len(raw_cands)} candidates.")
                if phase_callback:
                    phase_callback(
                        AnalysisPhase.CANDIDATE_DISCOVERY,
                        f"Tải kết quả khám phá ứng viên {node.node_id} từ cache",
                        {"index": idx + 1, "total": total_nodes, "cache": True, "candidates_count": len(raw_cands)},
                    )
            else:
                # Call AI client
                raw_resp = self.client.chat_json(
                    model=effective_model,
                    system=CANDIDATE_DISCOVERY_SYSTEM_PROMPT,
                    user_text=node.user_text,
                    thinking=effective_thinking,
                    cancel_event=cancellation_token,
                )

                # Schema validate (Requirement 3: malformed domain response not cache)
                if not validate_discovery_response_schema(raw_resp):
                    raise AnalysisError(f"Phản hồi candidate discovery cho node {node.node_id} sai cấu trúc schema, không lưu cache.")

                # Save atomically BEFORE cancel (Requirement 5: Save valid before cancel)
                self.cache.save_discovery_result(
                    node_id=node.node_id,
                    cache_key=node.cache_key,
                    data=raw_resp,
                    compaction_level=node.compaction_level.value,
                )

                raw_cands = raw_resp.get("discovered_candidates", raw_resp.get("candidate_proposals", []))
                self.logger.info(f"AI discovered {len(raw_cands)} candidates for node {node.node_id}.")
                if phase_callback:
                    phase_callback(
                        AnalysisPhase.CANDIDATE_DISCOVERY,
                        f"Đã khám phá {len(raw_cands)} ứng viên cho node {node.node_id}",
                        {"index": idx + 1, "total": total_nodes, "cache": False, "candidates_count": len(raw_cands)},
                    )

            # Cancellation check right after save
            if cancellation_token and cancellation_token.is_set():
                raise AnalysisCancelledError("Candidate discovery cancelled after node save.")

            # Ground validate each candidate (Requirement 4)
            for raw_cand in raw_cands:
                cand = CandidateProposal.from_dict(raw_cand)
                valid_cand = ground_validate_candidate(cand, allowed_episode_ids, episode_durations, self.logger)
                if valid_cand is not None:
                    new_proposals.append(valid_cand)

        # Combine seed and new proposals, dedup exact IDs only (Requirement 6)
        combined: list[CandidateProposal] = list(seed_proposals)
        existing_indices: dict[str, int] = {p.proposal_id: i for i, p in enumerate(combined) if p.proposal_id}

        for cand in new_proposals:
            if cand.proposal_id and cand.proposal_id in existing_indices:
                idx_to_replace = existing_indices[cand.proposal_id]
                combined[idx_to_replace] = cand
            else:
                combined.append(cand)
                if cand.proposal_id:
                    existing_indices[cand.proposal_id] = len(combined) - 1

        # Return all discovered candidates without artificial quota (Requirement 8)
        return combined

    def discover_candidates_season(self, *args: Any, **kwargs: Any) -> list[CandidateProposal]:
        """Alias for discover_season."""
        return self.discover_season(*args, **kwargs)


def discover_candidates_single(
    summary: CompactEpisodeSummary | dict[str, Any],
    connection: Any = None,
    coverage_ledger: Any = None,
    *,
    episode: SourceEpisode | None = None,
    settings: AppSettings | None = None,
    client: Any = None,
    policy: EditorialPolicy | None = None,
    cache: HierarchyCacheManager | None = None,
    phase_callback: PhaseCallback | None = None,
    cancellation_token: threading.Event | None = None,
    model: str | None = None,
    thinking: str | None = None,
) -> list[CandidateProposal]:
    """Functional API for single episode candidate discovery."""
    discoverer = CandidateDiscoverer(settings=settings, client=client, policy=policy, cache=cache)
    return discoverer.discover_single(
        summary=summary,
        connection=connection,
        coverage_ledger=coverage_ledger,
        episode=episode,
        phase_callback=phase_callback,
        cancellation_token=cancellation_token,
        model=model,
        thinking=thinking,
    )


def discover_candidates_season(
    summaries: Sequence[CompactEpisodeSummary | dict[str, Any]],
    connection: Any = None,
    coverage_ledger: Any = None,
    *,
    episodes: Sequence[SourceEpisode] | None = None,
    settings: AppSettings | None = None,
    client: Any = None,
    policy: EditorialPolicy | None = None,
    cache: HierarchyCacheManager | None = None,
    phase_callback: PhaseCallback | None = None,
    cancellation_token: threading.Event | None = None,
    model: str | None = None,
    thinking: str | None = None,
) -> list[CandidateProposal]:
    """Functional API for multi-episode season candidate discovery."""
    discoverer = CandidateDiscoverer(settings=settings, client=client, policy=policy, cache=cache)
    return discoverer.discover_season(
        summaries=summaries,
        connection=connection,
        coverage_ledger=coverage_ledger,
        episodes=episodes,
        phase_callback=phase_callback,
        cancellation_token=cancellation_token,
        model=model,
        thinking=thinking,
    )
