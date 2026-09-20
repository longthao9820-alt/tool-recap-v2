"""Season connection pass: hierarchical cross-episode link analysis, batching, caching, and merge."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from ..api_client import (
    APIError,
    OpenAICompatibleClient,
    SEASON_CONNECTION_TIMEOUT,
    _is_cancelled,
    estimate_request_size,
)
from ..domain.cache import (
    HierarchyCacheManager,
    compute_batch_cache_key,
    compute_connection_cache_key,
    compute_merge_cache_key,
    compute_summary_cache_key,
)
from ..domain.enums import CandidateScope, CompactionLevel
from ..domain.models import (
    CompactEpisodeSummary,
    CompactSummaryItem,
    EpisodeEvidence,
    SourceEpisode,
    build_compact_summary,
    compact_summary,
    split_summary_by_timeline,
)
from ..settings import AppSettings
from .errors import AnalysisCancelledError, AnalysisError, CoverageIncompleteError
from .phases import AnalysisPhase, PhaseCallback
from .prompts import (
    SEASON_BATCH_SYSTEM_PROMPT,
    SEASON_CONNECTION_SYSTEM_PROMPT,
    SEASON_MERGE_SYSTEM_PROMPT,
)

HARD_PAYLOAD_CEILING = 500_000
TARGET_PAYLOAD_CEILING = 480_000
HIERARCHY_ALGO_VERSION = "v3"
MAX_ITEMS_PER_BATCH_HINT = 4
MAX_PAYLOAD_BYTES = HARD_PAYLOAD_CEILING


def check_payload_size(payload: str, max_bytes: int = MAX_PAYLOAD_BYTES, context: str = "") -> None:
    """Ensure request payload does not exceed bounded size."""
    byte_len = len(payload.encode("utf-8"))
    if byte_len > max_bytes:
        raise AnalysisError(
            f"Payload size {byte_len} bytes exceeds max allowed limit of {max_bytes} bytes for {context}. "
            f"Raw evidence or oversized payload violates hierarchical compact mandates."
        )


def validate_batch_response_schema(raw: Any, context: str = "batch") -> None:
    """Validate AI batch or merge response schema before persisting to cache.

    Requires top-level dict and list types for connection structures.
    Malformed responses raise AnalysisError and must never be cached.
    """
    if not isinstance(raw, dict):
        raise AnalysisError(f"Phản hồi AI {context} phải là một dictionary, nhận được {type(raw).__name__}.")

    list_fields = [
        ("cross_episode_links", "links", "narrative_links"),
        ("candidate_proposals", "candidates", "proposals"),
        ("supporting_character_arcs", "supporting_arcs", "character_arcs"),
        ("rejected_or_merged", "rejected"),
    ]

    has_known_key = False
    for group in list_fields:
        for key in group:
            if key in raw:
                val = raw[key]
                if not isinstance(val, list):
                    raise AnalysisError(
                        f"Phản hồi AI {context} không hợp lệ: '{key}' phải là kiểu list, nhận được {type(val).__name__}."
                    )
                for idx, item in enumerate(val):
                    if not isinstance(item, dict):
                        raise AnalysisError(
                            f"Phản hồi AI {context} không hợp lệ: phần tử {idx} trong '{key}' phải là dict."
                        )
                has_known_key = True

    if not has_known_key and "outputs" not in raw and "batch_results" not in raw:
        raise AnalysisError(
            f"Phản hồi AI {context} thiếu các trường danh sách liên kết bắt buộc (cross_episode_links/candidate_proposals)."
        )


def format_node_id(
    level: str,
    span_start: int,
    span_end: int,
    content_hash: str,
    fragment_label: str = "",
) -> str:
    """Generic content/order based node ID: node_<level>_<span ordinals>_<hash> without episode names."""
    short_hash = content_hash[:8] if content_hash else "00000000"
    if fragment_label:
        return f"node_{level}_{span_start}_{span_end}_{fragment_label}_{short_hash}"
    return f"node_{level}_{span_start}_{span_end}_{short_hash}"


@dataclass
class AdaptiveBatchPlanItem:
    node_id: str
    span_start: int
    span_end: int
    summaries: list[CompactEpisodeSummary]
    compaction_level: CompactionLevel
    user_text: str
    cache_key: str
    estimated_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "episode_ids": [s.episode_id for s in self.summaries],
            "compaction_level": self.compaction_level.value,
            "estimated_bytes": self.estimated_bytes,
        }


def format_batch_user_text(
    node_id: str,
    summaries: list[CompactEpisodeSummary],
    coverage_notice: str,
    recap_prompt: str,
) -> str:
    batch_payload = {
        "node_id": node_id,
        "batch_id": node_id,
        "episodes": [s.to_dict() for s in summaries],
    }
    return (
        f"Analyze batch connections for {node_id} across {len(summaries)} episodes.\n"
        f"{coverage_notice}\n"
        f"Recap instructions:\n{recap_prompt or 'Standard video recap'}\n\n"
        f"Batch Compact Summaries:\n{json.dumps(batch_payload, ensure_ascii=False, indent=2)}\n"
    )


def estimate_batch_request_size(
    model: str,
    user_text: str,
    thinking: str = "auto",
    max_tokens: int = 32_000,
) -> int:
    return estimate_request_size(
        model=model,
        system=SEASON_BATCH_SYSTEM_PROMPT,
        user_text=user_text,
        thinking=thinking,
        max_tokens=max_tokens,
        variant=0,
    )


def format_merge_user_text(
    node_id: str,
    group: list[dict[str, Any]],
    coverage_notice: str,
    recap_prompt: str,
) -> str:
    merge_payload = {
        "node_id": node_id,
        "merge_id": node_id,
        "batch_results": group,
    }
    return (
        f"Merge and synthesize {len(group)} batch results into unified season connections for {node_id}.\n"
        f"{coverage_notice}\n"
        f"Recap instructions:\n{recap_prompt or 'Standard video recap'}\n\n"
        f"Batch Results:\n{json.dumps(merge_payload, ensure_ascii=False, indent=2)}\n"
    )


def estimate_merge_request_size(
    model: str,
    user_text: str,
    thinking: str = "auto",
    max_tokens: int = 32_000,
) -> int:
    return estimate_request_size(
        model=model,
        system=SEASON_MERGE_SYSTEM_PROMPT,
        user_text=user_text,
        thinking=thinking,
        max_tokens=max_tokens,
        variant=0,
    )


def normalize_and_dedup_merge_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Deterministically normalize and deduplicate AI result structures."""
    if not isinstance(raw, dict):
        return {}

    raw_links: list[dict[str, Any]] = []
    raw_proposals: list[dict[str, Any]] = []
    raw_arcs: list[dict[str, Any]] = []
    raw_rejected: list[dict[str, Any]] = []

    def _collect(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key in ("cross_episode_links", "links", "narrative_links"):
            if key in node and isinstance(node[key], list):
                raw_links.extend(item for item in node[key] if isinstance(item, dict))
        for key in ("candidate_proposals", "candidates", "proposals"):
            if key in node and isinstance(node[key], list):
                raw_proposals.extend(item for item in node[key] if isinstance(item, dict))
        for key in ("supporting_character_arcs", "supporting_arcs", "character_arcs"):
            if key in node and isinstance(node[key], list):
                raw_arcs.extend(item for item in node[key] if isinstance(item, dict))
        for key in ("rejected_or_merged", "rejected"):
            if key in node and isinstance(node[key], list):
                raw_rejected.extend(item for item in node[key] if isinstance(item, dict))
        if "batch_results" in node and isinstance(node["batch_results"], list):
            for sub in node["batch_results"]:
                _collect(sub)

    _collect(raw)

    seen_pids: set[str] = set()
    proposals: list[dict[str, Any]] = []
    for p in raw_proposals:
        pid = str(p.get("proposal_id", "")).strip()
        title = str(p.get("title", "")).strip()
        sig = pid or title.lower()
        if not sig or sig in seen_pids:
            continue
        seen_pids.add(sig)
        proposals.append(p)

    seen_links: set[str] = set()
    links: list[dict[str, Any]] = []
    for l in raw_links:
        tid = str(l.get("thread_id", "")).strip()
        theme = str(l.get("theme", "")).strip().lower()
        eps = "-".join(sorted(str(e) for e in l.get("episodes", [])))
        sig = tid or f"{theme}:{eps}"
        if not sig or sig in seen_links:
            continue
        seen_links.add(sig)
        links.append(l)

    seen_chars: set[str] = set()
    arcs: list[dict[str, Any]] = []
    for a in raw_arcs:
        c = str(a.get("character", a.get("name", ""))).strip().lower()
        if not c or c in seen_chars:
            continue
        seen_chars.add(c)
        arcs.append(a)

    seen_rej: set[str] = set()
    rejected: list[dict[str, Any]] = []
    for r in raw_rejected:
        rid = str(r.get("proposal_id", r.get("thread_id", r.get("title", "")))).strip()
        sig = rid or json.dumps(r, sort_keys=True)
        if not sig or sig in seen_rej:
            continue
        seen_rej.add(sig)
        rejected.append(r)

    res: dict[str, Any] = {
        "cross_episode_links": links,
        "candidate_proposals": proposals,
        "supporting_character_arcs": arcs,
        "rejected_or_merged": rejected,
    }
    if "node_id" in raw:
        res["node_id"] = raw["node_id"]
    if "batch_id" in raw:
        res["batch_id"] = raw["batch_id"]
    return res


def compact_merge_result(raw: dict[str, Any], level: CompactionLevel) -> dict[str, Any]:
    norm = normalize_and_dedup_merge_result(raw)
    if level == CompactionLevel.FULL:
        return norm

    props = norm.get("candidate_proposals", [])
    links = norm.get("cross_episode_links", [])
    arcs = norm.get("supporting_character_arcs", [])
    rej = norm.get("rejected_or_merged", [])

    if level == CompactionLevel.TRIMMED:
        out_props = []
        for p in props:
            p_copy = dict(p)
            if "editorial_reason" in p_copy:
                p_copy["editorial_reason"] = str(p_copy["editorial_reason"])[:120]
            if "description" in p_copy:
                p_copy["description"] = str(p_copy["description"])[:120]
            out_props.append(p_copy)
        out_links = []
        for l in links:
            l_copy = dict(l)
            if "summary" in l_copy:
                l_copy["summary"] = str(l_copy["summary"])[:120]
            out_links.append(l_copy)
        out_arcs = []
        for a in arcs:
            a_copy = dict(a)
            if "arc_summary" in a_copy:
                a_copy["arc_summary"] = str(a_copy["arc_summary"])[:120]
            out_arcs.append(a_copy)
        return {
            "node_id": norm.get("node_id", ""),
            "batch_id": norm.get("batch_id", ""),
            "cross_episode_links": out_links,
            "candidate_proposals": out_props,
            "supporting_character_arcs": out_arcs,
            "rejected_or_merged": rej[:10],
        }

    if level == CompactionLevel.PRIORITY:
        out_props = []
        for p in props:
            p_copy = dict(p)
            if "editorial_reason" in p_copy:
                p_copy["editorial_reason"] = str(p_copy["editorial_reason"])[:80]
            if "description" in p_copy:
                p_copy["description"] = str(p_copy["description"])[:80]
            out_props.append(p_copy)
        out_links = []
        for l in links:
            l_copy = dict(l)
            if "summary" in l_copy:
                l_copy["summary"] = str(l_copy["summary"])[:80]
            out_links.append(l_copy)
        out_arcs = []
        for a in arcs:
            a_copy = dict(a)
            if "arc_summary" in a_copy:
                a_copy["arc_summary"] = str(a_copy["arc_summary"])[:80]
            out_arcs.append(a_copy)
        return {
            "node_id": norm.get("node_id", ""),
            "batch_id": norm.get("batch_id", ""),
            "cross_episode_links": out_links,
            "candidate_proposals": out_props,
            "supporting_character_arcs": out_arcs,
            "rejected_or_merged": [],
        }

    # SKELETON
    out_props = []
    for p in props:
        prop_item = {
            "proposal_id": p.get("proposal_id", ""),
            "title": str(p.get("title", ""))[:120],
            "candidate_scope": p.get("candidate_scope", "CROSS_EPISODE"),
            "episodes": p.get("episodes", []),
            "characters": [str(c)[:80] for c in p.get("characters", [])[:5]],
            "editorial_reason": str(p.get("editorial_reason", ""))[:80] if p.get("editorial_reason") else "",
            "status": p.get("status", "keep"),
        }
        if "description" in p:
            prop_item["description"] = str(p.get("description", ""))[:80]
        out_props.append(prop_item)
    out_links = []
    for l in links:
        out_links.append({
            "thread_id": l.get("thread_id", ""),
            "theme": str(l.get("theme", ""))[:50],
            "episodes": l.get("episodes", []),
            "summary": str(l.get("summary", ""))[:80] if l.get("summary") else "",
        })
    out_arcs = []
    for a in arcs:
        out_arcs.append({
            "character": str(a.get("character", a.get("name", "")))[:80],
            "episodes": a.get("episodes", []),
            "arc_summary": str(a.get("arc_summary", ""))[:80] if a.get("arc_summary") else "",
            "has_dedicated_candidate": a.get("has_dedicated_candidate", True),
        })
    return {
        "node_id": norm.get("node_id", ""),
        "batch_id": norm.get("batch_id", ""),
        "cross_episode_links": out_links,
        "candidate_proposals": out_props,
        "supporting_character_arcs": out_arcs,
        "rejected_or_merged": [],
    }


def split_merge_result(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a merge result into two halves if oversized."""
    props = list(raw.get("candidate_proposals", []))
    links = list(raw.get("cross_episode_links", []))
    arcs = list(raw.get("supporting_character_arcs", []))

    if len(props) <= 1 and len(links) <= 1:
        return [raw]

    mid_p = max(1, len(props) // 2) if len(props) > 1 else len(props)
    mid_l = max(1, len(links) // 2) if len(links) > 1 else len(links)

    part1 = {
        "node_id": raw.get("node_id", ""),
        "merge_id": raw.get("merge_id", ""),
        "cross_episode_links": links[:mid_l],
        "candidate_proposals": props[:mid_p],
        "supporting_character_arcs": arcs,
        "rejected_or_merged": [],
    }
    part2 = {
        "node_id": raw.get("node_id", ""),
        "merge_id": raw.get("merge_id", ""),
        "cross_episode_links": links[mid_l:],
        "candidate_proposals": props[mid_p:],
        "supporting_character_arcs": arcs,
        "rejected_or_merged": [],
    }
    return [part1, part2]


def deterministic_cap_merge_item(raw: dict[str, Any], cap: int = 80) -> dict[str, Any]:
    """Deterministically cap string fields in a merge item to prevent indivisible string overflow."""
    norm = normalize_and_dedup_merge_result(raw)
    props = norm.get("candidate_proposals", [])
    links = norm.get("cross_episode_links", [])
    arcs = norm.get("supporting_character_arcs", [])
    rej = norm.get("rejected_or_merged", [])

    out_props = []
    for p in props:
        p_copy = dict(p)
        if "title" in p_copy:
            p_copy["title"] = str(p_copy["title"])[:cap]
        if "editorial_reason" in p_copy:
            p_copy["editorial_reason"] = str(p_copy["editorial_reason"])[:cap]
        if "characters" in p_copy and isinstance(p_copy["characters"], list):
            p_copy["characters"] = [str(c)[:cap] for c in p_copy["characters"][:5]]
        out_props.append(p_copy)

    out_links = []
    for l in links:
        l_copy = dict(l)
        if "theme" in l_copy:
            l_copy["theme"] = str(l_copy["theme"])[:cap]
        if "summary" in l_copy:
            l_copy["summary"] = str(l_copy["summary"])[:cap]
        out_links.append(l_copy)

    out_arcs = []
    for a in arcs:
        a_copy = dict(a)
        if "character" in a_copy:
            a_copy["character"] = str(a_copy["character"])[:cap]
        if "name" in a_copy:
            a_copy["name"] = str(a_copy["name"])[:cap]
        if "arc_summary" in a_copy:
            a_copy["arc_summary"] = str(a_copy["arc_summary"])[:cap]
        out_arcs.append(a_copy)

    return {
        "node_id": norm.get("node_id", ""),
        "batch_id": norm.get("batch_id", ""),
        "cross_episode_links": out_links,
        "candidate_proposals": out_props,
        "supporting_character_arcs": out_arcs,
        "rejected_or_merged": rej[:5] if cap > 40 else [],
    }


def _highest_compaction_level(levels: list[CompactionLevel]) -> CompactionLevel:
    order = [CompactionLevel.FULL, CompactionLevel.TRIMMED, CompactionLevel.PRIORITY, CompactionLevel.SKELETON]
    max_idx = max((order.index(lvl) for lvl in levels), default=0)
    return order[max_idx]


def _recursively_split_summary_to_fit(
    summary: CompactEpisodeSummary,
    test_fit_fn: Callable[[CompactEpisodeSummary], bool],
    base_check_fn: Callable[[], None],
) -> list[CompactEpisodeSummary]:
    """Recursively split a summary by timeline into fragments until each fits."""
    if test_fit_fn(summary):
        return [summary]

    if len(summary.items) <= 1:
        base_check_fn()
        raise AnalysisError(
            f"Atomic evidence item in episode {summary.episode_id} exceeds payload ceiling even at skeleton cap."
        )

    parts = split_summary_by_timeline(summary)
    result: list[CompactEpisodeSummary] = []
    for part in parts:
        result.extend(_recursively_split_summary_to_fit(part, test_fit_fn, base_check_fn))
    return result


def plan_adaptive_batches(
    ordered_summaries: list[CompactEpisodeSummary],
    model: str,
    thinking: str = "auto",
    recap_prompt: str = "",
    coverage_notice: str = "",
    target_ceiling: int = TARGET_PAYLOAD_CEILING,
    hard_ceiling: int = HARD_PAYLOAD_CEILING,
    max_items_hint: int = MAX_ITEMS_PER_BATCH_HINT,
) -> list[AdaptiveBatchPlanItem]:
    """Plan adaptive batches greedily bounded by request body size <= target_ceiling."""
    if not ordered_summaries:
        return []

    # 1. Baseline envelope check
    empty_text = format_batch_user_text("node_L0_baseline", [], coverage_notice, recap_prompt)
    empty_size = estimate_batch_request_size(model, empty_text, thinking)
    if empty_size >= target_ceiling:
        raise AnalysisError(
            f"Baseline batch prompt envelope size {empty_size} bytes exceeds target limit of {target_ceiling} bytes."
        )

    def _base_check() -> None:
        if empty_size >= target_ceiling:
            raise AnalysisError("Baseline envelope exceeds limit.")

    # 2. Pre-fit each summary into units that fit alone <= target_ceiling
    units: list[tuple[int, int, CompactEpisodeSummary, CompactionLevel]] = []

    for ord_idx, summ in enumerate(ordered_summaries):
        def _fits(s: CompactEpisodeSummary) -> bool:
            txt = format_batch_user_text(f"node_L0_{ord_idx}_{ord_idx}_test", [s], coverage_notice, recap_prompt)
            return estimate_batch_request_size(model, txt, thinking) <= target_ceiling

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
            frags = _recursively_split_summary_to_fit(skeleton_s, _fits, _base_check)
            for frag in frags:
                units.append((ord_idx, ord_idx, frag, CompactionLevel.SKELETON))

    # 3. Left-greedy batch packing
    batch_plans: list[AdaptiveBatchPlanItem] = []
    current_group: list[tuple[int, int, CompactEpisodeSummary, CompactionLevel]] = []

    def _build_plan(group: list[tuple[int, int, CompactEpisodeSummary, CompactionLevel]]) -> AdaptiveBatchPlanItem:
        span_start = group[0][0]
        span_end = group[-1][1]
        group_summs = [u[2] for u in group]
        compaction_level = _highest_compaction_level([u[3] for u in group])

        content_raw = json.dumps([s.to_dict() for s in group_summs], sort_keys=True)
        content_hash = hashlib.sha256(content_raw.encode("utf-8")).hexdigest()

        frag_label = ""
        for s in group_summs:
            if s.fragment_id:
                frag_label = f"f{s.fragment_index}"
                break

        node_id = format_node_id("L0", span_start, span_end, content_hash, frag_label)
        user_text = format_batch_user_text(node_id, group_summs, coverage_notice, recap_prompt)
        est_size = estimate_batch_request_size(model, user_text, thinking)

        if est_size > hard_ceiling:
            raise AnalysisError(
                f"Không thể tạo request batch an toàn cho node {node_id}: "
                f"{est_size} bytes vượt giới hạn {hard_ceiling} bytes sau mọi bước compact/split."
            )

        child_hashes = [s.canonical_hash() for s in group_summs]
        cache_key = compute_batch_cache_key(
            node_id=node_id,
            ordered_summary_hashes=child_hashes,
            model=model,
            thinking=thinking,
            recap_prompt=recap_prompt,
            algo=HIERARCHY_ALGO_VERSION,
            compaction_level=compaction_level.value,
        )
        return AdaptiveBatchPlanItem(
            node_id=node_id,
            span_start=span_start,
            span_end=span_end,
            summaries=group_summs,
            compaction_level=compaction_level,
            user_text=user_text,
            cache_key=cache_key,
            estimated_bytes=est_size,
        )

    for unit in units:
        if not current_group:
            current_group.append(unit)
        else:
            if len(current_group) < max_items_hint:
                test_group = current_group + [unit]
                test_summs = [u[2] for u in test_group]
                test_text = format_batch_user_text("test_node", test_summs, coverage_notice, recap_prompt)
                if estimate_batch_request_size(model, test_text, thinking) <= target_ceiling:
                    current_group.append(unit)
                    continue
            batch_plans.append(_build_plan(current_group))
            current_group = [unit]

    if current_group:
        batch_plans.append(_build_plan(current_group))

    return batch_plans


def partition_season_batches(items: list[Any]) -> list[tuple[str, list[Any]]]:
    """Deterministically partition items into batches of max 4, typical 3-4.

    Rules:
    - N <= 0: []
    - N == 1: 1 batch of size 1 ("batch_{id}_{id}")
    - N == 2: 1 batch of size 2 ("batch_{start}_{end}")
    - N == 3: 1 batch of size 3 ("batch_{start}_{end}")
    - N == 4: 1 batch of size 4 ("batch_{start}_{end}")
    - N == 5: 2 batches of sizes [3, 2] (unavoidable)
    - N >= 6: all batches have size 3 or 4 when mathematically possible.
    """
    n = len(items)
    if n == 0:
        return []

    def _get_id(x: Any) -> str:
        if hasattr(x, "episode_id"):
            return str(x.episode_id)
        if isinstance(x, dict) and "batch_id" in x:
            return str(x["batch_id"])
        if isinstance(x, dict) and "episode_id" in x:
            return str(x["episode_id"])
        return str(x)

    if n == 1:
        single_id = _get_id(items[0])
        return [(f"batch_{single_id}_{single_id}", list(items))]
    if n == 2:
        return [(f"batch_{_get_id(items[0])}_{_get_id(items[1])}", list(items))]
    if n == 3:
        return [(f"batch_{_get_id(items[0])}_{_get_id(items[2])}", list(items))]
    if n == 4:
        return [(f"batch_{_get_id(items[0])}_{_get_id(items[3])}", list(items))]
    if n == 5:
        sizes = [3, 2]
    else:
        k = n // 4
        rem = n % 4
        if rem == 0:
            sizes = [4] * k
        elif rem == 1:
            sizes = [4] * (k - 2) + [3] * 3
        elif rem == 2:
            sizes = [4] * (k - 1) + [3] * 2
        else:
            sizes = [4] * k + [3]

    batches: list[tuple[str, list[Any]]] = []
    idx = 0
    for s in sizes:
        batch_slice = items[idx : idx + s]
        idx += s
        start_id = _get_id(batch_slice[0])
        end_id = _get_id(batch_slice[-1])
        batch_id = f"batch_{start_id}_{end_id}"
        batches.append((batch_id, batch_slice))
    return batches


@dataclass
class CandidateProposal:
    proposal_id: str
    title: str
    candidate_scope: str = CandidateScope.CROSS_EPISODE.value
    episodes: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    description: str = ""
    editorial_reason: str = ""
    status: str = "keep"  # keep, reject, merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "candidate_scope": self.candidate_scope,
            "episodes": self.episodes,
            "characters": self.characters,
            "description": self.description,
            "editorial_reason": self.editorial_reason,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateProposal":
        return cls(
            proposal_id=str(data.get("proposal_id", "")),
            title=str(data.get("title", "")),
            candidate_scope=str(data.get("candidate_scope", CandidateScope.CROSS_EPISODE.value)),
            episodes=list(data.get("episodes", [])),
            characters=list(data.get("characters", [])),
            description=str(data.get("description", "")),
            editorial_reason=str(data.get("editorial_reason", "")),
            status=str(data.get("status", "keep")),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeasonConnectionResult":
        props = [
            CandidateProposal.from_dict(p) if isinstance(p, dict) else p
            for p in data.get("candidate_proposals", [])
        ]
        return cls(
            cross_episode_links=list(data.get("cross_episode_links", [])),
            candidate_proposals=props,
            supporting_character_arcs=list(data.get("supporting_character_arcs", [])),
            rejected_or_merged=list(data.get("rejected_or_merged", [])),
            is_complete=bool(data.get("is_complete", True)),
            missing_episodes=list(data.get("missing_episodes", [])),
        )


def compact_connection_result(
    conn: SeasonConnectionResult,
    level: CompactionLevel,
) -> SeasonConnectionResult:
    """Deterministically compact SeasonConnectionResult to FULL, TRIMMED, PRIORITY, or SKELETON."""
    norm = compact_merge_result(conn.to_dict(), level)
    props = [
        CandidateProposal.from_dict(p) if isinstance(p, dict) else p
        for p in norm.get("candidate_proposals", [])
    ]
    return SeasonConnectionResult(
        cross_episode_links=list(norm.get("cross_episode_links", [])),
        candidate_proposals=props,
        supporting_character_arcs=list(norm.get("supporting_character_arcs", [])),
        rejected_or_merged=list(norm.get("rejected_or_merged", [])),
        is_complete=conn.is_complete,
        missing_episodes=list(conn.missing_episodes),
    )


class SeasonConnector:
    """Performs hierarchical season-wide connection pass across all episode evidence."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        client: OpenAICompatibleClient | None = None,
        hierarchy_cache: HierarchyCacheManager | None = None,
        *,
        target_ceiling: int | None = None,
        hard_ceiling: int | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.client = client
        self.hierarchy_cache = hierarchy_cache or HierarchyCacheManager()
        self._target_ceiling = target_ceiling
        self._hard_ceiling = hard_ceiling

    @property
    def target_ceiling(self) -> int:
        return self._target_ceiling if self._target_ceiling is not None else TARGET_PAYLOAD_CEILING

    @property
    def hard_ceiling(self) -> int:
        return self._hard_ceiling if self._hard_ceiling is not None else HARD_PAYLOAD_CEILING

    def compute_connection_key(
        self,
        episodes: list[SourceEpisode],
        evidence_map: dict[str, EpisodeEvidence],
    ) -> str:
        """Deterministic cache key for season connection identity (algo v3)."""
        available = [ep for ep in episodes if ep.episode_id in evidence_map and evidence_map[ep.episode_id] is not None]
        sorted_eps = sorted(available, key=lambda x: x.episode_id)
        hashes: list[str] = []
        for ep in sorted_eps:
            ev = evidence_map[ep.episode_id]
            h = hashlib.sha256(json.dumps(ev.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:16]
            hashes.append(f"{h}")
        return compute_connection_cache_key(
            ordered_evidence_hashes=hashes,
            model=self.settings.finalizer_model,
            thinking=self.settings.finalizer_thinking,
            recap_prompt=self.settings.recap_prompt or "",
            algo=HIERARCHY_ALGO_VERSION,
        )

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
        """Hierarchically analyze connections across all episode evidence records."""
        if _is_cancelled(cancel_event):
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

        is_gateway_enabled = (
            self.settings.gateway_enabled
            and bool(self.settings.api_endpoint.strip())
            and self.settings.api_endpoint.strip().lower() != "offline"
            and self.client is not None
        )

        # 2. Check full connection cache first (fast skip if connection pass previously completed)
        conn_key = self.compute_connection_key(episodes, evidence_map)
        if is_gateway_enabled:
            cached_conn, conn_meta = self.hierarchy_cache.load_connection_result(conn_key)
            if conn_meta["hit"] and cached_conn is not None:
                avail_count = len([ep for ep in episodes if ep.episode_id in evidence_map and evidence_map[ep.episode_id] is not None])
                if log:
                    log(f"[Season Connection] cache=hit episodes={avail_count}")
                    log("Sử dụng kết quả liên kết mùa phim đã lưu trong bộ nhớ đệm (connection cache hit).")
                if on_phase:
                    on_phase(
                        AnalysisPhase.SEASON_CONNECTING,
                        "season",
                        {
                            "status": "cached",
                            "status_message": "Sử dụng liên kết mùa từ bộ nhớ đệm (cache hit)",
                            "cache_hit": True,
                            "stage": "complete",
                            "progress": 75,
                        },
                    )
                    on_phase(
                        AnalysisPhase.SEASON_MERGING,
                        "season_complete",
                        {
                            "status": "complete",
                            "status_message": "Hợp nhất liên kết mùa hoàn tất (từ cache)",
                            "cache_hit": True,
                            "stage": "complete",
                            "progress": 75,
                        },
                    )
                return self._parse_connection_result(
                    cached_conn,
                    is_complete=len(missing_episodes) == 0,
                    missing_episodes=missing_episodes,
                )

        # 3. Build or load compact summaries deterministically
        available_episodes = [
            ep for ep in episodes if ep.episode_id in evidence_map and evidence_map[ep.episode_id] is not None
        ]

        compact_summaries: dict[str, CompactEpisodeSummary] = {}
        for ep in available_episodes:
            if _is_cancelled(cancel_event):
                raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

            ev = evidence_map[ep.episode_id]
            ev_canonical = json.dumps(ev.to_dict(), sort_keys=True)
            ev_hash = hashlib.sha256(ev_canonical.encode("utf-8")).hexdigest()

            cached_summary, s_meta = self.hierarchy_cache.load_summary(ep.episode_id, ev_hash)
            if s_meta["hit"] and cached_summary is not None:
                summary = cached_summary
                hit = True
            else:
                summary = build_compact_summary(ep, ev)
                self.hierarchy_cache.save_summary(summary, ev_hash)
                hit = False

            compact_summaries[ep.episode_id] = summary

            if on_phase:
                on_phase(
                    AnalysisPhase.EPISODE_SUMMARIZING,
                    ep.episode_id,
                    {
                        "episode_id": ep.episode_id,
                        "cache_hit": hit,
                        "item_count": len(summary.items),
                    },
                )

        if not is_gateway_enabled or self.client is None:
            # Deterministic offline season connection
            return self._connect_offline(
                episodes=episodes,
                evidence_map=evidence_map,
                is_complete=len(missing_episodes) == 0,
                missing_episodes=missing_episodes,
            )

        coverage_notice = ""
        if missing_episodes:
            coverage_notice = (
                f"\nCRITICAL COVERAGE NOTICE:\n"
                f"The following episodes are MISSING or FAILED: {missing_episodes}.\n"
                f"This is a PARTIAL season analysis. NEVER claim, imply, or hallucinate full season coverage.\n"
                f"Restrict all narrative links and candidate proposals strictly to available episodes: "
                f"{[ep.episode_id for ep in available_episodes]}.\n"
            )

        # 4. Plan adaptive batches
        batch_plans = plan_adaptive_batches(
            ordered_summaries=[compact_summaries[ep.episode_id] for ep in available_episodes],
            model=self.settings.finalizer_model,
            thinking=self.settings.finalizer_thinking,
            recap_prompt=self.settings.recap_prompt or "",
            coverage_notice=coverage_notice,
            target_ceiling=self.target_ceiling,
            hard_ceiling=self.hard_ceiling,
            max_items_hint=MAX_ITEMS_PER_BATCH_HINT,
        )
        total_batches = len(batch_plans)

        # 5. Process each batch plan
        batch_results: list[dict[str, Any]] = []

        for batch_idx, plan in enumerate(batch_plans, start=1):
            if _is_cancelled(cancel_event):
                raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

            cached_batch, b_meta = self.hierarchy_cache.load_batch_result(plan.node_id, plan.cache_key)
            if b_meta["hit"] and cached_batch is not None:
                if log:
                    log(f"Batch {plan.node_id} đã có trong bộ nhớ đệm (cache hit).")
                if on_phase:
                    on_phase(
                        AnalysisPhase.SEASON_BATCH,
                        plan.node_id,
                        {
                            "batch_index": batch_idx,
                            "total_batches": total_batches,
                            "batch_id": plan.node_id,
                            "node_id": plan.node_id,
                            "cache_hit": True,
                        },
                    )
                batch_results.append(cached_batch)
                continue

            if on_phase:
                on_phase(
                    AnalysisPhase.SEASON_BATCH,
                    plan.node_id,
                    {
                        "batch_index": batch_idx,
                        "total_batches": total_batches,
                        "batch_id": plan.node_id,
                        "node_id": plan.node_id,
                        "cache_hit": False,
                    },
                )

            # Invariant assertion
            est_size = estimate_batch_request_size(
                model=self.settings.finalizer_model,
                user_text=plan.user_text,
                thinking=self.settings.finalizer_thinking,
            )
            if est_size > self.hard_ceiling:
                raise AnalysisError(
                    f"Planner tạo batch {plan.node_id} vượt giới hạn an toàn: "
                    f"{est_size}/{self.hard_ceiling} bytes."
                )

            if log:
                log(
                    f"[Season Connection] phase=season_batch batch_id={plan.node_id} "
                    f"model={self.settings.finalizer_model} cache=miss "
                    f"count={len(plan.summaries)} payload_bytes={est_size}"
                )

            def _batch_status_cb(msg: str) -> None:
                if on_phase:
                    on_phase(
                        AnalysisPhase.SEASON_BATCH,
                        plan.node_id,
                        {"status": msg, "status_message": msg, "batch_id": plan.node_id, "node_id": plan.node_id},
                    )

            call_kwargs: dict[str, Any] = {
                "model": self.settings.finalizer_model,
                "thinking": self.settings.finalizer_thinking,
                "system": SEASON_BATCH_SYSTEM_PROMPT,
                "user_text": plan.user_text,
                "cancel_event": cancel_event,
                "phase": "season_batch",
                "timeout": SEASON_CONNECTION_TIMEOUT,
                "on_status": _batch_status_cb,
                "log": log,
            }

            try:
                try:
                    raw_batch = self.client.chat_json(**call_kwargs)
                except TypeError as te:
                    if "unexpected keyword argument" in str(te):
                        filtered = {k: v for k, v in call_kwargs.items() if k not in ("phase", "timeout", "on_status", "log")}
                        raw_batch = self.client.chat_json(**filtered)
                    else:
                        raise
            except AnalysisCancelledError:
                raise
            except APIError as exc:
                if _is_cancelled(cancel_event) or "đã bị dừng" in str(exc) or "bị hủy" in str(exc):
                    raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.") from exc
                raise AnalysisError(
                    f"AI Gateway season batch {plan.node_id} lỗi sau các lần thử: {exc}. "
                    f"Các batch và compact summaries trước đó đã được lưu an toàn trong bộ nhớ đệm."
                ) from exc

            # Requirement 7: Validate + Save successful batch BEFORE cancellation check!
            if not isinstance(raw_batch, dict):
                raise AnalysisError(f"Batch {plan.node_id} returned invalid non-dict response.")

            raw_batch["node_id"] = plan.node_id
            raw_batch["batch_id"] = plan.node_id
            validate_batch_response_schema(raw_batch, context=f"batch {plan.node_id}")
            self.hierarchy_cache.save_batch_result(
                plan.node_id,
                plan.cache_key,
                raw_batch,
                algo=HIERARCHY_ALGO_VERSION,
                compaction_level=plan.compaction_level.value,
            )
            batch_results.append(raw_batch)

            if _is_cancelled(cancel_event):
                raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

        # 6. Hierarchical cross-batch merge
        if len(batch_results) == 1:
            final_raw = batch_results[0]
        else:
            final_raw = self._hierarchical_merge(
                batch_results=batch_results,
                coverage_notice=coverage_notice,
                cancel_event=cancel_event,
                on_phase=on_phase,
                log=log,
            )

        # Save connection result cache after schema validation
        validate_batch_response_schema(final_raw, context="connection")
        self.hierarchy_cache.save_connection_result(conn_key, final_raw, algo=HIERARCHY_ALGO_VERSION)

        if on_phase:
            on_phase(
                AnalysisPhase.SEASON_MERGING,
                "season_complete",
                {
                    "status": "complete",
                    "status_message": "Hợp nhất liên kết mùa hoàn tất",
                    "cache_hit": False,
                    "stage": "complete",
                    "progress": 75,
                },
            )

        return self._parse_connection_result(
            final_raw,
            is_complete=len(missing_episodes) == 0,
            missing_episodes=missing_episodes,
        )

    def _hierarchical_merge(
        self,
        batch_results: list[dict[str, Any]],
        coverage_notice: str,
        cancel_event: threading.Event | None,
        on_phase: PhaseCallback | None,
        log: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        """Merge batch results hierarchically using adaptive byte packing until 1 unified result remains."""
        empty_text = format_merge_user_text("node_L1_baseline", [], coverage_notice, self.settings.recap_prompt or "")
        empty_size = estimate_merge_request_size(self.settings.finalizer_model, empty_text, self.settings.finalizer_thinking)
        if empty_size >= self.target_ceiling:
            raise AnalysisError(
                f"Baseline merge prompt envelope size {empty_size} bytes exceeds target limit of {self.target_ceiling} bytes."
            )

        current_level = list(batch_results)
        round_idx = 1

        while len(current_level) > 1:
            if _is_cancelled(cancel_event):
                raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

            normalized_level = [normalize_and_dedup_merge_result(item) for item in current_level]

            fitted_items: list[dict[str, Any]] = []
            for item in normalized_level:
                fitted_items.extend(self._ensure_merge_item_fits_alone(item, coverage_notice))

            groups = self._group_merge_items(fitted_items, coverage_notice)

            # Avoid infinite loop when all groups are singletons
            if len(groups) == len(fitted_items) and len(fitted_items) > 1:
                groups = self._force_merge_pairs(fitted_items, coverage_notice)

            next_level: list[dict[str, Any]] = []
            for g_num, group in enumerate(groups, start=1):
                if _is_cancelled(cancel_event):
                    raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

                group_raw = json.dumps(group, sort_keys=True)
                content_hash = hashlib.sha256(group_raw.encode("utf-8")).hexdigest()
                node_id = format_node_id(f"L{round_idx}", g_num - 1, g_num - 1 + len(group) - 1, content_hash)

                merged_item = self._merge_group(
                    group=group,
                    merge_id=node_id,
                    coverage_notice=coverage_notice,
                    cancel_event=cancel_event,
                    on_phase=on_phase,
                    log=log,
                )
                next_level.append(merged_item)

            current_level = next_level
            round_idx += 1

        return current_level[0]

    def _ensure_merge_item_fits_alone(
        self,
        item: dict[str, Any],
        coverage_notice: str,
    ) -> list[dict[str, Any]]:
        ceiling = self.target_ceiling
        txt = format_merge_user_text("test_node", [item], coverage_notice, self.settings.recap_prompt or "")
        est = estimate_merge_request_size(self.settings.finalizer_model, txt, self.settings.finalizer_thinking)
        if est <= ceiling:
            return [item]

        for lvl in (CompactionLevel.TRIMMED, CompactionLevel.PRIORITY, CompactionLevel.SKELETON):
            compacted = compact_merge_result(item, lvl)
            txt = format_merge_user_text("test_node", [compacted], coverage_notice, self.settings.recap_prompt or "")
            if estimate_merge_request_size(self.settings.finalizer_model, txt, self.settings.finalizer_thinking) <= ceiling:
                return [compacted]

        skeleton_item = compact_merge_result(item, CompactionLevel.SKELETON)
        parts = split_merge_result(skeleton_item)
        if len(parts) > 1:
            result: list[dict[str, Any]] = []
            for p in parts:
                result.extend(self._ensure_merge_item_fits_alone(p, coverage_notice))
            return result

        # Cannot split further: indivisible strings exist. Deterministic cap fields.
        for cap in (80, 40, 20):
            capped = deterministic_cap_merge_item(skeleton_item, cap=cap)
            txt = format_merge_user_text("test_node", [capped], coverage_notice, self.settings.recap_prompt or "")
            if estimate_merge_request_size(self.settings.finalizer_model, txt, self.settings.finalizer_thinking) <= ceiling:
                return [capped]

        node_lbl = item.get("node_id") or item.get("batch_id") or "merge_item"
        raise AnalysisError(
            f"Merge item {node_lbl} exceeds payload ceiling even at skeleton cap."
        )

    def _group_merge_items(
        self,
        items: list[dict[str, Any]],
        coverage_notice: str,
    ) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        current_group: list[dict[str, Any]] = []

        for item in items:
            if not current_group:
                current_group.append(item)
            else:
                if len(current_group) < MAX_ITEMS_PER_BATCH_HINT:
                    test_group = current_group + [item]
                    test_txt = format_merge_user_text("test_group", test_group, coverage_notice, self.settings.recap_prompt or "")
                    if estimate_merge_request_size(self.settings.finalizer_model, test_txt, self.settings.finalizer_thinking) <= self.target_ceiling:
                        current_group.append(item)
                        continue
                groups.append(current_group)
                current_group = [item]

        if current_group:
            groups.append(current_group)
        return groups

    def _force_merge_pairs(
        self,
        items: list[dict[str, Any]],
        coverage_notice: str,
    ) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        i = 0
        while i < len(items):
            if i + 1 < len(items):
                p1 = compact_merge_result(items[i], CompactionLevel.SKELETON)
                p2 = compact_merge_result(items[i + 1], CompactionLevel.SKELETON)
                test_txt = format_merge_user_text("pair_test", [p1, p2], coverage_notice, self.settings.recap_prompt or "")
                if estimate_merge_request_size(self.settings.finalizer_model, test_txt, self.settings.finalizer_thinking) > self.target_ceiling:
                    p1 = deterministic_cap_merge_item(p1, cap=50)
                    p2 = deterministic_cap_merge_item(p2, cap=50)
                groups.append([p1, p2])
                i += 2
            else:
                groups.append([items[i]])
                i += 1
        return groups

    def _merge_group(
        self,
        group: list[dict[str, Any]],
        merge_id: str,
        coverage_notice: str,
        cancel_event: threading.Event | None,
        on_phase: PhaseCallback | None,
        log: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        if _is_cancelled(cancel_event):
            raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

        input_hashes = [
            hashlib.sha256(json.dumps(r, sort_keys=True).encode("utf-8")).hexdigest()
            for r in group
        ]
        merge_key = compute_merge_cache_key(
            node_id=merge_id,
            ordered_batch_hashes=input_hashes,
            model=self.settings.finalizer_model,
            thinking=self.settings.finalizer_thinking,
            recap_prompt=self.settings.recap_prompt or "",
            algo=HIERARCHY_ALGO_VERSION,
            merge_version="v3",
        )

        cached_merge, m_meta = self.hierarchy_cache.load_merge_result(merge_id, merge_key)
        if m_meta["hit"] and cached_merge is not None:
            if log:
                log(f"Merge group {merge_id} đã có trong bộ nhớ đệm (cache hit).")
            if on_phase:
                on_phase(
                    AnalysisPhase.SEASON_MERGING,
                    merge_id,
                    {"merge_id": merge_id, "node_id": merge_id, "cache_hit": True, "count": len(group)},
                )
            return cached_merge

        if on_phase:
            on_phase(
                AnalysisPhase.SEASON_MERGING,
                merge_id,
                {"merge_id": merge_id, "node_id": merge_id, "cache_hit": False, "count": len(group)},
            )

        user_text = format_merge_user_text(
            node_id=merge_id,
            group=group,
            coverage_notice=coverage_notice,
            recap_prompt=self.settings.recap_prompt or "Standard video recap",
        )

        # Invariant assertion
        est_bytes = estimate_merge_request_size(
            model=self.settings.finalizer_model,
            user_text=user_text,
            thinking=self.settings.finalizer_thinking,
        )
        if est_bytes > self.hard_ceiling:
            raise AnalysisError(
                f"Không thể tạo request merge an toàn cho node {merge_id}: "
                f"{est_bytes} bytes vượt giới hạn {self.hard_ceiling} bytes sau mọi bước compact/split."
            )

        if log:
            log(
                f"[Season Connection] phase=season_merging merge_id={merge_id} "
                f"model={self.settings.finalizer_model} cache=miss "
                f"count={len(group)} payload_bytes={est_bytes}"
            )

        def _merge_status_cb(msg: str) -> None:
            if on_phase:
                on_phase(
                    AnalysisPhase.SEASON_MERGING,
                    merge_id,
                    {"status": msg, "status_message": msg, "merge_id": merge_id, "node_id": merge_id},
                )

        call_kwargs: dict[str, Any] = {
            "model": self.settings.finalizer_model,
            "thinking": self.settings.finalizer_thinking,
            "system": SEASON_MERGE_SYSTEM_PROMPT,
            "user_text": user_text,
            "cancel_event": cancel_event,
            "phase": "season_merging",
            "timeout": SEASON_CONNECTION_TIMEOUT,
            "on_status": _merge_status_cb,
            "log": log,
        }

        try:
            try:
                raw_merge = self.client.chat_json(**call_kwargs)  # type: ignore[union-attr]
            except TypeError as te:
                if "unexpected keyword argument" in str(te):
                    filtered = {k: v for k, v in call_kwargs.items() if k not in ("phase", "timeout", "on_status", "log")}
                    raw_merge = self.client.chat_json(**filtered)  # type: ignore[union-attr]
                else:
                    raise
        except AnalysisCancelledError:
            raise
        except APIError as exc:
            if _is_cancelled(cancel_event) or "đã bị dừng" in str(exc) or "bị hủy" in str(exc):
                raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.") from exc
            raise AnalysisError(
                f"AI Gateway cross-batch merge {merge_id} lỗi sau các lần thử: {exc}. "
                f"Dữ liệu batch trước đó đã được lưu an toàn trong bộ nhớ đệm."
            ) from exc

        # Requirement 7: Validate + Save successful merge BEFORE cancellation check!
        if not isinstance(raw_merge, dict):
            raise AnalysisError(f"Merge {merge_id} returned invalid non-dict response.")

        raw_merge["node_id"] = merge_id
        raw_merge["merge_id"] = merge_id
        validate_batch_response_schema(raw_merge, context=f"merge {merge_id}")
        self.hierarchy_cache.save_merge_result(merge_id, merge_key, raw_merge, algo=HIERARCHY_ALGO_VERSION)

        if _is_cancelled(cancel_event):
            raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

        return raw_merge

    def _parse_connection_result(
        self,
        raw: dict[str, Any],
        is_complete: bool,
        missing_episodes: list[str],
    ) -> SeasonConnectionResult:
        """Parse AI response into SeasonConnectionResult, recursively collecting from aliases and merge structures."""
        raw_links: list[dict[str, Any]] = []
        raw_proposals: list[dict[str, Any]] = []
        raw_arcs: list[dict[str, Any]] = []
        raw_rejected: list[dict[str, Any]] = []

        def _collect(node: Any) -> None:
            if not isinstance(node, dict):
                return

            # Aliases for links
            for key in ("cross_episode_links", "links", "narrative_links"):
                if key in node and isinstance(node[key], list):
                    raw_links.extend(item for item in node[key] if isinstance(item, dict))

            # Aliases for candidate proposals
            for key in ("candidate_proposals", "candidates", "proposals"):
                if key in node and isinstance(node[key], list):
                    raw_proposals.extend(item for item in node[key] if isinstance(item, dict))

            # Aliases for supporting character arcs
            for key in ("supporting_character_arcs", "supporting_arcs", "character_arcs"):
                if key in node and isinstance(node[key], list):
                    raw_arcs.extend(item for item in node[key] if isinstance(item, dict))

            # Aliases for rejected or merged
            for key in ("rejected_or_merged", "rejected"):
                if key in node and isinstance(node[key], list):
                    raw_rejected.extend(item for item in node[key] if isinstance(item, dict))

            # Recursively collect from nested batch_results
            if "batch_results" in node and isinstance(node["batch_results"], list):
                for sub in node["batch_results"]:
                    _collect(sub)

        _collect(raw)

        # Deduplicate proposals by proposal_id and title
        seen_proposal_ids: set[str] = set()
        seen_proposal_titles: set[str] = set()
        proposals: list[CandidateProposal] = []
        for i, p in enumerate(raw_proposals, start=1):
            pid = str(p.get("proposal_id", f"prop_{i:02d}"))
            title = str(p.get("title", f"Candidate {i}")).strip()
            if pid in seen_proposal_ids or (title and title.lower() in seen_proposal_titles):
                continue
            seen_proposal_ids.add(pid)
            if title:
                seen_proposal_titles.add(title.lower())

            scope_val = str(p.get("candidate_scope", CandidateScope.CROSS_EPISODE.value))
            if scope_val not in {s.value for s in CandidateScope}:
                scope_val = CandidateScope.CROSS_EPISODE.value

            proposals.append(
                CandidateProposal(
                    proposal_id=pid,
                    title=title or f"Candidate {i}",
                    candidate_scope=scope_val,
                    episodes=[str(e) for e in p.get("episodes", []) if isinstance(e, str)],
                    characters=[str(c) for c in p.get("characters", []) if isinstance(c, str)],
                    editorial_reason=str(p.get("editorial_reason", "")),
                    status=str(p.get("status", "keep")),
                )
            )

        # Deduplicate links by thread_id or (theme + sorted episodes)
        seen_links: set[str] = set()
        links: list[dict[str, Any]] = []
        for l in raw_links:
            tid = str(l.get("thread_id", "")).strip()
            theme = str(l.get("theme", "")).strip().lower()
            eps = "-".join(sorted(str(e) for e in l.get("episodes", [])))
            key = tid if tid else f"{theme}:{eps}"
            if key and key in seen_links:
                continue
            if key:
                seen_links.add(key)
            links.append(l)

        # Deduplicate supporting arcs by character name
        seen_characters: set[str] = set()
        arcs: list[dict[str, Any]] = []
        for a in raw_arcs:
            char = str(a.get("character", a.get("name", ""))).strip().lower()
            if char and char in seen_characters:
                continue
            if char:
                seen_characters.add(char)
            arcs.append(a)

        # Deduplicate rejected_or_merged
        seen_rejected: set[str] = set()
        rejected: list[dict[str, Any]] = []
        for r in raw_rejected:
            rid = str(r.get("proposal_id", r.get("thread_id", r.get("title", "")))).strip()
            sig = rid if rid else json.dumps(r, sort_keys=True)
            if sig in seen_rejected:
                continue
            seen_rejected.add(sig)
            rejected.append(r)

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
