"""Season connection pass: hierarchical cross-episode link analysis, batching, caching, and merge."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from ..api_client import APIError, OpenAICompatibleClient, SEASON_CONNECTION_TIMEOUT, _is_cancelled
from ..domain.cache import (
    HierarchyCacheManager,
    compute_batch_cache_key,
    compute_connection_cache_key,
    compute_merge_cache_key,
    compute_summary_cache_key,
)
from ..domain.enums import CandidateScope
from ..domain.models import (
    CompactEpisodeSummary,
    EpisodeEvidence,
    SourceEpisode,
    build_compact_summary,
)
from ..settings import AppSettings
from .errors import AnalysisCancelledError, AnalysisError, CoverageIncompleteError
from .phases import AnalysisPhase, PhaseCallback
from .prompts import (
    SEASON_BATCH_SYSTEM_PROMPT,
    SEASON_CONNECTION_SYSTEM_PROMPT,
    SEASON_MERGE_SYSTEM_PROMPT,
)

MAX_PAYLOAD_BYTES = 500_000


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
    """Performs hierarchical season-wide connection pass across all episode evidence."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        client: OpenAICompatibleClient | None = None,
        hierarchy_cache: HierarchyCacheManager | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.client = client
        self.hierarchy_cache = hierarchy_cache or HierarchyCacheManager()

    def compute_connection_key(
        self,
        episodes: list[SourceEpisode],
        evidence_map: dict[str, EpisodeEvidence],
    ) -> str:
        """Deterministic cache key for season connection identity."""
        available = [ep for ep in episodes if ep.episode_id in evidence_map and evidence_map[ep.episode_id] is not None]
        sorted_eps = sorted(available, key=lambda x: x.episode_id)
        hashes: list[str] = []
        for ep in sorted_eps:
            ev = evidence_map[ep.episode_id]
            h = hashlib.sha256(json.dumps(ev.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:16]
            hashes.append(f"{ep.episode_id}:{h}")
        return compute_connection_cache_key(
            ordered_evidence_hashes=hashes,
            model=self.settings.finalizer_model,
            thinking=self.settings.finalizer_thinking,
            recap_prompt=self.settings.recap_prompt or "",
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

        # 4. Partition available episodes into deterministic batches
        batches = partition_season_batches(available_episodes)
        total_batches = len(batches)

        if not is_gateway_enabled or self.client is None:
            # Deterministic offline season connection
            return self._connect_offline(
                episodes=episodes,
                evidence_map=evidence_map,
                is_complete=len(missing_episodes) == 0,
                missing_episodes=missing_episodes,
            )

        # 5. Process each batch
        batch_results: list[dict[str, Any]] = []
        coverage_notice = ""
        if missing_episodes:
            coverage_notice = (
                f"\nCRITICAL COVERAGE NOTICE:\n"
                f"The following episodes are MISSING or FAILED: {missing_episodes}.\n"
                f"This is a PARTIAL season analysis. NEVER claim, imply, or hallucinate full season coverage.\n"
                f"Restrict all narrative links and candidate proposals strictly to available episodes: "
                f"{[ep.episode_id for ep in available_episodes]}.\n"
            )

        for batch_idx, (batch_id, batch_eps) in enumerate(batches, start=1):
            if _is_cancelled(cancel_event):
                raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

            batch_summs = [compact_summaries[ep.episode_id] for ep in batch_eps]
            ordered_hashes = [s.canonical_hash() for s in batch_summs]
            batch_key = compute_batch_cache_key(
                batch_id=batch_id,
                ordered_summary_hashes=ordered_hashes,
                model=self.settings.finalizer_model,
                thinking=self.settings.finalizer_thinking,
                recap_prompt=self.settings.recap_prompt or "",
            )

            cached_batch, b_meta = self.hierarchy_cache.load_batch_result(batch_id, batch_key)
            if b_meta["hit"] and cached_batch is not None:
                if log:
                    log(f"Batch {batch_id} đã có trong bộ nhớ đệm (cache hit).")
                if on_phase:
                    on_phase(
                        AnalysisPhase.SEASON_BATCH,
                        batch_id,
                        {
                            "batch_index": batch_idx,
                            "total_batches": total_batches,
                            "batch_id": batch_id,
                            "cache_hit": True,
                        },
                    )
                batch_results.append(cached_batch)
                continue

            if on_phase:
                on_phase(
                    AnalysisPhase.SEASON_BATCH,
                    batch_id,
                    {
                        "batch_index": batch_idx,
                        "total_batches": total_batches,
                        "batch_id": batch_id,
                        "cache_hit": False,
                    },
                )

            batch_payload = {
                "batch_id": batch_id,
                "episodes": [s.to_dict() for s in batch_summs],
            }
            user_text = (
                f"Analyze batch connections for {batch_id} across {len(batch_summs)} episodes.\n"
                f"{coverage_notice}\n"
                f"Recap instructions:\n{self.settings.recap_prompt or 'Standard video recap'}\n\n"
                f"Batch Compact Summaries:\n{json.dumps(batch_payload, ensure_ascii=False, indent=2)}\n"
            )
            check_payload_size(user_text, max_bytes=MAX_PAYLOAD_BYTES, context=f"batch {batch_id}")
            payload_bytes = len(user_text.encode("utf-8"))
            if log:
                log(
                    f"[Season Connection] phase=season_batch batch_id={batch_id} "
                    f"model={self.settings.finalizer_model} cache=miss "
                    f"count={len(batch_summs)} payload_bytes={payload_bytes}"
                )

            def _batch_status_cb(msg: str) -> None:
                if on_phase:
                    on_phase(AnalysisPhase.SEASON_BATCH, batch_id, {"status": msg, "status_message": msg, "batch_id": batch_id})

            call_kwargs: dict[str, Any] = {
                "model": self.settings.finalizer_model,
                "thinking": self.settings.finalizer_thinking,
                "system": SEASON_BATCH_SYSTEM_PROMPT,
                "user_text": user_text,
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
                    f"AI Gateway season batch {batch_id} lỗi sau các lần thử: {exc}. "
                    f"Các batch và compact summaries trước đó đã được lưu an toàn trong bộ nhớ đệm."
                ) from exc

            if _is_cancelled(cancel_event):
                raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

            if not isinstance(raw_batch, dict):
                raise AnalysisError(f"Batch {batch_id} returned invalid non-dict response.")

            raw_batch["batch_id"] = batch_id
            validate_batch_response_schema(raw_batch, context=f"batch {batch_id}")
            self.hierarchy_cache.save_batch_result(batch_id, batch_key, raw_batch)
            batch_results.append(raw_batch)

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
        self.hierarchy_cache.save_connection_result(conn_key, final_raw)

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
        """Merge batch results hierarchically in groups of 3-4 until 1 unified result remains."""
        current_level = list(batch_results)
        round_idx = 1

        while len(current_level) > 1:
            if _is_cancelled(cancel_event):
                raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

            if len(current_level) <= 4:
                merge_id = f"merge_round_{round_idx}_final"
                return self._merge_group(current_level, merge_id, coverage_notice, cancel_event, on_phase, log)

            grouped = partition_season_batches(current_level)
            next_level: list[dict[str, Any]] = []
            for g_num, (g_id, g_items) in enumerate(grouped, start=1):
                if _is_cancelled(cancel_event):
                    raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")
                round_merge_id = f"merge_round_{round_idx}_g{g_num}"
                merged_item = self._merge_group(g_items, round_merge_id, coverage_notice, cancel_event, on_phase, log)
                next_level.append(merged_item)

            current_level = next_level
            round_idx += 1

        return current_level[0]

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
            merge_id=merge_id,
            ordered_batch_hashes=input_hashes,
            model=self.settings.finalizer_model,
            thinking=self.settings.finalizer_thinking,
            recap_prompt=self.settings.recap_prompt or "",
        )

        cached_merge, m_meta = self.hierarchy_cache.load_merge_result(merge_id, merge_key)
        if m_meta["hit"] and cached_merge is not None:
            if log:
                log(f"Merge group {merge_id} đã có trong bộ nhớ đệm (cache hit).")
            if on_phase:
                on_phase(
                    AnalysisPhase.SEASON_MERGING,
                    merge_id,
                    {"merge_id": merge_id, "cache_hit": True, "count": len(group)},
                )
            return cached_merge

        if on_phase:
            on_phase(
                AnalysisPhase.SEASON_MERGING,
                merge_id,
                {"merge_id": merge_id, "cache_hit": False, "count": len(group)},
            )

        merge_payload = {
            "merge_id": merge_id,
            "batch_results": group,
        }
        user_text = (
            f"Merge and synthesize {len(group)} batch results into unified season connections.\n"
            f"{coverage_notice}\n"
            f"Recap instructions:\n{self.settings.recap_prompt or 'Standard video recap'}\n\n"
            f"Batch Results:\n{json.dumps(merge_payload, ensure_ascii=False, indent=2)}\n"
        )
        check_payload_size(user_text, max_bytes=MAX_PAYLOAD_BYTES, context=f"merge {merge_id}")
        payload_bytes = len(user_text.encode("utf-8"))
        if log:
            log(
                f"[Season Connection] phase=season_merging merge_id={merge_id} "
                f"model={self.settings.finalizer_model} cache=miss "
                f"count={len(group)} payload_bytes={payload_bytes}"
            )

        def _merge_status_cb(msg: str) -> None:
            if on_phase:
                on_phase(AnalysisPhase.SEASON_MERGING, merge_id, {"status": msg, "status_message": msg, "merge_id": merge_id})

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

        if _is_cancelled(cancel_event):
            raise AnalysisCancelledError("Phân tích liên kết mùa phim đã bị hủy.")

        if not isinstance(raw_merge, dict):
            raise AnalysisError(f"Merge {merge_id} returned invalid non-dict response.")

        validate_batch_response_schema(raw_merge, context=f"merge {merge_id}")
        self.hierarchy_cache.save_merge_result(merge_id, merge_key, raw_merge)
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
