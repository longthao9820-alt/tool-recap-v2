"""Targeted tests for adaptive-tests-r7 (Objective rev 7).

Matrix:
1. Exact estimator all captured batch/merge calls <= 500,000.
2. Regression: produce old candidate > 554,508 but adaptive splits sends bounded.
3. One dense summary compaction; one extreme temporal fragments same original episode ID.
4. Two dense split despite count 2; four small packed.
5. Counts 1, 5, 10, 20, 30, 37 uneven deterministic all item/ref coverage/order and bounded.
6. Oversized merge adaptive recursive, no singleton loop.
7. Stable node/cache append prefix; mutate one reuses unaffected calls/caches (actual connector two runs).
8. Supporting item survives skeleton.
9. Valid response saved then cancellation.
10. Baseline system/envelope too large fatal only base case clear.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from toolrecap_v2.analyzer.connection import (
    HARD_PAYLOAD_CEILING,
    TARGET_PAYLOAD_CEILING,
    AdaptiveBatchPlanItem,
    SeasonConnectionResult,
    SeasonConnector,
    compact_merge_result,
    deterministic_cap_merge_item,
    estimate_batch_request_size,
    estimate_merge_request_size,
    format_batch_user_text,
    format_merge_user_text,
    format_node_id,
    plan_adaptive_batches,
    split_merge_result,
)
from toolrecap_v2.analyzer.errors import AnalysisCancelledError, AnalysisError
from toolrecap_v2.analyzer.prompts import (
    SEASON_BATCH_SYSTEM_PROMPT,
    SEASON_MERGE_SYSTEM_PROMPT,
)
from toolrecap_v2.api_client import (
    APIError,
    estimate_request_size,
)
from toolrecap_v2.domain.cache import (
    HierarchyCacheManager,
    compute_batch_cache_key,
    compute_connection_cache_key,
    compute_merge_cache_key,
)
from toolrecap_v2.domain.enums import CandidateScope, CompactionLevel
from toolrecap_v2.domain.models import (
    CompactEpisodeSummary,
    CompactSummaryItem,
    EpisodeEvidence,
    SourceEpisode,
    compact_summary,
    split_summary_by_timeline,
)
from toolrecap_v2.settings import AppSettings


class MockR7AIClient:
    """Mock OpenAICompatibleClient for R7 testing."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.call_history: list[dict[str, Any]] = []
        self.endpoint = "http://mock-ai:20128/v1"
        self.api_key = "mock-key"

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user_text: str,
        thinking: str = "auto",
        images: Any = (),
        max_tokens: int = 32000,
        cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("Mock API call cancelled.")

        call_record = {
            "model": model,
            "system": system,
            "user_text": user_text,
            "thinking": thinking,
        }
        self.call_history.append(call_record)

        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if callable(resp):
                return resp(call_record)
            return resp

        # Default valid batch/merge response
        return {
            "cross_episode_links": [
                {
                    "thread_id": "thread_default",
                    "theme": "Continuity",
                    "episodes": ["E01", "E02"],
                    "summary": "Continuity across episodes.",
                }
            ],
            "candidate_proposals": [
                {
                    "proposal_id": "prop_default",
                    "title": "Default Proposal",
                    "candidate_scope": "CROSS_EPISODE",
                    "episodes": ["E01", "E02"],
                    "characters": ["Protagonist"],
                    "editorial_reason": "Standard arc",
                    "status": "keep",
                }
            ],
            "supporting_character_arcs": [
                {
                    "character": "Sidekick",
                    "episodes": ["E01"],
                    "arc_summary": "Sidekick journey",
                    "has_dedicated_candidate": True,
                }
            ],
            "rejected_or_merged": [],
        }


def _make_dummy_summary(
    episode_id: str,
    item_count: int = 3,
    item_text_len: int = 50,
    with_supporting: bool = True,
) -> CompactEpisodeSummary:
    items = []
    for i in range(item_count):
        chars = ["Protagonist"]
        item_type = "event"
        cats = ["major_scenes"]
        if with_supporting and i == item_count - 1:
            chars = ["Officer Miller"]
            item_type = "supporting_development"
            cats = ["supporting_characters"]

        items.append(
            CompactSummaryItem(
                refs=[f"scene:{i}"],
                episode_id=episode_id,
                start_sec=float(i * 30),
                end_sec=float((i + 1) * 30),
                characters=chars,
                summary=f"Event {i} in {episode_id}: " + "x" * item_text_len,
                categories=cats,
                item_type=item_type,
            )
        )
    return CompactEpisodeSummary(
        episode_id=episode_id,
        title=f"Episode {episode_id}",
        duration_seconds=float(item_count * 30),
        items=items,
    )


# ---------------------------------------------------------------------------
# 1. Exact Estimator All Captured Batch/Merge Calls <= 500,000
# ---------------------------------------------------------------------------

def test_exact_estimator_all_captured_batch_and_merge_calls_bounded(tmp_path: Path) -> None:
    """Exact estimator validates all captured batch and merge calls <= 500,000 bytes."""
    episodes = [SourceEpisode(episode_id=f"E{i:02d}", source_video=f"video_{i}.mp4") for i in range(1, 9)]
    evidence_map = {}
    for ep in episodes:
        ev = EpisodeEvidence(episode_id=ep.episode_id, source_video=ep.source_video, duration_seconds=300.0)
        ev.data = {
            "major_scenes": [
                {"start_sec": float(j * 30), "end_sec": float((j + 1) * 30), "summary": f"Scene {j} details " * 10}
                for j in range(10)
            ],
            "supporting_characters": [
                {"character": "Detective Kim", "arc_summary": "Kim uncovers evidence in E01"}
            ],
        }
        evidence_map[ep.episode_id] = ev

    mock_client = MockR7AIClient()
    h_cache = HierarchyCacheManager(tmp_path / "cache")
    connector = SeasonConnector(client=mock_client, hierarchy_cache=h_cache)

    result = connector.connect_season(episodes, evidence_map)
    assert isinstance(result, SeasonConnectionResult)
    assert len(mock_client.call_history) >= 2  # Batches + Merge

    for call in mock_client.call_history:
        # Check both the raw user_text byte size and the full estimator
        est_bytes = estimate_request_size(
            model=call["model"],
            system=call["system"],
            user=call["user_text"],
            thinking=call["thinking"],
            variant=0,
        )
        assert est_bytes <= HARD_PAYLOAD_CEILING, (
            f"Call exceeded hard ceiling: {est_bytes} > {HARD_PAYLOAD_CEILING}"
        )
        assert est_bytes <= 500_000


# ---------------------------------------------------------------------------
# 2. Regression: Old Candidate > 554,508 Bytes, Adaptive Splits & Bounded
# ---------------------------------------------------------------------------

def test_regression_old_candidate_over_554508_adaptive_splits_and_bounds() -> None:
    """Produce old candidate > 554,508 bytes, verify adaptive batching splits/compacts <= 500,000."""
    # Build an oversized summary where uncompacted payload exceeds 554,508 bytes
    # System prompt is ~3,000 bytes. We need user_text > 552,000 bytes.
    # 600 items each with ~950 characters of summary text -> ~570,000 bytes
    items = [
        CompactSummaryItem(
            refs=[f"scene:{i}"],
            episode_id="E01",
            start_sec=float(i),
            end_sec=float(i + 1),
            characters=["Protagonist", "Antagonist"],
            summary=f"Item {i:04d}: " + ("Detailed narrative event description " * 25)[:920],
            categories=["major_scenes"],
            item_type="event",
        )
        for i in range(600)
    ]
    raw_summary = CompactEpisodeSummary(
        episode_id="E01",
        title="Oversized Episode E01",
        duration_seconds=600.0,
        items=items,
    )

    # 1. Verify uncompacted estimate is indeed > 554,508 bytes
    raw_text = format_batch_user_text("test_node", [raw_summary], "", "")
    raw_est = estimate_batch_request_size("sub", raw_text, "auto")
    assert raw_est > 554_508, f"Regression baseline not reproduced: {raw_est} <= 554508"

    # 2. Adaptive batching must split and compact to stay strictly <= 500,000
    plans = plan_adaptive_batches(
        ordered_summaries=[raw_summary],
        model="sub",
        thinking="auto",
        target_ceiling=TARGET_PAYLOAD_CEILING,
        hard_ceiling=HARD_PAYLOAD_CEILING,
    )

    assert len(plans) >= 1
    for p in plans:
        assert p.estimated_bytes <= HARD_PAYLOAD_CEILING, (
            f"Planned batch {p.node_id} exceeds hard ceiling: {p.estimated_bytes} > {HARD_PAYLOAD_CEILING}"
        )
        assert p.estimated_bytes <= 500_000
        assert p.estimated_bytes <= TARGET_PAYLOAD_CEILING


# ---------------------------------------------------------------------------
# 3. Dense Summary Compaction & Extreme Temporal Fragments Same Episode ID
# ---------------------------------------------------------------------------

def test_one_dense_summary_compaction() -> None:
    """A dense summary that exceeds ceiling at FULL compacts to TRIMMED/PRIORITY/SKELETON."""
    # Use configurable ceiling (e.g. 10,000 bytes)
    ceil = 10_000
    # Create a summary whose FULL representation is ~15,000 bytes, but TRIMMED is ~8,000 bytes
    items = [
        CompactSummaryItem(
            refs=[f"scene:{i}"],
            episode_id="E01",
            start_sec=float(i * 10),
            end_sec=float((i + 1) * 10),
            characters=["Detective"],
            summary="A" * 300,  # 300 chars per item * 40 items = 12,000 chars
            categories=["major_scenes"],
            item_type="event",
        )
        for i in range(40)
    ]
    summary = CompactEpisodeSummary(episode_id="E01", title="Ep 1", duration_seconds=400.0, items=items)

    full_est = estimate_batch_request_size("sub", format_batch_user_text("n", [summary], "", ""), "auto")
    assert full_est > ceil

    plans = plan_adaptive_batches(
        ordered_summaries=[summary],
        model="sub",
        thinking="auto",
        target_ceiling=ceil,
        hard_ceiling=ceil + 2000,
    )

    assert len(plans) >= 1
    assert plans[0].compaction_level in (CompactionLevel.TRIMMED, CompactionLevel.PRIORITY, CompactionLevel.SKELETON)
    assert plans[0].estimated_bytes <= ceil


def test_one_extreme_temporal_fragments_same_original_episode_id() -> None:
    """Extreme density summary splits into temporal fragments all sharing original episode_id."""
    ceil = 6_000
    # 50 items with 150 chars each. Even at skeleton (capped at 80 chars), 50 * 80 + JSON overhead > 6,000
    items = [
        CompactSummaryItem(
            refs=[f"ref:{i}"],
            episode_id="E01",
            start_sec=float(i * 10),
            end_sec=float((i + 1) * 10),
            characters=["Agent Cooper"],
            summary=f"Key plot twist {i}: " + "y" * 120,
            categories=["reveals"],
            item_type="reveal",
        )
        for i in range(50)
    ]
    summary = CompactEpisodeSummary(episode_id="E01", title="Extreme Episode", duration_seconds=500.0, items=items)

    plans = plan_adaptive_batches(
        ordered_summaries=[summary],
        model="sub",
        thinking="auto",
        target_ceiling=ceil,
        hard_ceiling=ceil + 2000,
    )

    # Must split into multiple fragment batches
    assert len(plans) >= 2
    for p in plans:
        assert p.estimated_bytes <= ceil
        # Every summary across all plans must preserve original episode_id "E01"
        for s in p.summaries:
            assert s.episode_id == "E01"
            assert s.fragment_id != ""  # Fragment metadata populated
            assert s.fragment_id.startswith("E01")


# ---------------------------------------------------------------------------
# 4. Two Dense Split Despite Count 2; Four Small Packed
# ---------------------------------------------------------------------------

def test_two_dense_split_despite_count_two() -> None:
    """Two dense episodes exceed ceiling when combined and split into 2 separate batches."""
    ceil = 5_000
    # Each summary is ~4,100 bytes with baseline prompt. Alone <= 5,000; combined ~5,200 > 5,000.
    s1 = _make_dummy_summary("E01", item_count=3, item_text_len=80)
    s2 = _make_dummy_summary("E02", item_count=3, item_text_len=80)

    # Check sizing
    est_s1 = estimate_batch_request_size("sub", format_batch_user_text("test1", [s1], "", ""))
    est_s2 = estimate_batch_request_size("sub", format_batch_user_text("test2", [s2], "", ""))
    est_both = estimate_batch_request_size("sub", format_batch_user_text("test12", [s1, s2], "", ""))
    assert est_s1 <= ceil
    assert est_s2 <= ceil
    assert est_both > ceil

    plans = plan_adaptive_batches(
        ordered_summaries=[s1, s2],
        model="sub",
        thinking="auto",
        target_ceiling=ceil,
        hard_ceiling=ceil + 2000,
    )

    # Must be split into 2 batches despite count being only 2
    assert len(plans) == 2
    assert len(plans[0].summaries) == 1
    assert plans[0].summaries[0].episode_id == "E01"
    assert len(plans[1].summaries) == 1
    assert plans[1].summaries[0].episode_id == "E02"


def test_four_small_packed() -> None:
    """Four small episodes fit together within ceiling and pack into a single batch."""
    ceil = 15_000
    summaries = [_make_dummy_summary(f"E0{i}", item_count=2, item_text_len=30) for i in range(1, 5)]

    plans = plan_adaptive_batches(
        ordered_summaries=summaries,
        model="sub",
        thinking="auto",
        target_ceiling=ceil,
        hard_ceiling=ceil + 2000,
    )

    assert len(plans) == 1
    assert len(plans[0].summaries) == 4
    assert [s.episode_id for s in plans[0].summaries] == ["E01", "E02", "E03", "E04"]


# ---------------------------------------------------------------------------
# 5. Counts 1, 5, 10, 20, 30, 37 Uneven Deterministic Coverage and Order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_count", [1, 5, 10, 20, 30, 37])
def test_counts_uneven_deterministic_coverage_order_and_bounded(n_count: int) -> None:
    """Deterministically plan batches for counts 1, 5, 10, 20, 30, 37 with uneven items."""
    ceil = 12_000
    hard = 14_000

    # Create uneven episodes: varying item counts (1 to 7 items per episode)
    summaries = [
        _make_dummy_summary(f"E{i:02d}", item_count=(i % 7) + 1, item_text_len=40)
        for i in range(1, n_count + 1)
    ]

    # Run 1
    plans1 = plan_adaptive_batches(
        ordered_summaries=summaries,
        model="sub",
        thinking="auto",
        target_ceiling=ceil,
        hard_ceiling=hard,
    )

    # Run 2 (Determinism check)
    plans2 = plan_adaptive_batches(
        ordered_summaries=summaries,
        model="sub",
        thinking="auto",
        target_ceiling=ceil,
        hard_ceiling=hard,
    )

    assert len(plans1) == len(plans2)
    for p1, p2 in zip(plans1, plans2):
        assert p1.node_id == p2.node_id
        assert p1.span_start == p2.span_start
        assert p1.span_end == p2.span_end
        assert p1.cache_key == p2.cache_key
        assert p1.estimated_bytes == p2.estimated_bytes
        assert p1.estimated_bytes <= hard

    # All items covered and monotonic order
    covered_episodes: list[str] = []
    last_end = -1
    for p in plans1:
        assert p.span_start <= p.span_end
        assert p.span_start >= last_end or p.span_start == last_end
        last_end = p.span_end
        for s in p.summaries:
            covered_episodes.append(s.episode_id)

    # Every input episode is covered
    unique_covered = set(covered_episodes)
    for s in summaries:
        assert s.episode_id in unique_covered


# ---------------------------------------------------------------------------
# 6. Oversized Merge Adaptive Recursive, No Singleton Loop
# ---------------------------------------------------------------------------

def test_oversized_merge_adaptive_recursive_no_singleton_loop(tmp_path: Path) -> None:
    """Oversized merge items compact, split recursively, cap indivisible strings, and terminate without loop."""
    # Create 4 oversized batch results that exceed target ceiling when paired
    ceil = 4_000
    hard = 5_000

    batch_results = []
    for b_idx in range(1, 5):
        props = [
            {
                "proposal_id": f"prop_b{b_idx}_{p}",
                "title": f"Candidate Proposal {p} in Batch {b_idx} with long editorial context " * 3,
                "candidate_scope": "CROSS_EPISODE",
                "episodes": [f"E0{b_idx}", f"E0{b_idx+1}"],
                "characters": ["Character A", "Character B"],
                "editorial_reason": "Very long editorial reason explaining why this arc is crucial " * 3,
                "status": "keep",
            }
            for p in range(4)
        ]
        links = [
            {
                "thread_id": f"thread_b{b_idx}_{l}",
                "theme": f"Theme {l} connecting narrative elements across multiple episodes",
                "episodes": [f"E0{b_idx}", f"E0{b_idx+1}"],
                "summary": "Long summary of narrative continuity and thematic resonance " * 3,
            }
            for l in range(3)
        ]
        batch_results.append({
            "node_id": f"node_L0_{b_idx}_{b_idx}_dummy",
            "batch_id": f"node_L0_{b_idx}_{b_idx}_dummy",
            "candidate_proposals": props,
            "cross_episode_links": links,
            "supporting_character_arcs": [],
            "rejected_or_merged": [],
        })

    mock_client = MockR7AIClient()
    h_cache = HierarchyCacheManager(tmp_path / "cache")
    connector = SeasonConnector(
        client=mock_client,
        hierarchy_cache=h_cache,
        target_ceiling=ceil,
        hard_ceiling=hard,
    )

    # Hierarchical merge must successfully terminate with 1 unified result without infinite loop
    unified = connector._hierarchical_merge(
        batch_results=batch_results,
        coverage_notice="",
        cancel_event=None,
        on_phase=None,
        log=None,
    )

    assert isinstance(unified, dict)
    assert "candidate_proposals" in unified
    # Check that merge calls were made and all stayed <= hard ceiling
    merge_calls = [c for c in mock_client.call_history if "season_merge" in c.get("system", "").lower() or "merge" in c.get("user_text", "").lower()]
    assert len(merge_calls) >= 1
    for mc in merge_calls:
        est = estimate_merge_request_size(connector.settings.finalizer_model, mc["user_text"])
        assert est <= hard


# ---------------------------------------------------------------------------
# 7. Stable Node/Cache Append Prefix; Mutate One Reuses Unaffected
# ---------------------------------------------------------------------------

def test_stable_node_cache_append_prefix_and_mutate_one_reuses_unaffected(tmp_path: Path) -> None:
    """Prefix append and single episode mutation reuse unaffected batches from cache (actual connector two runs)."""
    h_cache = HierarchyCacheManager(tmp_path / "cache")
    mock_client = MockR7AIClient()
    connector = SeasonConnector(client=mock_client, hierarchy_cache=h_cache)

    ep1 = SourceEpisode(episode_id="E01", source_video="v1.mp4")
    ep2 = SourceEpisode(episode_id="E02", source_video="v2.mp4")
    ep3 = SourceEpisode(episode_id="E03", source_video="v3.mp4")
    ep4 = SourceEpisode(episode_id="E04", source_video="v4.mp4")
    episodes_run1 = [ep1, ep2, ep3, ep4]

    ev_map_run1 = {
        ep.episode_id: EpisodeEvidence(
            episode_id=ep.episode_id,
            source_video=ep.source_video,
            duration_seconds=120.0,
            data={"major_scenes": [{"start_sec": 0.0, "end_sec": 60.0, "summary": f"Scene in {ep.episode_id}"}]},
        )
        for ep in episodes_run1
    }

    # Run 1: 4 episodes -> 1 batch (E01-E04)
    connector.connect_season(episodes_run1, ev_map_run1)
    calls_run1 = len(mock_client.call_history)
    assert calls_run1 >= 1  # 1 batch (since 4 small fit in 1 batch)

    # Run 2: Append E05 (prefix E01..E04 unchanged)
    ep5 = SourceEpisode(episode_id="E05", source_video="v5.mp4")
    episodes_run2 = [ep1, ep2, ep3, ep4, ep5]
    ev_map_run2 = dict(ev_map_run1)
    ev_map_run2["E05"] = EpisodeEvidence(
        episode_id="E05",
        source_video="v5.mp4",
        duration_seconds=120.0,
        data={"major_scenes": [{"start_sec": 0.0, "end_sec": 60.0, "summary": "Scene in E05"}]},
    )

    mock_client.call_history.clear()
    connector.connect_season(episodes_run2, ev_map_run2)

    # In Run 2: E01-E04 batch hit cache! Only E05 batch and merge were called
    batch_calls = [c for c in mock_client.call_history if "batch connection analyst" in c.get("system", "").lower()]
    assert len(batch_calls) == 1
    assert "E05" in batch_calls[0]["user_text"]
    assert "E01" not in batch_calls[0]["user_text"]  # E01-E04 was reused from cache

    # Run 3: Mutate only E05 (prefix E01..E04 still unchanged)
    ev_map_run3 = dict(ev_map_run2)
    ev_map_run3["E05"] = EpisodeEvidence(
        episode_id="E05",
        source_video="v5.mp4",
        duration_seconds=120.0,
        data={"major_scenes": [{"start_sec": 0.0, "end_sec": 60.0, "summary": "Mutated scene in E05"}]},
    )
    mock_client.call_history.clear()
    connector.connect_season(episodes_run2, ev_map_run3)

    # In Run 3: E01-E04 batch is STILL reused from cache!
    batch_calls_run3 = [c for c in mock_client.call_history if "batch connection analyst" in c.get("system", "").lower()]
    assert len(batch_calls_run3) == 1
    assert "Mutated scene in E05" in batch_calls_run3[0]["user_text"]
    assert "E01" not in batch_calls_run3[0]["user_text"]


# ---------------------------------------------------------------------------
# 8. Supporting Item Survives Skeleton
# ---------------------------------------------------------------------------

def test_supporting_item_survives_skeleton() -> None:
    """Supporting character evidence explicitly survives skeleton compaction level."""
    items = [
        CompactSummaryItem(
            refs=["scene:0"],
            episode_id="E01",
            start_sec=0.0,
            end_sec=30.0,
            characters=["Main Protagonist"],
            summary="Protagonist enters the abandoned warehouse.",
            categories=["major_scenes"],
            item_type="event",
        ),
        CompactSummaryItem(
            refs=["supporting:0"],
            episode_id="E01",
            start_sec=40.0,
            end_sec=70.0,
            characters=["Officer Miller"],
            summary="Officer Miller discovers unauthorized transaction receipts.",
            categories=["supporting_characters"],
            item_type="supporting_development",
        ),
        CompactSummaryItem(
            refs=["dialogue:0"],
            episode_id="E01",
            start_sec=80.0,
            end_sec=100.0,
            characters=["Background Extra"],
            summary="Background noise and passerby chatter in hallway.",
            categories=["dialogue"],
            item_type="ambient",
        ),
    ]
    summary = CompactEpisodeSummary(episode_id="E01", title="Episode 1", duration_seconds=120.0, items=items)

    skeleton = compact_summary(summary, CompactionLevel.SKELETON)

    # Verify supporting item survived in skeleton
    surviving_chars = [c for it in skeleton.items for c in it.characters]
    assert "Officer Miller" in surviving_chars
    supporting_items = [it for it in skeleton.items if "Officer Miller" in it.characters]
    assert len(supporting_items) == 1
    assert supporting_items[0].item_type == "supporting_development"
    assert supporting_items[0].episode_id == "E01"
    assert len(supporting_items[0].summary) <= 80  # Skeleton text cap


# ---------------------------------------------------------------------------
# 9. Valid Response Saved Then Cancellation (Batch and Merge)
# ---------------------------------------------------------------------------

def test_valid_response_saved_then_cancellation(tmp_path: Path) -> None:
    """Valid AI response is persisted to cache before cancellation check raises AnalysisCancelledError."""
    cancel_event = threading.Event()
    h_cache = HierarchyCacheManager(tmp_path / "cache")

    ep1 = SourceEpisode(episode_id="E01", source_video="v1.mp4")
    ep2 = SourceEpisode(episode_id="E02", source_video="v2.mp4")
    evidence_map = {
        "E01": EpisodeEvidence(episode_id="E01", source_video="v1.mp4", duration_seconds=60.0),
        "E02": EpisodeEvidence(episode_id="E02", source_video="v2.mp4", duration_seconds=60.0),
    }

    valid_batch = {
        "cross_episode_links": [],
        "candidate_proposals": [{"proposal_id": "p1", "title": "Saved Before Cancel"}],
        "supporting_character_arcs": [],
        "rejected_or_merged": [],
    }

    def client_chat_with_cancel(call_record: dict[str, Any]) -> dict[str, Any]:
        # Cancel right as response is returned
        cancel_event.set()
        return valid_batch

    mock_client = MockR7AIClient([client_chat_with_cancel])
    connector = SeasonConnector(client=mock_client, hierarchy_cache=h_cache)

    with pytest.raises(AnalysisCancelledError):
        connector.connect_season(
            [ep1, ep2],
            evidence_map,
            cancel_event=cancel_event,
        )

    # Verify the batch was saved to cache despite the cancellation!
    saved_batch_files = list(h_cache.batch_dir.glob("*.batch.json"))
    assert len(saved_batch_files) >= 1
    saved_data = json.loads(saved_batch_files[0].read_text(encoding="utf-8"))
    assert saved_data["result"]["candidate_proposals"][0]["title"] == "Saved Before Cancel"


def test_valid_merge_response_saved_then_cancellation(tmp_path: Path) -> None:
    """Valid AI merge response is persisted to cache before cancellation check raises AnalysisCancelledError."""
    cancel_event = threading.Event()
    h_cache = HierarchyCacheManager(tmp_path / "cache")

    valid_merge = {
        "cross_episode_links": [
            {
                "thread_id": "t_merge",
                "theme": "Merge Continuity",
                "episodes": ["E01", "E02"],
                "summary": "Merged thread summary across batches.",
            }
        ],
        "candidate_proposals": [
            {
                "proposal_id": "p_merge",
                "title": "Saved Merge Before Cancel",
                "candidate_scope": "CROSS_EPISODE",
                "episodes": ["E01", "E02"],
                "characters": ["Protagonist"],
                "editorial_reason": "Merged arc",
                "status": "keep",
            }
        ],
        "supporting_character_arcs": [],
        "rejected_or_merged": [],
    }

    def client_merge_with_cancel(call_record: dict[str, Any]) -> dict[str, Any]:
        cancel_event.set()
        return valid_merge

    mock_client = MockR7AIClient([client_merge_with_cancel])
    connector = SeasonConnector(client=mock_client, hierarchy_cache=h_cache)

    batch_results = [
        {
            "node_id": "node_L0_1_1_e1",
            "batch_id": "node_L0_1_1_e1",
            "candidate_proposals": [{"proposal_id": "p1", "title": "B1 Prop"}],
            "cross_episode_links": [],
            "supporting_character_arcs": [],
            "rejected_or_merged": [],
        },
        {
            "node_id": "node_L0_2_2_e2",
            "batch_id": "node_L0_2_2_e2",
            "candidate_proposals": [{"proposal_id": "p2", "title": "B2 Prop"}],
            "cross_episode_links": [],
            "supporting_character_arcs": [],
            "rejected_or_merged": [],
        },
    ]

    with pytest.raises(AnalysisCancelledError):
        connector._hierarchical_merge(
            batch_results=batch_results,
            coverage_notice="",
            cancel_event=cancel_event,
            on_phase=None,
            log=None,
        )

    # Verify the merge was saved to cache despite the cancellation!
    saved_merge_files = list(h_cache.merge_dir.glob("*.merge.json"))
    assert len(saved_merge_files) >= 1
    saved_data = json.loads(saved_merge_files[0].read_text(encoding="utf-8"))
    assert saved_data["result"]["candidate_proposals"][0]["title"] == "Saved Merge Before Cancel"


def test_valid_merge_response_saved_then_cancellation_end_to_end(tmp_path: Path) -> None:
    """End-to-end connect_season: both batches and merge response are saved before cancellation raises."""
    cancel_event = threading.Event()
    h_cache = HierarchyCacheManager(tmp_path / "cache")

    ceil = 5_000
    hard = 7_000

    ep1 = SourceEpisode(episode_id="E01", source_video="v1.mp4")
    ep2 = SourceEpisode(episode_id="E02", source_video="v2.mp4")
    ev1 = EpisodeEvidence(
        episode_id="E01",
        source_video="v1.mp4",
        duration_seconds=450.0,
        data={"major_scenes": [{"start_sec": float(i * 30), "end_sec": float((i + 1) * 30), "summary": "Detailed scene description " * 5} for i in range(15)]},
    )
    ev2 = EpisodeEvidence(
        episode_id="E02",
        source_video="v2.mp4",
        duration_seconds=450.0,
        data={"major_scenes": [{"start_sec": float(i * 30), "end_sec": float((i + 1) * 30), "summary": "Detailed scene description " * 5} for i in range(15)]},
    )

    class MergeCancellingMockClient(MockR7AIClient):
        def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            call_record = {
                "model": kwargs.get("model", ""),
                "system": kwargs.get("system", ""),
                "user_text": kwargs.get("user_text", ""),
                "thinking": kwargs.get("thinking", ""),
            }
            self.call_history.append(call_record)
            if kwargs.get("system") == SEASON_MERGE_SYSTEM_PROMPT:
                cancel_event.set()
                return {
                    "cross_episode_links": [],
                    "candidate_proposals": [{"proposal_id": "m1", "title": "Saved Merge In E2E"}],
                    "supporting_character_arcs": [],
                    "rejected_or_merged": [],
                }
            return {
                "cross_episode_links": [],
                "candidate_proposals": [{"proposal_id": "b1", "title": "Saved Batch"}],
                "supporting_character_arcs": [],
                "rejected_or_merged": [],
            }

    mock_client = MergeCancellingMockClient()
    connector = SeasonConnector(
        client=mock_client,
        hierarchy_cache=h_cache,
        target_ceiling=ceil,
        hard_ceiling=hard,
    )

    with pytest.raises(AnalysisCancelledError):
        connector.connect_season([ep1, ep2], {"E01": ev1, "E02": ev2}, cancel_event=cancel_event)

    # Verify both batches AND merge were saved to cache
    saved_batches = list(h_cache.batch_dir.glob("*.batch.json"))
    saved_merges = list(h_cache.merge_dir.glob("*.merge.json"))
    assert len(saved_batches) >= 2
    assert len(saved_merges) >= 1
    merge_data = json.loads(saved_merges[0].read_text(encoding="utf-8"))
    assert merge_data["result"]["candidate_proposals"][0]["title"] == "Saved Merge In E2E"


# ---------------------------------------------------------------------------
# 10. Baseline System/Envelope Too Large Fatal Only Base Case Clear
# ---------------------------------------------------------------------------

def test_baseline_system_envelope_too_large_fatal_clear() -> None:
    """When baseline prompt envelope alone exceeds target ceiling, raises fatal AnalysisError immediately."""
    # Empty prompt baseline size is ~3,000 bytes. Setting target_ceiling=1000 causes immediate fatal
    s = _make_dummy_summary("E01", item_count=1)

    with pytest.raises(AnalysisError, match="Baseline batch prompt envelope size .* exceeds target limit"):
        plan_adaptive_batches(
            ordered_summaries=[s],
            model="sub",
            thinking="auto",
            target_ceiling=1_000,
            hard_ceiling=2_000,
        )


def test_baseline_merge_envelope_too_large_fatal_clear(tmp_path: Path) -> None:
    """When baseline merge prompt envelope exceeds target ceiling, raises fatal AnalysisError immediately."""
    mock_client = MockR7AIClient()
    h_cache = HierarchyCacheManager(tmp_path / "cache")
    connector = SeasonConnector(
        client=mock_client,
        hierarchy_cache=h_cache,
        target_ceiling=1_000,
        hard_ceiling=2_000,
    )

    with pytest.raises(AnalysisError, match="Baseline merge prompt envelope size .* exceeds target limit"):
        connector._hierarchical_merge(
            batch_results=[{"dummy": 1}, {"dummy": 2}],
            coverage_notice="",
            cancel_event=None,
            on_phase=None,
            log=None,
        )
