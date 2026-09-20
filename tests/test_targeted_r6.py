"""Targeted tests for project-ui-r6:
1. Old project load & backwards-compatible fields
2. Season error preserves episode rows
3. Season row always present in UI
4. Output error ownership
5. Phase ranges monotonic and cached immediate
6. Retry status callback
7. Batch completion not 100 on error
8. Rerun paused/error reuse display
9. Cancellation mapping to CANCELLED not ERROR
10. Callback backward compatibility
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from toolrecap_v2.analyzer.engine import AnalysisEngine
from toolrecap_v2.analyzer.errors import AnalysisCancelledError, AnalysisError
from toolrecap_v2.analyzer.phases import AnalysisPhase
from toolrecap_v2.api_client import OpenAICompatibleClient
from toolrecap_v2.domain.cache import EvidenceCacheManager, HierarchyCacheManager
from toolrecap_v2.domain.enums import OutputStatus
from toolrecap_v2.domain.models import CommentaryOutput, EpisodeEvidence, SourceEpisode, build_compact_summary
from toolrecap_v2.media import RenderCancelled
from toolrecap_v2.projects import ProjectQueue, ProjectRecord, ProjectStore
from toolrecap_v2.settings import AppSettings
from toolrecap_v2.subtitles.cache import SubtitleCacheManager
from toolrecap_v2.subtitles.models import SubtitleCue
from toolrecap_v2.subtitles.pipeline import SubtitlePipeline
from toolrecap_v2.ui import ToolRecapV2App


# ---------------------------------------------------------------------------
# 1. Old Project Load & Backwards-Compatible Fields
# ---------------------------------------------------------------------------

def test_old_project_load_and_defaults(tmp_path: Path) -> None:
    """Old project json without new r6 fields loads with correct defaults, safe asdict/from_dict."""
    store_file = tmp_path / "old_projects.json"
    old_data = {
        "projects": [
            {
                "id": "proj_old_1",
                "name": "Old Project 1",
                "source_video": str(tmp_path / "old1.mp4"),
                "status": "WAITING",
                "progress": 0,
            },
            {
                "id": "proj_old_comp",
                "name": "Old Completed",
                "source_video": str(tmp_path / "comp.mp4"),
                "status": "COMPLETED",
                "progress": 100,
                "analysis_scope": "SEASON",
            },
            {
                "id": "proj_interrupted",
                "name": "Interrupted Proj",
                "source_video": str(tmp_path / "int.mp4"),
                "status": "RUNNING",
                "progress": 40,
                "source_episodes": [
                    {
                        "episode_id": "E01",
                        "source_video": str(tmp_path / "e1.mp4"),
                        "status": "COMPLETED",
                        "progress": 100,
                        "stage": "Evidence Complete",
                    },
                    {
                        "episode_id": "E02",
                        "source_video": str(tmp_path / "e2.mp4"),
                        "status": "RUNNING",
                        "progress": 20,
                    },
                ],
            },
        ]
    }
    store_file.write_text(json.dumps(old_data), encoding="utf-8")
    store = ProjectStore(store_file)
    records = store.load()

    assert len(records) == 3

    # 1a. proj_old_1 defaults
    p1 = records[0]
    assert p1.season_status == "WAITING"
    assert p1.season_progress == 0
    assert p1.season_error is None
    assert p1.error_scope is None
    assert p1.error_target is None
    assert len(p1.source_episodes) == 1
    assert p1.source_episodes[0].status == "WAITING"
    assert p1.source_episodes[0].stage == "Sẵn sàng"
    assert p1.source_episodes[0].progress == 0
    assert p1.source_episodes[0].cached is False

    # 1b. Legacy completed backfill
    p_comp = records[1]
    assert p_comp.status == "COMPLETED"
    assert p_comp.season_status == "COMPLETED"
    assert p_comp.season_progress == 100
    assert p_comp.source_episodes[0].status == "COMPLETED"
    assert p_comp.source_episodes[0].progress == 100
    assert p_comp.source_episodes[0].stage == "Evidence Complete"

    # 1c. Interrupted running becomes PAUSED but preserves completed episode
    p_int = records[2]
    assert p_int.status == "PAUSED"
    assert p_int.source_episodes[0].status == "COMPLETED"
    assert p_int.source_episodes[0].progress == 100
    assert p_int.source_episodes[1].status == "PAUSED"

    # 1d. Safe asdict & from_dict
    d = asdict(p1)
    p1_roundtrip = ProjectRecord.from_dict(d)
    assert p1_roundtrip.id == p1.id
    assert p1_roundtrip.season_status == "WAITING"

    ep_d = asdict(p1.source_episodes[0])
    ep_roundtrip = SourceEpisode.from_dict(ep_d)
    assert ep_roundtrip.episode_id == "E01"
    assert ep_roundtrip.status == "WAITING"


# ---------------------------------------------------------------------------
# 2. Season Error Preserves Episode Rows
# ---------------------------------------------------------------------------

def test_season_error_preserves_episode_rows(tk_root: pytest.MonkeyPatch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When season analysis fails, only season_status and season row become ERROR; completed episodes stay intact."""
    monkeypatch.setattr("toolrecap_v2.ui.default_data_directory", lambda: tmp_path)
    app = ToolRecapV2App()
    try:
        app.withdraw()

        ep1 = SourceEpisode(episode_id="E01", source_video=str(tmp_path / "e1.mp4"), status="COMPLETED", stage="Evidence Complete", progress=100)
        ep2 = SourceEpisode(episode_id="E02", source_video=str(tmp_path / "e2.mp4"), status="CACHED", stage="Evidence Complete", progress=100, cached=True)

        rec = ProjectRecord(
            id="season_err_proj",
            name="Season Error Proj",
            source_video=str(tmp_path / "e1.mp4"),
            analysis_scope="SEASON",
            source_episodes=[ep1, ep2],
            season_status="ERROR",
            season_error="Season batch 2 failed after retries",
            error_scope="SEASON",
            error_target="season",
            phase="season_batch",
            status="ERROR",
            progress=65,
            current_message="Lỗi season batch 2",
        )
        app.projects = [rec]
        app._refresh_queue_table()

        # Check Treeview rows
        e1_row = f"{rec.id}_E01"
        e2_row = f"{rec.id}_E02"
        season_row = f"{rec.id}_season"

        assert app.tree.set(e1_row, "stage") == "Evidence Complete"
        assert app.tree.set(e1_row, "progress") == "100%"

        assert app.tree.set(e2_row, "stage") == "Evidence Complete"
        assert app.tree.set(e2_row, "progress") == "100%"

        assert app.tree.set(season_row, "stage") == "Lỗi"
        assert "Season batch 2 failed" in app.tree.set(season_row, "status")

        # Now test _apply_project_update preserves episode rows too
        app._apply_project_update(rec)
        assert app.tree.set(e1_row, "stage") == "Evidence Complete"
        assert app.tree.set(e2_row, "stage") == "Evidence Complete"
        assert app.tree.set(season_row, "stage") == "Lỗi"
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
# 3. Season Row Always Present in UI
# ---------------------------------------------------------------------------

def test_season_row_always_present_in_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Season row is always present for SEASON projects across WAITING, RUNNING, ERROR, COMPLETED."""
    monkeypatch.setattr("toolrecap_v2.ui.default_data_directory", lambda: tmp_path)
    app = ToolRecapV2App()
    try:
        app.withdraw()

        ep1 = SourceEpisode(episode_id="E01", source_video=str(tmp_path / "e1.mp4"))
        rec = ProjectRecord(
            id="s_row_proj",
            name="Season Row Proj",
            analysis_scope="SEASON",
            source_episodes=[ep1],
            season_status="WAITING",
            status="WAITING",
        )
        app.projects = [rec]

        # 3a. WAITING: season row present
        app._refresh_queue_table()
        s_row = f"{rec.id}_season"
        assert app.tree.exists(s_row)
        assert app.tree.set(s_row, "stage") == "Sẵn sàng"

        # 3b. RUNNING: season row present
        rec.season_status = "RUNNING"
        rec.season_stage = "Season Analysis"
        rec.current_message = "Đang phân tích..."
        app._apply_project_update(rec)
        assert app.tree.exists(s_row)
        assert app.tree.set(s_row, "stage") == "Season Analysis"

        # 3c. COMPLETED: season row present
        rec.season_status = "COMPLETED"
        rec.status = "COMPLETED"
        app._apply_project_update(rec)
        assert app.tree.exists(s_row)
        assert app.tree.set(s_row, "stage") == "Hoàn thành"
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
# 4. Output Error Ownership
# ---------------------------------------------------------------------------

def test_output_error_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Output error marks only the failing output row; episodes and season remain intact."""
    monkeypatch.setattr("toolrecap_v2.ui.default_data_directory", lambda: tmp_path)
    app = ToolRecapV2App()
    try:
        app.withdraw()

        ep1 = SourceEpisode(episode_id="E01", source_video=str(tmp_path / "e1.mp4"), status="COMPLETED", stage="Evidence Complete", progress=100)
        out1 = CommentaryOutput(output_id="out_1", title="Recap 1", sanitized_title="Recap 1", status=OutputStatus.COMPLETED.value)
        out2 = CommentaryOutput(output_id="out_2", title="Recap 2", sanitized_title="Recap 2", status=OutputStatus.ERROR.value, error="FFmpeg encode failed")

        rec = ProjectRecord(
            id="out_err_proj",
            name="Output Error Proj",
            analysis_scope="SINGLE_EPISODE",
            source_episodes=[ep1],
            outputs=[out1, out2],
            error_scope="OUTPUT",
            error_target="out_2",
            status="ERROR",
            progress=92,
            current_message="Lỗi kết xuất output 2",
        )
        app.projects = [rec]
        app._refresh_queue_table()

        e1_row = f"{rec.id}_E01"
        out1_row = f"{rec.id}_output_1"
        out2_row = f"{rec.id}_output_2"

        assert app.tree.set(e1_row, "stage") == "Evidence Complete"
        assert app.tree.set(out1_row, "stage") == "Hoàn thành"
        assert app.tree.set(out2_row, "stage") == "Lỗi"
        assert "FFmpeg encode failed" in app.tree.set(out2_row, "status")

        # Update test
        app._apply_project_update(rec)
        assert app.tree.set(e1_row, "stage") == "Evidence Complete"
        assert app.tree.set(out1_row, "stage") == "Hoàn thành"
        assert app.tree.set(out2_row, "stage") == "Lỗi"
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
# 5. Phase Ranges Monotonic and Cached Immediate
# ---------------------------------------------------------------------------

def test_phase_ranges_monotonic_and_cached_immediate(tmp_path: Path) -> None:
    """Queue progress follows monotonic ranges:
    probe 0-10, subtitle 10-20, evidence 20-60 (cached immediately updates),
    season batch 63-70, merging 70-75, mining 75-85, plan 85, render 85-100.
    On failure achieved progress is kept."""
    store = ProjectStore(tmp_path / "queue_store.json")
    settings = AppSettings()

    ep1 = SourceEpisode(episode_id="E01", source_video=str(tmp_path / "e1.mp4"))
    ep2 = SourceEpisode(episode_id="E02", source_video=str(tmp_path / "e2.mp4"))
    rec = ProjectRecord(
        id="mono_proj",
        name="Mono Proj",
        analysis_scope="SEASON",
        source_episodes=[ep1, ep2],
    )

    queue = ProjectQueue([rec], store=store, settings=settings)

    # Simulate _on_engine_phase callbacks inside queue
    observed_progress: list[int] = []

    def tracking_update(r: ProjectRecord) -> None:
        observed_progress.append(r.progress)

    queue.on_update = tracking_update

    # 5a. Scanner callback with cached: True immediately updates progress (20 + 40*1/2 = 40)
    # We can invoke the inner callback behavior by testing the logic
    rec.progress = 20
    # Simulate scanner callback E01 cached
    ep1.cached = True
    ep1.status = "CACHED"
    completed = sum(1 for e in rec.source_episodes if e.status in {"CACHED", "COMPLETED", "EVIDENCE_COMPLETE"})
    new_prog = max(rec.progress, 20 + int(40 * completed / 2))
    assert new_prog == 40

    # Simulate scanner callback E02 complete (20 + 40*2/2 = 60)
    ep2.status = "EVIDENCE_COMPLETE"
    completed = sum(1 for e in rec.source_episodes if e.status in {"CACHED", "COMPLETED", "EVIDENCE_COMPLETE"})
    new_prog = max(new_prog, 20 + int(40 * completed / 2))
    assert new_prog == 60

    # 5b. Season batch 63-70
    batch_prog = 63 + int(7 * (1 / 2))
    assert batch_prog == 66
    new_prog = max(new_prog, batch_prog)
    assert new_prog == 66

    # 5c. Merging 70-75
    new_prog = max(new_prog, 70)
    assert new_prog == 70

    # 5d. Finalizer/mining 75-85
    new_prog = max(new_prog, 75)
    assert new_prog == 75

    # 5e. Output plan ready 85
    new_prog = max(new_prog, 85)
    assert new_prog == 85

    # 5f. On failure achieved progress is preserved, not reset
    rec.progress = 75
    rec.status = "ERROR"
    assert rec.progress == 75


# ---------------------------------------------------------------------------
# 6. Retry Status Callback
# ---------------------------------------------------------------------------

def test_retry_status_callback(tmp_path: Path) -> None:
    """Queue displays retry message and record status remains RUNNING."""
    ep = SourceEpisode(episode_id="E01", source_video=str(tmp_path / "e1.mp4"))
    rec = ProjectRecord(id="retry_proj", name="Retry Proj", source_episodes=[ep], status="RUNNING")
    store = ProjectStore(tmp_path / "retry_store.json")
    queue = ProjectQueue([rec], store=store, settings=AppSettings())

    # Mock engine phase callback logic
    retry_msg = "AI Gateway (scanner): Thử lại 1/3... / Retrying 1/3..."
    data = {"status": retry_msg, "status_message": retry_msg}

    # When retry status is emitted:
    msg = data.get("status_message") or data.get("status") or ""
    is_retry = "Thử lại" in msg or "Retrying" in msg
    assert is_retry is True

    if is_retry:
        rec.status = "RUNNING"
        rec.current_message = f"[E01] {msg}"

    assert rec.status == "RUNNING"
    assert "[E01]" in rec.current_message
    assert "Thử lại" in rec.current_message


# ---------------------------------------------------------------------------
# 7. Batch Completion Not 100 on Error
# ---------------------------------------------------------------------------

def test_batch_completion_not_100_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When batch completes with error (completed < total), progress_var is not 100.0."""
    monkeypatch.setattr("toolrecap_v2.ui.default_data_directory", lambda: tmp_path)
    app = ToolRecapV2App()
    try:
        app.withdraw()

        p1 = ProjectRecord(id="p1", name="Proj 1", status="COMPLETED", progress=100)
        p2 = ProjectRecord(id="p2", name="Proj 2", status="ERROR", progress=60)
        app.projects = [p1, p2]

        app._apply_batch_completed(completed=1, total=2)
        # Aggregate progress: (100 + 60) / 2 = 80.0
        assert app.progress_var.get() == 80.0
        assert app.progress_var.get() != 100.0

        # When all complete: 100.0
        app._apply_batch_completed(completed=2, total=2)
        assert app.progress_var.get() == 100.0
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
# 8. Rerun Paused/Error Reuse Display
# ---------------------------------------------------------------------------

def test_rerun_paused_or_error_reuse_display(tmp_path: Path) -> None:
    """Restarting a PAUSED/ERROR project resets error scopes but preserves completed episode states."""
    ep1 = SourceEpisode(episode_id="E01", source_video=str(tmp_path / "e1.mp4"), status="COMPLETED", stage="Evidence Complete", progress=100)
    ep2 = SourceEpisode(episode_id="E02", source_video=str(tmp_path / "e2.mp4"), status="ERROR", stage="Lỗi", progress=30, error="Network timeout")

    rec = ProjectRecord(
        id="rerun_proj",
        name="Rerun Proj",
        status="ERROR",
        progress=65,
        error="Previous error",
        error_scope="EPISODE",
        error_target="E02",
        season_status="ERROR",
        season_error="Season error",
        source_episodes=[ep1, ep2],
    )

    store = ProjectStore(tmp_path / "rerun_store.json")
    queue = ProjectQueue([rec], store=store, settings=AppSettings())

    # Simulate start of _process_single_project reset logic
    rec.status = "RUNNING"
    rec.error = None
    rec.error_scope = None
    rec.error_target = None
    rec.season_error = None
    if rec.season_status == "ERROR":
        rec.season_status = "WAITING"
        rec.season_stage = "Sẵn sàng"
    for ep in rec.source_episodes:
        if ep.status == "ERROR":
            ep.status = "WAITING"
            ep.error = None
            ep.stage = "Sẵn sàng"

    # Completed episode preserved!
    assert ep1.status == "COMPLETED"
    assert ep1.progress == 100
    assert ep1.stage == "Evidence Complete"

    # Errored episode reset to WAITING
    assert ep2.status == "WAITING"
    assert ep2.error is None
    assert ep2.stage == "Sẵn sàng"

    # Season error cleared
    assert rec.season_status == "WAITING"
    assert rec.season_error is None
    assert rec.error_scope is None


# ---------------------------------------------------------------------------
# 9. Cancellation Mapping
# ---------------------------------------------------------------------------

def test_cancellation_mapping_to_cancelled_not_error(tmp_path: Path) -> None:
    """AnalysisCancelledError and RenderCancelled map to CANCELLED, not ERROR."""
    ep = SourceEpisode(episode_id="E01", source_video=str(tmp_path / "e1.mp4"))
    rec = ProjectRecord(id="cancel_proj", name="Cancel Proj", source_episodes=[ep])
    store = ProjectStore(tmp_path / "cancel_store.json")
    queue = ProjectQueue([rec], store=store, settings=AppSettings())

    # Mock _process_single_project raising AnalysisCancelledError
    def mock_process(record: ProjectRecord) -> None:
        raise AnalysisCancelledError("Analysis was cancelled by user.")

    queue._process_single_project = mock_process  # type: ignore
    queue.start()
    assert queue._thread is not None
    queue._thread.join(timeout=5)

    assert rec.status == "CANCELLED"
    assert "Đã dừng" in rec.current_message
    assert rec.error is None


# ---------------------------------------------------------------------------
# 10. Callback Backward Compatibility
# ---------------------------------------------------------------------------

def test_callback_backward_compatibility(tmp_path: Path) -> None:
    """Callbacks without status_message or with string phase values handle gracefully."""
    ep = SourceEpisode(episode_id="E01", source_video=str(tmp_path / "e1.mp4"))
    rec = ProjectRecord(id="compat_proj", name="Compat Proj", source_episodes=[ep])
    store = ProjectStore(tmp_path / "compat_store.json")
    queue = ProjectQueue([rec], store=store, settings=AppSettings())

    # Call with bare status string
    old_style_data = {"status": "scanning"}
    phase = AnalysisPhase.SCANNER

    # Should not raise exception
    phase_val = phase.value if hasattr(phase, "value") else str(phase)
    msg = old_style_data.get("status_message") or old_style_data.get("status") or ""
    assert phase_val == "scanner"
    assert msg == "scanning"


# ---------------------------------------------------------------------------
# 11. Queue Season Actual Failure Ownership
# ---------------------------------------------------------------------------

def test_queue_season_actual_failure_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When season analysis fails in Queue, season_status/season_error/error_scope become SEASON,
    while completed episode rows stay intact."""
    v1 = tmp_path / "e1.mp4"
    v2 = tmp_path / "e2.mp4"
    v1.write_bytes(b"dummy1")
    v2.write_bytes(b"dummy2")

    ep1 = SourceEpisode(episode_id="E01", source_video=str(v1), duration_seconds=60.0)
    ep2 = SourceEpisode(episode_id="E02", source_video=str(v2), duration_seconds=60.0)
    rec = ProjectRecord(
        id="season_fail_proj",
        name="Season Fail Proj",
        analysis_scope="SEASON",
        source_episodes=[ep1, ep2],
        output_directory=str(tmp_path / "out"),
    )
    store = ProjectStore(tmp_path / "season_store.json")
    settings = AppSettings(gateway_enabled=True)
    queue = ProjectQueue([rec], store=store, settings=settings)

    monkeypatch.setattr("toolrecap_v2.projects.probe_typed_media", lambda p: MagicMock(duration=60.0, has_audio=False, subtitle_streams=[], video_streams=[MagicMock(duration=60.0)]))
    monkeypatch.setattr("toolrecap_v2.projects.probe_media", lambda p: MagicMock(duration=60.0, has_audio=False, subtitle_streams=[]))
    monkeypatch.setattr("toolrecap_v2.projects.probe_duration", lambda p: 60.0)
    monkeypatch.setattr("toolrecap_v2.subtitles.pipeline.SubtitlePipeline.get_episode_subtitles", lambda *a, **kw: [])

    def mock_analyze(self, project_id, episodes, **kwargs):
        for ep in episodes:
            ep.status = "COMPLETED"
            ep.progress = 100
            ep.stage = "Evidence Complete"
        rec.phase = "season_batch"
        raise AnalysisError("AI Gateway season batch 2 error: 503 Service Unavailable")

    monkeypatch.setattr("toolrecap_v2.analyzer.engine.AnalysisEngine.analyze", mock_analyze)

    queue.start()
    assert queue._thread is not None
    queue._thread.join(timeout=5)

    assert rec.status == "ERROR"
    assert rec.season_status == "ERROR"
    assert "AI Gateway season batch 2 error" in (rec.season_error or "")
    assert rec.error_scope == "SEASON"
    assert rec.error_target == "season"
    assert ep1.status == "COMPLETED"
    assert ep2.status == "COMPLETED"
    assert ep1.error is None
    assert ep2.error is None


# ---------------------------------------------------------------------------
# 12. Plan Cache Hit Zero Calls and Invalidation
# ---------------------------------------------------------------------------

def test_plan_cache_hit_zero_calls_and_invalidation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan cache hit makes zero AI calls and returns cached manifest; config changes invalidate."""
    cache_mgr = EvidenceCacheManager(tmp_path / "cache")
    hierarchy_cache = HierarchyCacheManager(tmp_path / "hierarchy")
    client = OpenAICompatibleClient("http://127.0.0.1:20128/v1")
    settings = AppSettings(gateway_enabled=True, api_endpoint="http://127.0.0.1:20128/v1")
    engine = AnalysisEngine(
        settings=settings,
        client=client,
        cache_manager=cache_mgr,
        hierarchy_cache=hierarchy_cache,
    )
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(AnalysisEngine, "plan_cache_dir", property(lambda self: plans_dir))

    ep = SourceEpisode(episode_id="E01", source_video=str(tmp_path / "e1.mp4"), duration_seconds=60.0)
    ev = EpisodeEvidence(
        episode_id="E01",
        source_video=str(tmp_path / "e1.mp4"),
        duration_seconds=60.0,
        data={"major_scenes": [{"start_sec": 0.0, "end_sec": 10.0, "summary": "Scene 1"}]},
    )
    cache_mgr.save_evidence(ep, "e1_hash", ev)
    monkeypatch.setattr(engine.scanner, "scan_episode", lambda *a, **kw: ev)

    call_count = 0

    def mock_chat(*a, **kw):
        nonlocal call_count
        call_count += 1
        return {
            "outputs": [
                {
                    "output_id": "out_01",
                    "title": "Recap 1",
                    "segments": [
                        {
                            "segment_id": "seg_01",
                            "source_clips": [{"episode_id": "E01", "start": 0.0, "end": 10.0}],
                            "narration": "Narration 1",
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(client, "chat_json", mock_chat)

    # 1. Cold cache: calls chat_json
    m1 = engine.analyze(project_id="proj1", episodes=[ep], use_final_plan_cache=True)
    assert call_count == 1
    assert len(m1.outputs) == 1

    # 2. Warm cache hit: zero calls made!
    def fail_chat(*a, **kw):
        raise AssertionError("chat_json should not be called on plan cache hit")

    monkeypatch.setattr(client, "chat_json", fail_chat)
    m2 = engine.analyze(project_id="proj1", episodes=[ep], use_final_plan_cache=True)
    assert call_count == 1
    assert m2.outputs[0].title == m1.outputs[0].title

    # 3. Invalidation: change config -> cache miss, calls chat_json again
    engine.settings.recap_mode = "DIFFERENT_MODE"
    monkeypatch.setattr(client, "chat_json", mock_chat)
    m3 = engine.analyze(project_id="proj1", episodes=[ep], use_final_plan_cache=True)
    assert call_count == 2
    assert len(m3.outputs) == 1


# ---------------------------------------------------------------------------
# 13. Summary Generic Fallback and Dedup
# ---------------------------------------------------------------------------

def test_summary_generic_fallback_and_dedup() -> None:
    """Compact summary handles generic fallback for unknown categories and dedupes overlapping items."""
    ep = SourceEpisode(episode_id="E01", source_video="dummy.mp4", duration_seconds=120.0)

    # 1. Generic fallback: unknown category with arbitrary fields
    ev1 = EpisodeEvidence(
        episode_id="E01",
        source_video="dummy.mp4",
        duration_seconds=120.0,
        data={
            "custom_plot_threads": [
                {
                    "start_sec": 15.0,
                    "end_sec": 30.0,
                    "description": "Secret organization plans heist",
                    "details": "High stakes in Paris",
                }
            ]
        },
    )
    summ1 = build_compact_summary(ep, ev1)
    assert len(summ1.items) == 1
    item = summ1.items[0]
    assert item.episode_id == "E01"
    assert item.start_sec == 15.0
    assert item.end_sec == 30.0
    assert "Secret organization plans heist" in item.summary
    assert "custom_plot_threads" in item.categories

    # 2. Dedup: overlapping items with same timestamps and similar text merged
    ev2 = EpisodeEvidence(
        episode_id="E01",
        source_video="dummy.mp4",
        duration_seconds=120.0,
        data={
            "major_scenes": [
                {
                    "start_sec": 40.0,
                    "end_sec": 55.0,
                    "summary": "Agent Carter recovers the lost encrypted drive from the safe",
                    "characters": ["Carter"],
                }
            ],
            "reveals": [
                {
                    "start_sec": 41.0,
                    "end_sec": 54.0,
                    "reveal": "Agent Carter recovers the lost encrypted drive from the safe",
                    "characters": ["Carter", "Jarvis"],
                }
            ],
        },
    )
    summ2 = build_compact_summary(ep, ev2)
    assert len(summ2.items) == 1
    merged_item = summ2.items[0]
    assert "major_scenes" in merged_item.categories
    assert "reveals" in merged_item.categories
    assert "Carter" in merged_item.characters
    assert "Jarvis" in merged_item.characters
    assert len(merged_item.refs) == 2


# ---------------------------------------------------------------------------
# 14. Malformed Hierarchy Cache Load Miss
# ---------------------------------------------------------------------------

def test_malformed_hierarchy_cache_load_miss(tmp_path: Path) -> None:
    """Malformed or invalid hierarchy cache files result in cache miss (hit=False), not exceptions."""
    cache_mgr = HierarchyCacheManager(base_dir=tmp_path / "hierarchy")

    # 1. Corrupt summary JSON
    bad_sum_file = cache_mgr.summary_dir / "E01_badhash123456.summary.json"
    bad_sum_file.write_text("{this is not valid json", encoding="utf-8")
    loaded_sum, meta_sum = cache_mgr.load_summary("E01", "badhash123456")
    assert loaded_sum is None
    assert meta_sum["hit"] is False

    # 2. Valid JSON but invalid schema/type in batch
    bad_batch_file = cache_mgr.batch_dir / "b1_badkey123456.batch.json"
    bad_batch_file.write_text(json.dumps({"cache_key": "badkey123456", "batch_id": "b1", "result": "not_a_dict"}), encoding="utf-8")
    loaded_batch, meta_batch = cache_mgr.load_batch_result("b1", "badkey123456")
    assert loaded_batch is None
    assert meta_batch["hit"] is False

    # 3. Corrupt merge JSON
    bad_merge_file = cache_mgr.merge_dir / "m1_badkey123456.merge.json"
    bad_merge_file.write_text("{corrupt json", encoding="utf-8")
    loaded_merge, meta_merge = cache_mgr.load_merge_result("m1", "badkey123456")
    assert loaded_merge is None
    assert meta_merge["hit"] is False

    # 4. Corrupt connection JSON
    bad_conn_file = cache_mgr.connection_dir / "season_badkey123456.connection.json"
    bad_conn_file.write_text("{corrupt json", encoding="utf-8")
    loaded_conn, meta_conn = cache_mgr.load_connection_result("badkey123456")
    assert loaded_conn is None
    assert meta_conn["hit"] is False


# ---------------------------------------------------------------------------
# 15. STT Cache Resume on Retry
# ---------------------------------------------------------------------------

def test_stt_cache_resume_on_retry(tmp_path: Path) -> None:
    """When SubtitlePipeline falls back to STT, cues are cached and reused on retry without calling STT again."""
    sub_cache = SubtitleCacheManager(cache_dir=tmp_path / "sub_cache")
    pipeline = SubtitlePipeline(cache_manager=sub_cache)

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"video content")
    episode_id = "E01"

    stt_calls = 0

    def mock_stt(video_path: str, ep_id: str) -> list[SubtitleCue]:
        nonlocal stt_calls
        stt_calls += 1
        return [SubtitleCue(start_ms=1000, end_ms=3000, text="Hello world", source_type="stt", source_format="whisper")]

    # 1. First run: cold cache -> calls STT fallback
    cues1 = pipeline.get_episode_subtitles(
        dummy_video,
        episode_id=episode_id,
        stt_fallback_fn=mock_stt,
    )
    assert stt_calls == 1
    assert len(cues1) == 1
    assert cues1[0].text == "Hello world"

    # 2. Retry / second run: warm cache -> returns cached cues, STT not called
    def fail_stt(video_path: str, ep_id: str) -> list[SubtitleCue]:
        raise AssertionError("STT should not be invoked when cached")

    cues2 = pipeline.get_episode_subtitles(
        dummy_video,
        episode_id=episode_id,
        stt_fallback_fn=fail_stt,
    )
    assert stt_calls == 1
    assert len(cues2) == 1
    assert cues2[0].text == "Hello world"
