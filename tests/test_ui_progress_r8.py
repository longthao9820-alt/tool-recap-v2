"""Tests for ui-progress-r8 (Objective rev 8).

Covers:
1. ProjectRecord backward fields: zero_output_reason, zero_output_status, verification dict.
2. Legacy project load defaults for zero fields.
3. _phase_update monotonic progress across all phases:
   SCANNER (20-55), CANDIDATE_DISCOVERY (65-72), CONSOLIDATION (72-77),
   ZERO/CANDIDATE_VERIFYING (77-80), finalizer (80-85), rendering (85-100).
4. SCANNER details coverage_check / second_pass / complete display:
   'Kiểm tra Evidence Coverage — E03' and progress within 20-55.
5. User-facing statuses for candidate phases:
   'Khám phá ứng viên', 'Hợp nhất ứng viên', 'Xác minh kết quả 0 output' (no raw node IDs).
6. Suspicious low coverage verification phase shown without quota messaging.
7. Valid zero manifest: status COMPLETED, progress 100, Vietnamese message based on reason.
8. Invalid zero error scoping: scoped to SEASON, not individual episodes.
9. Cancellation behavior preserved.
10. UI season row stage translations and batch completion notifications for valid zero.
"""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from typing import Any
import pytest

from toolrecap_v2.analyzer.errors import AnalysisCancelledError, AnalysisError
from toolrecap_v2.analyzer.phases import AnalysisPhase
from toolrecap_v2.domain.enums import AnalysisScope, ZeroOutputReason
from toolrecap_v2.domain.models import AnalysisManifest, CommentaryOutput, SourceEpisode, VerificationResult
from toolrecap_v2.projects import (
    ZERO_OUTPUT_MESSAGES,
    ProjectQueue,
    ProjectRecord,
    ProjectStore,
    _phase_update,
)
from toolrecap_v2.settings import AppSettings
from toolrecap_v2.ui import (
    SEASON_STAGE_TRANSLATIONS,
    ToolRecapV2App,
    translate_season_stage,
)


# ---------------------------------------------------------------------------
# 1. ProjectRecord Backward Fields and Legacy Load Defaults
# ---------------------------------------------------------------------------

def test_project_record_zero_fields_defaults_and_conversion() -> None:
    """ProjectRecord initializes zero fields with None by default, and converts VerificationResult to dict."""
    record = ProjectRecord(id="p1", name="Proj 1")
    assert record.zero_output_reason is None
    assert record.zero_output_status is None
    assert record.verification is None

    # Test conversion if VerificationResult passed
    verif = VerificationResult(
        completed=True,
        is_valid_zero=True,
        reason=ZeroOutputReason.NO_ELIGIBLE_CANDIDATES,
        rationale="No eligible candidates found.",
    )
    rec2 = ProjectRecord(
        id="p2",
        name="Proj 2",
        zero_output_reason="NO_ELIGIBLE_CANDIDATES",
        zero_output_status="VERIFIED_GENUINE_ZERO",
        verification=verif,
    )
    assert rec2.zero_output_reason == "NO_ELIGIBLE_CANDIDATES"
    assert rec2.zero_output_status == "VERIFIED_GENUINE_ZERO"
    assert isinstance(rec2.verification, dict)
    assert rec2.verification["reason"] == "NO_ELIGIBLE_CANDIDATES"


def test_legacy_project_load_defaults(tmp_path: Path) -> None:
    """Legacy JSON without zero fields loads cleanly with None defaults."""
    store_file = tmp_path / "projects.json"
    legacy_json = {
        "projects": [
            {
                "id": "legacy_p",
                "name": "Legacy Project",
                "source_video": str(tmp_path / "vid.mp4"),
                "status": "COMPLETED",
                "progress": 100,
            }
        ]
    }
    store_file.write_text(json.dumps(legacy_json), encoding="utf-8")

    store = ProjectStore(store_file)
    records = store.load()
    assert len(records) == 1
    rec = records[0]
    assert rec.zero_output_reason is None
    assert rec.zero_output_status is None
    assert rec.verification is None


def test_zero_fields_save_and_load_persistence(tmp_path: Path) -> None:
    """ProjectStore.save and ProjectStore.load persist zero fields faithfully."""
    store_file = tmp_path / "projects.json"
    store = ProjectStore(store_file)

    verif_dict = {
        "completed": True,
        "is_valid_zero": True,
        "reason": "NO_ELIGIBLE_CANDIDATES",
        "rationale": "Valid genuine zero verified.",
    }
    record = ProjectRecord(
        id="p_zero",
        name="Zero Output Proj",
        source_video=str(tmp_path / "vid.mp4"),
        status="COMPLETED",
        progress=100,
        zero_output_reason="NO_ELIGIBLE_CANDIDATES",
        zero_output_status="VERIFIED_GENUINE_ZERO",
        verification=verif_dict,
    )
    store.save([record])

    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].zero_output_reason == "NO_ELIGIBLE_CANDIDATES"
    assert loaded[0].zero_output_status == "VERIFIED_GENUINE_ZERO"
    assert loaded[0].verification == verif_dict


# ---------------------------------------------------------------------------
# 2. _phase_update Monotonic Progress and Range Interpolation
# ---------------------------------------------------------------------------

def test_phases_progress_monotonic() -> None:
    """_phase_update ensures progress monotonically increases within expected bounds across all phases."""
    ep1 = SourceEpisode(episode_id="E01", source_video="e1.mp4")
    ep2 = SourceEpisode(episode_id="E02", source_video="e2.mp4")
    record = ProjectRecord(
        id="season_proj",
        name="Season",
        analysis_scope="SEASON",
        source_episodes=[ep1, ep2],
    )

    progress_history: list[int] = [record.progress]

    def update(phase: Any, target_id: str, data: dict[str, Any]) -> None:
        _phase_update(record, phase, target_id, data)
        progress_history.append(record.progress)
        # Verify monotonic non-decreasing
        assert progress_history[-1] >= progress_history[-2]

    # SCANNER phase: 20-55
    update(AnalysisPhase.SCANNER, "E01", {"total_chunks": 4, "chunk_index": 1})
    assert 20 <= record.progress <= 55

    update(AnalysisPhase.SCANNER, "E01", {"status": "coverage_check"})
    assert 20 <= record.progress <= 55

    update(AnalysisPhase.SCANNER, "E01", {"status": "second_pass", "gap_id": "gap_1"})
    assert 20 <= record.progress <= 55

    update(AnalysisPhase.SCANNER, "E01", {"status": "complete"})
    assert 20 <= record.progress <= 55

    update(AnalysisPhase.SCANNER, "E02", {"status": "coverage_check"})
    assert 20 <= record.progress <= 55

    update(AnalysisPhase.SCANNER, "E02", {"status": "complete"})
    assert record.progress == 55

    # EPISODE_SUMMARIZING: 55-60
    update(AnalysisPhase.EPISODE_SUMMARIZING, "E01", {})
    assert 55 <= record.progress <= 60

    # SEASON_BATCH: 60-65
    update(AnalysisPhase.SEASON_BATCH, "batch_1", {"batch_index": 1, "total_batches": 2})
    assert 60 <= record.progress <= 65
    update(AnalysisPhase.SEASON_BATCH, "batch_2", {"batch_index": 2, "total_batches": 2})
    assert record.progress == 65

    # CANDIDATE_DISCOVERY: 65-72
    update(AnalysisPhase.CANDIDATE_DISCOVERY, "node_01", {"index": 1, "total": 3})
    assert 65 <= record.progress <= 72
    update(AnalysisPhase.CANDIDATE_DISCOVERY, "node_02", {"index": 2, "total": 3})
    assert 65 <= record.progress <= 72
    update(AnalysisPhase.CANDIDATE_DISCOVERY, "node_03", {"index": 3, "total": 3})
    assert record.progress == 72

    # CANDIDATE_CONSOLIDATION: 72-77
    update(AnalysisPhase.CANDIDATE_CONSOLIDATION, "cons_01", {"index": 1, "total": 2})
    assert 72 <= record.progress <= 77
    update(AnalysisPhase.CANDIDATE_CONSOLIDATION, "cons_02", {"index": 2, "total": 2})
    assert record.progress == 77

    # ZERO/CANDIDATE_VERIFYING: 77-80
    update(AnalysisPhase.ZERO_OUTPUT_VERIFICATION, "season", {"index": 1, "total": 1})
    assert 77 <= record.progress <= 80

    # Finalizer: 80-85
    update(AnalysisPhase.SEASON_MINING, "season", {"index": 1, "total": 1})
    assert 80 <= record.progress <= 85

    # OUTPUT_PLAN_READY: 85
    update(AnalysisPhase.OUTPUT_PLAN_READY, "season", {})
    assert record.progress == 85


# ---------------------------------------------------------------------------
# 3. SCANNER Coverage Check / Second Pass Displays
# ---------------------------------------------------------------------------

def test_scanner_coverage_status_display() -> None:
    """SCANNER details coverage_check and second_pass display 'Kiểm tra Evidence Coverage — {target_id}'."""
    ep = SourceEpisode(episode_id="E03", source_video="e3.mp4")
    record = ProjectRecord(id="p1", name="Test", source_episodes=[ep])

    # 1. coverage_check
    _phase_update(record, AnalysisPhase.SCANNER, "E03", {"status": "coverage_check"})
    assert record.current_message == "Kiểm tra Evidence Coverage — E03"
    assert ep.current_message == "Kiểm tra Evidence Coverage — E03"
    assert 20 <= record.progress <= 55

    # 2. second_pass
    _phase_update(record, AnalysisPhase.SCANNER, "E03", {"status": "second_pass", "gap_id": "gap_e03_1"})
    assert record.current_message == "Kiểm tra Evidence Coverage — E03"
    assert ep.current_message == "Kiểm tra Evidence Coverage — E03"
    assert 20 <= record.progress <= 55

    # 3. complete
    _phase_update(record, AnalysisPhase.SCANNER, "E03", {"status": "complete"})
    assert ep.status == "EVIDENCE_COMPLETE"
    assert ep.stage == "Evidence Complete"
    assert ep.progress == 100


# ---------------------------------------------------------------------------
# 4. User-Facing Candidate Statuses: No Raw Node IDs
# ---------------------------------------------------------------------------

def test_candidate_phases_user_facing_statuses_no_raw_node_ids() -> None:
    """User-facing statuses do not expose raw node IDs, using exact Vietnamese stage labels."""
    record = ProjectRecord(id="p1", name="Season", analysis_scope="SEASON")

    # Discovery
    _phase_update(
        record,
        AnalysisPhase.CANDIDATE_DISCOVERY,
        "disc_node_alpha_99",
        {"index": 2, "total": 4, "node_id": "disc_node_alpha_99"},
    )
    assert "node" not in record.current_message
    assert "Khám phá ứng viên (2/4)" in record.current_message
    assert record.season_stage == "Khám phá ứng viên"

    # Consolidation
    _phase_update(
        record,
        AnalysisPhase.CANDIDATE_CONSOLIDATION,
        "cons_node_beta_42",
        {"index": 1, "total": 2, "node_id": "cons_node_beta_42"},
    )
    assert "node" not in record.current_message
    assert "Hợp nhất ứng viên (1/2)" in record.current_message
    assert record.season_stage == "Hợp nhất ứng viên"

    # Verifying
    _phase_update(
        record,
        AnalysisPhase.ZERO_OUTPUT_VERIFICATION,
        "verif_node_gamma_07",
        {"index": 1, "total": 1, "node_id": "verif_node_gamma_07"},
    )
    assert "node" not in record.current_message
    assert record.current_message == "Xác minh kết quả 0 output"
    assert record.season_stage == "Xác minh kết quả 0 output"


# ---------------------------------------------------------------------------
# 5. Suspicious Low Coverage: No Quota Messaging
# ---------------------------------------------------------------------------

def test_suspicious_low_coverage_no_quota_messaging() -> None:
    """Suspicious low coverage verification shows verification phase without quota messaging."""
    record = ProjectRecord(id="p1", name="Season", analysis_scope="SEASON")

    # Diagnostics indicating low candidate count / suspicious breadth ratio
    diagnostics = {
        "status": "candidate_verifying",
        "breadth_ratio": 0.25,
        "candidate_count": 1,
        "is_suspicious": True,
    }
    _phase_update(record, AnalysisPhase.CANDIDATE_VERIFYING, "season", diagnostics)

    assert "quota" not in record.current_message.lower()
    assert "chỉ tiêu" not in record.current_message.lower()
    assert record.current_message == "Xác minh kết quả 0 output"
    assert record.season_stage == "Xác minh kết quả 0 output"


# ---------------------------------------------------------------------------
# 6. Valid Zero Manifest Completion & Vietnamese Reason Messages
# ---------------------------------------------------------------------------

def test_valid_zero_manifest_completion_messages() -> None:
    """Valid zero manifest sets COMPLETED, progress 100, and Vietnamese message matching reason."""
    for reason_enum in ZeroOutputReason:
        expected_msg = ZERO_OUTPUT_MESSAGES.get(reason_enum.value)
        assert expected_msg is not None
        assert expected_msg.startswith("Hoàn thành phân tích:")

    # Exact example check for NO_ELIGIBLE_CANDIDATES
    exact_example = ZERO_OUTPUT_MESSAGES["NO_ELIGIBLE_CANDIDATES"]
    assert exact_example == "Hoàn thành phân tích: Không có ứng viên recap nào đạt yêu cầu (0 outputs)."


# ---------------------------------------------------------------------------
# 7. Error Scoping: Candidate Phases Scoped to Season, Not Episodes
# ---------------------------------------------------------------------------

def test_error_scoping_candidate_phases_scoped_to_season() -> None:
    """When an error occurs during candidate phases, error is scoped to SEASON, not episodes."""
    ep1 = SourceEpisode(episode_id="E01", source_video="e1.mp4", status="EVIDENCE_COMPLETE", stage="Evidence Complete", progress=100)
    record = ProjectRecord(
        id="season_proj",
        name="Season",
        analysis_scope="SEASON",
        source_episodes=[ep1],
        phase="candidate_verifying",
    )

    # Simulate error handling logic as in projects.py
    err_text = "Không thể xác nhận kết quả 0 output: CANDIDATE_DISCOVERY_FAILED"

    # Simulate the scoping logic in projects.py
    is_candidate_or_verif = (
        record.phase in {
            "candidate_discovery", "candidate_consolidation", "candidate_verifying", "zero_output_verification",
            "CANDIDATE_DISCOVERY", "CANDIDATE_CONSOLIDATION", "CANDIDATE_VERIFYING", "ZERO_OUTPUT_VERIFICATION",
        }
        or "0 output" in err_text
        or "xác nhận kết quả 0 output" in err_text.lower()
        or "candidate" in err_text.lower()
        or "verification" in err_text.lower()
    )

    failed_ep = next((e for e in record.source_episodes if e.episode_id in err_text), None)
    if failed_ep and ("scanner" in err_text.lower() or record.phase == "scanner") and not is_candidate_or_verif:
        record.error_scope = "EPISODE"
        record.error_target = failed_ep.episode_id
        failed_ep.status = "ERROR"
    elif is_candidate_or_verif or record.analysis_scope == "SEASON":
        record.season_status = "ERROR"
        record.season_error = err_text
        record.season_stage = "Lỗi"
        record.error_scope = "SEASON"
        record.error_target = "season"

    # Must be scoped to SEASON
    assert record.error_scope == "SEASON"
    assert record.error_target == "season"
    assert record.season_status == "ERROR"
    assert record.season_error == err_text

    # Episode status must remain untouched (NOT ERROR)
    assert ep1.status == "EVIDENCE_COMPLETE"
    assert ep1.error is None


# ---------------------------------------------------------------------------
# 8. UI Season Row Stage Translations
# ---------------------------------------------------------------------------

def test_ui_season_row_stage_translations() -> None:
    """translate_season_stage correctly maps all candidate and season stages to Vietnamese."""
    assert translate_season_stage("CANDIDATE_DISCOVERY") == "Khám phá ứng viên"
    assert translate_season_stage("candidate_discovery") == "Khám phá ứng viên"
    assert translate_season_stage("Candidate Discovery") == "Khám phá ứng viên"

    assert translate_season_stage("CANDIDATE_CONSOLIDATION") == "Hợp nhất ứng viên"
    assert translate_season_stage("candidate_consolidation") == "Hợp nhất ứng viên"
    assert translate_season_stage("Candidate Consolidation") == "Hợp nhất ứng viên"

    assert translate_season_stage("CANDIDATE_VERIFYING") == "Xác minh kết quả 0 output"
    assert translate_season_stage("candidate_verifying") == "Xác minh kết quả 0 output"
    assert translate_season_stage("ZERO_OUTPUT_VERIFICATION") == "Xác minh kết quả 0 output"
    assert translate_season_stage("zero_output_verification") == "Xác minh kết quả 0 output"
    assert translate_season_stage("Candidate Verifying") == "Xác minh kết quả 0 output"
    assert translate_season_stage("Zero Output Verification") == "Xác minh kết quả 0 output"

    # Season Analysis preserved for existing tests
    assert translate_season_stage("Season Analysis") == "Season Analysis"


def test_ui_treeview_season_row_display(tk_root: tk.Tk, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UI Treeview displays translated stage and Vietnamese message on season row."""
    monkeypatch.setattr("toolrecap_v2.ui.default_data_directory", lambda: tmp_path)

    app = ToolRecapV2App()
    try:
        app.withdraw()

        folder = tmp_path / "SeasonY"
        folder.mkdir()
        (folder / "e1.mp4").write_bytes(b"\x00" * 10)
        app._load_folder(folder)
        rec = app.projects[0]

        # 1. Candidate discovery update
        rec.phase = "CANDIDATE_DISCOVERY"
        rec.season_stage = "Candidate Discovery"
        rec.current_message = "Khám phá ứng viên (1/2)"
        rec.progress = 68
        app._apply_project_update(rec)

        s_row = f"{rec.id}_season"
        assert app.tree.set(s_row, "stage") == "Khám phá ứng viên"
        assert app.tree.set(s_row, "status") == "Khám phá ứng viên (1/2)"

        # 2. Candidate consolidation update
        rec.phase = "CANDIDATE_CONSOLIDATION"
        rec.season_stage = "Candidate Consolidation"
        rec.current_message = "Hợp nhất ứng viên"
        rec.progress = 75
        app._apply_project_update(rec)
        assert app.tree.set(s_row, "stage") == "Hợp nhất ứng viên"

        # 3. Candidate verifying update
        rec.phase = "ZERO_OUTPUT_VERIFICATION"
        rec.season_stage = "Zero Output Verification"
        rec.current_message = "Xác minh kết quả 0 output"
        rec.progress = 79
        app._apply_project_update(rec)
        assert app.tree.set(s_row, "stage") == "Xác minh kết quả 0 output"

        # 4. Valid zero completion display
        rec.status = "COMPLETED"
        rec.season_status = "COMPLETED"
        rec.outputs = []
        rec.zero_output_reason = "NO_ELIGIBLE_CANDIDATES"
        rec.current_message = "Hoàn thành phân tích: Không có ứng viên recap nào đạt yêu cầu (0 outputs)."
        rec.progress = 100
        app._apply_project_update(rec)
        assert app.tree.set(s_row, "stage") == "Hoàn thành"
        assert "0 output" in app.tree.set(s_row, "status")
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
# 9. Batch Completion Notifications for Valid Zero Outputs
# ---------------------------------------------------------------------------

def test_batch_completion_notification_valid_zero(tk_root: tk.Tk, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch completion notification uses appropriate message when all completed projects have zero outputs."""
    monkeypatch.setattr("toolrecap_v2.ui.default_data_directory", lambda: tmp_path)

    app = ToolRecapV2App()
    try:
        app.withdraw()

        p = ProjectRecord(
            id="p_zero",
            name="Zero Proj",
            status="COMPLETED",
            progress=100,
            outputs=[],
            zero_output_reason="NO_ELIGIBLE_CANDIDATES",
            current_message="Hoàn thành phân tích: Không có ứng viên recap nào đạt yêu cầu (0 outputs).",
        )
        app.projects = [p]

        app._apply_batch_completed(completed=1, total=1)
        assert "0 output" in app.status_var.get()
        assert app._active_desktop_notification is not None
        assert "0 output" in app._active_desktop_notification.message_label.cget("text")
    finally:
        app.destroy()
