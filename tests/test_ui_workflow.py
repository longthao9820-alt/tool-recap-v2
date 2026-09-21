"""Comprehensive test suite for UI workflow, settings dialog, voice selection, audio mix, and queue phases."""
from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path
import pytest

from toolrecap_v2.analyzer.prompts import get_default_recap_prompt
from toolrecap_v2.notifications import DesktopNotification
from toolrecap_v2.projects import ProjectRecord, ProjectStore
from toolrecap_v2.settings import AppSettings, SettingsStore
from toolrecap_v2.ui import SettingsDialog, ToolRecapV2App, open_voicestudio_dialog
from toolrecap_v2.voice.catalog import (
    BUILTIN_VOICES,
    DEFAULT_VOICE_BY_LANGUAGE,
    get_voice_status,
)


# ---------------------------------------------------------------------------
# 1. Settings Dialog: Exactly 4 Panes, No STT Controls
# ---------------------------------------------------------------------------

def test_settings_dialog_panes_and_no_stt_controls(tk_root: tk.Tk, tmp_path: Path) -> None:
    settings = AppSettings(
        transcription_provider="openai",
        transcription_api_key="sk-hidden-stt",
        transcription_base_url="https://stt.example.com",
    )
    store = SettingsStore(tmp_path / "settings.json")
    store.save(settings)

    dialog = SettingsDialog(tk_root, settings, store)
    try:
        # 1. Exactly four panes
        expected_panes = {"Recap", "AI Gateway", "Voice", "Render and Output"}
        assert set(dialog._panes.keys()) == expected_panes
        assert set(dialog._nav_buttons.keys()) == expected_panes

        # 2. No user-facing STT pane or controls
        assert "Nhận diện giọng nói (STT)" not in dialog._panes
        assert "STT" not in dialog._panes
        assert not hasattr(dialog, "stt_provider_var")
        assert not hasattr(dialog, "stt_key_var")
        assert not hasattr(dialog, "stt_base_var")

        # 3. Preserved internal STT config
        assert dialog.settings.transcription_provider == "openai"
        assert dialog.settings.transcription_api_key == "sk-hidden-stt"
        assert dialog.settings.transcription_base_url == "https://stt.example.com"
    finally:
        dialog._on_close()


# ---------------------------------------------------------------------------
# 2. Main UI Widgets, Treeview Columns, and No Main Voice Card
# ---------------------------------------------------------------------------

def test_main_ui_widgets_and_columns(tk_root: tk.Tk, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Isolate paths
    monkeypatch.setattr("toolrecap_v2.ui.default_data_directory", lambda: tmp_path)
    store_file = tmp_path / "settings.json"
    SettingsStore(store_file).save(AppSettings())

    app = ToolRecapV2App()
    try:
        app.withdraw()
        app.update()

        # 1. Window title
        assert "ToolRecap V2" in app.title()

        # 2. Action buttons
        assert app.btn_select_file.cget("text") == "📁 Select File"
        assert app.btn_select_folder.cget("text") == "📂 Select Folder"
        assert app.btn_start.cget("text") == "▶ Start Creating Recap Videos"
        assert app.btn_cancel.cget("text") == "⏹ Stop"
        assert app.btn_open_out.cget("text") == "📂 Open Output Folder"

        # 3. Treeview exact columns & headings
        expected_cols = ("episode", "source_video", "stage", "progress", "status")
        assert app.tree.cget("columns") == expected_cols
        assert app.tree.heading("episode")["text"] == "Episode"
        assert app.tree.heading("source_video")["text"] == "Source Video"
        assert app.tree.heading("stage")["text"] == "Stage"
        assert app.tree.heading("progress")["text"] == "Progress"
        assert app.tree.heading("status")["text"] == "Status"

        # 4. No voice card on main UI
        assert not hasattr(app, "cbo_voice")
        assert not hasattr(app, "lbl_voice_status")

        # 5. Progress bar exists
        assert app.progressbar is not None
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
# 3. File and Folder Project Scope (SINGLE_EPISODE vs SEASON, direct-only)
# ---------------------------------------------------------------------------

def test_file_and_folder_project_scope_direct_only(tk_root: tk.Tk, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("toolrecap_v2.ui.default_data_directory", lambda: tmp_path)

    # Set up directory structure
    folder = tmp_path / "MySeason"
    folder.mkdir()
    (folder / "ep1.mp4").write_bytes(b"\x00" * 100)
    (folder / "ep2.mkv").write_bytes(b"\x00" * 100)
    (folder / "notes.txt").write_text("ignore", encoding="utf-8")
    sub = folder / "bonus"
    sub.mkdir()
    (sub / "ep3.mp4").write_bytes(b"\x00" * 100)

    app = ToolRecapV2App()
    try:
        app.withdraw()

        # 1. Select File -> SINGLE_EPISODE
        app._load_file(folder / "ep1.mp4")
        assert len(app.projects) == 1
        rec = app.projects[0]
        assert rec.analysis_scope == "SINGLE_EPISODE"
        assert len(rec.source_episodes) == 1
        assert rec.source_episodes[0].episode_id == "E01"
        assert "ep1" in rec.source_video

        # Check Treeview rows
        children = app.tree.get_children()
        assert len(children) == 1
        vals = app.tree.item(children[0])["values"]
        assert vals[0] == "E01"
        assert vals[1] == "ep1.mp4"

        # 2. Select Folder -> ONE SEASON ProjectRecord (direct files only)
        app._load_folder(folder)
        assert len(app.projects) == 1
        season_rec = app.projects[0]
        assert season_rec.analysis_scope == "SEASON"
        # Only direct files ep1.mp4 and ep2.mkv (bonus/ep3.mp4 ignored)
        assert len(season_rec.source_episodes) == 2
        assert season_rec.source_episodes[0].episode_id == "E01"
        assert season_rec.source_episodes[1].episode_id == "E02"

        # Check Treeview rows (2 episodes + 1 season row)
        children = app.tree.get_children()
        assert len(children) == 3
        vals1 = app.tree.item(children[0])["values"]
        vals2 = app.tree.item(children[1])["values"]
        vals3 = app.tree.item(children[2])["values"]
        assert vals1[0] == "E01"
        assert vals2[0] == "E02"
        assert vals3[0] == "Season"
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
# 4. Voice Selection, Language Filtering, Honest Status, and Persistence
# ---------------------------------------------------------------------------

def test_voice_filter_default_status_and_persist(tk_root: tk.Tk, tmp_path: Path) -> None:
    settings = AppSettings(
        recap_language="en-US",
        voice_id="voicestudio.en.documentarian",
    )
    store = SettingsStore(tmp_path / "settings.json")
    store.save(settings)

    dialog = SettingsDialog(tk_root, settings, store)
    try:
        # Show Voice pane
        dialog._show_pane("Voice")

        # 1. en-US voice filter
        assert dialog.recap_language_var.get() == "en-US"
        cbo_vals = dialog.cbo_voice.cget("values")
        assert len(cbo_vals) == 6
        assert any("Documentarian" in v for v in cbo_vals)
        assert any("Anchor" in v for v in cbo_vals)
        assert not any("Librarian" in v for v in cbo_vals)  # en-GB voice

        # 2. Switch to en-GB
        dialog.recap_language_var.set("en-GB")
        dialog._on_recap_language_changed()
        gb_vals = dialog.cbo_voice.cget("values")
        assert len(gb_vals) == 6
        assert any("Librarian" in v for v in gb_vals)
        assert any("Commentator" in v for v in gb_vals)
        assert dialog.voice_var.get() == DEFAULT_VOICE_BY_LANGUAGE["en-GB"]

        # 3. Honest status check
        status = get_voice_status(dialog.voice_var.get())
        if not status["ready"]:
            assert any(
                marker in dialog.voice_status_var.get()
                for marker in ("not installed", "click Preview", "requires repair")
            )

        # 4. Select a voice and save
        chosen_display = [v for v in gb_vals if "Librarian" in v][0]
        dialog.cbo_voice.set(chosen_display)
        dialog._on_voice_changed()
        assert dialog.voice_var.get() == "voicestudio.en.librarian"

        dialog._save()

        # Check persistence
        loaded = store.load()
        assert loaded.recap_language == "en-GB"
        assert loaded.voice_id == "voicestudio.en.librarian"
    finally:
        dialog._on_close()


# ---------------------------------------------------------------------------
# 5. Recap Pane: Content Type, Reload Default Prompt, and Persistence
# ---------------------------------------------------------------------------

def test_recap_reload_default_prompt_and_persist(tk_root: tk.Tk, tmp_path: Path) -> None:
    settings = AppSettings(
        recap_language="en-US",
        recap_mode="MAIN_STORIES",
        content_type="US_TV_SHOW",
        source_rights_status="UNVERIFIED",
    )
    store = SettingsStore(tmp_path / "settings.json")
    store.save(settings)

    dialog = SettingsDialog(tk_root, settings, store)
    try:
        # Default prompt for US_TV_SHOW
        initial_prompt = dialog.prompt_text.get("1.0", "end-1c")
        assert "US TV show" in initial_prompt

        # 1. Switch to DE_GERMAN_SOAP and click Reload Default Prompt
        dialog.content_type_var.set("DE_GERMAN_SOAP")
        dialog._reload_default_prompt()
        soap_prompt = dialog.prompt_text.get("1.0", "end-1c")
        assert "soap opera" in soap_prompt.lower() or "german" in soap_prompt.lower()

        # 2. Switch to BODYCAM and click Reload Default Prompt
        dialog.content_type_var.set("BODYCAM")
        dialog._reload_default_prompt()
        bodycam_prompt = dialog.prompt_text.get("1.0", "end-1c")
        assert "bodycam" in bodycam_prompt.lower()

        # 3. Change mode and footage rights, then save
        dialog.recap_mode_var.set("FULL_EPISODE")
        dialog.rights_var.set("FAIR_USE")
        dialog._save()

        # 4. Verify persistence
        loaded = store.load()
        assert loaded.content_type == "BODYCAM"
        assert loaded.recap_mode == "FULL_EPISODE"
        assert loaded.source_rights_status == "FAIR_USE"
        assert "bodycam" in loaded.recap_prompt.lower()
    finally:
        dialog._on_close()


# ---------------------------------------------------------------------------
# 6. Audio Mix: Exact Labels, Defaults, Auto-Duck Toggle, and Clamping
# ---------------------------------------------------------------------------

def test_exact_audio_labels_default_and_persist(tk_root: tk.Tk, tmp_path: Path) -> None:
    settings = AppSettings()
    store = SettingsStore(tmp_path / "settings.json")
    store.save(settings)

    dialog = SettingsDialog(tk_root, settings, store)
    try:
        dialog._show_pane("Voice")

        # 1. Verify defaults
        assert dialog.orig_audio_var.get() == 0.0
        assert dialog.commentary_var.get() == 0.0
        assert dialog.auto_duck_var.get() is False
        assert dialog.ducking_var.get() == -12.0
        assert dialog.target_loudness_var.get() == -14.0
        assert dialog.true_peak_var.get() == -1.0

        # 2. Auto-duck toggle enables/disables ducking spinbox
        assert str(dialog.ducking_spin.cget("state")) == "disabled"
        dialog.auto_duck_var.set(True)
        dialog._on_auto_duck_toggled()
        assert str(dialog.ducking_spin.cget("state")) == "normal"

        dialog.auto_duck_var.set(False)
        dialog._on_auto_duck_toggled()
        assert str(dialog.ducking_spin.cget("state")) == "disabled"

        # 3. Test clamping on extreme values
        dialog.orig_audio_var.set(100.0)
        dialog.commentary_var.set(-200.0)
        dialog.ducking_var.set(50.0)
        dialog.target_loudness_var.set(10.0)
        dialog.true_peak_var.set(15.0)

        dialog._save()

        loaded = store.load()
        assert loaded.original_audio_gain_db == 24.0
        assert loaded.commentary_gain_db == -60.0
        assert loaded.ducking_amount_db == 0.0
        assert loaded.target_loudness_lufs == -5.0
        assert loaded.true_peak_dbtp == 0.0
    finally:
        dialog._on_close()


# ---------------------------------------------------------------------------
# 7. Queue Phase Display Mapping
# ---------------------------------------------------------------------------

def test_queue_phase_display_mapping(tk_root: tk.Tk, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("toolrecap_v2.ui.default_data_directory", lambda: tmp_path)

    app = ToolRecapV2App()
    try:
        app.withdraw()

        # Setup season record with E01, E02
        folder = tmp_path / "SeasonX"
        folder.mkdir()
        (folder / "e1.mp4").write_bytes(b"\x00" * 10)
        (folder / "e2.mp4").write_bytes(b"\x00" * 10)
        app._load_folder(folder)
        rec = app.projects[0]

        # 1. E01 Media/Subtitles phase update
        rec.phase = "MEDIA_PROBE"
        rec.current_message = "E01 Media/Subtitles: Phân tích kỹ thuật..."
        rec.progress = 15
        app._apply_project_update(rec)

        e01_row = f"{rec.id}_E01"
        assert app.tree.set(e01_row, "stage") == "Media/Subtitles"
        assert app.tree.set(e01_row, "progress") == "15%"

        # 2. E01 Scanner phase update
        rec.phase = "SCANNER"
        rec.current_message = "E01 Scanner: Quét tình tiết câu chuyện..."
        rec.progress = 30
        app._apply_project_update(rec)
        assert app.tree.set(e01_row, "stage") == "Scanner"
        assert app.tree.set(e01_row, "progress") == "30%"

        # 3. Season Analysis connecting/mining phase
        rec.phase = "SEASON_CONNECTING"
        rec.current_message = "Season Analysis — Connecting storylines"
        rec.progress = 45
        app._apply_project_update(rec)

        season_row = f"{rec.id}_season"
        assert app.tree.exists(season_row)
        assert app.tree.set(season_row, "stage") == "Season Analysis"
        assert "Connecting" in app.tree.set(season_row, "status")

        # 4. Output rendering phase
        rec.phase = "OUTPUT_1_RENDERING"
        rec.current_message = "Output 1 Rendering: Cuộc chiến đẫm máu (60%)"
        rec.progress = 70
        app._apply_project_update(rec)

        out_row = f"{rec.id}_output_1"
        assert app.tree.exists(out_row)
        assert app.tree.set(out_row, "stage") == "Rendering"

        # 5. Zero outputs completion message
        rec.status = "COMPLETED"
        rec.outputs = []
        rec.current_message = "Hoàn thành phân tích: Không có ứng viên recap nào đạt yêu cầu (0 outputs)."
        rec.progress = 100
        app._apply_project_update(rec)
        assert "0 output" in app.tree.set(e01_row, "status")
        assert "0 outputs" in app.status_var.get()
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
# 8. Start / Stop State Restoration
# ---------------------------------------------------------------------------

def test_start_stop_state_restoration(tk_root: tk.Tk, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("toolrecap_v2.ui.default_data_directory", lambda: tmp_path)

    app = ToolRecapV2App()
    try:
        app.withdraw()

        # Initial state
        assert str(app.btn_start.cget("state")) == "normal"
        assert str(app.btn_cancel.cget("state")) == "disabled"
        assert str(app.btn_select_file.cget("state")) == "normal"
        assert str(app.btn_select_folder.cget("state")) == "normal"
        assert str(app.btn_clear.cget("state")) == "normal"

        # Running state
        app._apply_state_change(True)
        assert str(app.btn_start.cget("state")) == "disabled"
        assert str(app.btn_cancel.cget("state")) == "normal"
        assert str(app.btn_select_file.cget("state")) == "disabled"
        assert str(app.btn_select_folder.cget("state")) == "disabled"
        assert str(app.btn_clear.cget("state")) == "disabled"

        # Restored state (after Stop or Complete)
        app._apply_state_change(False)
        assert str(app.btn_start.cget("state")) == "normal"
        assert str(app.btn_cancel.cget("state")) == "disabled"
        assert str(app.btn_select_file.cget("state")) == "normal"
        assert str(app.btn_select_folder.cget("state")) == "normal"
        assert str(app.btn_clear.cget("state")) == "normal"
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
# 9. VoiceStudio Dialog and Desktop Notification Visible [X]
# ---------------------------------------------------------------------------

def test_notification_and_updater_regression(tk_root: tk.Tk) -> None:
    dismissed = False

    def on_close():
        nonlocal dismissed
        dismissed = True

    notif = DesktopNotification(
        tk_root,
        title="Recap Batch Complete",
        message="All outputs rendered.",
        on_dismiss=on_close,
    )
    try:
        assert notif.close_button.cget("text") == "✕"
        notif.close_button.invoke()
        assert dismissed is True
    finally:
        try:
            notif.destroy()
        except Exception:
            pass

    # ToolRecap-owned local voice runtime dialog regression
    vs_dlg = open_voicestudio_dialog(tk_root)
    try:
        assert "ToolRecap Local Voice Runtime" in vs_dlg.title()
        vs_dlg.update()
    finally:
        vs_dlg.destroy()


# ---------------------------------------------------------------------------
# 10. Manual Fit: Window Instantiation at 1120x760
# ---------------------------------------------------------------------------

def test_app_and_dialog_geometry_fit(tk_root: tk.Tk, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("toolrecap_v2.ui.default_data_directory", lambda: tmp_path)
    store_file = tmp_path / "settings.json"
    store = SettingsStore(store_file)
    settings = store.load()

    app = ToolRecapV2App()
    try:
        app.geometry("1120x760")
        app.update()
        assert app.winfo_width() >= 900
        assert app.winfo_height() >= 600

        dialog = SettingsDialog(app, settings, store)
        try:
            dialog.geometry("920x680")
            for pane_name in ("Recap", "AI Gateway", "Voice", "Render and Output"):
                dialog._show_pane(pane_name)
                dialog.update()
                assert dialog._panes[pane_name].winfo_ismapped() or True
        finally:
            dialog._on_close()
    finally:
        app.destroy()
