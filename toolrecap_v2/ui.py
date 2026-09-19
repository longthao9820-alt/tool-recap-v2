"""Main graphical user interface for ToolRecap V2 built with Tkinter and ttk."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from .analyzer.prompts import get_default_recap_prompt
from .api_client import OpenAICompatibleClient
from .gpu import EncoderStatus, detect_gpu_encoder, get_acceleration_plan
from .notifications import DesktopNotification, NotificationBanner, show_desktop_notification
from .paths import default_data_directory, set_window_icon
from .projects import ProjectQueue, ProjectRecord, ProjectStore
from .scanner import scan_videos
from .settings import AppSettings, SettingsStore
from .updater import (
    ReleaseInfo,
    check_for_updates,
    download_and_stage_update,
    generate_apply_script,
)
from .version import __version__
from .voice.audio_preview import AudioPreviewPlayer
from .voice.catalog import (
    BUILTIN_VOICES,
    DEFAULT_VOICE_BY_LANGUAGE,
    DEFAULT_VOICE_ID,
    SUPPORTED_VOICE_STYLES,
    STYLE_NAMES,
    get_voice_spec,
    get_voice_status,
    is_voicestudio_ready,
)
from .voice.manager import get_voice_manager


def safe_after(widget: tk.Misc | None, ms: int, func: Callable, *args: Any) -> str | None:
    """Safely schedule a callback on a tkinter widget, catching post-destroy / mainloop exceptions."""
    if widget is None:
        return None

    def _wrapped() -> None:
        try:
            if hasattr(widget, "winfo_exists") and not widget.winfo_exists():
                return
            func(*args)
        except (tk.TclError, RuntimeError):
            pass
        except Exception:
            pass

    try:
        if hasattr(widget, "winfo_exists") and not widget.winfo_exists():
            return None
        return widget.after(ms, _wrapped)
    except (tk.TclError, RuntimeError):
        return None
    except Exception:
        return None


def open_voicestudio_dialog(parent: tk.Misc) -> tk.Toplevel:
    """Open VoiceStudio Subsystem updater dialog."""
    dlg = tk.Toplevel(parent)
    dlg.title("Cập nhật VoiceStudio Subsystem")
    dlg.geometry("520x370")
    dlg.resizable(False, False)
    dlg.transient(parent)
    dlg.grab_set()
    set_window_icon(dlg)

    frame = ttk.Frame(dlg, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Hệ thống VoiceStudio (debpalash/VoiceStudio)",
        font=("Segoe UI Semibold", 11),
    ).pack(anchor="w", pady=(0, 6))

    info_box = ttk.LabelFrame(frame, text="Thông tin phiên bản", padding=10)
    info_box.pack(fill="x", pady=(0, 10))

    lbl_installed = ttk.Label(info_box, text="Đang kiểm tra...", font=("Segoe UI", 9))
    lbl_installed.pack(anchor="w", pady=2)

    lbl_supported = ttk.Label(info_box, text="Phiên bản hỗ trợ: v0.5.3 (đã ghim tương thích)", font=("Segoe UI", 9))
    lbl_supported.pack(anchor="w", pady=2)

    lbl_latest = ttk.Label(info_box, text="Bản phát hành chính thức debpalash: Đang kết nối...", font=("Segoe UI", 9))
    lbl_latest.pack(anchor="w", pady=2)

    txt_status = tk.Text(frame, height=5, font=("Segoe UI", 9), wrap="word")
    txt_status.pack(fill="both", expand=True, pady=(0, 10))
    txt_status.insert("1.0", "Đang kết nối kiểm tra trạng thái VoiceStudio...")
    txt_status.config(state="disabled")

    btn_box = ttk.Frame(frame)
    btn_box.pack(fill="x")

    btn_update = ttk.Button(btn_box, text="Cập nhật adapter", style="Primary.TButton", state="disabled")
    btn_update.pack(side="left")

    btn_close = ttk.Button(btn_box, text="Để sau / Đóng", command=dlg.destroy)
    btn_close.pack(side="right")

    def _fetch_status() -> None:
        from .voice.voice_updater import (
            apply_voicestudio_subsystem_update,
            check_voicestudio_status,
            download_and_stage_voicestudio_adapter,
        )
        status = check_voicestudio_status()

        def _update_ui() -> None:
            if not dlg.winfo_exists():
                return
            inst_text = status.installed_version or "Chưa cài đặt"
            lbl_installed.config(text=f"Phiên bản đã cài đặt: {inst_text}")
            latest_tag = status.release_info.tag_name if status.release_info else "Không xác định (ngoại tuyến)"
            lbl_latest.config(text=f"Bản phát hành chính thức debpalash: {latest_tag}")

            txt_status.config(state="normal")
            txt_status.delete("1.0", "end")
            txt_status.insert("1.0", status.status_label)
            txt_status.config(state="disabled")

            if status.is_compatible and status.has_adapter and status.adapter_asset:
                btn_update.config(state="normal")
                def _do_update() -> None:
                    btn_update.config(state="disabled", text="Đang cập nhật...")
                    def _work() -> None:
                        try:
                            staged = download_and_stage_voicestudio_adapter(status.adapter_asset)
                            apply_voicestudio_subsystem_update(staged, version=status.supported_version)
                            safe_after(parent, 0, lambda: messagebox.showinfo(
                                "Cập nhật thành công",
                                "VoiceStudio adapter đã được cập nhật thành công!",
                                parent=dlg,
                            ))
                            safe_after(parent, 0, dlg.destroy)
                        except Exception as exc:
                            safe_after(parent, 0, lambda: messagebox.showerror(
                                "Lỗi cập nhật",
                                f"Cập nhật VoiceStudio thất bại, hệ thống cũ được giữ nguyên: {exc}",
                                parent=dlg,
                            ))
                    threading.Thread(target=_work, daemon=True).start()
                btn_update.config(command=_do_update)
            else:
                btn_update.config(state="disabled")

        safe_after(parent, 0, _update_ui)

    threading.Thread(target=_fetch_status, daemon=True).start()
    return dlg


class ToolRecapV2App(tk.Tk):
    """Main desktop application window close to Toolrecap Auto V1."""

    def __init__(self) -> None:
        for attempt in range(3):
            try:
                super().__init__()
                break
            except tk.TclError:
                if attempt == 2:
                    raise
                import time
                time.sleep(0.05)
        self.title(f"ToolRecap V2 — Tự Động Hóa Video Recap (v{__version__})")
        self.geometry("1120x760")
        self.minsize(920, 680)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        set_window_icon(self)

        # Storage & State
        self.settings_store = SettingsStore()
        self.settings = self.settings_store.load()
        self.project_store = ProjectStore()
        self.projects: list[ProjectRecord] = self.project_store.load()

        self.voice_manager = get_voice_manager()
        self.preview_player = AudioPreviewPlayer()

        # Tkinter variables
        self.source_var = tk.StringVar(value="Chưa chọn video hoặc thư mục.")
        self.status_var = tk.StringVar(value="Sẵn sàng.")
        self.voice_var = tk.StringVar(value=self.settings.voice_id)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label_var = tk.StringVar(value="0%")
        self.gpu_status_var = tk.StringVar(value="Đang kiểm tra phần cứng...")
        self._active_desktop_notification: DesktopNotification | None = None

        # Initialize Project Queue
        self.queue = ProjectQueue(
            self.projects,
            store=self.project_store,
            settings=self.settings,
            on_update=self._on_project_updated,
            on_batch_complete=self._on_batch_completed,
            on_state_change=self._on_queue_state_changed,
        )

        self._build_style()
        self._build_ui()
        self._refresh_queue_table()

        # Background tasks
        safe_after(self, 200, self._detect_gpu_background)
        safe_after(self, 2000, self._check_update_background)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        theme = "vista" if "vista" in style.theme_names() else "clam"
        style.theme_use(theme)

        style.configure("Title.TLabel", font=("Segoe UI Semibold", 16))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#555555")
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 10), padding=(16, 8))
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 5))
        style.configure("Danger.TButton", font=("Segoe UI Semibold", 10), padding=(14, 8))
        style.configure("Treeview", rowheight=32, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 9), padding=6)
        style.configure("Horizontal.TProgressbar", thickness=14)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)

        # 1. Top Header
        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text="ToolRecap V2", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Tự động sản xuất video recap hoàn chỉnh chỉ với một cú nhấp chuột",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        btn_box = ttk.Frame(header)
        btn_box.grid(row=0, column=2, rowspan=2, sticky="e")
        self.update_btn = ttk.Button(btn_box, text="🔄 Kiểm tra cập nhật", command=self._manual_check_update)
        self.update_btn.pack(side="left", padx=4)
        self.settings_btn = ttk.Button(btn_box, text="⚙ Settings", command=self._open_settings_dialog)
        self.settings_btn.pack(side="left", padx=4)

        # 2. Notification Banner (with visible [X])
        self.banner = NotificationBanner(container, on_dismiss=self._on_banner_dismissed)
        self.banner.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.banner.grid_remove()

        # 3. Movie Source Section (Clean, without voice selector)
        source_box = ttk.LabelFrame(container, text="Movie Source", padding=12)
        source_box.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        source_box.columnconfigure(0, weight=1)

        src_btn_bar = ttk.Frame(source_box)
        src_btn_bar.pack(fill="x", pady=(0, 6))
        self.btn_select_file = ttk.Button(src_btn_bar, text="📁 Select File", command=self._choose_file)
        self.btn_select_file.pack(side="left", padx=(0, 6))
        self.btn_select_folder = ttk.Button(src_btn_bar, text="📂 Select Folder", command=self._choose_folder)
        self.btn_select_folder.pack(side="left")

        self.lbl_source = ttk.Label(
            source_box,
            textvariable=self.source_var,
            font=("Segoe UI", 9),
            foreground="#0969da",
            wraplength=900,
        )
        self.lbl_source.pack(fill="x")

        # 4. Source Episodes & Queue Table Frame
        queue_frame = ttk.LabelFrame(container, text="Source Episodes & Output Queue", padding=8)
        queue_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(0, weight=1)

        columns = ("episode", "source_video", "stage", "progress", "status")
        self.tree = ttk.Treeview(queue_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("episode", text="Episode")
        self.tree.heading("source_video", text="Source Video")
        self.tree.heading("stage", text="Stage")
        self.tree.heading("progress", text="Progress")
        self.tree.heading("status", text="Status")

        self.tree.column("episode", width=90, anchor="center")
        self.tree.column("source_video", width=260, anchor="w")
        self.tree.column("stage", width=150, anchor="center")
        self.tree.column("progress", width=90, anchor="center")
        self.tree.column("status", width=420, anchor="w")

        scroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        # 5. Progress Section
        prog_frame = ttk.LabelFrame(container, text="Progress", padding=(10, 6))
        prog_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        prog_frame.columnconfigure(0, weight=1)

        self.progressbar = ttk.Progressbar(prog_frame, variable=self.progress_var, maximum=100)
        self.progressbar.grid(row=0, column=0, sticky="ew")

        # 6. Action Controls Frame
        action_frame = ttk.Frame(container)
        action_frame.grid(row=5, column=0, sticky="ew", pady=(0, 8))

        self.btn_start = ttk.Button(
            action_frame,
            text="▶ Start Creating Recap Videos",
            style="Primary.TButton",
            command=self._start_batch,
        )
        self.btn_start.pack(side="left", padx=(0, 8))

        self.btn_cancel = ttk.Button(
            action_frame,
            text="⏹ Stop",
            style="Danger.TButton",
            command=self._cancel_batch,
            state="disabled",
        )
        self.btn_cancel.pack(side="left", padx=(0, 8))

        self.btn_open_out = ttk.Button(
            action_frame,
            text="📂 Open Output Folder",
            command=self._open_output_folder,
        )
        self.btn_open_out.pack(side="left")

        self.btn_clear = ttk.Button(
            action_frame,
            text="Xóa danh sách",
            command=self._clear_queue,
        )
        self.btn_clear.pack(side="right")

        # 7. Status Bar Frame
        status_bar = ttk.Frame(container, relief="sunken", padding=(8, 6))
        status_bar.grid(row=6, column=0, sticky="ew")
        status_bar.columnconfigure(1, weight=1)

        ttk.Label(status_bar, text="Trạng thái:", font=("Segoe UI Semibold", 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Label(status_bar, textvariable=self.status_var, font=("Segoe UI", 9)).grid(row=0, column=1, sticky="w")
        ttk.Label(status_bar, textvariable=self.gpu_status_var, font=("Segoe UI", 8), foreground="#555").grid(row=0, column=2, sticky="e")

    # -------------------------------------------------------------------------
    # Source Loading (Single File -> SINGLE_EPISODE, Folder -> SEASON)
    # -------------------------------------------------------------------------

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[("Video files", "*.mp4;*.mkv;*.mov;*.avi;*.webm;*.m4v;*.ts"), ("All files", "*.*")],
        )
        if path:
            self._load_file(Path(path))

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Chọn thư mục chứa video")
        if folder:
            self._load_folder(Path(folder))

    def _load_file(self, path: Path) -> None:
        path = Path(path).resolve()
        if not path.is_file():
            self.banner.show(f"File video không tồn tại: {path.name}", level="warning")
            return
        default_out = Path(self.settings.output_dir) if self.settings.output_dir else (default_data_directory() / "recaps_xuat")
        self.source_var.set(f"File: {path.name}")
        record = ProjectRecord.from_video_path(path, default_out, voice_id=self.settings.voice_id)
        self.projects = [record]
        self.queue.projects = self.projects
        self.project_store.save(self.projects)
        self._refresh_queue_table()
        self.status_var.set(f"Đã nạp file {path.name}. Nhấn 'Start Creating Recap Videos' để bắt đầu.")
        self.banner.show(f"Đã chọn file: {path.name} (SINGLE_EPISODE)", level="info")

    def _load_folder(self, path: Path) -> None:
        path = Path(path).resolve()
        videos = scan_videos(path)
        if not videos:
            self.banner.show(f"Không tìm thấy video nào được hỗ trợ trực tiếp trong: {path.name}", level="warning")
            return
        default_out = Path(self.settings.output_dir) if self.settings.output_dir else (default_data_directory() / "recaps_xuat")
        self.source_var.set(f"Folder: {path.name} ({len(videos)} video)")
        record = ProjectRecord.from_season_paths(videos, default_out, voice_id=self.settings.voice_id, title=path.name)
        self.projects = [record]
        self.queue.projects = self.projects
        self.project_store.save(self.projects)
        self._refresh_queue_table()
        self.status_var.set(f"Đã nạp mùa phim {path.name} ({len(videos)} tập). Nhấn 'Start Creating Recap Videos' để bắt đầu.")
        self.banner.show(f"Đã tìm thấy {len(videos)} tập phim trong {path.name}. Sẵn sàng phân tích mùa phim!", level="success")

    def _load_source(self, path: Path) -> None:
        """Compatibility helper routing to file or folder loader."""
        p = Path(path).resolve()
        if p.is_file():
            self._load_file(p)
        else:
            self._load_folder(p)

    def _refresh_queue_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for p in self.projects:
            for ep in p.source_episodes:
                row_id = f"{p.id}_{ep.episode_id}"
                v_name = Path(ep.source_video).name if ep.source_video else "—"
                stage_val = "Hoàn thành" if p.status == "COMPLETED" else ("Lỗi" if p.status == "ERROR" else "Sẵn sàng")
                prog_val = f"{p.progress}%" if p.status in {"COMPLETED", "RUNNING"} else "0%"
                stat_val = p.current_message if p.status in {"COMPLETED", "ERROR", "RUNNING"} else "Sẵn sàng"
                self.tree.insert(
                    "",
                    "end",
                    iid=row_id,
                    values=(ep.episode_id, v_name, stage_val, prog_val, stat_val),
                )
            if p.analysis_scope == "SEASON" and p.phase in {"SEASON_BARRIER", "SEASON_CONNECTING", "SEASON_MINING"}:
                row_id = f"{p.id}_season"
                self.tree.insert(
                    "",
                    "end",
                    iid=row_id,
                    values=("Season", "Toàn bộ mùa phim", "Season Analysis", f"{p.progress}%", p.current_message),
                )
            if p.outputs:
                for idx, out in enumerate(p.outputs, start=1):
                    row_id = f"{p.id}_output_{idx}"
                    out_name = Path(out.publication_video_path).name if getattr(out, "publication_video_path", None) else "—"
                    self.tree.insert(
                        "",
                        "end",
                        iid=row_id,
                        values=(f"Output {idx}", out_name, "Rendering" if p.status == "RUNNING" else "Hoàn thành", "100%" if p.status == "COMPLETED" else "0%", out.title),
                    )

    # -------------------------------------------------------------------------
    # Queue Callbacks and State Mapping
    # -------------------------------------------------------------------------

    def _on_project_updated(self, record: ProjectRecord) -> None:
        safe_after(self, 0, self._apply_project_update, record)

    def _apply_project_update(self, record: ProjectRecord) -> None:
        msg = record.current_message or ""
        phase = record.phase or ""

        # Update matching source episode row
        matched_ep = None
        for ep in record.source_episodes:
            if ep.episode_id in msg or (phase in {"MEDIA_PROBE", "SUBTITLES", "SCANNER"} and ep.episode_id in msg):
                matched_ep = ep
                break

        if matched_ep:
            row_id = f"{record.id}_{matched_ep.episode_id}"
            if self.tree.exists(row_id):
                if "Media/Subtitles" in msg or phase in {"MEDIA_PROBE", "SUBTITLES"}:
                    stage_name = "Media/Subtitles"
                elif "Scanner" in msg or phase == "SCANNER":
                    stage_name = "Scanner"
                else:
                    stage_name = phase
                self.tree.set(row_id, "stage", stage_name)
                self.tree.set(row_id, "progress", f"{record.progress}%")
                self.tree.set(row_id, "status", msg)

        # Dynamic season connection/mining row
        if record.analysis_scope == "SEASON" and (
            phase in {"SEASON_BARRIER", "SEASON_CONNECTING", "SEASON_MINING"}
            or "Season Analysis" in msg
        ):
            season_row_id = f"{record.id}_season"
            if not self.tree.exists(season_row_id):
                self.tree.insert(
                    "",
                    "end",
                    iid=season_row_id,
                    values=("Season", "Toàn bộ mùa phim", "Season Analysis", f"{record.progress}%", msg),
                )
            else:
                self.tree.set(season_row_id, "stage", "Season Analysis")
                self.tree.set(season_row_id, "progress", f"{record.progress}%")
                self.tree.set(season_row_id, "status", msg)

        # Dynamic output rendering rows
        if "Output" in msg and "Rendering" in msg:
            m = re.search(r"Output\s+(\d+)\s+Rendering", msg)
            out_idx = int(m.group(1)) if m else 1
            out_row_id = f"{record.id}_output_{out_idx}"
            if not self.tree.exists(out_row_id):
                self.tree.insert(
                    "",
                    "end",
                    iid=out_row_id,
                    values=(f"Output {out_idx}", "—", "Rendering", f"{record.progress}%", msg),
                )
            else:
                self.tree.set(out_row_id, "stage", "Rendering")
                self.tree.set(out_row_id, "progress", f"{record.progress}%")
                self.tree.set(out_row_id, "status", msg)

        # Handle completion or error
        if record.status == "COMPLETED":
            if not record.outputs:
                for ep in record.source_episodes:
                    row_id = f"{record.id}_{ep.episode_id}"
                    if self.tree.exists(row_id):
                        self.tree.set(row_id, "stage", "Hoàn thành")
                        self.tree.set(row_id, "progress", "100%")
                        self.tree.set(row_id, "status", "0 output (không có ứng viên phù hợp)")
            else:
                for idx, out in enumerate(record.outputs, start=1):
                    out_row_id = f"{record.id}_output_{idx}"
                    if self.tree.exists(out_row_id):
                        out_name = Path(out.publication_video_path).name if getattr(out, "publication_video_path", None) else "—"
                        self.tree.set(out_row_id, "source_video", out_name)
                        self.tree.set(out_row_id, "stage", "Hoàn thành")
                        self.tree.set(out_row_id, "progress", "100%")
                        self.tree.set(out_row_id, "status", "Hoàn tất")
                for ep in record.source_episodes:
                    row_id = f"{record.id}_{ep.episode_id}"
                    if self.tree.exists(row_id):
                        self.tree.set(row_id, "stage", "Hoàn thành")
                        self.tree.set(row_id, "progress", "100%")
        elif record.status == "ERROR":
            for ep in record.source_episodes:
                row_id = f"{record.id}_{ep.episode_id}"
                if self.tree.exists(row_id):
                    self.tree.set(row_id, "stage", "Lỗi")
                    self.tree.set(row_id, "status", record.error or msg)

        self.status_var.set(f"{record.name}: {msg}")
        self.progress_var.set(float(record.progress))

    def _on_queue_state_changed(self, is_running: bool) -> None:
        safe_after(self, 0, self._apply_state_change, is_running)

    def _apply_state_change(self, is_running: bool) -> None:
        if is_running:
            self.btn_start.config(state="disabled")
            self.btn_cancel.config(state="normal")
            self.btn_select_file.config(state="disabled")
            self.btn_select_folder.config(state="disabled")
            self.btn_clear.config(state="disabled")
        else:
            self.btn_start.config(state="normal")
            self.btn_cancel.config(state="disabled")
            self.btn_select_file.config(state="normal")
            self.btn_select_folder.config(state="normal")
            self.btn_clear.config(state="normal")

    def _on_batch_completed(self, completed: int, total: int) -> None:
        safe_after(self, 0, self._apply_batch_completed, completed, total)

    def _apply_batch_completed(self, completed: int, total: int) -> None:
        self.progress_var.set(100.0)

        def _cleanup_notification() -> None:
            self._active_desktop_notification = None

        has_zero_outputs = any(
            p.status == "COMPLETED" and not p.outputs for p in getattr(self, "projects", [])
        )

        if completed == total and total > 0:
            if has_zero_outputs and not any(p.outputs for p in getattr(self, "projects", [])):
                msg = "Hoàn thành phân tích: Không có ứng viên recap nào đạt yêu cầu (0 outputs)."
                level = "info"
            else:
                msg = f"Đã hoàn thành toàn bộ {total} dự án video recap thành công!"
                level = "success"
            self.status_var.set(msg)
            self.banner.show(msg, level=level)
            self._active_desktop_notification = show_desktop_notification(
                title="ToolRecap V2",
                message=msg,
                level=level,
                parent=self,
                on_dismiss=_cleanup_notification,
                play_sound=True,
            )
        elif completed < total:
            msg = f"Đã xử lý {completed}/{total} dự án. Một số tác vụ bị dừng hoặc lỗi."
            self.status_var.set(msg)
            self.banner.show(msg, level="warning")
            self._active_desktop_notification = show_desktop_notification(
                title="ToolRecap V2",
                message=msg,
                level="warning",
                parent=self,
                on_dismiss=_cleanup_notification,
                play_sound=True,
            )

    def _start_batch(self) -> None:
        if not self.projects:
            self.banner.show("Chưa có video nào trong danh sách. Hãy chọn file hoặc thư mục trước.", level="warning")
            return
        self.queue.start()

    def _cancel_batch(self) -> None:
        self.status_var.set("Đang dừng các tác vụ và đóng tiến trình...")
        self.queue.cancel()

    def _clear_queue(self) -> None:
        if self.queue.is_running:
            return
        self.projects = []
        self.queue.projects = []
        self.project_store.save([])
        self._refresh_queue_table()
        self.source_var.set("Chưa chọn video hoặc thư mục.")
        self.status_var.set("Đã xóa danh sách.")
        self.progress_var.set(0)

    def _open_output_folder(self) -> None:
        out_dir = Path(self.settings.output_dir) if self.settings.output_dir else (default_data_directory() / "recaps_xuat")
        out_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(out_dir))
        else:
            subprocess.run(["xdg-open", str(out_dir)], check=False)

    def _open_voicestudio_dialog(self) -> None:
        open_voicestudio_dialog(self)

    # -------------------------------------------------------------------------
    # GPU and Updater Background Checks
    # -------------------------------------------------------------------------

    def _detect_gpu_background(self) -> None:
        def _check() -> None:
            plan = get_acceleration_plan()
            label = plan.summary_label
            safe_after(self, 0, lambda: self.gpu_status_var.set(label))

        threading.Thread(target=_check, daemon=True).start()

    def _check_update_background(self) -> None:
        def _check() -> None:
            rel = check_for_updates()
            if rel:
                safe_after(self, 0, lambda: self._prompt_update(rel))

        threading.Thread(target=_check, daemon=True).start()

    def _manual_check_update(self) -> None:
        self.update_btn.config(state="disabled", text="Đang kiểm tra...")

        def _check() -> None:
            try:
                rel = check_for_updates()
                if rel:
                    safe_after(self, 0, lambda: self._prompt_update(rel))
                else:
                    safe_after(self, 0, lambda: self.banner.show(f"Bạn đang sử dụng phiên bản mới nhất (v{__version__}).", level="info"))
            finally:
                safe_after(self, 0, lambda: self.update_btn.config(state="normal", text="🔄 Kiểm tra cập nhật"))

        threading.Thread(target=_check, daemon=True).start()

    def _prompt_update(self, release: ReleaseInfo) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Bản cập nhật mới")
        dialog.geometry("500x320")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        set_window_icon(dialog)

        frame = ttk.Frame(dialog, padding=18)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text=f"Đã có phiên bản mới: v{release.version}",
            font=("Segoe UI Semibold", 12),
        ).pack(anchor="w", pady=(0, 2))

        ttk.Label(
            frame,
            text=f"Phiên bản hiện tại: v{__version__}  ➜  Phiên bản mới: v{release.version}",
            font=("Segoe UI", 9),
            foreground="#2563eb",
        ).pack(anchor="w", pady=(0, 8))

        notes = release.body or "Cải thiện hiệu năng và sửa lỗi."
        txt = tk.Text(frame, height=6, font=("Segoe UI", 9), wrap="word")
        txt.insert("1.0", notes)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True, pady=(0, 12))

        btn_box = ttk.Frame(frame)
        btn_box.pack(fill="x")

        def _do_update() -> None:
            dialog.destroy()
            self._execute_update(release)

        ttk.Button(btn_box, text="Cập nhật ngay", style="Primary.TButton", command=_do_update).pack(side="left")
        ttk.Button(btn_box, text="Để sau", command=dialog.destroy).pack(side="right")

    def _execute_update(self, release: ReleaseInfo) -> None:
        self.banner.show(f"Đang tải bản cập nhật v{release.version}...", level="info")

        def _worker() -> None:
            try:
                payload_dir = download_and_stage_update(release)
                app_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
                script = generate_apply_script(payload_dir, app_dir)
                safe_after(self, 0, lambda: self._prompt_restart_update(script))
            except Exception as exc:
                safe_after(self, 0, lambda: self.banner.show(f"Lỗi cập nhật: {exc}", level="error"))

        threading.Thread(target=_worker, daemon=True).start()

    def _prompt_restart_update(self, script_path: Path) -> None:
        if messagebox.askyesno(
            "Khởi động lại để cập nhật",
            "Bản cập nhật đã sẵn sàng. Khởi động lại ứng dụng ngay bây giờ để hoàn tất?",
            parent=self,
        ):
            subprocess.Popen([str(script_path)], shell=True)
            self.destroy()
            sys.exit(0)

    # -------------------------------------------------------------------------
    # Settings Dialog and App Lifecycle
    # -------------------------------------------------------------------------

    def _open_settings_dialog(self) -> None:
        def _on_saved(new_settings: AppSettings) -> None:
            self.settings = new_settings
            self.queue.settings = new_settings
            self.voice_var.set(new_settings.voice_id)
            for p in self.projects:
                if p.status == "WAITING":
                    p.voice_id = new_settings.voice_id
            self.project_store.save(self.projects)
            self.banner.show("Đã lưu cài đặt thành công!", level="success")

        SettingsDialog(self, self.settings, self.settings_store, on_save=_on_saved)

    def _on_banner_dismissed(self) -> None:
        pass

    def _on_close(self) -> None:
        if self.queue.is_running:
            if not messagebox.askyesno(
                "Đang xử lý",
                "Đang có tiến trình render hoạt động. Bạn có chắc chắn muốn dừng và thoát?",
                parent=self,
            ):
                return
            self.queue.cancel()
        self.preview_player.stop()
        self.destroy()


class SettingsDialog(tk.Toplevel):
    """Central settings window with exactly four panes: Recap, AI Gateway, Voice, Render and Output."""

    def __init__(
        self,
        parent: tk.Misc,
        settings: AppSettings,
        store: SettingsStore,
        *,
        on_save: Callable[[AppSettings], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.store = store
        self.on_save = on_save or (lambda s: None)
        self._panes: dict[str, ttk.Frame] = {}
        self._nav_buttons: dict[str, ttk.Button] = {}
        self.preview_player = AudioPreviewPlayer()

        self.title("Cài đặt — ToolRecap V2")
        self.geometry("920x680")
        self.minsize(840, 600)
        self.transient(parent)
        set_window_icon(self)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._make_variables()
        self._build()
        self._show_pane("Recap")
        self.grab_set()

    def _make_variables(self) -> None:
        val = self.settings

        # Recap variables
        self.recap_language_var = tk.StringVar(value=val.recap_language or "en-US")
        self.recap_mode_var = tk.StringVar(value=val.recap_mode or "MAIN_STORIES")
        self.content_type_var = tk.StringVar(value=val.content_type or "US_TV_SHOW")
        self.rights_var = tk.StringVar(value=val.source_rights_status or "UNVERIFIED")

        # Voice variables
        self.voice_var = tk.StringVar(value=val.voice_id or DEFAULT_VOICE_ID)
        self.voice_style_var = tk.StringVar(value=val.voice_style or "film_recap")
        self.voice_status_var = tk.StringVar(value="")

        # Audio Mix variables
        self.orig_audio_var = tk.DoubleVar(value=val.original_audio_gain_db)
        self.commentary_var = tk.DoubleVar(value=val.commentary_gain_db)
        self.auto_duck_var = tk.BooleanVar(value=val.auto_duck)
        self.ducking_var = tk.DoubleVar(value=val.ducking_amount_db)
        self.target_loudness_var = tk.DoubleVar(value=val.target_loudness_lufs)
        self.true_peak_var = tk.DoubleVar(value=val.true_peak_dbtp)

        # AI Gateway variables
        self.gateway_enabled_var = tk.BooleanVar(value=val.gateway_enabled)
        self.endpoint_var = tk.StringVar(value=val.api_endpoint)
        self.key_var = tk.StringVar(value=val.api_key)
        self.show_key_var = tk.BooleanVar(value=False)
        self.scanner_model_var = tk.StringVar(value=val.scanner_model)
        self.scanner_thinking_var = tk.StringVar(value=val.scanner_thinking)
        self.scanner_supports_vision_var = tk.BooleanVar(value=val.scanner_supports_vision)
        self.finalizer_model_var = tk.StringVar(value=val.finalizer_model)
        self.finalizer_thinking_var = tk.StringVar(value=val.finalizer_thinking)
        self.parallel_var = tk.IntVar(value=val.scanner_parallelism)
        self.chunk_var = tk.IntVar(value=val.api_chunk_seconds)
        self.ai_status_var = tk.StringVar(value="Scanner và Finalizer chưa được kiểm tra.")

        # Render variables
        self.quality_var = tk.StringVar(value=val.quality)
        self.gpu_var = tk.BooleanVar(value=val.use_gpu)
        self.burn_var = tk.BooleanVar(value=val.burn_subtitles)
        self.output_var = tk.StringVar(value=val.output_dir)

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # Sidebar with ONLY four panes: Recap, AI Gateway, Voice, Render and Output
        sidebar = ttk.Frame(root, padding=(0, 4, 12, 4))
        sidebar.grid(row=0, column=0, sticky="ns")
        for name in ("Recap", "AI Gateway", "Voice", "Render and Output"):
            btn = ttk.Button(sidebar, text=name, width=20, command=lambda target=name: self._show_pane(target))
            btn.pack(fill="x", pady=3)
            self._nav_buttons[name] = btn

        host = ttk.Frame(root, padding=(16, 8))
        host.grid(row=0, column=1, sticky="nsew")
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)

        self._panes["Recap"] = self._build_recap(host)
        self._panes["AI Gateway"] = self._build_ai(host)
        self._panes["Voice"] = self._build_voice(host)
        self._panes["Render and Output"] = self._build_render(host)

        for pane in self._panes.values():
            pane.grid(row=0, column=0, sticky="nsew")

        bottom = ttk.Frame(root)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(bottom, text="Hủy", command=self._on_close).pack(side="right")
        ttk.Button(bottom, text="Lưu cài đặt", style="Primary.TButton", command=self._save).pack(side="right", padx=(0, 8))

    def _show_pane(self, name: str) -> None:
        if name in self._panes:
            self._panes[name].tkraise()
            for key, button in self._nav_buttons.items():
                button.state(["disabled"] if key == name else ["!disabled"])

    # -------------------------------------------------------------------------
    # Recap Pane
    # -------------------------------------------------------------------------

    def _build_recap(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(6, weight=1)

        ttk.Label(frame, text="Cấu hình Kịch bản & Recap", font=("Segoe UI Semibold", 14)).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 16)
        )

        # 1. Recap Language
        ttk.Label(frame, text="Recap language:").grid(row=1, column=0, sticky="w", pady=5)
        self.recap_lang_combo = ttk.Combobox(
            frame,
            textvariable=self.recap_language_var,
            values=["en-US", "en-GB"],
            state="readonly",
        )
        self.recap_lang_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)
        self.recap_lang_combo.bind("<<ComboboxSelected>>", self._on_recap_language_changed)

        # 2. Recap Mode
        ttk.Label(frame, text="Recap mode:").grid(row=2, column=0, sticky="w", pady=5)
        self.recap_mode_combo = ttk.Combobox(
            frame,
            textvariable=self.recap_mode_var,
            values=["MAIN_STORIES", "FULL_EPISODE"],
            state="readonly",
        )
        self.recap_mode_combo.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)

        # 3. Content Type
        ttk.Label(frame, text="Content type:").grid(row=3, column=0, sticky="w", pady=5)
        self.content_type_combo = ttk.Combobox(
            frame,
            textvariable=self.content_type_var,
            values=["US_TV_SHOW", "DE_GERMAN_SOAP", "BODYCAM", "FEATURE_FILM", "OTHER"],
            state="readonly",
        )
        self.content_type_combo.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)

        # 4. Footage Rights
        ttk.Label(frame, text="Footage rights:").grid(row=4, column=0, sticky="w", pady=5)
        self.rights_combo = ttk.Combobox(
            frame,
            textvariable=self.rights_var,
            values=["UNVERIFIED", "OWNED", "LICENSED", "FIRST_PUBLICATION_RIGHTS", "FAIR_USE"],
            state="readonly",
        )
        self.rights_combo.grid(row=4, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)

        # 5. Recap Prompt
        prompt_header = ttk.Frame(frame)
        prompt_header.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        ttk.Label(prompt_header, text="Recap Prompt:", font=("Segoe UI Semibold", 10)).pack(side="left")
        self.reload_prompt_btn = ttk.Button(
            prompt_header,
            text="Reload Default Prompt",
            command=self._reload_default_prompt,
        )
        self.reload_prompt_btn.pack(side="right")

        text_frame = ttk.Frame(frame)
        text_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(0, 4))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.prompt_text = tk.Text(text_frame, wrap="word", height=10, font=("Segoe UI", 9))
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.prompt_text.yview)
        self.prompt_text.configure(yscrollcommand=scroll.set)
        self.prompt_text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        initial_prompt = self.settings.recap_prompt or get_default_recap_prompt(self.settings.content_type)
        self.prompt_text.insert("1.0", initial_prompt)

        return frame

    def _reload_default_prompt(self) -> None:
        ctype = self.content_type_var.get().strip()
        default_prompt = get_default_recap_prompt(ctype)
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", default_prompt)

    def _on_recap_language_changed(self, event: tk.Event | None = None) -> None:
        lang = self.recap_language_var.get()
        default_id = DEFAULT_VOICE_BY_LANGUAGE.get(lang, DEFAULT_VOICE_ID)
        self.voice_var.set(default_id)
        if hasattr(self, "cbo_voice"):
            self._refresh_voice_choices()
            self._update_voice_status_label()

    # -------------------------------------------------------------------------
    # Voice & Audio Mix Pane
    # -------------------------------------------------------------------------

    def _build_voice(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Cấu hình Giọng đọc & Âm thanh", font=("Segoe UI Semibold", 14)).pack(
            anchor="w", pady=(0, 10)
        )

        # 1. Voice Config Frame
        voice_box = ttk.LabelFrame(frame, text="Voice Configuration", padding=12)
        voice_box.pack(fill="x", pady=(0, 10))
        voice_box.columnconfigure(1, weight=1)

        # Language display
        ttk.Label(voice_box, text="Language:").grid(row=0, column=0, sticky="w", pady=4)
        self.lbl_voice_lang = ttk.Label(
            voice_box,
            textvariable=self.recap_language_var,
            font=("Segoe UI Semibold", 9),
            foreground="#0969da",
        )
        self.lbl_voice_lang.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=4)

        # Voice selection
        ttk.Label(voice_box, text="Voice selection:").grid(row=1, column=0, sticky="w", pady=4)
        self.cbo_voice = ttk.Combobox(voice_box, state="readonly", width=42)
        self.cbo_voice.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=4)
        self.cbo_voice.bind("<<ComboboxSelected>>", self._on_voice_changed)

        # Voice style
        ttk.Label(voice_box, text="Voice style:").grid(row=2, column=0, sticky="w", pady=4)
        style_choices = [(s, STYLE_NAMES.get(s, s)) for s in SUPPORTED_VOICE_STYLES]
        self.style_map = {name: key for key, name in style_choices}
        self.cbo_style = ttk.Combobox(
            voice_box,
            values=[name for _, name in style_choices],
            state="readonly",
            width=42,
        )
        cur_style_name = STYLE_NAMES.get(self.voice_style_var.get(), self.voice_style_var.get())
        self.cbo_style.set(cur_style_name)
        self.cbo_style.bind("<<ComboboxSelected>>", self._on_style_changed)
        self.cbo_style.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=4)

        # Action bar: Preview, progress, status, update
        voice_action_bar = ttk.Frame(voice_box)
        voice_action_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        self.btn_preview = ttk.Button(voice_action_bar, text="🔊 Nghe thử giọng", command=self._toggle_preview)
        self.btn_preview.pack(side="left")

        self.voice_preview_prog = ttk.Progressbar(voice_action_bar, length=80, mode="determinate")
        self.voice_preview_prog.pack(side="left", padx=(6, 0))

        self.btn_voicestudio = ttk.Button(
            voice_action_bar,
            text="🎙 Cập nhật VoiceStudio",
            command=lambda: open_voicestudio_dialog(self),
        )
        self.btn_voicestudio.pack(side="right")

        # Voice status label
        self.lbl_voice_status = ttk.Label(
            voice_box,
            textvariable=self.voice_status_var,
            font=("Segoe UI", 8),
            foreground="#6b7280",
            wraplength=520,
        )
        self.lbl_voice_status.grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self._refresh_voice_choices()
        self._update_voice_status_label()

        # 2. Audio Mix Frame
        mix_box = ttk.LabelFrame(frame, text="Audio Mix", padding=12)
        mix_box.pack(fill="x")
        mix_box.columnconfigure(1, weight=1)

        # Original audio [ 0.00 dB ]
        ttk.Label(mix_box, text="Original audio").grid(row=0, column=0, sticky="w", pady=4)
        orig_bar = ttk.Frame(mix_box)
        orig_bar.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=4)
        self.orig_audio_spin = ttk.Spinbox(
            orig_bar,
            from_=-60.0,
            to=24.0,
            increment=0.5,
            textvariable=self.orig_audio_var,
            width=8,
        )
        self.orig_audio_spin.pack(side="left")
        ttk.Label(orig_bar, text="dB").pack(side="left", padx=(6, 0))

        # Commentary voice [ 0.00 dB ]
        ttk.Label(mix_box, text="Commentary voice").grid(row=1, column=0, sticky="w", pady=4)
        comm_bar = ttk.Frame(mix_box)
        comm_bar.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=4)
        self.commentary_spin = ttk.Spinbox(
            comm_bar,
            from_=-60.0,
            to=24.0,
            increment=0.5,
            textvariable=self.commentary_var,
            width=8,
        )
        self.commentary_spin.pack(side="left")
        ttk.Label(comm_bar, text="dB").pack(side="left", padx=(6, 0))

        # Auto-duck original audio during commentary
        self.chk_auto_duck = ttk.Checkbutton(
            mix_box,
            text="Auto-duck original audio during commentary",
            variable=self.auto_duck_var,
            command=self._on_auto_duck_toggled,
        )
        self.chk_auto_duck.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)

        # Ducking amount [ -12.00 dB ]
        ttk.Label(mix_box, text="Ducking amount").grid(row=3, column=0, sticky="w", pady=4)
        duck_bar = ttk.Frame(mix_box)
        duck_bar.grid(row=3, column=1, sticky="w", padx=(10, 0), pady=4)
        self.ducking_spin = ttk.Spinbox(
            duck_bar,
            from_=-60.0,
            to=0.0,
            increment=1.0,
            textvariable=self.ducking_var,
            width=8,
        )
        self.ducking_spin.pack(side="left")
        ttk.Label(duck_bar, text="dB").pack(side="left", padx=(6, 0))

        # Target loudness [ -14.00 LUFS ]
        ttk.Label(mix_box, text="Target loudness").grid(row=4, column=0, sticky="w", pady=4)
        lufs_bar = ttk.Frame(mix_box)
        lufs_bar.grid(row=4, column=1, sticky="w", padx=(10, 0), pady=4)
        self.target_loudness_spin = ttk.Spinbox(
            lufs_bar,
            from_=-70.0,
            to=-5.0,
            increment=1.0,
            textvariable=self.target_loudness_var,
            width=8,
        )
        self.target_loudness_spin.pack(side="left")
        ttk.Label(lufs_bar, text="LUFS").pack(side="left", padx=(6, 0))

        # True peak [ -1.00 dBTP ]
        ttk.Label(mix_box, text="True peak").grid(row=5, column=0, sticky="w", pady=4)
        tp_bar = ttk.Frame(mix_box)
        tp_bar.grid(row=5, column=1, sticky="w", padx=(10, 0), pady=4)
        self.true_peak_spin = ttk.Spinbox(
            tp_bar,
            from_=-9.0,
            to=0.0,
            increment=0.5,
            textvariable=self.true_peak_var,
            width=8,
        )
        self.true_peak_spin.pack(side="left")
        ttk.Label(tp_bar, text="dBTP").pack(side="left", padx=(6, 0))

        self._on_auto_duck_toggled()

        return frame

    def _refresh_voice_choices(self) -> None:
        lang = self.recap_language_var.get()
        choices = [
            (vid, f"{spec.display_name} [{spec.gender}]")
            for vid, spec in BUILTIN_VOICES.items()
            if spec.language == lang
        ]
        cur_id = self.voice_var.get()
        if cur_id and cur_id not in [c[0] for c in choices]:
            spec = get_voice_spec(cur_id)
            if spec:
                label = f"{spec.display_name} (Tương thích nội bộ)" if "piper" in cur_id else spec.display_name
                choices.insert(0, (cur_id, label))

        self.voice_map = dict(choices)
        self.voice_reverse_map = {v: k for k, v in choices}
        self.cbo_voice.config(values=list(self.voice_map.values()))

        if cur_id in self.voice_map:
            self.cbo_voice.set(self.voice_map[cur_id])
        elif choices:
            default_id = DEFAULT_VOICE_BY_LANGUAGE.get(lang, choices[0][0])
            self.voice_var.set(default_id)
            self.cbo_voice.set(self.voice_map.get(default_id, choices[0][1]))

    def _on_voice_changed(self, event: tk.Event | None = None) -> None:
        disp = self.cbo_voice.get()
        vid = self.voice_reverse_map.get(disp, DEFAULT_VOICE_ID)
        self.voice_var.set(vid)
        self._update_voice_status_label()

    def _on_style_changed(self, event: tk.Event | None = None) -> None:
        disp = self.cbo_style.get()
        style_key = self.style_map.get(disp, "film_recap")
        self.voice_style_var.set(style_key)

    def _on_auto_duck_toggled(self) -> None:
        if self.auto_duck_var.get():
            self.ducking_spin.config(state="normal")
        else:
            self.ducking_spin.config(state="disabled")

    def _update_voice_status_label(self) -> None:
        cur_id = self.voice_var.get()
        status = get_voice_status(cur_id)
        if not status["ready"]:
            self.voice_status_var.set(status["status_label"])
            self.lbl_voice_status.config(foreground="#dc2626")
            self.voice_preview_prog["value"] = 0
        else:
            mgr = get_voice_manager()
            if mgr.is_voice_installed(cur_id):
                self.voice_status_var.set("✓ Giọng đọc đã cài đặt và sẵn sàng")
                self.lbl_voice_status.config(foreground="#16a34a")
                self.voice_preview_prog["value"] = 100
            else:
                self.voice_status_var.set("Model giọng đọc sẽ tự động tải khi bắt đầu render")
                self.lbl_voice_status.config(foreground="#6b7280")
                self.voice_preview_prog["value"] = 0

    def _toggle_preview(self) -> None:
        if self.preview_player.is_playing:
            self.preview_player.stop()
            self.btn_preview.config(text="🔊 Nghe thử giọng")
            return

        spec = get_voice_spec(self.voice_var.get())
        mgr = get_voice_manager()
        self.btn_preview.config(text="⏹ Dừng nghe")

        def _generate_and_play() -> None:
            preview_wav = default_data_directory() / "cache" / f"preview_{spec.voice_id}.wav"
            try:
                if not preview_wav.is_file():
                    def _prog(current: int, total: int, pct: float) -> None:
                        def _update_ui() -> None:
                            if total > 0 and current < total:
                                self.voice_status_var.set(f"Đang tải model giọng ({pct:.0f}%)...")
                                self.voice_preview_prog["value"] = pct
                            else:
                                self.voice_status_var.set("Đang tạo âm thanh mẫu...")
                                self.voice_preview_prog["value"] = 100
                        safe_after(self, 0, _update_ui)

                    mgr.synthesize(
                        spec.voice_id,
                        spec.preview_text,
                        preview_wav,
                        style=self.voice_style_var.get(),
                        progress_callback=_prog,
                    )
                    safe_after(self, 0, self._update_voice_status_label)
                else:
                    safe_after(self, 0, self._update_voice_status_label)

                self.preview_player.play(
                    preview_wav,
                    on_finished=lambda: safe_after(self, 0, lambda: self.btn_preview.config(text="🔊 Nghe thử giọng")),
                )
            except Exception as exc:
                safe_after(self, 0, lambda: self.voice_status_var.set(f"Không thể phát nghe thử: {exc}"))
                safe_after(self, 0, lambda: self.btn_preview.config(text="🔊 Nghe thử giọng"))
                safe_after(self, 0, self._update_voice_status_label)

        threading.Thread(target=_generate_and_play, daemon=True).start()

    # -------------------------------------------------------------------------
    # Render and Output Pane
    # -------------------------------------------------------------------------

    def _build_render(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Cấu hình Render & Đầu ra", font=("Segoe UI Semibold", 14)).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 16)
        )

        ttk.Label(frame, text="Chất lượng video:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Combobox(
            frame,
            textvariable=self.quality_var,
            values=["standard", "high", "source"],
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=5)

        ttk.Checkbutton(
            frame,
            text="Bật tăng tốc phần cứng GPU (NVENC/AMF/QSV)",
            variable=self.gpu_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=5)

        ttk.Checkbutton(
            frame,
            text="Nhúng thẳng phụ đề vào video (Burn subtitles)",
            variable=self.burn_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=5)

        ttk.Label(frame, text="Thư mục xuất video:").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.output_var).grid(row=4, column=1, sticky="ew", padx=(10, 6), pady=5)

        def _choose_dir() -> None:
            folder = filedialog.askdirectory(parent=self, title="Chọn thư mục xuất video")
            if folder:
                self.output_var.set(folder)

        ttk.Button(frame, text="Duyệt...", command=_choose_dir).grid(row=4, column=2, sticky="w", pady=5)

        # Updates & Subsystems
        sub_box = ttk.LabelFrame(frame, text="Hệ thống phụ & Cập nhật", padding=10)
        sub_box.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(16, 0))

        ttk.Button(
            sub_box,
            text="🔄 Kiểm tra VoiceStudio Subsystem",
            command=lambda: open_voicestudio_dialog(self),
        ).pack(side="left", padx=(0, 8))

        return frame

    # -------------------------------------------------------------------------
    # AI Gateway Pane
    # -------------------------------------------------------------------------

    def _build_ai(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Cấu hình AI Gateway", font=("Segoe UI Semibold", 14)).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 16)
        )

        chk_gw = ttk.Checkbutton(
            frame,
            text="Kích hoạt phân tích kịch bản bằng AI Gateway (Scanner + Finalizer)",
            variable=self.gateway_enabled_var,
        )
        chk_gw.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(frame, text="API endpoint:").grid(row=2, column=0, sticky="w", pady=5)
        self.endpoint_entry = ttk.Entry(frame, textvariable=self.endpoint_var)
        self.endpoint_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)

        ttk.Label(frame, text="API key:").grid(row=3, column=0, sticky="w", pady=5)
        self.key_entry = ttk.Entry(frame, textvariable=self.key_var, show="●")
        self.key_entry.grid(row=3, column=1, sticky="ew", padx=(10, 6), pady=5)
        self.show_key_check = ttk.Checkbutton(frame, text="Hiện", variable=self.show_key_var, command=self._toggle_key)
        self.show_key_check.grid(row=3, column=2, sticky="w")

        ttk.Label(frame, text="Scanner model:").grid(row=4, column=0, sticky="w", pady=5)
        self.scanner_model_entry = ttk.Entry(frame, textvariable=self.scanner_model_var)
        self.scanner_model_entry.grid(row=4, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)

        ttk.Label(frame, text="Scanner thinking:").grid(row=5, column=0, sticky="w", pady=5)
        self.scanner_thinking_combo = ttk.Combobox(
            frame,
            textvariable=self.scanner_thinking_var,
            values=["auto", "low", "medium", "high", "xhigh", "max"],
            state="readonly",
        )
        self.scanner_thinking_combo.grid(row=5, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)

        self.chk_scanner_vision = ttk.Checkbutton(
            frame,
            text="Scanner model supports image/Vision input",
            variable=self.scanner_supports_vision_var,
        )
        self.chk_scanner_vision.grid(row=6, column=0, columnspan=3, sticky="w", pady=5)

        ttk.Label(frame, text="Finalizer model:").grid(row=7, column=0, sticky="w", pady=5)
        self.finalizer_model_entry = ttk.Entry(frame, textvariable=self.finalizer_model_var)
        self.finalizer_model_entry.grid(row=7, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)

        ttk.Label(frame, text="Finalizer thinking:").grid(row=8, column=0, sticky="w", pady=5)
        self.finalizer_thinking_combo = ttk.Combobox(
            frame,
            textvariable=self.finalizer_thinking_var,
            values=["auto", "low", "medium", "high", "xhigh", "max"],
            state="readonly",
        )
        self.finalizer_thinking_combo.grid(row=8, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)

        ttk.Label(frame, text="Scanner parallelism:").grid(row=9, column=0, sticky="w", pady=5)
        self.parallel_spin = ttk.Spinbox(frame, from_=1, to=4, textvariable=self.parallel_var, width=8)
        self.parallel_spin.grid(row=9, column=1, sticky="w", padx=(10, 0), pady=5)

        ttk.Label(frame, text="Độ dài đoạn:").grid(row=10, column=0, sticky="w", pady=5)
        chunk_box = ttk.Frame(frame)
        chunk_box.grid(row=10, column=1, sticky="w", padx=(10, 0), pady=5)
        self.chunk_spin = ttk.Spinbox(chunk_box, from_=60, to=900, increment=30, textvariable=self.chunk_var, width=8)
        self.chunk_spin.pack(side="left")
        ttk.Label(chunk_box, text="giây").pack(side="left", padx=(6, 0))

        test_box = ttk.Frame(frame)
        test_box.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(14, 6))
        test_box.columnconfigure((0, 1), weight=1)
        self.test_scanner_btn = ttk.Button(test_box, text="Test Scanner", command=lambda: self._test_api("scanner"))
        self.test_scanner_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.test_finalizer_btn = ttk.Button(test_box, text="Test Finalizer", command=lambda: self._test_api("finalizer"))
        self.test_finalizer_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.ai_status_lbl = ttk.Label(
            frame,
            textvariable=self.ai_status_var,
            foreground="#075fc9",
            wraplength=520,
            font=("Segoe UI", 9),
        )
        self.ai_status_lbl.grid(row=12, column=0, columnspan=3, sticky="w", pady=(4, 8))

        note = ttk.Label(
            frame,
            text="* Lưu ý: API key được lưu cục bộ dưới dạng văn bản trong %LOCALAPPDATA%\\ToolRecapV2\\settings.json.",
            font=("Segoe UI", 8),
            foreground="#6b7280",
            wraplength=520,
        )
        note.grid(row=13, column=0, columnspan=3, sticky="w", pady=(8, 0))

        return frame

    def _toggle_key(self) -> None:
        if hasattr(self, "key_entry") and self.key_entry is not None:
            self.key_entry.configure(show="" if self.show_key_var.get() else "●")

    def _test_api(self, stage: str) -> None:
        endpoint = self.endpoint_var.get().strip()
        key = self.key_var.get().strip()
        if stage == "scanner":
            model = self.scanner_model_var.get().strip()
            thinking = self.scanner_thinking_var.get().strip()
            label = "Scanner"
        else:
            model = self.finalizer_model_var.get().strip()
            thinking = self.finalizer_thinking_var.get().strip()
            label = "Finalizer"

        if not endpoint or not model:
            messagebox.showwarning("AI Gateway", "Endpoint và model không được để trống.", parent=self)
            return

        self.ai_status_var.set(f"Đang kiểm tra {label} — {model}…")

        import queue
        result_queue: queue.Queue[str] = queue.Queue()

        def _work() -> None:
            import time
            started = time.monotonic()
            try:
                result = OpenAICompatibleClient(endpoint, key, timeout=30).test(model, thinking)
                elapsed = time.monotonic() - started
                result_queue.put(f"{label} hoạt động — {model} / {thinking} — {elapsed:.1f}s — {result}")
            except Exception as exc:
                result_queue.put(f"{label} lỗi — {model}: {exc}")

        def _poll() -> None:
            try:
                msg = result_queue.get_nowait()
                self.ai_status_var.set(msg)
            except queue.Empty:
                if self.winfo_exists():
                    safe_after(self, 25, _poll)

        threading.Thread(target=_work, daemon=True).start()
        safe_after(self, 25, _poll)

    # -------------------------------------------------------------------------
    # Save & Close
    # -------------------------------------------------------------------------

    def _save(self) -> None:
        try:
            parallel = max(1, min(4, int(self.parallel_var.get())))
            chunk = max(60, min(900, int(self.chunk_var.get())))
        except (tk.TclError, ValueError):
            messagebox.showerror("Cài đặt", "Scanner parallelism hoặc Độ dài đoạn không hợp lệ.", parent=self)
            return

        endpoint = self.endpoint_var.get().strip()
        scanner = self.scanner_model_var.get().strip()
        finalizer = self.finalizer_model_var.get().strip()
        gateway_enabled = self.gateway_enabled_var.get()

        if gateway_enabled and (not endpoint or not scanner or not finalizer):
            messagebox.showerror("Cài đặt", "Khi bật AI Gateway, Endpoint và hai model không được để trống.", parent=self)
            return

        try:
            orig_audio = max(-60.0, min(24.0, float(self.orig_audio_var.get())))
            commentary = max(-60.0, min(24.0, float(self.commentary_var.get())))
            ducking = max(-60.0, min(0.0, float(self.ducking_var.get())))
            target_lufs = max(-70.0, min(-5.0, float(self.target_loudness_var.get())))
            true_peak = max(-9.0, min(0.0, float(self.true_peak_var.get())))
        except (tk.TclError, ValueError):
            messagebox.showerror("Cài đặt", "Giá trị thông số Audio Mix không hợp lệ.", parent=self)
            return

        # Recap settings
        self.settings.recap_language = self.recap_language_var.get()
        self.settings.recap_mode = self.recap_mode_var.get()
        self.settings.content_type = self.content_type_var.get()
        self.settings.source_rights_status = self.rights_var.get()
        self.settings.recap_prompt = self.prompt_text.get("1.0", "end-1c").strip()

        # Voice settings
        self.settings.voice_id = self.voice_var.get()
        self.settings.voice_style = self.voice_style_var.get()

        # Audio Mix settings
        self.settings.original_audio_gain_db = orig_audio
        self.settings.commentary_gain_db = commentary
        self.settings.auto_duck = bool(self.auto_duck_var.get())
        self.settings.ducking_amount_db = ducking
        self.settings.target_loudness_lufs = target_lufs
        self.settings.true_peak_dbtp = true_peak

        # Render settings
        self.settings.quality = self.quality_var.get()
        self.settings.use_gpu = self.gpu_var.get()
        self.settings.burn_subtitles = self.burn_var.get()
        self.settings.output_dir = self.output_var.get().strip()

        # AI Gateway settings
        self.settings.api_endpoint = endpoint
        self.settings.api_key = self.key_var.get().strip()
        self.settings.scanner_model = scanner
        self.settings.scanner_thinking = self.scanner_thinking_var.get().strip()
        self.settings.scanner_supports_vision = bool(self.scanner_supports_vision_var.get())
        self.settings.finalizer_model = finalizer
        self.settings.finalizer_thinking = self.finalizer_thinking_var.get().strip()
        self.settings.scanner_parallelism = parallel
        self.settings.api_chunk_seconds = chunk
        self.settings.gateway_enabled = gateway_enabled

        self.store.save(self.settings)
        self.on_save(self.settings)
        self._on_close()

    def _on_close(self) -> None:
        self.preview_player.stop()
        self.destroy()


def run_app() -> None:
    app = ToolRecapV2App()
    app.mainloop()
