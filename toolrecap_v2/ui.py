"""Main graphical user interface for ToolRecap V2 built with Tkinter and ttk."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

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
from .voice.catalog import BUILTIN_VOICES, DEFAULT_VOICE_ID, get_voice_spec
from .voice.manager import get_voice_manager


class ToolRecapV2App(tk.Tk):
    """Main desktop application window."""

    def __init__(self) -> None:
        super().__init__()
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
        self.after(200, self._detect_gpu_background)
        self.after(2000, self._check_update_background)

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
        self.settings_btn = ttk.Button(btn_box, text="⚙ Cài đặt", command=self._open_settings_dialog)
        self.settings_btn.pack(side="left", padx=4)

        # 2. Notification Banner (with visible [X])
        self.banner = NotificationBanner(container, on_dismiss=self._on_banner_dismissed)
        self.banner.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.banner.grid_remove()  # Hidden by default

        # 3. Source & Voice Selection Frame
        control_frame = ttk.Frame(container)
        control_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        control_frame.columnconfigure(0, weight=3)
        control_frame.columnconfigure(1, weight=2)

        # 3a. Source Selector Box
        source_box = ttk.LabelFrame(control_frame, text="Nguồn video", padding=12)
        source_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        source_box.columnconfigure(0, weight=1)

        src_btn_bar = ttk.Frame(source_box)
        src_btn_bar.pack(fill="x", pady=(0, 6))
        self.btn_select_file = ttk.Button(src_btn_bar, text="📁 Chọn 1 file video...", command=self._choose_file)
        self.btn_select_file.pack(side="left", padx=(0, 6))
        self.btn_select_folder = ttk.Button(src_btn_bar, text="📂 Chọn thư mục chứa video...", command=self._choose_folder)
        self.btn_select_folder.pack(side="left")

        self.lbl_source = ttk.Label(
            source_box,
            textvariable=self.source_var,
            font=("Segoe UI", 9),
            foreground="#0969da",
            wraplength=520,
        )
        self.lbl_source.pack(fill="x")

        # 3b. Voice Selector Box
        voice_box = ttk.LabelFrame(control_frame, text="Giọng đọc thuyết minh (English)", padding=12)
        voice_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        voice_choices = [
            (vid, f"{spec.display_name} [{spec.gender}]")
            for vid, spec in BUILTIN_VOICES.items()
        ]
        self.voice_map = dict(voice_choices)
        self.voice_reverse_map = {v: k for k, v in voice_choices}

        self.cbo_voice = ttk.Combobox(
            voice_box,
            values=list(self.voice_map.values()),
            state="readonly",
            width=32,
        )
        current_display = self.voice_map.get(self.settings.voice_id, list(self.voice_map.values())[0])
        self.cbo_voice.set(current_display)
        self.cbo_voice.bind("<<ComboboxSelected>>", self._on_voice_changed)
        self.cbo_voice.pack(fill="x", pady=(0, 8))

        voice_action_bar = ttk.Frame(voice_box)
        voice_action_bar.pack(fill="x")
        self.btn_preview = ttk.Button(voice_action_bar, text="🔊 Nghe thử giọng", command=self._toggle_preview)
        self.btn_preview.pack(side="left")

        self.voice_preview_prog = ttk.Progressbar(voice_action_bar, length=80, mode="determinate")
        self.voice_preview_prog.pack(side="left", padx=(6, 0))

        self.lbl_voice_status = ttk.Label(voice_action_bar, text="", font=("Segoe UI", 8), foreground="#666")
        self.lbl_voice_status.pack(side="left", padx=(6, 0))
        self._update_voice_status_label()

        self.btn_voicestudio = ttk.Button(voice_action_bar, text="🎙 VoiceStudio", command=self._open_voicestudio_dialog)
        self.btn_voicestudio.pack(side="right")

        # 4. Queue Table Frame
        queue_frame = ttk.LabelFrame(container, text="Danh sách tập sẽ xử lý tuần tự", padding=8)
        queue_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(0, weight=1)

        columns = ("idx", "name", "progress", "status", "output")
        self.tree = ttk.Treeview(queue_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("idx", text="#")
        self.tree.heading("name", text="Tập phim")
        self.tree.heading("progress", text="Tiến trình")
        self.tree.heading("status", text="Trạng thái")
        self.tree.heading("output", text="Video đầu ra")

        self.tree.column("idx", width=45, anchor="center")
        self.tree.column("name", width=320, anchor="w")
        self.tree.column("progress", width=90, anchor="center")
        self.tree.column("status", width=360, anchor="w")
        self.tree.column("output", width=220, anchor="w")

        scroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        # 5. Action Controls Frame
        action_frame = ttk.Frame(container)
        action_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))

        self.btn_start = ttk.Button(
            action_frame,
            text="▶ Bắt đầu tự động",
            style="Primary.TButton",
            command=self._start_batch,
        )
        self.btn_start.pack(side="left", padx=(0, 8))

        self.btn_cancel = ttk.Button(
            action_frame,
            text="⏹ Dừng xử lý",
            style="Danger.TButton",
            command=self._cancel_batch,
            state="disabled",
        )
        self.btn_cancel.pack(side="left", padx=(0, 8))

        self.btn_open_out = ttk.Button(
            action_frame,
            text="📂 Mở thư mục kết quả",
            command=self._open_output_folder,
        )
        self.btn_open_out.pack(side="left")

        self.btn_clear = ttk.Button(
            action_frame,
            text="Xóa danh sách",
            command=self._clear_queue,
        )
        self.btn_clear.pack(side="right")

        # 6. Status Bar Frame
        status_bar = ttk.Frame(container, relief="sunken", padding=(8, 6))
        status_bar.grid(row=5, column=0, sticky="ew")
        status_bar.columnconfigure(1, weight=1)

        ttk.Label(status_bar, text="Trạng thái:", font=("Segoe UI Semibold", 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Label(status_bar, textvariable=self.status_var, font=("Segoe UI", 9)).grid(row=0, column=1, sticky="w")
        ttk.Label(status_bar, textvariable=self.gpu_status_var, font=("Segoe UI", 8), foreground="#555").grid(row=0, column=2, sticky="e")

        self.progressbar = ttk.Progressbar(container, variable=self.progress_var, maximum=100)
        self.progressbar.grid(row=6, column=0, sticky="ew", pady=(4, 0))

    # -------------------------------------------------------------------------
    # Queue and Event Handling
    # -------------------------------------------------------------------------

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[("Video files", "*.mp4;*.mkv;*.mov;*.avi;*.webm;*.m4v;*.ts"), ("All files", "*.*")],
        )
        if path:
            self._load_source(Path(path))

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Chọn thư mục chứa video")
        if folder:
            self._load_source(Path(folder))

    def _load_source(self, path: Path) -> None:
        videos = scan_videos(path)
        if not videos:
            self.banner.show(
                f"Không tìm thấy video nào được hỗ trợ trong: {path.name}",
                level="warning",
            )
            return

        self.source_var.set(f"Đã chọn: {path} ({len(videos)} video)")
        default_out = Path(self.settings.output_dir) if self.settings.output_dir else (default_data_directory() / "recaps_xuat")

        # Populate project queue
        self.projects = [
            ProjectRecord.from_video_path(v, default_out, voice_id=self.settings.voice_id)
            for v in videos
        ]
        self.queue.projects = self.projects
        self.project_store.save(self.projects)
        self._refresh_queue_table()
        self.status_var.set(f"Đã nạp {len(videos)} video. Nhấn 'Bắt đầu tự động' để chạy.")
        self.banner.show(f"Đã tìm thấy {len(videos)} video. Sẵn sàng xử lý tuần tự!", level="success")

    def _refresh_queue_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for i, p in enumerate(self.projects, start=1):
            out_name = Path(p.output_video).name if p.output_video else "—"
            self.tree.insert(
                "",
                "end",
                iid=p.id,
                values=(i, p.name, f"{p.progress}%", p.current_message, out_name),
            )

    def _on_project_updated(self, record: ProjectRecord) -> None:
        # Schedule update on UI main thread
        self.after(0, self._apply_project_update, record)

    def _apply_project_update(self, record: ProjectRecord) -> None:
        if self.tree.exists(record.id):
            out_name = Path(record.output_video).name if record.output_video else "—"
            self.tree.set(record.id, "progress", f"{record.progress}%")
            self.tree.set(record.id, "status", record.current_message)
            self.tree.set(record.id, "output", out_name)

        self.status_var.set(f"{record.name}: {record.current_message}")
        self.progress_var.set(float(record.progress))

    def _on_queue_state_changed(self, is_running: bool) -> None:
        self.after(0, self._apply_state_change, is_running)

    def _apply_state_change(self, is_running: bool) -> None:
        if is_running:
            self.btn_start.config(state="disabled")
            self.btn_cancel.config(state="normal")
            self.btn_select_file.config(state="disabled")
            self.btn_select_folder.config(state="disabled")
            self.btn_clear.config(state="disabled")
        else:
            # Safely restore all UI controls
            self.btn_start.config(state="normal")
            self.btn_cancel.config(state="disabled")
            self.btn_select_file.config(state="normal")
            self.btn_select_folder.config(state="normal")
            self.btn_clear.config(state="normal")

    def _on_batch_completed(self, completed: int, total: int) -> None:
        self.after(0, self._apply_batch_completed, completed, total)

    def _apply_batch_completed(self, completed: int, total: int) -> None:
        self.progress_var.set(100.0)

        def _cleanup_notification() -> None:
            self._active_desktop_notification = None

        if completed == total and total > 0:
            msg = f"Đã hoàn thành toàn bộ {total} tập video recap thành công!"
            self.status_var.set(msg)
            self.banner.show(msg, level="success")
            self._active_desktop_notification = show_desktop_notification(
                title="ToolRecap V2",
                message=msg,
                level="success",
                parent=self,
                on_dismiss=_cleanup_notification,
                play_sound=True,
            )
        elif completed < total:
            msg = f"Đã hoàn thành {completed}/{total} tập. Một số tập bị dừng hoặc lỗi."
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

    # -------------------------------------------------------------------------
    # Voice Handling
    # -------------------------------------------------------------------------

    def _on_voice_changed(self, event: tk.Event | None = None) -> None:
        display_val = self.cbo_voice.get()
        voice_id = self.voice_reverse_map.get(display_val, DEFAULT_VOICE_ID)
        self.settings.voice_id = voice_id
        self.settings_store.save(self.settings)
        for p in self.projects:
            if p.status == "WAITING":
                p.voice_id = voice_id
        self.project_store.save(self.projects)
        self._update_voice_status_label()

    def _update_voice_status_label(self) -> None:
        spec = get_voice_spec(self.settings.voice_id)
        if self.voice_manager.is_voice_installed(spec.voice_id):
            self.lbl_voice_status.config(text="✓ Đã sẵn sàng", foreground="#16a34a")
            self.voice_preview_prog["value"] = 100
        else:
            self.lbl_voice_status.config(text="Sẽ tự động tải khi render", foreground="#6b7280")
            self.voice_preview_prog["value"] = 0

    def _toggle_preview(self) -> None:
        if self.preview_player.is_playing:
            self.preview_player.stop()
            self.btn_preview.config(text="🔊 Nghe thử giọng")
            return

        spec = get_voice_spec(self.settings.voice_id)
        self.btn_preview.config(text="⏹ Dừng nghe")

        def _generate_and_play() -> None:
            preview_wav = default_data_directory() / "cache" / f"preview_{spec.voice_id}.wav"
            try:
                if not preview_wav.is_file():
                    def _prog(current: int, total: int, pct: float) -> None:
                        def _update_ui() -> None:
                            if total > 0 and current < total:
                                self.lbl_voice_status.config(
                                    text=f"Đang tải model giọng ({pct:.0f}%)...",
                                    foreground="#0284c7",
                                )
                                self.voice_preview_prog["value"] = pct
                            else:
                                self.lbl_voice_status.config(
                                    text="Đang tạo âm thanh mẫu...",
                                    foreground="#0284c7",
                                )
                                self.voice_preview_prog["value"] = 100
                        self.after(0, _update_ui)

                    self.voice_manager.synthesize(
                        spec.voice_id,
                        spec.preview_text,
                        preview_wav,
                        progress_callback=_prog,
                    )
                    self.after(0, self._update_voice_status_label)
                else:
                    self.after(0, self._update_voice_status_label)

                self.preview_player.play(
                    preview_wav,
                    on_finished=lambda: self.after(0, lambda: self.btn_preview.config(text="🔊 Nghe thử giọng")),
                )
            except Exception as exc:
                self.after(0, lambda: self.banner.show(f"Không thể phát nghe thử: {exc}", level="error"))
                self.after(0, lambda: self.btn_preview.config(text="🔊 Nghe thử giọng"))
                self.after(0, self._update_voice_status_label)

        threading.Thread(target=_generate_and_play, daemon=True).start()

    def _open_voicestudio_dialog(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Cập nhật VoiceStudio Subsystem")
        dlg.geometry("520x370")
        dlg.resizable(False, False)
        dlg.transient(self)
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
                check_voicestudio_status,
                download_and_stage_voicestudio_adapter,
                apply_voicestudio_subsystem_update,
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
                                self.after(0, lambda: messagebox.showinfo(
                                    "Cập nhật thành công",
                                    "VoiceStudio adapter đã được cập nhật thành công!",
                                    parent=dlg,
                                ))
                                self.after(0, dlg.destroy)
                            except Exception as exc:
                                self.after(0, lambda: messagebox.showerror(
                                    "Lỗi cập nhật",
                                    f"Cập nhật VoiceStudio thất bại, hệ thống cũ được giữ nguyên: {exc}",
                                    parent=dlg,
                                ))
                        threading.Thread(target=_work, daemon=True).start()
                    btn_update.config(command=_do_update)
                else:
                    btn_update.config(state="disabled")

            self.after(0, _update_ui)

        threading.Thread(target=_fetch_status, daemon=True).start()

    # -------------------------------------------------------------------------
    # GPU and Updater Background Checks
    # -------------------------------------------------------------------------

    def _detect_gpu_background(self) -> None:
        def _check() -> None:
            plan = get_acceleration_plan()
            label = plan.summary_label
            self.after(0, lambda: self.gpu_status_var.set(label))

        threading.Thread(target=_check, daemon=True).start()

    def _check_update_background(self) -> None:
        def _check() -> None:
            rel = check_for_updates()
            if rel:
                self.after(0, lambda: self._prompt_update(rel))

        threading.Thread(target=_check, daemon=True).start()

    def _manual_check_update(self) -> None:
        self.update_btn.config(state="disabled", text="Đang kiểm tra...")

        def _check() -> None:
            try:
                rel = check_for_updates()
                if rel:
                    self.after(0, lambda: self._prompt_update(rel))
                else:
                    self.after(0, lambda: self.banner.show(f"Bạn đang sử dụng phiên bản mới nhất (v{__version__}).", level="info"))
            finally:
                self.after(0, lambda: self.update_btn.config(state="normal", text="🔄 Kiểm tra cập nhật"))

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
                self.after(0, lambda: self._prompt_restart_update(script))
            except Exception as exc:
                self.after(0, lambda: self.banner.show(f"Lỗi cập nhật: {exc}", level="error"))

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
    # Settings Dialog
    # -------------------------------------------------------------------------

    def _open_settings_dialog(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Cài đặt ToolRecap V2")
        dlg.geometry("520x460")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()
        set_window_icon(dlg)

        frame = ttk.Frame(dlg, padding=16)
        frame.pack(fill="both", expand=True)

        # Video & Render Section
        vid_group = ttk.LabelFrame(frame, text="Cấu hình Video & Render", padding=10)
        vid_group.pack(fill="x", pady=(0, 10))

        # Quality
        ttk.Label(vid_group, text="Chất lượng video xuất:").grid(row=0, column=0, sticky="w", pady=4)
        quality_var = tk.StringVar(value=self.settings.quality)
        q_cbo = ttk.Combobox(vid_group, textvariable=quality_var, values=["standard", "high", "source"], state="readonly", width=18)
        q_cbo.grid(row=0, column=1, sticky="w", pady=4)

        # GPU acceleration
        gpu_var = tk.BooleanVar(value=self.settings.use_gpu)
        chk_gpu = ttk.Checkbutton(vid_group, text="Bật tăng tốc phần cứng GPU (NVENC/AMF/QSV)", variable=gpu_var)
        chk_gpu.grid(row=1, column=0, columnspan=3, sticky="w", pady=4)

        # Burn subtitles
        sub_var = tk.BooleanVar(value=self.settings.burn_subtitles)
        chk_sub = ttk.Checkbutton(vid_group, text="Nhúng thẳng phụ đề vào video (Burn subtitles)", variable=sub_var)
        chk_sub.grid(row=2, column=0, columnspan=3, sticky="w", pady=4)

        # Output folder
        ttk.Label(vid_group, text="Thư mục xuất video:").grid(row=3, column=0, sticky="w", pady=4)
        out_var = tk.StringVar(value=self.settings.output_dir)
        e_out = ttk.Entry(vid_group, textvariable=out_var, width=32)
        e_out.grid(row=3, column=1, sticky="ew", pady=4)

        def _choose_out() -> None:
            f = filedialog.askdirectory(parent=dlg, title="Chọn thư mục xuất video")
            if f:
                out_var.set(f)

        btn_browse = ttk.Button(vid_group, text="Duyệt...", command=_choose_out)
        btn_browse.grid(row=3, column=2, padx=(6, 0), pady=4)

        # Transcription / STT Section
        stt_group = ttk.LabelFrame(frame, text="Nhận diện giọng nói & Thoại (STT)", padding=10)
        stt_group.pack(fill="x", pady=(0, 12))

        provider_map = {
            "local": "Nội bộ (faster-whisper, tải mô hình khi cần)",
            "openai": "API OpenAI Whisper (yêu cầu API Key)",
        }
        rev_provider_map = {v: k for k, v in provider_map.items()}
        current_prov_display = provider_map.get(self.settings.transcription_provider, provider_map["local"])

        ttk.Label(stt_group, text="Bộ nhận diện (STT):").grid(row=0, column=0, sticky="w", pady=4)
        prov_var = tk.StringVar(value=current_prov_display)
        cbo_prov = ttk.Combobox(stt_group, textvariable=prov_var, values=list(provider_map.values()), state="readonly", width=38)
        cbo_prov.grid(row=0, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(stt_group, text="API Key (OpenAI):").grid(row=1, column=0, sticky="w", pady=4)
        api_key_var = tk.StringVar(value=self.settings.api_key)
        e_api_key = ttk.Entry(stt_group, textvariable=api_key_var, show="*", width=38)
        e_api_key.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(stt_group, text="API Base URL (tùy chọn):").grid(row=2, column=0, sticky="w", pady=4)
        api_base_var = tk.StringVar(value=self.settings.api_base_url)
        e_api_base = ttk.Entry(stt_group, textvariable=api_base_var, width=38)
        e_api_base.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

        lbl_note = ttk.Label(
            stt_group,
            text="* Chế độ nội bộ chạy hoàn toàn offline trên máy. Cấu hình bảo mật được lưu ngoài ứng dụng.",
            font=("Segoe UI", 8),
            foreground="#6b7280",
        )
        lbl_note.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # Save button
        def _save() -> None:
            self.settings.quality = quality_var.get()
            self.settings.use_gpu = gpu_var.get()
            self.settings.burn_subtitles = sub_var.get()
            self.settings.output_dir = out_var.get().strip()
            self.settings.transcription_provider = rev_provider_map.get(prov_var.get(), "local")
            self.settings.api_key = api_key_var.get().strip()
            self.settings.api_base_url = api_base_var.get().strip()
            self.settings_store.save(self.settings)
            self.queue.settings = self.settings
            dlg.destroy()
            self.banner.show("Đã lưu cài đặt thành công!", level="success")

        btn_save = ttk.Button(frame, text="Lưu thay đổi", style="Primary.TButton", command=_save)
        btn_save.pack(anchor="center")

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


def run_app() -> None:
    app = ToolRecapV2App()
    app.mainloop()
