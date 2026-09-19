"""Real Windows notifications and dismissible in-app banner widget for ToolRecap V2."""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk
from typing import Callable


class NotificationBanner(ttk.Frame):
    """Dismissible in-app banner with a visible 'X' button, using consistent grid geometry."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        on_dismiss: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, padding=(10, 6))
        self.on_dismiss = on_dismiss
        self._visible = False
        self._auto_dismiss_job: str | None = None

        # Internal grid configuration
        self.columnconfigure(1, weight=1)

        # Icon / Type indicator
        self.icon_label = ttk.Label(self, text="ℹ", font=("Segoe UI", 11, "bold"))
        self.icon_label.grid(row=0, column=0, padx=(4, 8), sticky="w")

        # Message text
        self.message_label = ttk.Label(self, text="", font=("Segoe UI", 9), wraplength=700)
        self.message_label.grid(row=0, column=1, sticky="ew")

        # Visible [X] dismiss button
        self.close_button = ttk.Button(
            self,
            text="✕",
            width=3,
            command=self.dismiss,
            style="NotificationClose.TButton",
        )
        self.close_button.grid(row=0, column=2, padx=(8, 4), sticky="e")

        self._configure_styles()

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.configure("NotificationClose.TButton", font=("Segoe UI", 9, "bold"), padding=2)

    @property
    def is_visible(self) -> bool:
        return self._visible

    def show(
        self,
        message: str,
        level: str = "info",
        *,
        auto_dismiss_ms: int = 0,
        row: int = 1,
        column: int = 0,
    ) -> None:
        """Display notification message using grid geometry consistently with parent container."""
        if self._auto_dismiss_job:
            self.after_cancel(self._auto_dismiss_job)
            self._auto_dismiss_job = None

        icons = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✕",
        }
        self.icon_label.config(text=icons.get(level, "ℹ"))
        self.message_label.config(text=message)

        # Use grid exclusively - never call pack()
        self.grid(row=row, column=column, sticky="ew", pady=(0, 8))
        self._visible = True

        if auto_dismiss_ms > 0:
            self._auto_dismiss_job = self.after(auto_dismiss_ms, self.dismiss)

    def dismiss(self) -> None:
        """Hide notification via grid_remove and invoke dismiss callback."""
        if self._auto_dismiss_job:
            self.after_cancel(self._auto_dismiss_job)
            self._auto_dismiss_job = None

        if self._visible:
            self.grid_remove()
            self._visible = False

        if self.on_dismiss:
            self.on_dismiss()


class DesktopNotification(tk.Toplevel):
    """Real Windows top-level notification window placed at screen bottom-right with visible [X]."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel | None = None,
        *,
        title: str = "ToolRecap V2",
        message: str = "Hoàn thành xử lý!",
        level: str = "info",
        auto_dismiss_ms: int = 8000,
        on_dismiss: Callable[[], None] | None = None,
        play_sound: bool = True,
    ) -> None:
        super().__init__(parent)
        self.on_dismiss = on_dismiss
        self._auto_dismiss_job: str | None = None
        self._dismissed = False

        # Configure window: topmost, borderless or floating dialog
        self.title(title)
        self.attributes("-topmost", True)
        self.resizable(False, False)

        width = 380
        height = 96
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        pos_x = max(10, screen_w - width - 24)
        pos_y = max(10, screen_h - height - 64)
        self.geometry(f"{width}x{height}+{pos_x}+{pos_y}")

        # Main frame
        container = ttk.Frame(self, padding=(12, 8))
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)

        icons = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✕",
        }
        icon_symbol = icons.get(level, "ℹ")

        # Top row: Icon + Title + Visible [X] Button
        lbl_icon = ttk.Label(container, text=icon_symbol, font=("Segoe UI", 11, "bold"))
        lbl_icon.grid(row=0, column=0, padx=(0, 6), sticky="w")

        lbl_title = ttk.Label(container, text=title, font=("Segoe UI", 9, "bold"))
        lbl_title.grid(row=0, column=1, sticky="w")

        self.close_button = ttk.Button(
            container,
            text="✕",
            width=3,
            command=self.dismiss,
            style="NotificationClose.TButton",
        )
        self.close_button.grid(row=0, column=2, sticky="e")

        # Body row: Message text
        self.message_label = ttk.Label(
            container,
            text=message,
            font=("Segoe UI", 9),
            wraplength=340,
        )
        self.message_label.grid(row=1, column=0, columnspan=3, pady=(6, 0), sticky="w")

        # Optional audio cue on Windows
        if play_sound and sys.platform == "win32":
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception:
                pass

        if auto_dismiss_ms > 0:
            self._auto_dismiss_job = self.after(auto_dismiss_ms, self.dismiss)

    def dismiss(self) -> None:
        """Immediately dismiss the top-level notification and trigger callback."""
        if self._dismissed:
            return
        self._dismissed = True

        if self._auto_dismiss_job:
            try:
                self.after_cancel(self._auto_dismiss_job)
            except Exception:
                pass
            self._auto_dismiss_job = None

        if self.on_dismiss:
            try:
                self.on_dismiss()
            except Exception:
                pass

        try:
            self.destroy()
        except Exception:
            pass


def show_desktop_notification(
    title: str,
    message: str,
    level: str = "info",
    parent: tk.Tk | tk.Toplevel | None = None,
    *,
    auto_dismiss_ms: int = 8000,
    on_dismiss: Callable[[], None] | None = None,
    play_sound: bool = True,
) -> DesktopNotification:
    """Convenience helper to spawn a desktop notification window."""
    return DesktopNotification(
        parent=parent,
        title=title,
        message=message,
        level=level,
        auto_dismiss_ms=auto_dismiss_ms,
        on_dismiss=on_dismiss,
        play_sound=play_sound,
    )
