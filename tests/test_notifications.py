"""Tests for dismissible notification banner widget and real Windows desktop notification."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import pytest

from toolrecap_v2.notifications import (
    DesktopNotification,
    NotificationBanner,
    show_desktop_notification,
)


@pytest.fixture(scope="module")
def tk_root() -> tk.Tk:
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


def test_notification_banner_lifecycle(tk_root: tk.Tk) -> None:
    dismissed = False

    def _on_dismiss() -> None:
        nonlocal dismissed
        dismissed = True

    banner = NotificationBanner(tk_root, on_dismiss=_on_dismiss)
    assert banner.is_visible is False

    # Show info notification
    banner.show("Hoàn thành xử lý!", level="success")
    assert banner.is_visible is True
    assert banner.message_label.cget("text") == "Hoàn thành xử lý!"
    assert banner.icon_label.cget("text") == "✓"

    # Close button must exist and have visible '✕'
    assert banner.close_button.cget("text") == "✕"

    # Click close button to dismiss
    banner.close_button.invoke()
    assert banner.is_visible is False
    assert dismissed is True


def test_notification_banner_grid_compatibility(tk_root: tk.Tk) -> None:
    """Ensure banner works inside a parent frame managed exclusively by grid without TclError."""
    container = ttk.Frame(tk_root)
    container.grid(row=0, column=0)

    # Add a normal grid slave
    label = ttk.Label(container, text="Header")
    label.grid(row=0, column=0)

    banner = NotificationBanner(container)
    banner.grid(row=1, column=0)
    banner.grid_remove()

    # Calling show must not raise TclError (which happened when pack was used)
    banner.show("Test message", level="info", row=1, column=0)
    assert banner.is_visible is True

    banner.dismiss()
    assert banner.is_visible is False


def test_desktop_notification_lifecycle(tk_root: tk.Tk) -> None:
    dismissed = False

    def _on_dismiss() -> None:
        nonlocal dismissed
        dismissed = True

    notif = show_desktop_notification(
        title="ToolRecap V2",
        message="Đã hoàn thành 3/3 video recap!",
        level="success",
        parent=tk_root,
        auto_dismiss_ms=0,
        on_dismiss=_on_dismiss,
        play_sound=False,
    )

    assert isinstance(notif, DesktopNotification)
    assert notif.title() == "ToolRecap V2"
    assert notif.attributes("-topmost") == 1
    assert notif.message_label.cget("text") == "Đã hoàn thành 3/3 video recap!"
    assert notif.close_button.cget("text") == "✕"

    # Invoke visible [X] button immediately
    notif.close_button.invoke()
    assert dismissed is True


def test_ui_apply_batch_completed_notification_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure UI _apply_batch_completed calls real show_desktop_notification for both success and partial failure."""
    from toolrecap_v2.ui import ToolRecapV2App

    class MockApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.withdraw()
            self.progress_var = tk.DoubleVar(value=0.0)
            self.status_var = tk.StringVar(value="")
            self.banner = NotificationBanner(self)
            self._active_desktop_notification: DesktopNotification | None = None

        def _apply_batch_completed(self, completed: int, total: int) -> None:
            ToolRecapV2App._apply_batch_completed(self, completed, total)

    app = MockApp()
    try:
        # 1. Test success path (completed == total)
        app._apply_batch_completed(3, 3)
        assert app._active_desktop_notification is not None
        assert isinstance(app._active_desktop_notification, DesktopNotification)
        assert app._active_desktop_notification.master == app
        assert "3" in app._active_desktop_notification.message_label.cget("text")
        assert app._active_desktop_notification.close_button.cget("text") == "✕"

        # Dismiss via visible [X]
        app._active_desktop_notification.close_button.invoke()
        assert app._active_desktop_notification is None

        # 2. Test partial/failure path (completed < total)
        app._apply_batch_completed(1, 3)
        assert app._active_desktop_notification is not None
        assert isinstance(app._active_desktop_notification, DesktopNotification)
        assert "1/3" in app._active_desktop_notification.message_label.cget("text")
        assert app._active_desktop_notification.close_button.cget("text") == "✕"

        # Dismiss via visible [X]
        app._active_desktop_notification.close_button.invoke()
        assert app._active_desktop_notification is None
    finally:
        try:
            app.destroy()
        except Exception:
            pass
