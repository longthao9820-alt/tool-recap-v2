"""Path utilities, asset discovery, and platform integration for ToolRecap V2."""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
import tkinter as tk


def default_data_directory() -> Path:
    """Return user data directory outside the application directory.
    Default: %LOCALAPPDATA%\\ToolRecapV2 on Windows, ~/.toolrecap_v2 on other systems.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        data_dir = base / "ToolRecapV2"
    else:
        data_dir = Path.home() / ".toolrecap_v2"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_stt_model_cache_dir() -> Path:
    """Return model cache directory for local STT models (%LOCALAPPDATA%\\ToolRecapV2\\models\\stt\\)."""
    model_dir = default_data_directory() / "models" / "stt"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def application_root() -> Path:
    """Return root directory of the application."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # Development directory
    return Path(__file__).resolve().parents[1]


def get_asset_path(filename: str) -> Path | None:
    """Find an asset file across PyInstaller bundles, executable dirs, and dev paths."""
    candidates: list[Path] = []

    # 1. PyInstaller _MEIPASS
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "assets" / filename)
        candidates.append(Path(meipass) / filename)

    # 2. Frozen executable directory
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "assets" / filename)
        candidates.append(exe_dir / filename)

    # 3. Development / repo layout
    root = application_root()
    candidates.append(root / "assets" / filename)
    candidates.append(root / "toolrecap_v2" / "assets" / filename)

    for cand in candidates:
        if cand.is_file():
            return cand
    return None


# Robust lifecycle handling for Tk Image and Variable deletion
_orig_image_del = tk.Image.__del__


def _robust_image_del(self: tk.Image) -> None:
    if not getattr(self, "name", None):
        return
    try:
        _orig_image_del(self)
    except (tk.TclError, RuntimeError):
        pass


if tk.Image.__del__ is not _robust_image_del:
    tk.Image.__del__ = _robust_image_del

_orig_var_del = tk.Variable.__del__


def _robust_var_del(self: tk.Variable) -> None:
    try:
        _orig_var_del(self)
    except (tk.TclError, RuntimeError):
        pass


if tk.Variable.__del__ is not _robust_var_del:
    tk.Variable.__del__ = _robust_var_del


def set_window_icon(window: tk.Tk | tk.Toplevel) -> None:
    """Set window icon and configure Windows Taskbar AppUserModelID."""
    if sys.platform == "win32":
        try:
            app_id = "ToolRecapV2.DesktopApp.1.0"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass

    ico_path = get_asset_path("icon.ico")
    if ico_path and ico_path.is_file():
        try:
            window.iconbitmap(default=str(ico_path))
        except Exception:
            try:
                window.iconbitmap(str(ico_path))
            except Exception:
                pass

    png_path = get_asset_path("icon-256.png") or get_asset_path("icon.png")
    if png_path and png_path.is_file():
        try:
            img = tk.PhotoImage(master=window, file=str(png_path))
            window.iconphoto(True, img)
            window._icon_photo_ref = img  # type: ignore[attr-defined]

            def _cleanup_icon(event: object = None) -> None:
                if getattr(event, "widget", None) is window or event is None:
                    ref = getattr(window, "_icon_photo_ref", None)
                    if ref is not None:
                        try:
                            if ref.name and getattr(window, "tk", None) is not None:
                                window.tk.call("image", "delete", ref.name)
                        except Exception:
                            pass
                        ref.name = None
                        try:
                            delattr(window, "_icon_photo_ref")
                        except AttributeError:
                            pass

            window.bind("<Destroy>", _cleanup_icon, add="+")
        except Exception:
            pass
