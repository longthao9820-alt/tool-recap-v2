"""Main application entry point for ToolRecap V2 desktop application."""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from toolrecap_v2.paths import default_data_directory, get_asset_path
from toolrecap_v2.ui import run_app
from toolrecap_v2.version import __version__


def _report_startup_error(exc: BaseException) -> None:
    log_dir = default_data_directory() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "startup_error.log"
    log_file.write_text(traceback.format_exc(), encoding="utf-8")

    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "ToolRecap V2 không thể khởi động",
            f"Lỗi: {exc}\n\nChi tiết lỗi đã được ghi vào tệp:\n{log_file}",
            parent=root,
        )
        root.destroy()
    except Exception:
        pass


def _run_self_check() -> int:
    from toolrecap_v2.gpu import bundled_binary, get_acceleration_plan

    errors: list[str] = []

    # 1. Check icon (mandatory)
    icon = get_asset_path("icon.ico")
    icon_ok = icon is not None and icon.is_file()
    if not icon_ok:
        errors.append("Thiếu icon ứng dụng (icon.ico).")

    # 2. Check FFmpeg (mandatory)
    ffmpeg = bundled_binary("ffmpeg")
    ffmpeg_ok = False
    if ffmpeg:
        try:
            res = subprocess.run([str(ffmpeg), "-version"], capture_output=True, text=True, timeout=10)
            ffmpeg_ok = res.returncode == 0
        except Exception as exc:
            errors.append(f"Không thể chạy FFmpeg ({ffmpeg}): {exc}")
    else:
        errors.append("Không tìm thấy FFmpeg trong runtime hoặc PATH.")

    # 3. Check voice synthesis runtime (Piper builtin - mandatory)
    voice_ok = False
    try:
        import piper
        from piper import PiperVoice
        voice_ok = True
    except Exception as exc:
        voice_ok = False
        errors.append(f"Không thể tải runtime giọng nói Piper ({exc}).")

    # 4. Check GUI module (mandatory)
    gui_ok = False
    try:
        import tkinter
        gui_ok = True
    except Exception as exc:
        errors.append(f"Không thể tải giao diện đồ họa Tkinter: {exc}")

    # 5. Check OCR runtime package (mandatory package)
    ocr_pkg_ok = False
    ocr_pkg_name = "None"
    ocr_pkg_ver = ""
    try:
        import rapidocr_onnxruntime
        ocr_pkg_ok = True
        ocr_pkg_name = "rapidocr_onnxruntime"
        try:
            import importlib.metadata
            ocr_pkg_ver = importlib.metadata.version("rapidocr_onnxruntime")
        except Exception:
            ocr_pkg_ver = getattr(rapidocr_onnxruntime, "__version__", "")
    except ImportError:
        try:
            import rapidocr
            ocr_pkg_ok = True
            ocr_pkg_name = "rapidocr"
            try:
                import importlib.metadata
                ocr_pkg_ver = importlib.metadata.version("rapidocr")
            except Exception:
                ocr_pkg_ver = getattr(rapidocr, "__version__", "")
        except ImportError as exc:
            errors.append(f"Không tìm thấy gói runtime OCR (rapidocr_onnxruntime/rapidocr): {exc}")

    # 6. Check Lazy Models downloaded: yes/no (no is completely valid)
    ocr_models_downloaded = False
    try:
        from toolrecap_v2.subtitles.ocr import OcrModelManager
        ocr_models_downloaded = OcrModelManager().are_models_available()
    except Exception:
        ocr_models_downloaded = False

    voice_models_downloaded = False
    try:
        from toolrecap_v2.voice.manager import get_voice_manager
        vm = get_voice_manager()
        voice_models_downloaded = vm._is_omnivoice_model_cached() or any(
            vm.is_voice_installed(vid) for vid in ("piper.en_US-lessac-medium", "voicestudio.en.neighbor")
        )
    except Exception:
        voice_models_downloaded = False

    stt_models_downloaded = False
    try:
        stt_cache = default_data_directory() / "models" / "stt"
        stt_models_downloaded = stt_cache.is_dir() and any(stt_cache.iterdir())
    except Exception:
        stt_models_downloaded = False

    # 7. Check Voice backend runtime: official / isolated / install-on-first-use
    voice_backend_mode = "install-on-first-use"
    voice_backend_path = None
    try:
        from toolrecap_v2.voice.catalog import detect_official_voicestudio_runtime, get_isolated_runtime_python
        official = detect_official_voicestudio_runtime()
        if official is not None and official.is_file():
            voice_backend_mode = "official"
            voice_backend_path = str(official)
        else:
            isolated = get_isolated_runtime_python()
            if isolated is not None and isolated.is_file():
                voice_backend_mode = "isolated"
                voice_backend_path = str(isolated)
            else:
                voice_backend_mode = "install-on-first-use"
    except Exception as exc:
        voice_backend_mode = f"install-on-first-use (probe error: {exc})"

    # 8. Check hardware acceleration and FFmpeg capabilities
    accel_summary = "Không khả dụng"
    hybrid_report = ""
    if ffmpeg_ok and ffmpeg:
        try:
            plan = get_acceleration_plan(use_gpu=True)
            accel_summary = f"{plan.summary_label} ({plan.encoder.encoder})"
            caps = plan.capabilities
            hybrid_type = "Hybrid (QSV decode + NVENC encode)" if plan.is_hybrid else f"Tiêu chuẩn ({plan.decoder.method} decode + {plan.encoder.encoder} encode)"
            hybrid_report = (
                f"  - Kế hoạch tăng tốc: {hybrid_type}\n"
                f"  - Phần cứng: {accel_summary}\n"
                f"  - Khả năng FFmpeg/GPU: QSV_decode={caps.qsv_decode}, NVENC={caps.nvenc_encode}, AMF={caps.amf_encode}, QSV_encode={caps.qsv_encode}\n"
                f"  - GPU phát hiện: {', '.join(caps.gpu_names) if caps.gpu_names else 'None'}"
            )
        except Exception as exc:
            accel_summary = f"Lỗi phát hiện: {exc}"

    if not errors:
        print(f"ToolRecap V2 v{__version__} self-check PASSED (ffmpeg={ffmpeg_ok}, voice={voice_ok}, icon={icon_ok}, gui={gui_ok})")
        print(f"  - OCR runtime package: {ocr_pkg_name} {ocr_pkg_ver} (available={ocr_pkg_ok})")
        print(f"  - Models downloaded: OCR={'yes' if ocr_models_downloaded else 'no'}, Voice={'yes' if voice_models_downloaded else 'no'}, STT={'yes' if stt_models_downloaded else 'no'} (lazy assets, 'no' is valid)")
        print(f"  - Voice backend runtime: {voice_backend_mode} ({voice_backend_path or 'install-on-first-use'})")
        if hybrid_report:
            print(hybrid_report)
        return 0
    else:
        print(f"ToolRecap V2 v{__version__} self-check FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1


if __name__ == "__main__":
    if "--version" in sys.argv:
        print(f"ToolRecap V2 v{__version__}")
        sys.exit(0)

    if "--self-check" in sys.argv:
        sys.exit(_run_self_check())

    try:
        run_app()
    except BaseException as exc:
        _report_startup_error(exc)
        raise
