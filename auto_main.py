"""Main application entry point for ToolRecap V2 desktop application."""
from __future__ import annotations

import os
import shutil
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

    # 3. Check ToolRecap's managed voice architecture (mutable runtime installs on demand).
    voice_ok = False
    try:
        from toolrecap_v2.voice.runtime import VOICE_RUNTIME_VERSION, runtime_fingerprint
        voice_ok = bool(VOICE_RUNTIME_VERSION and runtime_fingerprint())
    except Exception as exc:
        voice_ok = False
        errors.append(f"Không thể tải kiến trúc ToolRecap Local Voice ({exc}).")

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

    # 7. Check ToolRecap-owned voice runtime only.
    voice_backend_mode = "install-on-first-use"
    voice_backend_path = None
    try:
        from toolrecap_v2.voice.runtime import VoiceRuntimeInspector
        voice_health = VoiceRuntimeInspector().inspect(run_imports=False)
        if voice_health.runtime_present and voice_health.runtime_version_ok:
            voice_backend_mode = "toolrecap-managed"
            voice_backend_path = str(VoiceRuntimeInspector().python_executable)
        else:
            voice_backend_mode = "install-or-repair-on-first-use"
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

    # 9. Check FFmpeg filters and encoders for robust rendering
    filters_ok = True
    encoders_ok = True
    if ffmpeg_ok and ffmpeg:
        try:
            f_res = subprocess.run([str(ffmpeg), "-filters"], capture_output=True, text=True, timeout=10)
            f_text = (f_res.stdout or "").lower()
            required_filters = ["scale", "pad", "fps", "aresample", "aformat", "anullsrc", "concat"]
            missing_filters = [f for f in required_filters if f not in f_text]
            if missing_filters:
                filters_ok = False
                errors.append(f"FFmpeg thiếu các bộ lọc bắt buộc cho robust renderer: {', '.join(missing_filters)}")

            e_res = subprocess.run([str(ffmpeg), "-encoders"], capture_output=True, text=True, timeout=10)
            e_text = (e_res.stdout or "").lower()
            required_encoders = ["libx264", "aac"]
            missing_encoders = [e for e in required_encoders if e not in e_text]
            if missing_encoders:
                encoders_ok = False
                errors.append(f"FFmpeg thiếu các bộ mã hóa bắt buộc: {', '.join(missing_encoders)}")
        except Exception as exc:
            errors.append(f"Lỗi kiểm tra bộ lọc/mã hóa FFmpeg: {exc}")

    if not errors:
        print(f"ToolRecap V2 v{__version__} self-check PASSED (ffmpeg={ffmpeg_ok}, voice={voice_ok}, icon={icon_ok}, gui={gui_ok})")
        print(f"  - OCR runtime package: {ocr_pkg_name} {ocr_pkg_ver} (available={ocr_pkg_ok})")
        print(f"  - Models downloaded: OCR={'yes' if ocr_models_downloaded else 'no'}, Voice={'yes' if voice_models_downloaded else 'no'}, STT={'yes' if stt_models_downloaded else 'no'} (lazy assets, 'no' is valid)")
        print(f"  - Voice backend runtime: {voice_backend_mode} ({voice_backend_path or 'install-on-first-use'})")
        print(f"  - Robust renderer filters: {'OK' if filters_ok else 'FAIL'}, encoders: {'OK' if encoders_ok else 'FAIL'}")
        if hybrid_report:
            print(hybrid_report)
        return 0
    else:
        print(f"ToolRecap V2 v{__version__} self-check FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1


def _run_concat_check() -> int:
    """Run a fast synthetic multi-source concat self-check using bundled binary."""
    from toolrecap_v2.gpu import bundled_binary
    from toolrecap_v2.media import concat_media_clips

    ffmpeg = bundled_binary("ffmpeg")
    if not ffmpeg:
        print("Lỗi: Không tìm thấy FFmpeg để kiểm tra concat!")
        return 1

    temp_dir = default_data_directory() / "concat_selfcheck"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Clip 1: 320x240, 25fps, 44100Hz audio, 1.0s
        c1 = temp_dir / "clip1.mp4"
        cmd1 = [
            str(ffmpeg), "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=1.0:r=25",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1.0",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            "-shortest", str(c1),
        ]
        subprocess.run(cmd1, capture_output=True, check=True, timeout=15)

        # Clip 2: 640x360, 30fps, NO audio, 1.0s
        c2 = temp_dir / "clip2.mp4"
        cmd2 = [
            str(ffmpeg), "-y",
            "-f", "lavfi", "-i", "color=c=red:s=640x360:d=1.0:r=30",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-an", str(c2),
        ]
        subprocess.run(cmd2, capture_output=True, check=True, timeout=15)

        # Run concat_media_clips
        out_concat = temp_dir / "normalized_concat.mp4"
        probe_res = concat_media_clips([c1, c2], output=out_concat, require_audio=True)

        assert out_concat.is_file(), "Tệp sau ghép nối không tồn tại!"
        assert abs(probe_res.duration - 2.0) < 0.4, f"Thời lượng {probe_res.duration}s không khớp dự kiến 2.0s!"
        assert probe_res.has_audio and probe_res.audio_streams, "Tệp ghép nối thiếu luồng âm thanh!"
        audio_stream = probe_res.audio_streams[0]
        assert audio_stream.sample_rate == 48000, f"Sample rate {audio_stream.sample_rate} != 48000!"
        assert audio_stream.channels == 2, f"Channels {audio_stream.channels} != 2 (stereo)!"

        print(f"ToolRecap V2 v{__version__} concat self-check PASSED:")
        print(f"  - Output: {out_concat.name} ({probe_res.duration:.2f}s)")
        print(f"  - Audio: {audio_stream.sample_rate}Hz, {audio_stream.channels}ch (normalized stereo)")
        print(f"  - Video: {probe_res.width}x{probe_res.height}")
        return 0
    except Exception as exc:
        print(f"ToolRecap V2 v{__version__} concat self-check FAILED: {exc}")
        return 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    if "--version" in sys.argv:
        print(f"ToolRecap V2 v{__version__}")
        sys.exit(0)

    if "--self-check" in sys.argv:
        sys.exit(_run_self_check())

    if "--concat-check" in sys.argv:
        sys.exit(_run_concat_check())

    try:
        run_app()
    except BaseException as exc:
        _report_startup_error(exc)
        raise
