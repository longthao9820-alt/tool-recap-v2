"""Automated packaging script for ToolRecap V2 portable Windows onedir distribution."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from toolrecap_v2.version import __version__


def find_system_ffmpeg() -> tuple[Path | None, Path | None]:
    """Find ffmpeg.exe and ffprobe.exe on the system."""
    ffmpeg_cand = shutil.which("ffmpeg")
    ffprobe_cand = shutil.which("ffprobe")

    ffmpeg_path = Path(ffmpeg_cand).resolve() if ffmpeg_cand else None
    ffprobe_path = Path(ffprobe_cand).resolve() if ffprobe_cand else None

    # Fallback to known winget path if which didn't find
    if not ffmpeg_path or not ffprobe_path:
        winget_base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        for p in winget_base.glob("**/ffmpeg.exe"):
            ffmpeg_path = p
            break
        for p in winget_base.glob("**/ffprobe.exe"):
            ffprobe_path = p
            break

    return ffmpeg_path, ffprobe_path


def verify_ffmpeg_capabilities(ffmpeg_path: Path) -> dict[str, bool]:
    """Inspect FFmpeg binary for compiled encoder support (NVENC/AMF/QSV)."""
    try:
        res = subprocess.run([str(ffmpeg_path), "-encoders"], capture_output=True, text=True, timeout=10)
        text = res.stdout.lower()
        return {
            "nvenc": "nvenc" in text,
            "amf": "amf" in text,
            "qsv": "qsv" in text,
        }
    except Exception:
        return {"nvenc": False, "amf": False, "qsv": False}


def create_portable_zip(app_dir: Path, zip_dest: Path) -> Path:
    """Create portable zip containing ToolRecapV2 root folder, excluding all dev/cache junk."""
    if zip_dest.exists():
        zip_dest.unlink()

    parent_dir = app_dir.parent
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in app_dir.rglob("*"):
            if not file_path.is_file():
                continue
            # Exclude unwanted source/cache/prime files
            parts = [p.lower() for p in file_path.parts]
            if any(p in parts for p in (".prime", ".git", ".pytest_cache", "__pycache__", "tests")):
                continue
            if file_path.name.lower().endswith((".pyc", ".pyo", ".tmp")):
                continue
            rel_archive_path = file_path.relative_to(parent_dir)
            zf.write(file_path, str(rel_archive_path))
    return zip_dest


def create_sha256_file(zip_path: Path) -> Path:
    """Create exact companion .zip.sha256.txt with `<hash> *<filename>` format."""
    h = hashlib.sha256(zip_path.read_bytes()).hexdigest().lower()
    sha_file = zip_path.parent / f"{zip_path.name}.sha256.txt"
    sha_file.write_text(f"{h} *{zip_path.name}\n", encoding="utf-8")
    return sha_file


def build_portable_package() -> int:
    repo_root = Path(__file__).resolve().parent
    spec_path = repo_root / "ToolRecapV2.spec"
    dist_dir = repo_root / "release"
    build_dir = repo_root / "build" / "ToolRecapV2"
    app_dir = dist_dir / "ToolRecapV2"

    print("==================================================")
    print(f"  BẮT ĐẦU ĐÓNG GÓI TOOLRECAP V2 v{__version__} PORTABLE WINDOWS")
    print("==================================================")

    # 0. Clean old build outputs
    print("[0/6] Làm sạch thư mục build cũ...")
    if app_dir.exists():
        shutil.rmtree(app_dir, ignore_errors=True)
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run PyInstaller
    print("[1/6] Chạy PyInstaller...")
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        "--distpath", str(dist_dir),
        str(spec_path),
    ]
    res = subprocess.run(cmd, cwd=str(repo_root))
    if res.returncode != 0:
        print("Lỗi: PyInstaller thất bại.")
        return 1

    exe_path = app_dir / "ToolRecapV2.exe"
    if not exe_path.is_file():
        print(f"Lỗi: Không tìm thấy file {exe_path}")
        return 1

    # 2. Bundle FFmpeg and FFprobe
    print("[2/6] Đóng gói FFmpeg và FFprobe vào runtime ứng dụng...")
    ffmpeg_bin, ffprobe_bin = find_system_ffmpeg()
    runtime_bin = app_dir / "runtime" / "ffmpeg" / "bin"
    runtime_bin.mkdir(parents=True, exist_ok=True)

    # Clean up duplicate top-level binaries if present
    for dup in (app_dir / "ffmpeg.exe", app_dir / "ffprobe.exe"):
        if dup.is_file():
            dup.unlink(missing_ok=True)

    if ffmpeg_bin and ffmpeg_bin.is_file():
        shutil.copy2(ffmpeg_bin, runtime_bin / "ffmpeg.exe")
        caps = verify_ffmpeg_capabilities(ffmpeg_bin)
        print(f"  + FFmpeg nhúng vào runtime: {runtime_bin / 'ffmpeg.exe'}")
        print(f"  + Khả năng mã hóa FFmpeg: NVENC={caps['nvenc']}, AMF={caps['amf']}, QSV={caps['qsv']}")
    else:
        print("  ! Cảnh báo: Không tìm thấy FFmpeg để đóng gói tự động.")

    if ffprobe_bin and ffprobe_bin.is_file():
        shutil.copy2(ffprobe_bin, runtime_bin / "ffprobe.exe")
        print(f"  + FFprobe nhúng vào runtime: {runtime_bin / 'ffprobe.exe'}")
    else:
        print("  ! Cảnh báo: Không tìm thấy FFprobe để đóng gói tự động.")

    # 3. Bundle Licenses and Guides
    print("[3/6] Đóng gói giấy phép, khởi chạy và hướng dẫn sử dụng...")
    if (repo_root / "LICENSE").is_file():
        shutil.copy2(repo_root / "LICENSE", app_dir / "LICENSE")
    if (repo_root / "THIRD_PARTY_LICENSES.md").is_file():
        shutil.copy2(repo_root / "THIRD_PARTY_LICENSES.md", app_dir / "THIRD_PARTY_LICENSES.md")

    guide_content = r"""========================================================================
             HƯỚNG DẪN SỬ DỤNG TOOLRECAP V2 (PORTABLE WINDOWS v0.2.0)
========================================================================

1. CÁCH MỞ ỨNG DỤNG:
   - Nhấp đúp chuột vào tệp: ToolRecapV2.exe (hoặc Chay-ToolRecapV2.cmd).
   - Ứng dụng chạy hoàn toàn độc lập (Portable), không cần cài đặt Python,
     không cần cài đặt FFmpeg hay bất kỳ phần mềm nào khác.

2. CÁC BƯỚC SỬ DỤNG:
   - Bước 1: Nhấn "Chọn 1 file video..." hoặc "Chọn thư mục chứa video...".
             Ứng dụng tự động lọc video (.mp4, .mkv, .mov, v.v.) và sắp xếp
             theo đúng thứ tự tập (ep1, ep2, ep10).
   - Bước 2: Chọn giọng đọc tiếng Anh mong muốn ở ô bên phải.
             Có thể nhấn "Nghe thử giọng" để kiểm tra âm thanh mẫu.
   - Bước 3: Cấu hình AI Gateway (⚙ Cài đặt):
             Mặc định ứng dụng kết nối tới AI Gateway tại http://127.0.0.1:20128/v1.
             + Scanner model: sub (thinking: max)
             + Finalizer model: prime (thinking: high)
             + Song song (parallelism): 2, Độ dài đoạn: 300s.
             Nhấn "Test Scanner" và "Test Finalizer" trong Cài đặt để kiểm tra kết nối.
             (Nếu muốn chạy offline hoàn toàn không cần gateway, bỏ chọn "Kích hoạt AI Gateway").
   - Bước 4: Nhấn nút to màu xanh "▶ Bắt đầu tự động".
             Ứng dụng sẽ tự động:
               + Phân tích nội dung thoại thực tế (phụ đề / faster-whisper).
               + Phân tích kịch bản 2 giai đoạn (Scanner -> Finalizer) qua AI Gateway.
               + Đọc lời dẫn thuyết minh chân thực bằng AI (Piper TTS).
               + Cắt cảnh khớp thời lượng, ghép giọng và tạo phụ đề SRT.
               + Xuất video recap hoàn chỉnh với tăng tốc phần cứng GPU.
   - Bước 5: Nhấn "📂 Mở thư mục kết quả" để xem video đã hoàn thành.

3. DỪNG AN TOÀN (CANCELLATION):
   - Bất kỳ lúc nào đang xử lý, bạn có thể nhấn "⏹ Dừng xử lý".
   - Ứng dụng sẽ lập tức dừng các phân đoạn, đóng các tiến trình FFmpeg và mở khóa lại giao diện.

4. NƠI LƯU TRỮ DỮ LIỆU & BỘ NHỚ ĐỆM (CACHE):
   - Cài đặt, lịch sử và mô hình AI được lưu tại: %LOCALAPPDATA%\ToolRecapV2
   - Kết quả phân tích AI Gateway được lưu đệm tự động tại:
     %LOCALAPPDATA%\ToolRecapV2\cache\gateway_analysis
     giúp các lần chạy lại không tốn API call khi nội dung và cấu hình không đổi.
   - API key nằm dạng văn bản trong settings.json cục bộ, không mã hóa.
     Không chia sẻ tệp này cho người khác. Không lưu API key vào cache.

5. NẾU CÓ LỖI XẢY RA:
   - Kiểm tra file log tại: %LOCALAPPDATA%\ToolRecapV2\logs
   - Nếu gateway chưa bật hoặc cổng 20128 chưa mở, ứng dụng sẽ báo lỗi rõ ràng.
     Vào Cài đặt để kiểm tra kết nối bằng nút Test hoặc tắt gateway nếu chạy offline.
   - Đảm bảo ổ đĩa còn đủ dung lượng trống để chứa video đầu ra.
========================================================================
"""
    (app_dir / "HUONG_DAN_SU_DUNG.txt").write_text(guide_content, encoding="utf-8")
    (dist_dir / "HUONG_DAN_CHAY.txt").write_text(guide_content, encoding="utf-8")

    run_cmd_content = """@echo off
start "" "%~dp0ToolRecapV2.exe"
exit
"""
    (app_dir / "Chay-ToolRecapV2.cmd").write_text(run_cmd_content, encoding="utf-8")

    # 4. Verify release files
    print("[4/6] Kiểm tra tính toàn vẹn của gói phát hành...")
    has_exe = (app_dir / "ToolRecapV2.exe").is_file()
    has_ffmpeg = (runtime_bin / "ffmpeg.exe").is_file()
    has_ffprobe = (runtime_bin / "ffprobe.exe").is_file()
    has_guide = (app_dir / "HUONG_DAN_SU_DUNG.txt").is_file()
    has_license = (app_dir / "LICENSE").is_file()
    has_third_party = (app_dir / "THIRD_PARTY_LICENSES.md").is_file()

    print(f"  - ToolRecapV2.exe: {'OK' if has_exe else 'THIẾU'}")
    print(f"  - runtime/ffmpeg/bin/ffmpeg.exe: {'OK' if has_ffmpeg else 'THIẾU'}")
    print(f"  - runtime/ffmpeg/bin/ffprobe.exe: {'OK' if has_ffprobe else 'THIẾU'}")
    print(f"  - Hướng dẫn: {'OK' if has_guide else 'THIẾU'}")
    print(f"  - Giấy phép LICENSE: {'OK' if has_license else 'THIẾU'}")
    print(f"  - THIRD_PARTY_LICENSES: {'OK' if has_third_party else 'THIẾU'}")

    if not (has_exe and has_ffmpeg and has_ffprobe and has_guide and has_license and has_third_party):
        print("Lỗi: Bản phát hành chưa đầy đủ các tệp cần thiết.")
        return 1

    # 5. Run --version and --self-check on built executable
    print("[5/6] Kiểm tra tự động trên file exe đã dựng...")
    try:
        ver_res = subprocess.run(
            [str(exe_path), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        print(f"  Version: {(ver_res.stdout or '').strip()}")
        chk_res = subprocess.run(
            [str(exe_path), "--self-check"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        out_msg = (chk_res.stdout or "").strip()
        print(f"  Self-check:\n{out_msg}")
        if chk_res.returncode != 0:
            print(f"Lỗi: Self-check trả về mã lỗi {chk_res.returncode}")
            return 1
    except Exception as exc:
        print(f"Lỗi khi chạy thử file exe: {exc}")
        return 1

    # 6. Create ZIP and SHA256 companion file
    print("[6/6] Tạo file nén Portable ZIP và mã băm SHA256...")
    zip_name = f"ToolRecapV2-v{__version__}-windows-portable.zip"
    zip_path = dist_dir / zip_name
    create_portable_zip(app_dir, zip_path)
    sha_path = create_sha256_file(zip_path)
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)

    print(f"  + ZIP Portable: {zip_path.name} ({zip_size_mb:.1f} MB)")
    print(f"  + SHA256 File:  {sha_path.name}")
    print(f"  + Checksum:     {sha_path.read_text(encoding='utf-8').strip()}")

    print("\n>>> ĐÓNG GÓI THÀNH CÔNG! BẢN PHÁT HÀNH TẠI:")
    print(f"    Thư mục: {app_dir}")
    print(f"    Tệp nén: {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(build_portable_package())
