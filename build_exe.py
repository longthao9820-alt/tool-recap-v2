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
            if file_path.name.lower().endswith((".pyc", ".pyo", ".tmp", ".onnx", ".bin", ".pt", ".safetensors", ".model", ".tflite")):
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

    # Remove any model weights and test cache that might have leaked into app_dir
    model_exts = (".onnx", ".bin", ".pt", ".safetensors", ".model", ".tflite")
    for f in app_dir.rglob("*"):
        if f.is_file() and f.name.lower().endswith(model_exts):
            f.unlink()

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
             HƯỚNG DẪN SỬ DỤNG TOOLRECAP V2 (PORTABLE WINDOWS v0.4.0)
========================================================================

1. CÁCH MỞ ỨNG DỤNG:
   - Nhấp đúp chuột vào tệp: ToolRecapV2.exe (hoặc Chay-ToolRecapV2.cmd).
   - Ứng dụng chạy hoàn toàn độc lập (Portable), đã tích hợp sẵn runtime FFmpeg
     và bộ thư viện cần thiết, không cần cài đặt Python hay môi trường ngoài.

2. NGUỒN VIDEO (SINGLE VS SEASON & DIRECT-ONLY):
   - Xử lý 1 tập đơn lẻ: Nhấn "📁 Select File" để chọn 1 tệp video duy nhất.
   - Xử lý trọn bộ mùa phim: Nhấn "📂 Select Folder" để chọn thư mục mùa.
   - Quét trực tiếp (Direct-only): Ứng dụng chỉ quét các video nằm ngay trong
     thư mục được chọn, KHÔNG quét đệ quy thư mục con. Tự động nhận diện
     các định dạng video (.mp4, .mkv, .mov, .avi, .webm, .m4v, .ts) và sắp xếp
     theo thứ tự tập tự nhiên (ep1, ep2, ep10).

3. MỘT CÂU NHẮC ĐIỀU KHIỂN TOÀN BỘ CHU TRÌNH (ONE PROMPT FULL PIPELINE):
   - Người dùng chỉ cần nhập một câu nhắc duy nhất tại khung "Recap Prompt" trong
     Cài đặt (Tab 1 - Recap).
   - Hệ thống tự động phân tích ngoại tuyến câu nhắc này thành các chỉ thị chuyên biệt
     cho tất cả các khâu: Scanner (quét bằng chứng), Coverage (quét phủ lần hai),
     Connection (kết nối mùa), Candidate Discovery & Consolidation (khám phá và
     hợp nhất ứng viên), và Finalizer (hoàn thiện kịch bản). Không cần cấu hình rời rạc.

4. QUÉT PHỦ LẦN HAI & SỔ CÁI ĐỘ PHỦ (COVERAGE SECOND PASS):
   - Hệ thống tự động lập sổ cái (Coverage Ledger) theo dõi dòng thời gian và bằng chứng.
   - Khi phát hiện khoảng trống dữ liệu (gap) hoặc thiếu hụt diễn biến trọng tâm,
     hệ thống tự động kích hoạt lượt quét thứ hai có mục tiêu (second-pass) nhắm đúng
     vào khoảng thời gian thiếu, đảm bảo không bỏ sót tình tiết cốt lõi trong tập dài.

5. KHÁM PHÁ & HỢP NHẤT ỨNG VIÊN (CANDIDATE DISCOVERY & CONSOLIDATION):
   - Khám phá ứng viên: Tự động tìm kiếm các ý tưởng kịch bản tiềm năng (cảnh đơn lẻ,
     chuỗi cảnh liên kết, hoặc toàn bộ tuyến truyện mùa) bám sát mốc thời gian thực tế.
   - Hợp nhất ứng viên: Tự động gom nhóm các ứng viên trùng lặp hoặc bổ trợ cho nhau,
     giữ lại các góc nhìn độc đáo (kể cả nhân vật phụ), loại bỏ sự trùng lặp mà không
     làm mất tính phong phú của nội dung.

6. PHÂN ĐỊNH RÕ RÀNG LÝ DO 0 OUTPUT (GENUINE ZERO REASONS):
   - Khi một tập hoặc mùa phim không tạo ra video recap nào (0 output), hệ thống phân
     biệt chính xác:
     + Genuine Zero (Hợp lệ): Do nội dung nguồn không có sự kiện nào khớp với tiêu chí
       biên tập yêu cầu. Hệ thống kiểm toán độc lập, xác nhận hợp lệ và báo hoàn tất kèm lý do.
     + Lỗi kỹ thuật: Sự cố kết nối mạng, lỗi giải mã phụ đề hoặc lỗi dịch vụ AI. Hệ thống
       báo lỗi chính xác kèm mã lỗi để người dùng dễ nhận biết.

7. KHÔNG ÁP ĐẶT CHỈ TIÊU SỐ LƯỢNG CỨNG (NO QUOTA):
   - Tuyệt đối không áp đặt hạn mức hay chỉ tiêu số lượng video nhân tạo.
   - Nội dung có bao nhiêu câu chuyện chất lượng được chứng minh bằng bằng chứng thực tế
     thì tạo bấy nhiêu video recap tương ứng; không bịa đặt thêm để đủ số lượng.

8. CƠ CHẾ LÀM MỚI BỘ NHỚ ĐỆM THÔNG MINH (CACHE INVALIDATION):
   - Mã băm bộ nhớ đệm phụ thuộc trực tiếp vào câu nhắc biên tập (prompt hash), cấu hình
     và dấu vết tệp video nguồn.
   - Khi thay đổi câu nhắc hoặc đổi tệp video, bộ nhớ đệm tự động được làm mới chính xác.
   - Khi chạy lại với cùng câu nhắc và video cũ, hệ thống tái sử dụng ngay kết quả cache
     mà không tốn thêm bất kỳ cuộc gọi AI nào.

9. GIỚI HẠN THỰC TẾ VỀ LỜI THOẠI & HÌNH ẢNH (TRANSCRIPT-ONLY LIMITATION):
   - Hệ thống vận hành chủ yếu dựa trên phụ đề và lời thoại bóc tách từ video (Transcript-only).
   - Những diễn biến phim thuần túy bằng hình ảnh im lặng (không có lời thoại và không có
     phụ đề miêu tả) sẽ có giới hạn phản ánh trong kịch bản, trừ khi người dùng bật checkbox
     "Scanner model supports image/Vision input" trong Cài đặt để gửi hình ảnh lên mô hình Vision.

10. KIẾN TRÚC KỸ THUẬT (ENGINEERING ARCHITECTURE):
    - Nguyên nhân gốc rễ (Root Cause): Các phiên bản trước chỉ dùng 1 lượt quét đơn tuyến
      dễ để lại khoảng trống dữ liệu trên video dài, câu nhắc bị phân mảnh gán cứng, thiếu
      bước khám phá và gộp nhóm ứng viên độc lập, và không phân định được 0 output hợp lệ.
    - Các mô-đun mới (New Modules v0.4.0):
      + toolrecap_v2.domain.policy: Giải mã một câu nhắc duy nhất thành tập hợp chỉ thị cấu trúc.
      + toolrecap_v2.analyzer.coverage: Sổ cái độ phủ, phát hiện khoảng trống và quét bù lần 2.
      + toolrecap_v2.analyzer.candidates: Gồm discovery.py (khám phá), consolidation.py (hợp nhất)
        và verifier.py (kiểm toán độc lập 0 output và xác nhận Genuine Zero).

11. BẢNG CÀI ĐẶT (CHÍNH XÁC 4 TAB):
    Nhấn "⚙ Settings" trên thanh công cụ để mở cửa sổ cấu hình gồm đúng 4 tab:
    - Tab 1 - Recap: Ngôn ngữ kịch bản (en-US, en-GB), chế độ recap (MAIN_STORIES,
      FULL_EPISODE), thể loại (US_TV_SHOW, DE_GERMAN_SOAP, BODYCAM, FEATURE_FILM, OTHER),
      bản quyền tư liệu và khung nhập Recap Prompt (kèm nút Reload Default Prompt).
    - Tab 2 - AI Gateway: Kích hoạt AI Gateway, API endpoint (mặc định
      http://127.0.0.1:20128/v1), API key (lưu an toàn cục bộ), Scanner model
      (sub - thinking max), checkbox rõ ràng "Scanner model supports image/Vision input",
      Finalizer model (prime - thinking high), số luồng song song (1-4) và độ dài đoạn (60-900s).
    - Tab 3 - Voice: Lựa chọn 12 giọng thiết kế chuẩn (Neighbor, Companion...), phong cách
      giọng đọc, nút "🔊 Nghe thử giọng", nút "🎙 Cập nhật VoiceStudio", và khu vực
      Audio Mix chuyên nghiệp (âm lượng gốc dB, âm lượng thuyết minh dB, Auto-ducking,
      Target loudness -14 LUFS, True peak -1 dBTP).
    - Tab 4 - Render and Output: Chất lượng video (standard/high/source), bật/tắt GPU
      (NVENC/AMF/QSV), nhúng phụ đề (Burn subtitles), thư mục xuất và kiểm tra subsystem.

12. ĐẦU RA CHÍNH XÁC 3 TỆP (OUTPUTS EXACT THREE):
    Mỗi phân đoạn video recap được xuất vào thư mục riêng với ĐÚNG 3 tệp thành phẩm:
    1. {safe_title}.mp4: Video recap hoàn chỉnh chất lượng cao.
    2. {safe_title}.original.srt: Phụ đề các đoạn thoại gốc giữ lại trong video.
    3. {safe_title}.narration.srt: Phụ đề lời dẫn thuyết minh AI chuẩn xác.
    Tuyệt đối sạch sẽ, không có tệp tạm hay tệp rác.

13. DỪNG AN TOÀN (STOP / CANCELLATION):
    - Nhấn "⏹ Stop" bất kỳ lúc nào để dừng xử lý ngay lập tức.
    - Hệ thống ngắt chu kỳ chờ thử lại (backoff delay), hủy các giai đoạn kế tiếp,
      đóng sạch cây tiến trình con FFmpeg, mở khóa lại giao diện và đánh dấu CANCELLED.
    - Giới hạn kỹ thuật chính xác: Yêu cầu mạng HTTP urlopen đang gửi/nhận dở dang trên
      socket chỉ có thể kết thúc khi nhận được phản hồi hoặc hết socket timeout; sau đó
      tiến trình dừng hoàn toàn theo cờ Stop mà không sinh tiến trình mồ côi (orphan).

14. CẬP NHẬT & DỮ LIỆU:
    - Tự động kiểm tra GitHub Releases chính thức từ longthao9820-alt/tool-recap-v2 kèm SHA256.
    - Dữ liệu lưu ngoài thư mục ứng dụng tại %LOCALAPPDATA%\ToolRecapV2.
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
