"""End-to-end smoke test for ToolRecap V2 pipeline and GUI instantiation."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from toolrecap_v2.gpu import bundled_binary
from toolrecap_v2.media import probe_duration, probe_media
from toolrecap_v2.paths import default_data_directory
from toolrecap_v2.projects import ProjectQueue, ProjectRecord, ProjectStore
from toolrecap_v2.scanner import scan_videos
from toolrecap_v2.settings import AppSettings
from toolrecap_v2.ui import ToolRecapV2App


def run_smoke_test() -> int:
    print("==================================================")
    print("  CHẠY SMOKE TEST TOÀN DIỆN TOOLRECAP V2")
    print("==================================================")

    test_dir = default_data_directory() / "smoke_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a test video with audio and meaningful companion SRT
    ffmpeg = bundled_binary("ffmpeg")
    if not ffmpeg:
        print("Lỗi: Không tìm thấy FFmpeg!")
        return 1

    sample_video = test_dir / "episode_01.mp4"
    sample_srt = test_dir / "episode_01.srt"
    print(f"[1/4] Tạo video kiểm thử và phụ đề sidecar SRT tại: {sample_video.name}...")
    cmd = [
        str(ffmpeg),
        "-y",
        "-f", "lavfi", "-i", "color=c=navy:s=640x360:d=6:r=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        str(sample_video),
    ]
    res = subprocess.run(cmd, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if res.returncode != 0 or not sample_video.is_file():
        print(f"Lỗi tạo video mẫu: {res.stderr.decode()}")
        return 1

    # Write meaningful companion subtitles so narration analysis uses real dialogue
    srt_content = """1
00:00:00,500 --> 00:00:02,500
The hero begins his difficult journey across the mountains.

2
00:00:03,000 --> 00:00:05,500
He uncovers an unexpected secret that changes everything.
"""
    sample_srt.write_text(srt_content, encoding="utf-8")

    # Verify Piper TTS runtime and model availability
    try:
        import piper
        from piper import PiperVoice
        from toolrecap_v2.voice.manager import VoiceModelManager

        vm = VoiceModelManager()
        model_dir = vm.ensure_voice_model("piper.en_US-lessac-medium")
        print(f"  + Piper TTS sẵn sàng với model: {model_dir.name}")
    except Exception as exc:
        print(f"Lỗi: Piper TTS không khả dụng ({exc})!")
        return 1

    # 2. Test scanner
    print("[2/4] Kiểm tra bộ quét video (scanner)...")
    scanned = scan_videos(test_dir)
    assert sample_video.resolve() in scanned, "Scanner không tìm thấy video mẫu!"
    print(f"  + Tìm thấy {len(scanned)} video hợp lệ.")

    # 3. Test full batch sequential execution
    print("[3/4] Chạy chu trình sản xuất recap hoàn chỉnh...")
    out_dir = test_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    store = ProjectStore(test_dir / "projects.json")
    settings = AppSettings(quality="standard", use_gpu=False, burn_subtitles=True)

    record = ProjectRecord.from_video_path(sample_video, out_dir)
    finished = False

    def _batch_done(comp: int, tot: int) -> None:
        nonlocal finished
        finished = True

    queue = ProjectQueue(
        [record],
        store=store,
        settings=settings,
        on_batch_complete=_batch_done,
    )

    queue.start()
    for _ in range(60):
        if finished or not queue.is_running:
            break
        time.sleep(0.5)

    assert record.status == "COMPLETED", f"Pipeline thất bại: {record.error}"
    assert record.output_video and Path(record.output_video).is_file(), "Không tìm thấy video recap đầu ra!"
    assert record.output_srt and Path(record.output_srt).is_file(), "Không tìm thấy phụ đề SRT đầu ra!"

    out_probe = probe_media(record.output_video)
    print(f"  + Video recap hoàn thành: {Path(record.output_video).name} ({out_probe['duration']:.1f}s)")
    print(f"  + Phụ đề SRT hoàn thành: {Path(record.output_srt).name}")

    # 4. Test UI instantiation and key controls
    print("[4/4] Kiểm tra khởi tạo giao diện người dùng (GUI)...")
    try:
        app = ToolRecapV2App()
        app.update()  # Processes all UI events
        assert app.winfo_exists(), "Cửa sổ giao diện không tồn tại!"
        assert app.btn_start.winfo_exists(), "Nút Bắt đầu tự động không tồn tại!"
        assert app.btn_preview.winfo_exists(), "Nút Nghe thử giọng không tồn tại!"
        assert app.btn_voicestudio.winfo_exists(), "Nút VoiceStudio không tồn tại!"
        assert app.cbo_voice.winfo_exists(), "Combobox chọn giọng không tồn tại!"
        assert app.tree.winfo_exists(), "Bảng hàng đợi không tồn tại!"
        print(f"  + Giao diện khởi tạo thành công với đầy đủ nút bấm: {app.title()} ({app.winfo_width()}x{app.winfo_height()})")
        app.destroy()
    except Exception as exc:
        print(f"Lỗi khởi tạo giao diện: {exc}")
        return 1

    print("\n>>> TẤT CẢ CÁC BƯỚC SMOKE TEST ĐỀU ĐẠT THÀNH CÔNG (PASS)!")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
