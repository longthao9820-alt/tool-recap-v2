"""End-to-end smoke test for ToolRecap V2 pipeline, season multi-source, and GUI instantiation."""
from __future__ import annotations

import shutil
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
from toolrecap_v2.output_validation import validate_publication_folder
from toolrecap_v2.paths import default_data_directory
from toolrecap_v2.projects import ProjectQueue, ProjectRecord, ProjectStore
from toolrecap_v2.scanner import scan_videos
from toolrecap_v2.settings import AppSettings, SettingsStore
from toolrecap_v2.ui import SettingsDialog, ToolRecapV2App


def run_smoke_test() -> int:
    print("==================================================")
    print("  CHẠY SMOKE TEST TOÀN DIỆN TOOLRECAP V2")
    print("==================================================")

    test_dir = default_data_directory() / "smoke_test"
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a test video with audio and meaningful companion SRT
    ffmpeg = bundled_binary("ffmpeg")
    if not ffmpeg:
        print("Lỗi: Không tìm thấy FFmpeg!")
        return 1

    sample_video = test_dir / "episode_01.mp4"
    sample_srt = test_dir / "episode_01.srt"
    print(f"[1/5] Tạo video kiểm thử và phụ đề sidecar SRT tại: {sample_video.name}...")
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
        from toolrecap_v2.voice.manager import VoiceModelManager

        vm = VoiceModelManager()
        model_dir = vm.ensure_voice_model("piper.en_US-lessac-medium")
        print(f"  + Piper TTS sẵn sàng với model: {model_dir.name}")
    except Exception as exc:
        print(f"Lỗi: Piper TTS không khả dụng ({exc})!")
        return 1

    # 2. Test scanner (direct-only)
    print("[2/5] Kiểm tra bộ quét video (scanner direct-only)...")
    scanned = scan_videos(test_dir)
    assert sample_video.resolve() in scanned, "Scanner không tìm thấy video mẫu!"
    print(f"  + Tìm thấy {len(scanned)} video hợp lệ (direct-only non-recursive).")

    # 3. Test single episode batch sequential execution
    print("[3/5] Chạy chu trình sản xuất recap tập đơn lẻ (Single Episode)...")
    out_dir = test_dir / "output_single"
    out_dir.mkdir(parents=True, exist_ok=True)

    store = ProjectStore(test_dir / "projects_single.json")
    settings = AppSettings(quality="standard", use_gpu=False, burn_subtitles=True, gateway_enabled=False)

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
    for _ in range(180):
        if finished or not queue.is_running:
            break
        time.sleep(0.5)

    assert record.status == "COMPLETED", f"Pipeline tập đơn lẻ thất bại: {record.error}"
    assert record.output_video and Path(record.output_video).is_file(), "Không tìm thấy video recap đầu ra!"
    assert record.output_srt and Path(record.output_srt).is_file(), "Không tìm thấy phụ đề SRT đầu ra!"

    out_probe = probe_media(record.output_video)
    print(f"  + Video recap hoàn thành: {Path(record.output_video).name} ({out_probe['duration']:.1f}s)")
    print(f"  + Phụ đề SRT hoàn thành: {Path(record.output_srt).name}")

    # 4. Test season multi-source smoke execution
    print("[4/5] Chạy chu trình sản xuất recap cả mùa (Season Multi-Source)...")
    season_dir = test_dir / "season_source"
    season_dir.mkdir(parents=True, exist_ok=True)
    ep1_path = season_dir / "ep01.mp4"
    ep2_path = season_dir / "ep02.mp4"
    shutil.copy2(sample_video, ep1_path)
    shutil.copy2(sample_video, ep2_path)

    ep1_srt = season_dir / "ep01.srt"
    ep2_srt = season_dir / "ep02.srt"
    ep1_srt.write_text("""1\n00:00:00,500 --> 00:00:02,500\nEpisode one introduces the main detectives.\n\n2\n00:00:03,000 --> 00:00:05,500\nA mysterious crime occurs at midnight.\n""", encoding="utf-8")
    ep2_srt.write_text("""1\n00:00:00,500 --> 00:00:02,500\nEpisode two reveals a critical forensic lead.\n\n2\n00:00:03,000 --> 00:00:05,500\nThe suspect tries to flee the country.\n""", encoding="utf-8")

    season_scanned = scan_videos(season_dir)
    assert len(season_scanned) == 2, f"Scanner season mong muốn 2 video, nhận được {len(season_scanned)}"
    assert season_scanned[0].name == "ep01.mp4" and season_scanned[1].name == "ep02.mp4", "Thứ tự sắp xếp natural sort season không chuẩn xác!"

    season_out_dir = test_dir / "output_season"
    season_out_dir.mkdir(parents=True, exist_ok=True)
    season_store = ProjectStore(test_dir / "projects_season.json")
    season_record = ProjectRecord.from_season_paths(
        season_scanned,
        season_out_dir,
        title="Season01",
    )

    season_finished = False

    def _season_done(comp: int, tot: int) -> None:
        nonlocal season_finished
        season_finished = True

    season_queue = ProjectQueue(
        [season_record],
        store=season_store,
        settings=settings,
        on_batch_complete=_season_done,
    )

    season_queue.start()
    for _ in range(360):
        if season_finished or not season_queue.is_running:
            break
        time.sleep(0.5)

    assert season_record.status == "COMPLETED", f"Pipeline Season thất bại: {season_record.error}"
    assert len(season_record.outputs) >= 1, "Season analysis không tạo ra output nào!"
    print(f"  + Season analysis tạo thành công {len(season_record.outputs)} outputs.")

    # Validate exact three publication files for each output
    for out in season_record.outputs:
        assert out.publication_video_path, "Output thiếu publication_video_path"
        pub_folder = Path(out.publication_video_path).parent
        assert pub_folder.is_dir(), f"Thư mục xuất bản không tồn tại: {pub_folder}"
        val_res = validate_publication_folder(pub_folder, out.sanitized_title)
        assert val_res["video_path"], "Video MP4 không hợp lệ"
        assert val_res["original_srt_path"], "Phụ đề original.srt không hợp lệ"
        assert val_res["narration_srt_path"], "Phụ đề narration.srt không hợp lệ"
        print(f"  + Xuất bản Season Output hợp lệ (Outputs exact three): {out.sanitized_title}")

    # 5. Test UI instantiation and key controls (Main window + 4 Settings Panes)
    print("[5/5] Kiểm tra khởi tạo giao diện người dùng (GUI & 4 Settings Panes)...")
    try:
        app = ToolRecapV2App()
        app.update()
        assert app.winfo_exists(), "Cửa sổ giao diện không tồn tại!"
        assert app.btn_start.winfo_exists(), "Nút Start Creating Recap Videos không tồn tại!"
        assert app.btn_cancel.winfo_exists(), "Nút Stop không tồn tại!"
        assert app.btn_select_file.winfo_exists(), "Nút Select File không tồn tại!"
        assert app.btn_select_folder.winfo_exists(), "Nút Select Folder không tồn tại!"
        assert app.settings_btn.winfo_exists(), "Nút Settings không tồn tại!"
        assert app.tree.winfo_exists(), "Bảng hàng đợi không tồn tại!"

        # Test Settings Dialog: exactly 4 panes
        diag = SettingsDialog(app, settings, store)
        diag.update()
        assert set(diag._panes.keys()) == {"Recap", "AI Gateway", "Voice", "Render and Output"}
        for pane_name in ("Recap", "AI Gateway", "Voice", "Render and Output"):
            diag._show_pane(pane_name)
            diag.update()
        assert diag.btn_preview.winfo_exists(), "Nút Nghe thử giọng không tồn tại trong tab Voice!"
        assert diag.cbo_voice.winfo_exists(), "Combobox chọn giọng không tồn tại trong tab Voice!"
        assert diag.btn_voicestudio.winfo_exists(), "Nút VoiceStudio không tồn tại trong tab Voice!"
        diag.destroy()
        app.destroy()
        print(f"  + Giao diện và 4 tab Cài đặt khởi tạo thành công không lỗi.")
    except Exception as exc:
        print(f"Lỗi khởi tạo giao diện: {exc}")
        return 1

    print("\n>>> TẤT CẢ CÁC BƯỚC SMOKE TEST ĐỀU ĐẠT THÀNH CÔNG (PASS)!")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
