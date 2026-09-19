"""Project queue, state persistence, and sequential batch execution with safe cancellation."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .gpu import video_encode_args
from .media import (
    MediaError,
    RenderCancelled,
    cut_clip,
    find_binary,
    probe_duration,
    probe_media,
    render_final_video,
    run_command,
    write_srt_file,
)
from .narration import RecapManifest, prepare_narration_for_video
from .paths import default_data_directory
from .settings import AppSettings
from .voice.catalog import DEFAULT_VOICE_ID
from .voice.manager import get_voice_manager


ProjectCallback = Callable[["ProjectRecord"], None]
BatchCallback = Callable[[int, int], None]


@dataclass
class ProjectRecord:
    id: str
    name: str
    source_video: str
    manifest_path: str = ""
    output_directory: str = ""
    voice_id: str = DEFAULT_VOICE_ID
    status: str = "WAITING"  # "WAITING", "RUNNING", "COMPLETED", "CANCELLED", "ERROR", "PAUSED"
    progress: int = 0
    current_message: str = "Sẵn sàng"
    error: str | None = None
    output_video: str | None = None
    output_srt: str | None = None

    @classmethod
    def from_video_path(
        cls,
        video_path: Path,
        output_root: Path,
        voice_id: str = DEFAULT_VOICE_ID,
    ) -> "ProjectRecord":
        stem = video_path.stem
        out_dir = output_root / stem
        return cls(
            id=f"{stem}-{uuid.uuid4().hex[:8]}",
            name=stem,
            source_video=str(video_path),
            output_directory=str(out_dir),
            voice_id=voice_id,
            current_message="Sẵn sàng",
        )


class ProjectStore:
    """Atomic persistent storage for project queue in %LOCALAPPDATA%."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (default_data_directory() / "projects.json")
        self._lock = threading.RLock()

    def load(self) -> list[ProjectRecord]:
        with self._lock:
            if not self.path.is_file():
                return []
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                records: list[ProjectRecord] = []
                allowed_fields = ProjectRecord.__dataclass_fields__
                for item in data.get("projects", []):
                    if not isinstance(item, dict) or not item.get("source_video"):
                        continue
                    filtered = {k: v for k, v in item.items() if k in allowed_fields}
                    record = ProjectRecord(**filtered)
                    # State recovery: reset interrupted states to PAUSED
                    if record.status in {"RUNNING", "QUEUED"}:
                        record.status = "PAUSED"
                        record.current_message = "Đã dừng khi ứng dụng đóng trước đó"
                    records.append(record)
                return records
            except Exception:
                return []

    def save(self, projects: list[ProjectRecord]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            data = {"projects": [asdict(p) for p in projects]}
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)


class ProjectQueue:
    """Manages sequential full-batch processing of video recap projects with safe cancellation."""

    def __init__(
        self,
        projects: list[ProjectRecord],
        store: ProjectStore,
        settings: AppSettings,
        *,
        on_update: ProjectCallback | None = None,
        on_batch_complete: BatchCallback | None = None,
        on_state_change: Callable[[bool], None] | None = None,
    ) -> None:
        self.projects = projects
        self.store = store
        self.settings = settings
        self.on_update = on_update
        self.on_batch_complete = on_batch_complete
        self.on_state_change = on_state_change  # Called with is_running: bool to enable/disable UI controls

        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._is_running = False
        self._lock = threading.RLock()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self) -> None:
        """Start sequential processing thread."""
        with self._lock:
            if self._is_running:
                return
            self._cancel_event.clear()
            self._is_running = True
            if self.on_state_change:
                self.on_state_change(True)

            self._thread = threading.Thread(target=self._process_queue, daemon=True)
            self._thread.start()

    def cancel(self) -> None:
        """Trigger cancellation for the active project and sequential queue."""
        with self._lock:
            self._cancel_event.set()

    def _notify_update(self, record: ProjectRecord) -> None:
        self.store.save(self.projects)
        if self.on_update:
            self.on_update(record)

    def _process_queue(self) -> None:
        total = len(self.projects)
        completed_count = 0
        try:
            for idx, record in enumerate(self.projects):
                if self._cancel_event.is_set():
                    if record.status == "WAITING":
                        record.status = "CANCELLED"
                        record.current_message = "Đã dừng bởi người dùng"
                        self._notify_update(record)
                    continue

                if record.status in {"COMPLETED"}:
                    completed_count += 1
                    continue

                # Process single record
                try:
                    self._process_single_project(record)
                    if record.status == "COMPLETED":
                        completed_count += 1
                except RenderCancelled:
                    record.status = "CANCELLED"
                    record.current_message = "Đã dừng theo yêu cầu của người dùng"
                    self._notify_update(record)
                    # Mark all subsequent waiting projects as CANCELLED
                    for remaining in self.projects[idx + 1:]:
                        if remaining.status == "WAITING":
                            remaining.status = "CANCELLED"
                            remaining.current_message = "Đã dừng bởi người dùng"
                            self._notify_update(remaining)
                    break
                except Exception as exc:
                    record.status = "ERROR"
                    record.error = str(exc)
                    record.current_message = f"Lỗi: {exc}"
                    self._notify_update(record)
                    # Continue to next item in queue despite single error
        finally:
            with self._lock:
                self._is_running = False
                if self.on_state_change:
                    self.on_state_change(False)
                if self.on_batch_complete:
                    self.on_batch_complete(completed_count, total)
                self.store.save(self.projects)

    def _process_single_project(self, record: ProjectRecord) -> None:
        """Execute full pipeline for one project record sequentially."""
        record.status = "RUNNING"
        record.progress = 5
        record.current_message = "Bắt đầu phân tích video..."
        self._notify_update(record)

        video_path = Path(record.source_video).resolve()
        if not video_path.is_file():
            raise MediaError(f"Không tìm thấy file video nguồn: {video_path}")

        out_dir = Path(record.output_directory).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = out_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        voice_mgr = get_voice_manager()

        # Step 1: Narration Preparation (Real & clearly defined)
        if self._cancel_event.is_set():
            raise RenderCancelled()

        record.current_message = "Chuẩn bị kịch bản thuyết minh (narration)..."
        record.progress = 15
        self._notify_update(record)

        def _narration_log(msg: str) -> None:
            record.current_message = msg
            self._notify_update(record)

        manifest = prepare_narration_for_video(
            video_path,
            out_dir,
            language="en-US",
            cancel_event=self._cancel_event,
            log=_narration_log,
            settings=self.settings,
        )
        record.manifest_path = str(out_dir / f"{video_path.stem}_manifest.json")

        # Step 2: Voice Audio Synthesis
        if self._cancel_event.is_set():
            raise RenderCancelled()

        record.current_message = f"Tổng hợp âm thanh thuyết minh ({record.voice_id})..."
        record.progress = 30
        self._notify_update(record)

        total_segs = len(manifest.segments)
        audio_files: list[Path] = []
        for s_idx, seg in enumerate(manifest.segments, start=1):
            if self._cancel_event.is_set():
                raise RenderCancelled()

            seg_audio = temp_dir / f"audio_seg_{s_idx:02d}.wav"

            def _synth_prog(current: int, total: int, pct: float) -> None:
                if total > 0 and current < total:
                    record.current_message = f"Đang tải model giọng {record.voice_id} ({pct:.0f}%)..."
                    record.progress = min(35, 30 + int(pct * 0.05))
                else:
                    seg_pct = min(48, int(35 + (s_idx / max(1, total_segs)) * 13))
                    record.progress = seg_pct
                    record.current_message = f"Tổng hợp âm thanh phân đoạn {s_idx}/{total_segs}..."
                self._notify_update(record)

            voice_mgr.synthesize(
                record.voice_id,
                seg.narration_text,
                seg_audio,
                progress_callback=_synth_prog,
                cancel_event=self._cancel_event,
            )
            audio_files.append(seg_audio)

        # Step 3: Video Scene Cutting and Audio Integration
        if self._cancel_event.is_set():
            raise RenderCancelled()

        record.current_message = "Cắt và dựng các phân đoạn recap..."
        record.progress = 50
        self._notify_update(record)

        cut_clips: list[Path] = []
        for s_idx, (seg, audio_file) in enumerate(zip(manifest.segments, audio_files), start=1):
            if self._cancel_event.is_set():
                raise RenderCancelled()

            clip_path = temp_dir / f"clip_seg_{s_idx:02d}.mp4"
            audio_dur = probe_duration(audio_file)
            # Match visual duration to speech duration with a small breathing buffer
            visual_dur = max(audio_dur + 0.3, seg.duration_sec)
            seg_end = seg.start_sec + visual_dur

            # Cut visual clip
            cut_clip(
                video_path,
                clip_path,
                seg.start_sec,
                seg_end,
                use_gpu=self.settings.use_gpu,
                cancel_event=self._cancel_event,
            )

            # Combine clip with narration audio
            merged_clip = temp_dir / f"merged_seg_{s_idx:02d}.mp4"
            merge_cmd = [
                find_binary("ffmpeg"),
                "-y",
                "-i", str(clip_path),
                "-i", str(audio_file),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(merged_clip),
            ]
            run_command(merge_cmd, cancel_event=self._cancel_event)
            cut_clips.append(merged_clip)

        # Step 4: Concatenate Scenes into Final Recap Video
        if self._cancel_event.is_set():
            raise RenderCancelled()

        record.current_message = "Ghép nối video hoàn chỉnh..."
        record.progress = 75
        self._notify_update(record)

        concat_list = temp_dir / "concat.txt"
        concat_lines = [f"file '{c.resolve()}'" for c in cut_clips]
        concat_list.write_text("\n".join(concat_lines), encoding="utf-8")

        raw_recap_video = temp_dir / "recap_raw.mp4"
        concat_cmd = [
            find_binary("ffmpeg"),
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(raw_recap_video),
        ]
        run_command(concat_cmd, cancel_event=self._cancel_event)

        # Step 5: Subtitle Generation & Burning
        if self._cancel_event.is_set():
            raise RenderCancelled()

        record.current_message = "Tạo phụ đề SRT..."
        record.progress = 85
        self._notify_update(record)

        # Build subtitle timestamps
        srt_rows: list[tuple[float, float, str]] = []
        cur_time = 0.0
        for seg, audio_f in zip(manifest.segments, audio_files):
            dur = probe_duration(audio_f)
            srt_rows.append((cur_time, cur_time + dur, seg.narration_text))
            cur_time += dur + 0.3

        out_srt = out_dir / f"{video_path.stem}.narration.srt"
        write_srt_file(srt_rows, out_srt)
        record.output_srt = str(out_srt)

        # Step 6: Final Render with Encoding Options
        if self._cancel_event.is_set():
            raise RenderCancelled()

        record.current_message = "Mã hóa video đầu ra cuối cùng..."
        record.progress = 90
        self._notify_update(record)

        final_video = out_dir / f"{video_path.stem}_recap.mp4"
        srt_to_burn = out_srt if self.settings.burn_subtitles else None

        render_final_video(
            raw_recap_video,
            final_video,
            srt_path=srt_to_burn,
            quality=self.settings.quality,
            use_gpu=self.settings.use_gpu,
            cancel_event=self._cancel_event,
        )

        # Cleanup temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

        record.output_video = str(final_video)
        record.status = "COMPLETED"
        record.progress = 100
        record.current_message = "Hoàn tất xuất sắc!"
        self._notify_update(record)
