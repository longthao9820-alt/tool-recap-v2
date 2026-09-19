"""Project queue, state persistence, and sequential batch execution with safe cancellation."""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .analyzer.engine import AnalysisEngine
from .analyzer.phases import AnalysisPhase
from .api_client import OpenAICompatibleClient
from .domain import CommentaryOutput, MediaSelection, SourceEpisode
from .domain.cache import EvidenceCacheManager
from .gpu import video_encode_args
from .media import (
    MediaError,
    MediaProbeResult,
    RenderCancelled,
    cut_clip,
    find_binary,
    probe_duration,
    probe_media,
    probe_typed_media,
    render_final_video,
    run_command,
    write_srt_file,
)
from .narration import (
    RecapManifest,
    extract_audio_from_video,
    prepare_narration_for_video,
    transcribe_local_whisper,
)
from .paths import default_data_directory
from .renderer import PublicationRenderer
from .settings import AppSettings
from .subtitles.cache import SubtitleCacheManager
from .subtitles.models import SubtitleCue
from .subtitles.pipeline import SubtitlePipeline
from .voice.catalog import DEFAULT_VOICE_ID
from .voice.manager import get_voice_manager


ProjectCallback = Callable[["ProjectRecord"], None]
BatchCallback = Callable[[int, int], None]


@dataclass
class ProjectRecord:
    id: str
    name: str
    source_video: str = ""
    manifest_path: str = ""
    output_directory: str = ""
    voice_id: str = DEFAULT_VOICE_ID
    status: str = "WAITING"  # "WAITING", "RUNNING", "COMPLETED", "CANCELLED", "ERROR", "PAUSED"
    progress: int = 0
    current_message: str = "Sẵn sàng"
    error: str | None = None
    output_video: str | None = None
    output_srt: str | None = None
    analysis_scope: str = "SINGLE_EPISODE"
    source_episodes: list[SourceEpisode] = field(default_factory=list)
    outputs: list[CommentaryOutput] = field(default_factory=list)
    phase: str = "IDLE"
    output_original_srt: str | None = None
    output_narration_srt: str | None = None

    def __post_init__(self) -> None:
        # Convert nested dicts to dataclass instances if needed
        if self.source_episodes:
            self.source_episodes = [
                ep if isinstance(ep, SourceEpisode) else SourceEpisode.from_dict(ep)
                for ep in self.source_episodes
                if isinstance(ep, (dict, SourceEpisode))
            ]

        if self.outputs:
            self.outputs = [
                out if isinstance(out, CommentaryOutput) else CommentaryOutput.from_dict(out)
                for out in self.outputs
                if isinstance(out, (dict, CommentaryOutput))
            ]

        # Backward compatibility: populate source_episodes from source_video or vice versa
        if self.source_video and not self.source_episodes:
            self.source_episodes = [
                SourceEpisode(
                    episode_id="E01",
                    source_video=self.source_video,
                    title=self.name,
                )
            ]
        elif self.source_episodes and not self.source_video:
            self.source_video = self.source_episodes[0].source_video

        # Aliases for subtitle paths
        if self.output_srt:
            if not self.output_original_srt:
                self.output_original_srt = self.output_srt
            if not self.output_narration_srt:
                self.output_narration_srt = self.output_srt
        elif self.output_original_srt and not self.output_srt:
            self.output_srt = self.output_original_srt
        elif self.output_narration_srt and not self.output_srt:
            self.output_srt = self.output_narration_srt

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # Keep aliases synchronized
        if name == "output_srt" and value:
            if not getattr(self, "output_original_srt", None):
                super().__setattr__("output_original_srt", value)
            if not getattr(self, "output_narration_srt", None):
                super().__setattr__("output_narration_srt", value)
        elif name == "output_original_srt" and value:
            if not getattr(self, "output_srt", None):
                super().__setattr__("output_srt", value)

    @classmethod
    def from_video_path(
        cls,
        video_path: Path,
        output_root: Path,
        voice_id: str = DEFAULT_VOICE_ID,
    ) -> "ProjectRecord":
        stem = video_path.stem
        out_dir = output_root / stem
        ep = SourceEpisode(
            episode_id="E01",
            source_video=str(video_path),
            title=stem,
        )
        return cls(
            id=f"{stem}-{uuid.uuid4().hex[:8]}",
            name=stem,
            source_video=str(video_path),
            output_directory=str(out_dir),
            voice_id=voice_id,
            current_message="Sẵn sàng",
            analysis_scope="SINGLE_EPISODE",
            source_episodes=[ep],
            outputs=[],
            phase="IDLE",
        )

    @classmethod
    def from_season_paths(
        cls,
        video_paths: Sequence[Path | str],
        output_root: Path | str,
        voice_id: str = DEFAULT_VOICE_ID,
        title: str = "",
    ) -> "ProjectRecord":
        paths = [Path(p).resolve() for p in video_paths]
        if not paths:
            raise ValueError("video_paths không được rỗng khi tạo dự án Season.")

        first_path = paths[0]
        name = title or first_path.parent.name or "Season"
        out_root = Path(output_root).resolve()
        out_dir = out_root / name

        episodes = [
            SourceEpisode(
                episode_id=f"E{idx:02d}",
                source_video=str(p),
                title=p.stem,
            )
            for idx, p in enumerate(paths, start=1)
        ]

        return cls(
            id=f"{name}-{uuid.uuid4().hex[:8]}",
            name=name,
            source_video=str(first_path),
            output_directory=str(out_dir),
            voice_id=voice_id,
            current_message="Sẵn sàng",
            analysis_scope="SEASON",
            source_episodes=episodes,
            outputs=[],
            phase="IDLE",
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
                    if not isinstance(item, dict):
                        continue
                    if not item.get("source_video") and not item.get("source_episodes"):
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
        """Execute full canonical pipeline for one project record sequentially."""
        if self._cancel_event.is_set():
            raise RenderCancelled("Xử lý đã bị dừng theo yêu cầu của người dùng.")

        record.status = "RUNNING"
        record.phase = "IDLE"
        record.progress = 5
        record.current_message = "Bắt đầu phân tích video..."
        self._notify_update(record)

        # 1. Validate source episodes
        if not record.source_episodes and record.source_video:
            record.source_episodes = [
                SourceEpisode(
                    episode_id="E01",
                    source_video=record.source_video,
                    title=record.name,
                )
            ]

        if not record.source_episodes:
            raise MediaError("Dự án không có tập phim nguồn (source_episodes).")

        out_dir = Path(record.output_directory).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        for ep in record.source_episodes:
            p = Path(ep.source_video).resolve()
            if not p.is_file():
                raise MediaError(f"Không tìm thấy file video nguồn: {p}")

        # 2. Probe all episodes & select media tracks
        if self._cancel_event.is_set():
            raise RenderCancelled()

        record.phase = "ANALYZING"
        record.progress = 10
        record.current_message = f"Đang phân tích kỹ thuật các tập phim ({len(record.source_episodes)} tập)..."
        self._notify_update(record)

        probes_by_episode: dict[str, MediaProbeResult] = {}
        for ep in record.source_episodes:
            if self._cancel_event.is_set():
                raise RenderCancelled()
            probe = probe_typed_media(ep.source_video)
            ep.duration_seconds = probe.duration
            sel_audio_idx = 0
            if probe.has_audio:
                if probe.selected_audio and probe.selected_audio.selected_stream:
                    sel_audio_idx = probe.selected_audio.selected_stream.audio_index
            ep.media_selection = MediaSelection(video_path=ep.source_video, audio_track=sel_audio_idx)
            probes_by_episode[ep.episode_id] = probe

        # 3. Subtitle extraction pipeline
        if self._cancel_event.is_set():
            raise RenderCancelled()

        record.progress = 20
        record.current_message = "Trích xuất và chuẩn hóa phụ đề đối thoại..."
        self._notify_update(record)

        sub_cache_mgr = SubtitleCacheManager()
        sub_pipeline = SubtitlePipeline(cache_manager=sub_cache_mgr)
        transcript_cues_by_episode: dict[str, list[SubtitleCue]] = {}
        client: OpenAICompatibleClient | None = None
        if (
            self.settings.gateway_enabled
            and self.settings.api_endpoint
            and self.settings.api_endpoint.strip().lower() != "offline"
        ):
            client = OpenAICompatibleClient(
                base_url=self.settings.api_endpoint,
                api_key=self.settings.api_key,
            )

        for ep in record.source_episodes:
            if self._cancel_event.is_set():
                raise RenderCancelled()

            probe = probes_by_episode.get(ep.episode_id)
            selected_audio_index: int | None = None
            if probe and probe.selected_audio and probe.selected_audio.selected_stream:
                selected_audio_index = probe.selected_audio.selected_stream.audio_index

            def _status(message: str, episode_id: str = ep.episode_id) -> None:
                record.current_message = f"{episode_id} Media/Subtitles: {message}"
                self._notify_update(record)

            def _stt_fallback(source_video: str, episode_id: str) -> list[SubtitleCue]:
                if self._cancel_event.is_set():
                    raise RenderCancelled()
                _status("Không có phụ đề English Full; đang nhận dạng lời thoại nội bộ...")
                stt_dir = default_data_directory() / "cache" / "stt_audio"
                stt_dir.mkdir(parents=True, exist_ok=True)
                wav_path = stt_dir / f"{record.id}_{episode_id}.wav"
                if not extract_audio_from_video(
                    Path(source_video),
                    wav_path,
                    audio_stream_index=selected_audio_index,
                    cancel_event=self._cancel_event,
                ):
                    raise MediaError(f"Không thể trích xuất English program audio cho {episode_id}.")
                try:
                    rows = transcribe_local_whisper(
                        wav_path,
                        language="en",
                        log=_status,
                        cancel_event=self._cancel_event,
                    )
                finally:
                    wav_path.unlink(missing_ok=True)
                return [
                    SubtitleCue(
                        start_ms=round(start * 1000),
                        end_ms=round(end * 1000),
                        text=text,
                        source_type="stt",
                        source_format="faster-whisper",
                        language="en",
                        confidence=0.85,
                        episode_id=episode_id,
                        source_video=source_video,
                    )
                    for start, end, text in rows
                    if text.strip() and end > start
                ]

            vision_enabled = bool(self.settings.scanner_supports_vision and client is not None)

            def _vision_fallback(image: Any) -> str | None:
                if not vision_enabled or client is None:
                    return None
                if self._cancel_event.is_set():
                    raise RenderCancelled()
                vision_dir = default_data_directory() / "cache" / "ocr_vision"
                vision_dir.mkdir(parents=True, exist_ok=True)
                image_path = vision_dir / f"{record.id}_{ep.episode_id}_{uuid.uuid4().hex[:8]}.png"
                image.save(image_path, format="PNG")
                try:
                    result = client.chat_json(
                        model=self.settings.scanner_model,
                        thinking=self.settings.scanner_thinking,
                        system="Read only the English subtitle text visible in this cropped subtitle bitmap. Return JSON only.",
                        user_text='Return {"text":"..."}. Use an empty string when unreadable. Never invent text.',
                        images=[image_path],
                        max_tokens=500,
                        cancel_event=self._cancel_event,
                    )
                    text = result.get("text")
                    return str(text).strip() if text else None
                finally:
                    image_path.unlink(missing_ok=True)

            cues = sub_pipeline.get_episode_subtitles(
                video_path=ep.source_video,
                episode_id=ep.episode_id,
                probe_result=probe,
                stt_fallback_fn=_stt_fallback,
                ai_vision_fallback=_vision_fallback if vision_enabled else None,
                vision_supported=vision_enabled,
                cancel_event=self._cancel_event,
                progress_callback=_status,
            )
            transcript_cues_by_episode[ep.episode_id] = cues

        # 4. AnalysisEngine execution
        if self._cancel_event.is_set():
            raise RenderCancelled()

        ev_cache_mgr = EvidenceCacheManager()
        engine = AnalysisEngine(
            settings=self.settings,
            client=client,
            cache_manager=ev_cache_mgr,
            subtitle_pipeline=sub_pipeline,
        )

        def _on_engine_phase(phase: AnalysisPhase, target_id: str, data: dict[str, Any]) -> None:
            record.phase = phase.value if hasattr(phase, "value") else str(phase)
            record.current_message = f"Phân tích [{record.phase}]: {target_id}"
            self._notify_update(record)

        def _engine_log(msg: str) -> None:
            record.current_message = msg
            self._notify_update(record)

        manifest = engine.analyze(
            project_id=record.id,
            episodes=record.source_episodes,
            scope=record.analysis_scope,
            injected_transcripts=transcript_cues_by_episode,
            injected_probes=probes_by_episode,
            cancel_event=self._cancel_event,
            on_phase=_on_engine_phase,
            log=_engine_log,
            legacy_wrapper=not self.settings.gateway_enabled,
        )

        record.manifest_path = str(out_dir / f"{record.name}_manifest.json")
        Path(record.manifest_path).write_text(manifest.to_json(indent=2), encoding="utf-8")
        record.outputs = manifest.outputs

        # 5. Check zero outputs: complete analysis with status COMPLETED and no publication files
        if not manifest.outputs:
            record.status = "COMPLETED"
            record.phase = "COMPLETED"
            record.progress = 100
            record.current_message = "Hoàn thành phân tích: Không có ứng viên recap nào đạt yêu cầu (0 outputs)."
            self._notify_update(record)
            return

        # 6. Render all outputs using PublicationRenderer
        if self._cancel_event.is_set():
            raise RenderCancelled()

        record.phase = "RENDERING"
        record.progress = 50
        record.current_message = f"Bắt đầu kết xuất {len(manifest.outputs)} video recap..."
        self._notify_update(record)

        def _render_prog(out_idx: int, total_out: int, cur_out: CommentaryOutput, pct: float, msg: str) -> None:
            record.current_message = f"[{out_idx}/{total_out}] {cur_out.title}: {msg}"
            record.progress = min(98, 50 + int(((out_idx - 1 + pct / 100.0) / max(1, total_out)) * 48))
            self._notify_update(record)

        renderer = PublicationRenderer(voice_manager=get_voice_manager())
        rendered_outputs = renderer.render_manifest(
            manifest=manifest,
            settings=self.settings,
            voice_id=record.voice_id,
            transcript_cues_by_episode=transcript_cues_by_episode,
            output_root=out_dir,
            callbacks=_render_prog,
            cancel_event=self._cancel_event,
        )

        record.outputs = rendered_outputs
        manifest.outputs = rendered_outputs
        manifest_tmp = Path(record.manifest_path).with_suffix(".tmp")
        manifest_tmp.write_text(manifest.to_json(indent=2), encoding="utf-8")
        os.replace(manifest_tmp, Path(record.manifest_path))
        if rendered_outputs:
            first = rendered_outputs[0]
            record.output_video = first.publication_video_path
            record.output_original_srt = first.publication_original_srt_path
            record.output_narration_srt = first.publication_narration_srt_path
            record.output_srt = first.publication_narration_srt_path

        record.status = "COMPLETED"
        record.phase = "COMPLETED"
        record.progress = 100
        record.current_message = "Hoàn tất xuất sắc!"
        self._notify_update(record)
