"""Project queue, state persistence, and sequential batch execution with safe cancellation."""
from __future__ import annotations

import io
import hashlib
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
from .analyzer.errors import AnalysisCancelledError
from .analyzer.phases import AnalysisPhase
from .api_client import OpenAICompatibleClient
from .domain import AnalysisManifest, CommentaryOutput, MediaSelection, SourceEpisode
from .domain.cache import EvidenceCacheManager
from .domain.enums import OutputStatus
from .gpu import video_encode_args
from .media import (
    MediaError,
    MediaProbeResult,
    RenderCancelled,
    RenderStageError,
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
from .renderer import PublicationRenderer, compute_output_render_signature
from .settings import AppSettings
from .subtitles.cache import SubtitleCacheManager
from .subtitles.discovery import discover_sidecars
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
    season_status: str = "WAITING"
    season_stage: str = "Sẵn sàng"
    season_progress: int = 0
    season_error: str | None = None
    stage: str = "Sẵn sàng"
    error_scope: str | None = None
    error_target: str | None = None
    zero_output_reason: str | None = None
    zero_output_status: str | None = None
    verification: dict[str, Any] | None = None
    analysis_signature: str = ""
    render_signature: str = ""

    def __post_init__(self) -> None:
        if self.verification is not None and hasattr(self.verification, "to_dict"):
            self.verification = self.verification.to_dict()

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

        # Legacy completed backfill only when truly legacy/default
        if self.status == "COMPLETED":
            for ep in self.source_episodes:
                if ep.status == "WAITING" and ep.progress == 0 and ep.error is None and not ep.cached:
                    ep.status = "COMPLETED"
                    ep.stage = "Evidence Complete"
                    ep.progress = 100
                    ep.current_message = "Hoàn tất"
            if self.analysis_scope == "SEASON":
                if self.season_status == "WAITING" and self.season_progress == 0 and self.season_error is None:
                    self.season_status = "COMPLETED"
                    self.season_stage = "Hoàn thành"
                    self.season_progress = 100

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectRecord":
        allowed_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in data.items() if k in allowed_fields}
        return cls(**filtered)

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


ZERO_OUTPUT_MESSAGES: dict[str, str] = {
    "VALID_EMPTY_OUTPUT": "Finalizer đã trả về một Final JSON hợp lệ với 0 output.",
    "NO_ELIGIBLE_CANDIDATES": "Hoàn thành phân tích: Không có ứng viên recap nào đạt yêu cầu (0 outputs).",
    "INSUFFICIENT_EVIDENCE": "Hoàn thành phân tích: Không đủ dữ liệu bằng chứng để tạo recap.",
    "CANDIDATE_DISCOVERY_FAILED": "Hoàn thành phân tích: Khám phá ứng viên không thành công.",
    "CANDIDATES_REJECTED_BY_VALIDATION": "Hoàn thành phân tích: Tất cả ứng viên bị loại sau khi xác minh.",
    "FINALIZER_FAILED": "Hoàn thành phân tích: Tạo kịch bản hoàn tất thất bại.",
    "MALFORMED_AI_RESPONSE": "Hoàn thành phân tích: Phản hồi AI không đúng định dạng.",
    "PARSER_FAILURE": "Hoàn thành phân tích: Lỗi phân tích cú pháp dữ liệu.",
    "COVERAGE_INCOMPLETE": "Hoàn thành phân tích: Phạm vi bằng chứng chưa đầy đủ.",
    "LOW_COVERAGE_SUSPECT": "Hoàn thành phân tích: Độ bao phủ thấp đáng ngờ.",
}


def compute_project_analysis_signature(record: ProjectRecord, settings: AppSettings) -> str:
    """Hash only dependencies that can change Scanner evidence or Final JSON."""
    sources: list[dict[str, Any]] = []
    for ep in record.source_episodes:
        path = Path(ep.source_video)
        try:
            stat = path.stat()
            identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            identity = {"size": None, "mtime_ns": None}
        sidecars: list[dict[str, Any]] = []
        if path.is_file():
            for track in discover_sidecars(path, ep.episode_id):
                if not track.source_file:
                    continue
                sidecar_path = Path(track.source_file)
                related = [sidecar_path]
                if sidecar_path.suffix.casefold() == ".idx":
                    related.append(sidecar_path.with_suffix(".sub"))
                for related_path in related:
                    try:
                        sidecar_stat = related_path.stat()
                        sidecars.append(
                            {
                                "path": str(related_path.resolve()),
                                "size": sidecar_stat.st_size,
                                "mtime_ns": sidecar_stat.st_mtime_ns,
                            }
                        )
                    except OSError:
                        sidecars.append({"path": str(related_path.resolve()), "size": None, "mtime_ns": None})
        sources.append({"episode_id": ep.episode_id, "path": str(path.resolve()), **identity, "sidecars": sidecars})
    payload = {
        "version": "final-json-v1",
        "scope": record.analysis_scope,
        "sources": sources,
        "scanner": {
            "model": settings.scanner_model,
            "thinking": settings.scanner_thinking,
            "vision": settings.scanner_supports_vision,
            "chunk_seconds": settings.api_chunk_seconds,
        },
        "finalizer": {"model": settings.finalizer_model, "thinking": settings.finalizer_thinking},
        "recap_prompt": settings.recap_prompt,
        "recap_language": settings.recap_language,
        "recap_mode": settings.recap_mode,
        "content_type": settings.content_type,
        "source_rights_status": settings.source_rights_status,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def compute_project_render_signature(
    analysis_signature: str,
    record: ProjectRecord,
    settings: AppSettings,
) -> str:
    """Hash render-only dependencies; changing these never invalidates AI work."""
    payload = {
        "version": "render-v1",
        "analysis_signature": analysis_signature,
        "voice_id": record.voice_id,
        "voice_style": settings.voice_style,
        "quality": settings.quality,
        "use_gpu": settings.use_gpu,
        "generate_srt": settings.generate_srt,
        "burn_subtitles": settings.burn_subtitles,
        "original_audio_gain_db": settings.original_audio_gain_db,
        "commentary_gain_db": settings.commentary_gain_db,
        "auto_duck": settings.auto_duck,
        "ducking_amount_db": settings.ducking_amount_db,
        "target_loudness_lufs": settings.target_loudness_lufs,
        "true_peak_dbtp": settings.true_peak_dbtp,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _persist_manifest_operational(manifest: AnalysisManifest | None, manifest_path: str) -> None:
    if manifest is None or not manifest_path:
        return
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")

    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and "outputs" in existing:
                existing_outputs_by_id = {
                    out.get("output_id"): out
                    for out in existing.get("outputs", [])
                    if isinstance(out, dict) and out.get("output_id")
                }
                for mem_out in manifest.outputs:
                    out_dict = mem_out.to_dict()
                    target = existing_outputs_by_id.get(mem_out.output_id)
                    if target is not None:
                        for op_field in (
                            "status",
                            "progress",
                            "error",
                            "publication_video_path",
                            "publication_original_srt_path",
                            "publication_narration_srt_path",
                            "render_signature",
                            "sanitized_title",
                        ):
                            target[op_field] = out_dict.get(op_field)
                    else:
                        existing.setdefault("outputs", []).append(out_dict)
                tmp_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp_path, path)
                return
        except Exception:
            pass

    tmp_path.write_text(manifest.to_json(indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _phase_update(
    record: ProjectRecord,
    phase: AnalysisPhase | str,
    target_id: str,
    data: dict[str, Any],
    notify_fn: Callable[[ProjectRecord], None] | None = None,
) -> None:
    phase_val = phase.value if hasattr(phase, "value") else str(phase)
    record.phase = phase_val
    msg = data.get("status_message") or data.get("status") or ""

    def _update_prog(val: int) -> None:
        record.progress = max(record.progress, min(100, val))

    num_eps = max(1, len(record.source_episodes))

    # Check if retry message: ensure queue displays retry and remains RUNNING
    is_retry = "Thử lại" in msg or "Retrying" in msg or "retry" in msg.lower()
    if is_retry:
        record.status = "RUNNING"
        record.current_message = f"[{target_id}] {msg}"

    # 4a. Scanner callbacks (progress within 20-55; details coverage_check/second_pass/complete)
    if phase in {AnalysisPhase.SCANNER, "scanner"}:
        ep = next((e for e in record.source_episodes if e.episode_id == target_id), None)
        st = data.get("status")

        if data.get("cached") is True:
            if ep:
                ep.cached = True
                ep.status = "CACHED"
                ep.stage = "Evidence Complete"
                ep.progress = 100
                ep.current_message = "Đã có trong bộ nhớ đệm"
        elif st == "complete":
            if ep:
                ep.status = "EVIDENCE_COMPLETE"
                ep.stage = "Evidence Complete"
                ep.progress = 100
                ep.current_message = "Hoàn tất"
        elif st in {"coverage_check", "second_pass"}:
            if ep:
                if ep.status not in {"COMPLETED", "CACHED", "EVIDENCE_COMPLETE"}:
                    ep.status = "RUNNING"
                ep.stage = "Scanner"
                ep.current_message = f"Kiểm tra Evidence Coverage — {target_id}"
            if not is_retry:
                record.current_message = f"Kiểm tra Evidence Coverage — {target_id}"
        else:
            if ep and ep.status not in {"COMPLETED", "CACHED", "EVIDENCE_COMPLETE"}:
                ep.status = "RUNNING"
                ep.stage = "Scanner"
                if msg:
                    ep.current_message = msg

        # Dynamic monotonic interpolation within 20-55
        ep_ratios: list[float] = []
        for e in record.source_episodes:
            if e.status in {"CACHED", "COMPLETED", "EVIDENCE_COMPLETE"}:
                ep_ratios.append(1.0)
            elif e.episode_id == target_id and st in {"coverage_check", "second_pass"}:
                ep_ratios.append(0.85)
            elif e.episode_id == target_id and data.get("total_chunks"):
                chunk_idx = int(data.get("chunk_index") or 0)
                tot_chunks = max(1, int(data.get("total_chunks") or 1))
                ep_ratios.append(0.7 * (chunk_idx / tot_chunks))
            elif e.status == "RUNNING":
                ep_ratios.append(0.5)
            else:
                ep_ratios.append(0.0)

        completed_eps = sum(1 for e in record.source_episodes if e.status in {"CACHED", "COMPLETED", "EVIDENCE_COMPLETE"})
        avg_ratio = sum(ep_ratios) / num_eps
        scanner_prog = 20 + int(35 * avg_ratio)
        _update_prog(min(55, scanner_prog))

        if not is_retry and st not in {"coverage_check", "second_pass"}:
            record.current_message = f"Phân tích [scanner]: {target_id} ({completed_eps}/{num_eps})"

    # Canonical Finalizer and technical Final JSON boundary (55-85).
    elif phase in {AnalysisPhase.FINALIZER, "finalizer"}:
        record.season_status = "RUNNING"
        record.season_stage = "Finalizer"
        _update_prog(70)
        if not is_retry:
            record.current_message = "Finalizer đang tạo Final JSON..."
    elif phase in {AnalysisPhase.JSON_VALIDATION, "json_validation"}:
        record.season_status = "RUNNING"
        record.season_stage = "Technical JSON Validation"
        _update_prog(80)
        if not is_retry:
            record.current_message = "Đang kiểm tra kỹ thuật Final JSON..."
    elif phase in {AnalysisPhase.JSON_REPAIR, "json_repair"}:
        record.season_status = "RUNNING"
        record.season_stage = "Final JSON Repair"
        _update_prog(80)
        attempt = int(data.get("attempt") or 1)
        if not is_retry:
            record.current_message = f"Finalizer đang sửa lỗi kỹ thuật JSON (lần {attempt})..."

    # Legacy-only stage mappings retained for persisted projects and diagnostics.
    # 4b. EPISODE_SUMMARIZING / BATCH_SUMMARIZING (55-60)
    elif phase_val in {"episode_summarizing", "batch_summarizing"} or phase in {
        AnalysisPhase.EPISODE_SUMMARIZING,
        AnalysisPhase.BATCH_SUMMARIZING,
    }:
        record.season_status = "RUNNING"
        record.season_stage = "Episode Summarizing"
        _update_prog(58)
        if not is_retry:
            record.current_message = f"Phân tích tóm tắt tập: {target_id}"

    # 4c. SEASON_BATCH per batch (60-65 using data index/total)
    elif phase in {AnalysisPhase.SEASON_BATCH, "season_batch"}:
        record.season_status = "RUNNING"
        record.season_stage = "Season Batch"
        batch_idx = int(data.get("batch_index") or 1)
        total_batches = int(data.get("total_batches") or 1)
        batch_prog = 60 + int(5 * (batch_idx / max(1, total_batches)))
        _update_prog(min(65, batch_prog))
        record.season_progress = int((batch_idx / max(1, total_batches)) * 100)
        if not is_retry:
            record.current_message = f"Phân tích nhóm mùa phim {target_id} ({batch_idx}/{total_batches})"

    # 4d. SEASON_MERGING (63-65)
    elif phase in {AnalysisPhase.SEASON_MERGING, "season_merging"}:
        record.season_status = "RUNNING"
        record.season_stage = "Season Merging"
        record.season_progress = 75
        _update_prog(65)
        if not is_retry:
            record.current_message = f"Hợp nhất dữ liệu cốt truyện: {target_id}"

    # 4e. CANDIDATE_DISCOVERY (65-72)
    elif phase in {AnalysisPhase.CANDIDATE_DISCOVERY, "candidate_discovery"}:
        record.season_status = "RUNNING"
        record.season_stage = "Khám phá ứng viên"
        idx = int(data.get("index") or 1)
        total = int(data.get("total") or 1)
        disc_prog = 65 + int(7 * (idx / max(1, total)))
        _update_prog(min(72, disc_prog))
        record.season_progress = int((idx / max(1, total)) * 100)
        if not is_retry:
            record.current_message = f"Khám phá ứng viên ({idx}/{total})" if total > 1 else "Khám phá ứng viên"

    # 4f. CANDIDATE_CONSOLIDATION (72-77)
    elif phase in {AnalysisPhase.CANDIDATE_CONSOLIDATION, "candidate_consolidation"}:
        record.season_status = "RUNNING"
        record.season_stage = "Hợp nhất ứng viên"
        idx = int(data.get("index") or 1)
        total = int(data.get("total") or 1)
        cons_prog = 72 + int(5 * (idx / max(1, total)))
        _update_prog(min(77, cons_prog))
        record.season_progress = int((idx / max(1, total)) * 100)
        if not is_retry:
            record.current_message = f"Hợp nhất ứng viên ({idx}/{total})" if total > 1 else "Hợp nhất ứng viên"

    # 4g. ZERO/CANDIDATE_VERIFYING (77-80)
    elif phase in {
        AnalysisPhase.CANDIDATE_VERIFYING,
        AnalysisPhase.ZERO_OUTPUT_VERIFICATION,
        "candidate_verifying",
        "zero_output_verification",
    }:
        record.season_status = "RUNNING"
        record.season_stage = "Xác minh kết quả 0 output"
        idx = int(data.get("index") or 1)
        total = int(data.get("total") or 1)
        verif_prog = 77 + int(3 * (idx / max(1, total)))
        _update_prog(min(80, verif_prog))
        record.season_progress = int((idx / max(1, total)) * 100)
        if not is_retry:
            record.current_message = "Xác minh kết quả 0 output"

    # 4h. SEASON_MINING / finalizer (80-85)
    elif phase in {AnalysisPhase.SEASON_MINING, "season_mining"}:
        record.season_status = "RUNNING"
        record.season_stage = "Season Mining"
        idx = int(data.get("index") or data.get("group_index") or 1)
        total = int(data.get("total") or data.get("total_groups") or 1)
        fin_prog = 80 + int(5 * (idx / max(1, total)))
        _update_prog(min(85, fin_prog))
        record.season_progress = int((idx / max(1, total)) * 100)
        if not is_retry:
            record.current_message = f"Khai thác cốt truyện: {target_id}" if "node" not in str(target_id).lower() else "Khai thác cốt truyện"

    # 4i. OUTPUT_PLAN_READY (85)
    elif phase in {AnalysisPhase.OUTPUT_PLAN_READY, "output_plan_ready"}:
        _update_prog(85)
        record.season_status = "COMPLETED"
        record.season_stage = "Hoàn thành"
        record.season_progress = 100
        if not is_retry:
            record.current_message = "Kế hoạch recap sẵn sàng"

    # Other season phases (55-60)
    elif phase in {AnalysisPhase.SEASON_BARRIER, "season_barrier", AnalysisPhase.SEASON_CONNECTING, "season_connecting"}:
        record.season_status = "RUNNING"
        record.season_stage = "Season Analysis"
        _update_prog(60)
        if not is_retry:
            record.current_message = f"Season Analysis — {target_id}"
    else:
        if not is_retry:
            record.current_message = f"Phân tích [{record.phase}]: {target_id}"

    if notify_fn:
        notify_fn(record)


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
                        for ep in record.source_episodes:
                            if ep.status not in {"COMPLETED", "CACHED", "EVIDENCE_COMPLETE"}:
                                ep.status = "PAUSED"
                                ep.current_message = "Đã dừng khi ứng dụng đóng trước đó"
                        if record.season_status not in {"COMPLETED"}:
                            record.season_status = "PAUSED"
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

    def _phase_update(
        self,
        record: ProjectRecord,
        phase: AnalysisPhase | str,
        target_id: str,
        data: dict[str, Any],
    ) -> None:
        _phase_update(record, phase, target_id, data, notify_fn=self._notify_update)

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
                except (RenderCancelled, AnalysisCancelledError):
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
                    if self._cancel_event.is_set() or "bị dừng" in str(exc).lower() or "bị hủy" in str(exc).lower():
                        record.status = "CANCELLED"
                        record.current_message = "Đã dừng theo yêu cầu của người dùng"
                        self._notify_update(record)
                        for remaining in self.projects[idx + 1:]:
                            if remaining.status == "WAITING":
                                remaining.status = "CANCELLED"
                                remaining.current_message = "Đã dừng bởi người dùng"
                                self._notify_update(remaining)
                        break

                    record.status = "ERROR"
                    hint = "Các kết quả phân tích trước đó đã được lưu an toàn trong bộ nhớ đệm."
                    err_str = str(exc)
                    if "bộ nhớ đệm" not in err_str and "cache" not in err_str.lower():
                        record.error = f"{err_str}. {hint}"
                    else:
                        record.error = err_str
                    record.current_message = f"Lỗi: {record.error}"
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

        # Reset errors on restart/rerun while preserving completed episode states
        record.status = "RUNNING"
        record.error = None
        record.error_scope = None
        record.error_target = None
        record.season_error = None
        if record.season_status == "ERROR":
            record.season_status = "WAITING"
            record.season_stage = "Sẵn sàng"
        for ep in record.source_episodes:
            if ep.status == "ERROR":
                ep.status = "WAITING"
                ep.error = None
                ep.stage = "Sẵn sàng"
        for out in record.outputs:
            if getattr(out, "status", None) == "ERROR":
                out.status = "WAITING"
                out.error = None

        def _update_progress(val: int) -> None:
            record.progress = max(record.progress, min(100, val))

        record.phase = "IDLE"
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

        # 2. Probe all episodes & select media tracks (setup: 0-10)
        if self._cancel_event.is_set():
            raise RenderCancelled()

        record.phase = "ANALYZING"
        num_eps = len(record.source_episodes)
        probes_by_episode: dict[str, MediaProbeResult] = {}
        for idx, ep in enumerate(record.source_episodes):
            if self._cancel_event.is_set():
                raise RenderCancelled()
            if ep.status not in {"COMPLETED", "CACHED", "EVIDENCE_COMPLETE"}:
                ep.status = "RUNNING"
                ep.stage = "Media Probe"
                ep.current_message = f"Phân tích kỹ thuật {ep.episode_id}..."
            _update_progress(int((idx / num_eps) * 10))
            record.current_message = f"Đang phân tích kỹ thuật các tập phim ({idx + 1}/{num_eps})..."
            self._notify_update(record)

            try:
                probe = probe_typed_media(ep.source_video)
                ep.duration_seconds = probe.duration
                sel_audio_idx = 0
                if probe.has_audio:
                    if probe.selected_audio and probe.selected_audio.selected_stream:
                        sel_audio_idx = probe.selected_audio.selected_stream.audio_index
                ep.media_selection = MediaSelection(video_path=ep.source_video, audio_track=sel_audio_idx)
                probes_by_episode[ep.episode_id] = probe
            except Exception as exc:
                if self._cancel_event.is_set() or isinstance(exc, (RenderCancelled, AnalysisCancelledError)):
                    raise
                record.error_scope = "EPISODE"
                record.error_target = ep.episode_id
                ep.status = "ERROR"
                ep.stage = "Lỗi"
                ep.error = str(exc)
                ep.current_message = f"Lỗi: {exc}"
                raise

            if ep.status not in {"COMPLETED", "CACHED", "EVIDENCE_COMPLETE"}:
                ep.progress = 10
            _update_progress(int(((idx + 1) / num_eps) * 10))
            self._notify_update(record)

        # 3. Subtitle extraction pipeline (subtitles: 10-20)
        if self._cancel_event.is_set():
            raise RenderCancelled()

        _update_progress(10)
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

        for idx, ep in enumerate(record.source_episodes):
            if self._cancel_event.is_set():
                raise RenderCancelled()

            if ep.status not in {"COMPLETED", "CACHED", "EVIDENCE_COMPLETE"}:
                ep.status = "RUNNING"
                ep.stage = "Subtitles"
            _update_progress(10 + int((idx / num_eps) * 10))
            record.current_message = f"Trích xuất và chuẩn hóa phụ đề đối thoại ({idx + 1}/{num_eps})..."
            self._notify_update(record)

            probe = probes_by_episode.get(ep.episode_id)
            selected_audio_index: int | None = None
            if probe and probe.selected_audio and probe.selected_audio.selected_stream:
                selected_audio_index = probe.selected_audio.selected_stream.audio_index

            def _status(message: str, episode_id: str = ep.episode_id) -> None:
                record.current_message = f"{episode_id} Media/Subtitles: {message}"
                if ep.status not in {"COMPLETED", "CACHED", "EVIDENCE_COMPLETE"}:
                    ep.current_message = message
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
                        phase="ocr",
                        timeout=120,
                        log=lambda msg: _status(msg, ep.episode_id),
                        on_status=lambda msg: _status(msg, ep.episode_id),
                    )
                    text = result.get("text")
                    return str(text).strip() if text else None
                finally:
                    image_path.unlink(missing_ok=True)

            try:
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
            except Exception as exc:
                if self._cancel_event.is_set() or isinstance(exc, (RenderCancelled, AnalysisCancelledError)):
                    raise
                record.error_scope = "EPISODE"
                record.error_target = ep.episode_id
                ep.status = "ERROR"
                ep.stage = "Lỗi"
                ep.error = str(exc)
                ep.current_message = f"Lỗi: {exc}"
                raise

            _update_progress(10 + int(((idx + 1) / num_eps) * 10))
            self._notify_update(record)

        # 4. Analysis execution. A valid persisted Final JSON is reused when only
        # voice/render settings changed; Scanner/Finalizer dependencies have their
        # own signature and caches.
        if self._cancel_event.is_set():
            raise RenderCancelled()

        analysis_signature = compute_project_analysis_signature(record, self.settings)
        previous_outputs = {out.output_id: out for out in record.outputs}
        manifest: AnalysisManifest | None = None
        persisted_manifest = Path(record.manifest_path) if record.manifest_path else None
        if (
            record.analysis_signature == analysis_signature
            and persisted_manifest is not None
            and persisted_manifest.is_file()
        ):
            try:
                persisted_raw = json.loads(persisted_manifest.read_text(encoding="utf-8"))
                if not isinstance(persisted_raw, dict) or "outputs" not in persisted_raw:
                    raise ValueError("Persisted manifest is not a Final JSON document.")
                if not persisted_raw.get("outputs") and persisted_raw.get("zero_output_status") != "VALID_EMPTY_OUTPUT":
                    raise ValueError("Persisted empty output is not an explicit VALID_EMPTY_OUTPUT result.")
                manifest = AnalysisManifest.from_dict(persisted_raw, validate=True)
                record.current_message = "Sử dụng Final JSON hợp lệ đã lưu; bỏ qua Scanner và Finalizer."
                self._notify_update(record)
            except Exception:
                manifest = None

        ev_cache_mgr = EvidenceCacheManager()
        engine = AnalysisEngine(
            settings=self.settings,
            client=client,
            cache_manager=ev_cache_mgr,
            subtitle_pipeline=sub_pipeline,
        )

        def _on_engine_phase(phase: AnalysisPhase, target_id: str, data: dict[str, Any]) -> None:
            self._phase_update(record, phase, target_id, data)

        def _engine_log(msg: str) -> None:
            record.current_message = msg
            self._notify_update(record)

        try:
            if manifest is None:
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
        except Exception as exc:
            if self._cancel_event.is_set() or isinstance(exc, (RenderCancelled, AnalysisCancelledError)):
                raise
            err_text = str(exc)
            failed_ep = next((e for e in record.source_episodes if e.episode_id in err_text), None)
            is_candidate_or_verif = (
                record.phase in {
                    "candidate_discovery", "candidate_consolidation", "candidate_verifying", "zero_output_verification",
                    "CANDIDATE_DISCOVERY", "CANDIDATE_CONSOLIDATION", "CANDIDATE_VERIFYING", "ZERO_OUTPUT_VERIFICATION",
                }
                or "0 output" in err_text
                or "xác nhận kết quả 0 output" in err_text.lower()
                or "candidate" in err_text.lower()
                or "verification" in err_text.lower()
            )
            if failed_ep and ("scanner" in err_text.lower() or record.phase == "scanner") and not is_candidate_or_verif:
                record.error_scope = "EPISODE"
                record.error_target = failed_ep.episode_id
                failed_ep.status = "ERROR"
                failed_ep.stage = "Lỗi"
                failed_ep.error = err_text
                failed_ep.current_message = f"Lỗi: {err_text}"
            elif is_candidate_or_verif or record.analysis_scope == "SEASON" or "season" in err_text.lower() or "batch" in err_text.lower() or "merge" in err_text.lower() or record.phase in {"season_barrier", "season_connecting", "episode_summarizing", "season_batch", "season_merging", "season_mining"}:
                record.season_status = "ERROR"
                record.season_error = err_text
                record.season_stage = "Lỗi"
                record.error_scope = "SEASON"
                record.error_target = "season"
            else:
                if len(record.source_episodes) == 1:
                    ep0 = record.source_episodes[0]
                    record.error_scope = "EPISODE"
                    record.error_target = ep0.episode_id
                    ep0.status = "ERROR"
                    ep0.stage = "Lỗi"
                    ep0.error = err_text
            raise

        if manifest is None:
            raise RuntimeError("Analysis completed without a Final JSON manifest.")
        record.manifest_path = str(out_dir / f"{record.name}_manifest.json")
        manifest_tmp = Path(record.manifest_path).with_suffix(".tmp")
        manifest_tmp.write_text(manifest.to_json(indent=2), encoding="utf-8")
        os.replace(manifest_tmp, Path(record.manifest_path))
        record.analysis_signature = analysis_signature

        # Always copy prior fields by output id regardless of global render signature
        for out in manifest.outputs:
            prior = previous_outputs.get(out.output_id)
            if prior is not None:
                out.status = prior.status
                out.progress = prior.progress
                out.error = prior.error
                out.publication_video_path = prior.publication_video_path
                out.publication_original_srt_path = prior.publication_original_srt_path
                out.publication_narration_srt_path = prior.publication_narration_srt_path
                out.render_signature = getattr(prior, "render_signature", "") or getattr(out, "render_signature", "")
                if getattr(prior, "sanitized_title", None):
                    out.sanitized_title = prior.sanitized_title

        # Compute target signature for each manifest output from analysis_signature/settings/voice
        target_signatures = {
            out.output_id: compute_output_render_signature(
                out=out,
                settings=self.settings,
                voice_id=record.voice_id,
                analysis_signature=analysis_signature,
            )
            for out in manifest.outputs
        }

        render_signature = compute_project_render_signature(analysis_signature, record, self.settings)
        record.render_signature = render_signature
        record.outputs = manifest.outputs
        record.zero_output_reason = manifest.zero_output_reason
        record.zero_output_status = manifest.zero_output_status
        record.verification = (
            manifest.verification.to_dict()
            if hasattr(manifest.verification, "to_dict")
            else (manifest.verification if isinstance(manifest.verification, dict) else None)
        )

        # 5. Check zero outputs: complete analysis with status COMPLETED and no publication files
        if not manifest.outputs:
            record.status = "COMPLETED"
            record.phase = "COMPLETED"
            _update_progress(100)
            if record.analysis_scope == "SEASON":
                record.season_status = "COMPLETED"
                record.season_stage = "Hoàn thành"
                record.season_progress = 100
            msg = ZERO_OUTPUT_MESSAGES.get(
                manifest.zero_output_reason or "",
                "Finalizer đã trả về một Final JSON hợp lệ với 0 output.",
            )
            record.current_message = msg
            self._notify_update(record)
            return

        # 6. Render all outputs using PublicationRenderer (render 85-100)
        if self._cancel_event.is_set():
            raise RenderCancelled()

        record.phase = "RENDERING"
        _update_progress(85)
        record.current_message = f"Bắt đầu kết xuất {len(manifest.outputs)} video recap..."
        self._notify_update(record)

        # Ensure manifest object and record.outputs are the same list before call
        record.outputs = manifest.outputs

        def _render_prog(out_idx: int, total_out: int, cur_out: CommentaryOutput, pct: float, msg: str) -> None:
            record.current_message = f"[{out_idx}/{total_out}] {cur_out.title}: {msg}"
            _update_progress(min(99, 85 + int(((out_idx - 1 + pct / 100.0) / max(1, total_out)) * 15)))
            self._notify_update(record)

        def _on_output_complete(completed_out: CommentaryOutput, out_idx: int, total_out: int) -> None:
            if not record.output_video and completed_out.publication_video_path:
                record.output_video = completed_out.publication_video_path
                record.output_original_srt = completed_out.publication_original_srt_path
                record.output_narration_srt = completed_out.publication_narration_srt_path
                record.output_srt = completed_out.publication_narration_srt_path
            _persist_manifest_operational(manifest, record.manifest_path)
            self.store.save(self.projects)
            if self.on_update:
                self.on_update(record)

        renderer = PublicationRenderer(voice_manager=get_voice_manager())
        try:
            rendered_outputs = renderer.render_manifest(
                manifest=manifest,
                settings=self.settings,
                voice_id=record.voice_id,
                transcript_cues_by_episode=transcript_cues_by_episode,
                output_root=out_dir,
                callbacks=_render_prog,
                cancel_event=self._cancel_event,
                resume_completed=True,
                analysis_signature=analysis_signature,
                target_signatures=target_signatures,
                on_output_complete=_on_output_complete,
            )
            record.outputs = rendered_outputs
            manifest.outputs = rendered_outputs
            if rendered_outputs:
                first = rendered_outputs[0]
                record.output_video = first.publication_video_path
                record.output_original_srt = first.publication_original_srt_path
                record.output_narration_srt = first.publication_narration_srt_path
                record.output_srt = first.publication_narration_srt_path

            record.status = "COMPLETED"
            record.phase = "COMPLETED"
            _update_progress(100)
            record.current_message = "Hoàn tất xuất sắc!"
            self._notify_update(record)
        except RenderCancelled:
            raise
        except RenderStageError as exc:
            record.error_scope = "OUTPUT"
            target_id = exc.output_id or (manifest.outputs[0].output_id if manifest.outputs else None)
            record.error_target = target_id
            failed_idx: int | None = None
            for idx, out in enumerate(manifest.outputs):
                if out.output_id == target_id:
                    out.status = OutputStatus.ERROR.value
                    out.error = str(exc)
                    failed_idx = idx
                    break
            if failed_idx is not None:
                for rem in manifest.outputs[failed_idx + 1:]:
                    rem.status = OutputStatus.WAITING.value
                    rem.progress = 0
            elif manifest.outputs:
                manifest.outputs[0].status = OutputStatus.ERROR.value
                manifest.outputs[0].error = str(exc)
                record.error_target = manifest.outputs[0].output_id
                for rem in manifest.outputs[1:]:
                    rem.status = OutputStatus.WAITING.value
                    rem.progress = 0
            raise
        except Exception as exc:
            if self._cancel_event.is_set() or isinstance(exc, (RenderCancelled, AnalysisCancelledError)):
                raise
            record.error_scope = "OUTPUT"
            failed_out = next(
                (o for o in manifest.outputs if getattr(o, "status", None) in {"ERROR", OutputStatus.ERROR.value}),
                None,
            )
            if failed_out is not None:
                record.error_target = failed_out.output_id
                try:
                    fail_idx = manifest.outputs.index(failed_out)
                    for rem in manifest.outputs[fail_idx + 1:]:
                        rem.status = OutputStatus.WAITING.value
                        rem.progress = 0
                except ValueError:
                    pass
            elif manifest.outputs:
                failed_out = manifest.outputs[0]
                failed_out.status = OutputStatus.ERROR.value
                failed_out.error = str(exc)
                record.error_target = failed_out.output_id
                for rem in manifest.outputs[1:]:
                    rem.status = OutputStatus.WAITING.value
                    rem.progress = 0
            raise
        finally:
            _persist_manifest_operational(manifest, record.manifest_path)
            self.store.save(self.projects)
