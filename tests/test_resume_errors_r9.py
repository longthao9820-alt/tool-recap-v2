"""Tests for R9 resume behavior and render stage error wrapping.

Acceptance Tests:
1. OutputStatus no NameError via ProjectQueue fail
2. Partial output persistence: 1 complete / 2 fail / 3 waiting in manifest + projects
3. Restart reuse output 1 retry others with fake renderer/validation
4. Outputs 1-4 resume output 5
5. Render setting change: analysis skipped but rerender calls
6. One output segment change only
7. Exact id/category per stage failure
8. Cancellation: no callback and partial cleanup
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from toolrecap_v2.domain.enums import AudioPolicy, OutputStatus
from toolrecap_v2.domain.models import (
    AnalysisManifest,
    CommentaryOutput,
    Segment,
    SourceClip,
    SourceEpisode,
)
from toolrecap_v2.media import (
    CutClipError,
    MediaError,
    MediaErrorCategory,
    MediaProbeResult,
    RenderCancelled,
    RenderStageError,
)
from toolrecap_v2.output_validation import OutputValidationError
from toolrecap_v2.projects import (
    ProjectQueue,
    ProjectRecord,
    ProjectStore,
    compute_output_render_signature,
    compute_project_analysis_signature,
    compute_project_render_signature,
)
from toolrecap_v2.renderer import PublicationRenderer
from toolrecap_v2.settings import AppSettings
from toolrecap_v2.subtitles.models import SubtitleCue


def _make_dummy_output(
    output_id: str,
    title: str,
    sanitized_title: str,
    status: str = OutputStatus.WAITING.value,
    narration: str = "Thuyết minh thử nghiệm.",
    video_path: str = "dummy.mp4",
) -> CommentaryOutput:
    return CommentaryOutput(
        output_id=output_id,
        title=title,
        sanitized_title=sanitized_title,
        status=status,
        segments=[
            Segment(
                segment_id=f"seg_{output_id}",
                source_clips=[
                    SourceClip(
                        episode_id="E01",
                        source_video=video_path,
                        start=0.0,
                        end=2.0,
                    )
                ],
                narration=narration,
                audio_policy=AudioPolicy.DUCK.value,
            )
        ],
    )


def _setup_project(
    tmp_path: Path,
    outputs: list[CommentaryOutput],
    settings: AppSettings,
    project_id: str = "proj_test",
) -> tuple[ProjectRecord, ProjectStore, Path]:
    settings.gateway_enabled = False
    settings.api_endpoint = "offline"

    video_file = tmp_path / "source.mp4"
    video_file.write_bytes(b"dummy video content")

    for out in outputs:
        for seg in out.segments:
            for clip in seg.source_clips:
                clip.source_video = str(video_file)

    ep = SourceEpisode(
        episode_id="E01",
        source_video=str(video_file),
        status="COMPLETED",
        stage="Evidence Complete",
        progress=100,
        duration_seconds=10.0,
    )

    manifest_path = tmp_path / "outputs" / f"{project_id}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = AnalysisManifest(
        project_id=project_id,
        source_episodes=[ep],
        outputs=outputs,
    )
    manifest_path.write_text(manifest.to_json(indent=2), encoding="utf-8")

    store_file = tmp_path / "queue_store.json"
    store = ProjectStore(store_file)

    record = ProjectRecord(
        id=project_id,
        name=project_id,
        source_video=str(video_file),
        manifest_path=str(manifest_path),
        output_directory=str(tmp_path / "outputs"),
        source_episodes=[ep],
        outputs=outputs,
        status="WAITING",
    )
    record.analysis_signature = compute_project_analysis_signature(record, settings)
    store.save([record])

    return record, store, manifest_path


def _mock_subtitles(*args: Any, **kwargs: Any) -> list[SubtitleCue]:
    return [
        SubtitleCue(
            start_ms=0,
            end_ms=2000,
            text="Mock subtitle",
            source_type="embedded",
            source_format="srt",
            episode_id="E01",
        )
    ]


def _mock_probe(path: Any, *args: Any, **kwargs: Any) -> MediaProbeResult:
    return MediaProbeResult(
        path=str(path),
        duration=10.0,
        width=1920,
        height=1080,
        has_video=True,
        has_audio=True,
        video_codec="h264",
        audio_codec="aac",
        video_streams=[],
        audio_streams=[],
    )


# ---------------------------------------------------------------------------
# 1. OutputStatus no NameError via ProjectQueue fail
# ---------------------------------------------------------------------------
def test_output_status_no_name_error_via_project_queue_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Handling render errors in _process_single_project never raises NameError for OutputStatus."""
    settings = AppSettings()
    out1 = _make_dummy_output("out_01", "Output 1", "output-1")
    out2 = _make_dummy_output("out_02", "Output 2", "output-2")
    record, store, _ = _setup_project(tmp_path, [out1, out2], settings)

    monkeypatch.setattr("toolrecap_v2.projects.probe_typed_media", _mock_probe)
    monkeypatch.setattr(
        "toolrecap_v2.subtitles.pipeline.SubtitlePipeline.get_episode_subtitles",
        _mock_subtitles,
    )

    def mock_no_analyze(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("AnalysisEngine.analyze không được phép gọi khi tái sử dụng manifest!")

    monkeypatch.setattr("toolrecap_v2.projects.AnalysisEngine.analyze", mock_no_analyze)

    def mock_fail_render(*args: Any, **kwargs: Any) -> None:
        raise RenderStageError(
            message="Giả lập lỗi render stage",
            output_id="out_01",
            output_title="Output 1",
            stage="voice",
            category="VOICE_ERROR",
        )

    monkeypatch.setattr(PublicationRenderer, "render_manifest", mock_fail_render)

    queue = ProjectQueue([record], store=store, settings=settings)

    with pytest.raises(RenderStageError):
        queue._process_single_project(record)

    # Ensure no NameError occurred and record error state is recorded accurately
    assert record.error_scope == "OUTPUT"
    assert record.error_target == "out_01"
    assert record.outputs[0].status == OutputStatus.ERROR.value
    assert record.outputs[1].status == OutputStatus.WAITING.value


# ---------------------------------------------------------------------------
# 2. Partial output persistence 1 complete / 2 fail / 3 waiting
# ---------------------------------------------------------------------------
def test_partial_output_persistence_1_complete_2_fail_3_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When output 1 succeeds, output 2 fails, output 3 waits: manifest and store persist exact states."""
    settings = AppSettings()
    out1 = _make_dummy_output("out_01", "Output 1", "output-1")
    out2 = _make_dummy_output("out_02", "Output 2", "output-2")
    out3 = _make_dummy_output("out_03", "Output 3", "output-3")
    record, store, manifest_path = _setup_project(tmp_path, [out1, out2, out3], settings)

    monkeypatch.setattr("toolrecap_v2.projects.probe_typed_media", _mock_probe)
    monkeypatch.setattr(
        "toolrecap_v2.subtitles.pipeline.SubtitlePipeline.get_episode_subtitles",
        _mock_subtitles,
    )

    def mock_no_analyze(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("AnalysisEngine.analyze không được phép gọi khi tái sử dụng manifest!")

    monkeypatch.setattr("toolrecap_v2.projects.AnalysisEngine.analyze", mock_no_analyze)

    def mock_render_manifest(
        self: Any,
        manifest: AnalysisManifest,
        on_output_complete: Any = None,
        **kwargs: Any,
    ) -> list[CommentaryOutput]:
        # Output 1 succeeds
        out_first = manifest.outputs[0]
        out_first.status = OutputStatus.COMPLETED.value
        out_first.publication_video_path = str(tmp_path / "out1.mp4")
        if on_output_complete:
            on_output_complete(out_first, 1, 3)

        # Output 2 fails
        out_second = manifest.outputs[1]
        out_second.status = OutputStatus.ERROR.value
        out_second.error = "[FINAL_ENCODE_ERROR] Mã hóa thất bại"

        # Output 3 remains WAITING
        manifest.outputs[2].status = OutputStatus.WAITING.value

        raise RenderStageError(
            message="Mã hóa thất bại",
            output_id=out_second.output_id,
            output_title=out_second.title,
            stage="final_encode",
            category="FINAL_ENCODE_ERROR",
        )

    monkeypatch.setattr(PublicationRenderer, "render_manifest", mock_render_manifest)

    queue = ProjectQueue([record], store=store, settings=settings)

    with pytest.raises(RenderStageError):
        queue._process_single_project(record)

    # In-memory record checks
    assert record.outputs[0].status == OutputStatus.COMPLETED.value
    assert record.outputs[1].status == OutputStatus.ERROR.value
    assert record.outputs[2].status == OutputStatus.WAITING.value
    assert record.error_scope == "OUTPUT"
    assert record.error_target == "out_02"

    # Manifest file persistence check
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["outputs"][0]["status"] == OutputStatus.COMPLETED.value
    assert manifest_data["outputs"][1]["status"] == OutputStatus.ERROR.value
    assert manifest_data["outputs"][2]["status"] == OutputStatus.WAITING.value

    # Store file persistence check
    loaded_records = store.load()
    assert loaded_records[0].outputs[0].status == OutputStatus.COMPLETED.value
    assert loaded_records[0].outputs[1].status == OutputStatus.ERROR.value
    assert loaded_records[0].outputs[2].status == OutputStatus.WAITING.value


# ---------------------------------------------------------------------------
# 3. Restart reuse output1 retry others with fake renderer/validation
# ---------------------------------------------------------------------------
def test_restart_reuse_output1_retry_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrying project reuses already completed output 1 and re-renders only outputs 2 and 3."""
    settings = AppSettings()
    out1 = _make_dummy_output("out_01", "Output 1", "output-1", status=OutputStatus.COMPLETED.value)
    out2 = _make_dummy_output("out_02", "Output 2", "output-2", status=OutputStatus.ERROR.value)
    out3 = _make_dummy_output("out_03", "Output 3", "output-3", status=OutputStatus.WAITING.value)

    record, store, manifest_path = _setup_project(tmp_path, [out1, out2, out3], settings)

    # Set up signature and publication folder for output 1 so resume verification passes
    pub_dir_1 = tmp_path / "outputs" / "output-1"
    pub_dir_1.mkdir(parents=True, exist_ok=True)
    out1.render_signature = compute_output_render_signature(
        out=out1,
        settings=settings,
        voice_id=record.voice_id,
        analysis_signature=record.analysis_signature,
    )
    out1.publication_video_path = str(pub_dir_1 / "output-1.mp4")
    out1.publication_original_srt_path = str(pub_dir_1 / "output-1.original.srt")
    out1.publication_narration_srt_path = str(pub_dir_1 / "output-1.narration.srt")

    monkeypatch.setattr("toolrecap_v2.projects.probe_typed_media", _mock_probe)
    monkeypatch.setattr(
        "toolrecap_v2.subtitles.pipeline.SubtitlePipeline.get_episode_subtitles",
        _mock_subtitles,
    )

    def mock_no_analyze(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("AnalysisEngine.analyze không được phép gọi khi tái sử dụng manifest!")

    monkeypatch.setattr("toolrecap_v2.projects.AnalysisEngine.analyze", mock_no_analyze)

    # Mock publication validation for output 1
    def mock_validate_pub(pub_dir: Any, safe_title: str, **kwargs: Any) -> dict[str, Any]:
        p = Path(pub_dir)
        return {
            "video_path": str(p / f"{safe_title}.mp4"),
            "original_srt_path": str(p / f"{safe_title}.original.srt"),
            "narration_srt_path": str(p / f"{safe_title}.narration.srt"),
            "duration": 5.0,
        }

    monkeypatch.setattr("toolrecap_v2.renderer.validate_publication_folder", mock_validate_pub)

    rendered_ids: list[str] = []

    def mock_render_single(
        self: Any,
        out: CommentaryOutput,
        safe_title: str,
        pub_dir: Path,
        **kwargs: Any,
    ) -> None:
        rendered_ids.append(out.output_id)
        pub_dir.mkdir(parents=True, exist_ok=True)
        out.status = OutputStatus.COMPLETED.value
        out.publication_video_path = str(pub_dir / f"{safe_title}.mp4")
        out.publication_original_srt_path = str(pub_dir / f"{safe_title}.original.srt")
        out.publication_narration_srt_path = str(pub_dir / f"{safe_title}.narration.srt")

    monkeypatch.setattr(PublicationRenderer, "_render_single_output", mock_render_single)

    queue = ProjectQueue([record], store=store, settings=settings)
    queue._process_single_project(record)

    # Output 1 was reused (not in rendered_ids), outputs 2 and 3 were rendered
    assert "out_01" not in rendered_ids
    assert "out_02" in rendered_ids
    assert "out_03" in rendered_ids
    assert record.status == "COMPLETED"
    assert record.phase == "COMPLETED"
    assert all(o.status == OutputStatus.COMPLETED.value for o in record.outputs)


# ---------------------------------------------------------------------------
# 4. Outputs 1-4 resume output 5
# ---------------------------------------------------------------------------
def test_outputs1_4_resume_output5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When outputs 1-4 are already completed, only output 5 is rendered on resume."""
    settings = AppSettings()
    outputs = [
        _make_dummy_output(f"out_{i:02d}", f"Output {i}", f"output-{i}", status=OutputStatus.COMPLETED.value)
        for i in range(1, 5)
    ]
    out5 = _make_dummy_output("out_05", "Output 5", "output-5", status=OutputStatus.WAITING.value)
    outputs.append(out5)

    record, store, _ = _setup_project(tmp_path, outputs, settings)

    # Set valid signatures and publication folders for outputs 1-4
    for o in outputs[:4]:
        pub = tmp_path / "outputs" / o.sanitized_title
        pub.mkdir(parents=True, exist_ok=True)
        o.render_signature = compute_output_render_signature(
            out=o,
            settings=settings,
            voice_id=record.voice_id,
            analysis_signature=record.analysis_signature,
        )
        o.publication_video_path = str(pub / f"{o.sanitized_title}.mp4")

    monkeypatch.setattr("toolrecap_v2.projects.probe_typed_media", _mock_probe)
    monkeypatch.setattr(
        "toolrecap_v2.subtitles.pipeline.SubtitlePipeline.get_episode_subtitles",
        _mock_subtitles,
    )

    def mock_no_analyze(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("AnalysisEngine.analyze không được phép gọi khi tái sử dụng manifest!")

    monkeypatch.setattr("toolrecap_v2.projects.AnalysisEngine.analyze", mock_no_analyze)

    def mock_validate_pub(pub_dir: Any, safe_title: str, **kwargs: Any) -> dict[str, Any]:
        p = Path(pub_dir)
        return {
            "video_path": str(p / f"{safe_title}.mp4"),
            "original_srt_path": str(p / f"{safe_title}.original.srt"),
            "narration_srt_path": str(p / f"{safe_title}.narration.srt"),
            "duration": 5.0,
        }

    monkeypatch.setattr("toolrecap_v2.renderer.validate_publication_folder", mock_validate_pub)

    rendered_ids: list[str] = []

    def mock_render_single(
        self: Any,
        out: CommentaryOutput,
        safe_title: str,
        pub_dir: Path,
        **kwargs: Any,
    ) -> None:
        rendered_ids.append(out.output_id)
        pub_dir.mkdir(parents=True, exist_ok=True)
        out.status = OutputStatus.COMPLETED.value
        out.publication_video_path = str(pub_dir / f"{safe_title}.mp4")

    monkeypatch.setattr(PublicationRenderer, "_render_single_output", mock_render_single)

    queue = ProjectQueue([record], store=store, settings=settings)
    queue._process_single_project(record)

    assert rendered_ids == ["out_05"]
    assert record.status == "COMPLETED"
    assert all(o.status == OutputStatus.COMPLETED.value for o in record.outputs)


# ---------------------------------------------------------------------------
# 5. Render setting change analysis skipped but rerender calls
# ---------------------------------------------------------------------------
def test_render_setting_change_analysis_skipped_rerender_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When render settings change, AnalysisEngine is skipped but renderer is called."""
    settings = AppSettings(quality="1080p")
    out1 = _make_dummy_output("out_01", "Output 1", "output-1")
    record, store, manifest_path = _setup_project(tmp_path, [out1], settings)

    # Change render setting
    settings.quality = "720p"

    monkeypatch.setattr("toolrecap_v2.projects.probe_typed_media", _mock_probe)
    monkeypatch.setattr(
        "toolrecap_v2.subtitles.pipeline.SubtitlePipeline.get_episode_subtitles",
        _mock_subtitles,
    )

    # Assert AnalysisEngine.analyze is NEVER called
    def mock_analyze(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("AnalysisEngine.analyze không được phép gọi khi chỉ đổi render setting!")

    monkeypatch.setattr("toolrecap_v2.projects.AnalysisEngine.analyze", mock_analyze)

    render_manifest_called = False

    def mock_render_manifest(self: Any, manifest: AnalysisManifest, **kwargs: Any) -> list[CommentaryOutput]:
        nonlocal render_manifest_called
        render_manifest_called = True
        for o in manifest.outputs:
            o.status = OutputStatus.COMPLETED.value
            o.publication_video_path = str(tmp_path / f"{o.sanitized_title}.mp4")
        return manifest.outputs

    monkeypatch.setattr(PublicationRenderer, "render_manifest", mock_render_manifest)

    queue = ProjectQueue([record], store=store, settings=settings)
    queue._process_single_project(record)

    assert render_manifest_called is True
    assert record.status == "COMPLETED"


# ---------------------------------------------------------------------------
# 6. One output segment change only
# ---------------------------------------------------------------------------
def test_one_output_segment_change_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When only one output's segment changes, only that output is rerendered while the other is reused."""
    settings = AppSettings()
    out1 = _make_dummy_output("out_01", "Output 1", "output-1", status=OutputStatus.COMPLETED.value)
    out2 = _make_dummy_output("out_02", "Output 2", "output-2", status=OutputStatus.COMPLETED.value)
    record, store, _ = _setup_project(tmp_path, [out1, out2], settings)

    # Compute signatures when both were completed
    out1.render_signature = compute_output_render_signature(
        out=out1,
        settings=settings,
        voice_id=record.voice_id,
        analysis_signature=record.analysis_signature,
    )
    pub1 = tmp_path / "outputs" / "output-1"
    pub1.mkdir(parents=True, exist_ok=True)
    out1.publication_video_path = str(pub1 / "output-1.mp4")

    # Change only out2's segment narration (invalidating out2's signature)
    out2.render_signature = "old_stale_signature"
    out2.segments[0].narration = "Nội dung narration hoàn toàn mới"

    def mock_validate_pub(pub_dir: Any, safe_title: str, **kwargs: Any) -> dict[str, Any]:
        p = Path(pub_dir)
        return {
            "video_path": str(p / f"{safe_title}.mp4"),
            "original_srt_path": str(p / f"{safe_title}.original.srt"),
            "narration_srt_path": str(p / f"{safe_title}.narration.srt"),
            "duration": 5.0,
        }

    monkeypatch.setattr("toolrecap_v2.renderer.validate_publication_folder", mock_validate_pub)

    rendered_ids: list[str] = []

    def mock_render_single(
        self: Any,
        out: CommentaryOutput,
        safe_title: str,
        pub_dir: Path,
        **kwargs: Any,
    ) -> None:
        rendered_ids.append(out.output_id)
        pub_dir.mkdir(parents=True, exist_ok=True)
        out.status = OutputStatus.COMPLETED.value
        out.publication_video_path = str(pub_dir / f"{safe_title}.mp4")

    monkeypatch.setattr(PublicationRenderer, "_render_single_output", mock_render_single)

    renderer = PublicationRenderer()
    manifest = AnalysisManifest(project_id="test", outputs=[out1, out2])

    rendered = renderer.render_manifest(
        manifest=manifest,
        settings=settings,
        voice_id=record.voice_id,
        output_root=tmp_path / "outputs",
        resume_completed=True,
        analysis_signature=record.analysis_signature,
    )

    # out1 was reused, out2 was rerendered
    assert "out_01" not in rendered_ids
    assert "out_02" in rendered_ids
    assert len(rendered) == 2


# ---------------------------------------------------------------------------
# 7. Exact id/category per stage failure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "stage_name,expected_category,mock_target",
    [
        ("extract_audio", "AUDIO_LAYOUT_ERROR", "toolrecap_v2.renderer.validate_wav_audio"),
        ("voice", "VOICE_ERROR", "toolrecap_v2.renderer.validate_wav_audio"),
        ("timeline", "AUDIO_LAYOUT_ERROR", "toolrecap_v2.renderer.validate_wav_audio"),
        ("mix", "AUDIO_LAYOUT_ERROR", "toolrecap_v2.renderer.validate_mixed_audio"),
        ("subtitle", "SUBTITLE_ERROR", "toolrecap_v2.renderer.write_srt_file"),
        ("final_encode", "FINAL_ENCODE_ERROR", "toolrecap_v2.renderer.render_final_video"),
        ("publication_validate", "INTERMEDIATE_VALIDATION_ERROR", "toolrecap_v2.renderer.validate_publication_folder"),
        ("cut_clip", "CUT_CLIP_ERROR", "toolrecap_v2.renderer.cut_clip"),
    ],
)
def test_exact_id_and_category_per_stage_failure(
    stage_name: str,
    expected_category: str,
    mock_target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each render stage failure produces RenderStageError with exact output_id and exact category."""
    settings = AppSettings(burn_subtitles=True)
    out = _make_dummy_output("out_target_01", "Target Output", "target-output")
    record, store, _ = _setup_project(tmp_path, [out], settings)

    monkeypatch.setattr("toolrecap_v2.projects.probe_typed_media", _mock_probe)
    monkeypatch.setattr("toolrecap_v2.renderer.probe_typed_media", _mock_probe)
    monkeypatch.setattr(
        "toolrecap_v2.subtitles.pipeline.SubtitlePipeline.get_episode_subtitles",
        _mock_subtitles,
    )
    monkeypatch.setattr("toolrecap_v2.renderer.find_binary", lambda name: "mock_ffmpeg")

    def create_mock_output(args: list[str], **kwargs: Any) -> MagicMock:
        output = Path(args[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mock-media")
        return MagicMock()

    def create_mock_concat(*args: Any, **kwargs: Any) -> Any:
        output = Path(kwargs.get("output_path") or kwargs.get("output") or args[1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mock-video")
        return _mock_probe(output)

    monkeypatch.setattr("toolrecap_v2.renderer.run_command", create_mock_output)
    def create_mock_clip(*args: Any, **kwargs: Any) -> Path:
        output = Path(kwargs.get("output_path") or args[1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"clip")
        return output

    monkeypatch.setattr("toolrecap_v2.renderer.cut_clip", create_mock_clip)
    monkeypatch.setattr("toolrecap_v2.renderer.concat_media_clips", create_mock_concat)
    monkeypatch.setattr("toolrecap_v2.renderer.remap_subtitles", lambda *args, **kwargs: [MagicMock()])
    monkeypatch.setattr("toolrecap_v2.renderer.cues_to_srt_rows", lambda *args, **kwargs: [(0.0, 1.0, "text")])

    fake_voice = MagicMock()
    fake_voice.synthesize.return_value = None
    monkeypatch.setattr("toolrecap_v2.renderer.get_voice_manager", lambda: fake_voice)
    monkeypatch.setattr("toolrecap_v2.projects.get_voice_manager", lambda: fake_voice)

    def mock_no_analyze(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("AnalysisEngine.analyze không được phép gọi khi tái sử dụng manifest!")

    monkeypatch.setattr("toolrecap_v2.projects.AnalysisEngine.analyze", mock_no_analyze)

    # Defaults for intermediate stages so preceding stages succeed cleanly
    monkeypatch.setattr("toolrecap_v2.renderer.probe_duration", lambda path: 1.0)
    monkeypatch.setattr("toolrecap_v2.renderer.validate_mixed_audio", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr("toolrecap_v2.renderer.write_srt_file", lambda *args, **kwargs: None)
    monkeypatch.setattr("toolrecap_v2.renderer.render_final_video", lambda *args, **kwargs: None)
    monkeypatch.setattr("toolrecap_v2.renderer.validate_publication_folder", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr("toolrecap_v2.renderer.validate_wav_audio", lambda path: None)

    # Trigger failure at target stage
    def mock_fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"Lỗi thử nghiệm tại stage {stage_name}")

    if stage_name == "voice":
        fake_voice.synthesize.side_effect = mock_fail
    elif stage_name == "extract_audio":
        def fail_extract(p: Any) -> None:
            if "original_audio" in str(p):
                mock_fail()
        monkeypatch.setattr("toolrecap_v2.renderer.validate_wav_audio", fail_extract)
    elif stage_name == "timeline":
        def fail_timeline(p: Any) -> None:
            if "commentary_timeline" in str(p):
                mock_fail()
        monkeypatch.setattr("toolrecap_v2.renderer.validate_wav_audio", fail_timeline)
    else:
        monkeypatch.setattr(mock_target, mock_fail)

    queue = ProjectQueue([record], store=store, settings=settings)

    with pytest.raises(RenderStageError) as exc_info:
        queue._process_single_project(record)

    err = exc_info.value
    assert err.output_id == "out_target_01"
    assert err.category == expected_category, (str(err), repr(err.__cause__))
    assert record.error_scope == "OUTPUT"
    assert record.error_target == "out_target_01"
    assert record.outputs[0].status == OutputStatus.ERROR.value
    assert expected_category in record.outputs[0].error


# ---------------------------------------------------------------------------
# 8. Cancellation no callback and partial cleanup
# ---------------------------------------------------------------------------
def test_cancellation_no_callback_and_partial_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation prevents on_output_complete callback, cleans partial file, and marks CANCELLED."""
    settings = AppSettings()
    out = _make_dummy_output("out_cancel_01", "Cancel Output", "cancel-output")
    record, store, _ = _setup_project(tmp_path, [out], settings)

    pub_dir = tmp_path / "outputs" / "cancel-output"
    pub_dir.mkdir(parents=True, exist_ok=True)
    partial_file = pub_dir / "cancel-output.mp4"
    partial_file.write_bytes(b"partial video data")

    monkeypatch.setattr("toolrecap_v2.projects.probe_typed_media", _mock_probe)
    monkeypatch.setattr(
        "toolrecap_v2.subtitles.pipeline.SubtitlePipeline.get_episode_subtitles",
        _mock_subtitles,
    )

    cancel_evt = threading.Event()

    callback_called = False

    def on_complete(completed_out: CommentaryOutput, idx: int, total: int) -> None:
        nonlocal callback_called
        callback_called = True

    def mock_render_single(
        self: Any,
        out: CommentaryOutput,
        safe_title: str,
        pub_dir: Path,
        cancel_evt: threading.Event | None = None,
        **kwargs: Any,
    ) -> None:
        if cancel_evt:
            cancel_evt.set()
        raise RenderCancelled("Hủy render giữa chừng")

    monkeypatch.setattr(PublicationRenderer, "_render_single_output", mock_render_single)

    renderer = PublicationRenderer()
    manifest = AnalysisManifest(project_id="test", outputs=[out])

    with pytest.raises(RenderCancelled):
        renderer.render_manifest(
            manifest=manifest,
            settings=settings,
            output_root=tmp_path / "outputs",
            cancel_event=cancel_evt,
            on_output_complete=on_complete,
        )

    # Callback was NOT called for cancelled output
    assert callback_called is False

    # Partial file was removed
    assert not partial_file.exists()

    # Output status marked CANCELLED
    assert out.status == OutputStatus.CANCELLED.value


# ---------------------------------------------------------------------------
# 9. Stale Publication File Cleanup with Mocked Renderer Success
# ---------------------------------------------------------------------------
def test_stale_publication_file_cleanup_with_mocked_renderer_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-existing stale/temporary files in publication directory are cleaned up before render."""
    settings = AppSettings()
    safe_title = "stale-cleanup-test"
    out = _make_dummy_output("out_stale_01", "Stale Cleanup Test", safe_title)
    output_root = tmp_path / "outputs"
    pub_dir = output_root / safe_title
    pub_dir.mkdir(parents=True, exist_ok=True)

    # Populate stale files from earlier interrupted/failed runs
    stale_tmp = pub_dir / "stale_partial.tmp"
    stale_tmp.write_text("corrupted temporary data")
    stale_log = pub_dir / "failed_run.log"
    stale_log.write_text("old error log")
    stale_extra = pub_dir / "extra_audio.wav"
    stale_extra.write_bytes(b"RIFFoldwav")

    assert stale_tmp.is_file()
    assert stale_log.is_file()
    assert stale_extra.is_file()

    def mock_render_single(
        self: PublicationRenderer,
        *,
        out: CommentaryOutput,
        safe_title: str,
        pub_dir: Path,
        **kwargs: Any,
    ) -> None:
        # Prior stale files must have been cleaned by render_manifest before invoking single render
        assert not (pub_dir / "stale_partial.tmp").exists()
        assert not (pub_dir / "failed_run.log").exists()
        assert not (pub_dir / "extra_audio.wav").exists()

        # Generate expected 3 publication files
        (pub_dir / f"{safe_title}.mp4").write_bytes(b"rendered video stream")
        (pub_dir / f"{safe_title}.original.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nOriginal cue\n", encoding="utf-8"
        )
        (pub_dir / f"{safe_title}.narration.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nNarration cue\n", encoding="utf-8"
        )
        out.status = OutputStatus.COMPLETED.value
        out.progress = 100
        out.publication_video_path = str(pub_dir / f"{safe_title}.mp4")
        out.publication_original_srt_path = str(pub_dir / f"{safe_title}.original.srt")
        out.publication_narration_srt_path = str(pub_dir / f"{safe_title}.narration.srt")

    monkeypatch.setattr(PublicationRenderer, "_render_single_output", mock_render_single)

    renderer = PublicationRenderer()
    manifest = AnalysisManifest(project_id="test_stale_clean", outputs=[out])

    rendered = renderer.render_manifest(
        manifest=manifest,
        settings=settings,
        output_root=output_root,
    )

    assert len(rendered) == 1
    assert rendered[0].status == OutputStatus.COMPLETED.value
    assert not stale_tmp.exists()
    assert not stale_log.exists()
    assert not stale_extra.exists()

    expected_files = {f"{safe_title}.mp4", f"{safe_title}.original.srt", f"{safe_title}.narration.srt"}
    actual_files = {f.name for f in pub_dir.iterdir()}
    assert actual_files == expected_files
