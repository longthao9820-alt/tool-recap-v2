from __future__ import annotations

from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import pytest

from toolrecap_v2.analyzer.engine import AnalysisEngine, compute_final_plan_cache_key
from toolrecap_v2.analyzer.final_json import (
    CanonicalProjectFinalizer,
    FinalJsonValidationError,
    build_finalizer_project_prompt,
    validate_final_json,
)
from toolrecap_v2.domain.models import EpisodeEvidence, SourceEpisode
from toolrecap_v2.projects import ProjectRecord, compute_project_analysis_signature, compute_project_render_signature
from toolrecap_v2.settings import AppSettings
from toolrecap_v2.updater import generate_apply_script
from toolrecap_v2.renderer import PublicationRenderer
from toolrecap_v2.api_client import GLOBAL_GATEWAY_CONCURRENCY, OpenAICompatibleClient


class RecordingClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _episode(tmp_path: Path, episode_id: str = "E01", duration: float = 120.0) -> SourceEpisode:
    source = tmp_path / f"{episode_id}.mkv"
    source.write_bytes(b"source")
    return SourceEpisode(episode_id, str(source), duration_seconds=duration, title=episode_id)


def _output(ep: SourceEpisode, index: int = 1, *, end_ms: int = 10_000) -> dict[str, Any]:
    return {
        "output_id": f"out_{index:02d}",
        "title": f"Story {index}",
        "file_name": f"story-{index}.mp4",
        "output_type": "recap",
        "language": "en-US",
        "segments": [
            {
                "segment_id": "seg_01",
                "segment_type": "narration",
                "purpose": "hook",
                "narration_text": "A grounded narration.",
                "original_dialogue_text": "",
                "audio_policy": "duck",
                "subtitle_policy": "both",
                "recommended_visual_speed": 1.0,
                "source_clips": [
                    {
                        "episode_id": ep.episode_id,
                        "source_file": ep.source_video,
                        "start_ms": 0,
                        "end_ms": end_ms,
                    }
                ],
            }
        ],
    }


def test_generic_models_raw_prompt_and_all_outputs_reach_renderer_contract(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    prompt = "RAW user instruction: make any number of useful stories."
    settings = AppSettings(
        scanner_model="abc-worker",
        finalizer_model="xyz-editor",
        recap_prompt=prompt,
    )
    client = RecordingClient([{"outputs": [_output(ep, i) for i in range(1, 6)]}])
    finalizer = CanonicalProjectFinalizer(settings, client)  # type: ignore[arg-type]
    manifest = finalizer.finalize(
        project_id="season-project",
        episodes=[ep],
        evidence_map={"E01": EpisodeEvidence("E01", ep.source_video, 120.0, data={"events": []})},
        scope="SINGLE_EPISODE",
        require_source_files=True,
    )

    assert len(manifest.outputs) == 5
    assert client.calls[0]["model"] == "xyz-editor"
    assert prompt in client.calls[0]["user_text"]
    assert "candidate_proposals" not in client.calls[0]["user_text"]


def test_missing_outputs_is_schema_error_not_zero_output(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    with pytest.raises(FinalJsonValidationError) as exc:
        validate_final_json(
            {"message": "nothing"},
            project_id="p",
            episodes=[ep],
            scope="SINGLE_EPISODE",
            settings=AppSettings(),
            require_source_files=True,
        )
    assert exc.value.issues[0].code == "SCHEMA_ERROR"
    assert exc.value.issues[0].path == "$.outputs"


def test_invalid_source_reference_is_not_silently_substituted(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    bad = _output(ep)
    bad["segments"][0]["source_clips"][0]["source_file"] = "not-an-episode.mkv"
    with pytest.raises(FinalJsonValidationError) as exc:
        validate_final_json(
            {"outputs": [bad]},
            project_id="p",
            episodes=[ep],
            scope="SINGLE_EPISODE",
            settings=AppSettings(),
            require_source_files=True,
        )
    assert any(issue.code == "SOURCE_RESOLUTION_ERROR" for issue in exc.value.issues)


def test_explicit_empty_outputs_is_valid_editorial_result(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    manifest = validate_final_json(
        {"outputs": []},
        project_id="p",
        episodes=[ep],
        scope="SINGLE_EPISODE",
        settings=AppSettings(),
        require_source_files=True,
    )
    assert manifest.outputs == []
    assert manifest.zero_output_status == "VALID_EMPTY_OUTPUT"


def test_exact_timestamp_error_is_sent_back_for_repair(tmp_path: Path) -> None:
    ep = _episode(tmp_path, duration=5.0)
    invalid = {"outputs": [_output(ep, end_ms=7_000)]}
    repaired = {"outputs": [_output(ep, end_ms=5_000)]}
    client = RecordingClient([invalid, repaired])
    settings = AppSettings(finalizer_model="editor-model-y", final_json_repair_attempts=2)
    manifest = CanonicalProjectFinalizer(settings, client).finalize(  # type: ignore[arg-type]
        project_id="p",
        episodes=[ep],
        evidence_map={"E01": EpisodeEvidence("E01", ep.source_video, 5.0)},
        scope="SINGLE_EPISODE",
        require_source_files=True,
    )
    assert len(client.calls) == 2
    assert "TIMESTAMP_OUT_OF_RANGE" in client.calls[1]["user_text"]
    assert '"source_duration_ms": 5000' in client.calls[1]["user_text"]
    assert manifest.outputs[0].segments[0].source_clips[0].end == 5.0


def test_engine_bypasses_legacy_editorial_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ep = _episode(tmp_path)
    settings = AppSettings(
        scanner_model="worker-model-x",
        finalizer_model="editor-model-y",
        recap_prompt="Choose freely.",
    )
    client = RecordingClient([{"outputs": [_output(ep)]}])
    engine = AnalysisEngine(settings=settings, client=client)  # type: ignore[arg-type]
    monkeypatch.setattr(
        engine.scanner,
        "scan_episode",
        lambda *a, **k: EpisodeEvidence("E01", ep.source_video, 120.0, data={"events": []}),
    )
    monkeypatch.setattr(engine.connector, "connect_season", lambda *a, **k: pytest.fail("legacy connector called"))
    monkeypatch.setattr(engine.discoverer, "discover_single", lambda *a, **k: pytest.fail("candidate discovery called"))
    monkeypatch.setattr(engine.consolidator, "consolidate", lambda *a, **k: pytest.fail("candidate consolidation called"))
    monkeypatch.setattr(engine.verifier, "verify", lambda *a, **k: pytest.fail("candidate verification called"))

    manifest = engine.analyze("p", [ep], use_final_plan_cache=False)
    assert len(manifest.outputs) == 1
    assert [call["model"] for call in client.calls] == ["editor-model-y"]


def test_cache_dependencies_prompt_and_models_but_not_render_settings(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    evidence = {"E01": EpisodeEvidence("E01", ep.source_video, 120.0, data={"events": []})}
    base = AppSettings(scanner_model="worker", finalizer_model="editor", recap_prompt="A")
    k1 = compute_final_plan_cache_key(evidence, base, "SINGLE_EPISODE")
    base.voice_id = "another-voice"
    base.quality = "low"
    k2 = compute_final_plan_cache_key(evidence, base, "SINGLE_EPISODE")
    assert k1 == k2
    base.recap_prompt = "B"
    assert compute_final_plan_cache_key(evidence, base, "SINGLE_EPISODE") != k1
    base.recap_prompt = "A"
    base.finalizer_model = "editor-v2"
    assert compute_final_plan_cache_key(evidence, base, "SINGLE_EPISODE") != k1


def test_project_prompt_contains_source_mapping_and_scanner_observations(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    settings = AppSettings(recap_prompt="My exact prompt")
    text = build_finalizer_project_prompt(
        "p",
        [ep],
        {"E01": EpisodeEvidence("E01", ep.source_video, 120.0, data={"dialogue": [{"quote": "Hi"}]})},
        settings,
        "SINGLE_EPISODE",
    )
    assert "My exact prompt" in text
    assert Path(ep.source_video).name in text
    assert '"dialogue"' in text


def test_analysis_and_render_dependency_layers_are_separate(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    record = ProjectRecord(id="p", name="Project", source_episodes=[ep], analysis_scope="SINGLE_EPISODE")
    settings = AppSettings(scanner_model="worker", finalizer_model="editor", recap_prompt="Prompt A")
    ai_1 = compute_project_analysis_signature(record, settings)
    render_1 = compute_project_render_signature(ai_1, record, settings)

    settings.quality = "low"
    settings.use_gpu = False
    ai_2 = compute_project_analysis_signature(record, settings)
    render_2 = compute_project_render_signature(ai_2, record, settings)
    assert ai_2 == ai_1
    assert render_2 != render_1

    settings.recap_prompt = "Prompt B"
    assert compute_project_analysis_signature(record, settings) != ai_1


def test_updater_verifies_backup_before_cleaning_installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setattr("toolrecap_v2.updater.default_data_directory", lambda: data_dir)
    staged = tmp_path / "staged"
    target = tmp_path / "app"
    staged.mkdir()
    target.mkdir()
    script = generate_apply_script(staged, target)
    text = script.read_text(encoding="utf-8")
    verify_pos = text.index('if not exist "%BACKUP%\\%EXE_NAME%" goto backup_failed')
    clean_pos = text.index("Làm sạch thư mục ứng dụng")
    assert verify_pos < clean_pos
    assert ":apply_failed" in text
    assert ":rollback_failed" in text


def test_renderer_resume_skips_verified_completed_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ep = _episode(tmp_path)
    manifest = validate_final_json(
        {"outputs": [_output(ep)]},
        project_id="p",
        episodes=[ep],
        scope="SINGLE_EPISODE",
        settings=AppSettings(),
        require_source_files=True,
    )
    out = manifest.outputs[0]
    out.status = "COMPLETED"
    safe = out.sanitized_title
    pub = tmp_path / "published" / safe
    pub.mkdir(parents=True)
    validation = {
        "video_path": str(pub / f"{safe}.mp4"),
        "original_srt_path": str(pub / f"{safe}.original.srt"),
        "narration_srt_path": str(pub / f"{safe}.narration.srt"),
    }
    monkeypatch.setattr("toolrecap_v2.renderer.validate_publication_folder", lambda **kwargs: validation)
    monkeypatch.setattr(
        PublicationRenderer,
        "_render_single_output",
        lambda *a, **k: pytest.fail("completed output was rendered again"),
    )
    rendered = PublicationRenderer(voice_manager=object()).render_manifest(
        manifest,
        AppSettings(output_dir=str(tmp_path / "published")),
        output_root=tmp_path / "published",
        resume_completed=True,
    )
    assert rendered == [out]
    assert out.publication_video_path == validation["video_path"]


def test_global_gateway_scheduler_bounds_all_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    class Response:
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            nonlocal active
            with lock:
                active -= 1

        def read(self) -> bytes:
            time.sleep(0.03)
            return json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode()

    def fake_urlopen(*args: Any, **kwargs: Any) -> Response:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        return Response()

    monkeypatch.setattr("toolrecap_v2.api_client.request.urlopen", fake_urlopen)

    def invoke(_: int) -> dict[str, Any]:
        return OpenAICompatibleClient("http://gateway.test").chat_json(
            model="worker", system="JSON", user_text="{}", retry_delays=(0.0,),
        )

    with ThreadPoolExecutor(max_workers=GLOBAL_GATEWAY_CONCURRENCY * 2) as pool:
        results = list(pool.map(invoke, range(GLOBAL_GATEWAY_CONCURRENCY * 2)))
    assert all(result == {"ok": True} for result in results)
    assert peak <= GLOBAL_GATEWAY_CONCURRENCY
