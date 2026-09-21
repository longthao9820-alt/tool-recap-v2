from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
import wave
from unittest.mock import MagicMock

import pytest

from toolrecap_v2.projects import ProjectQueue, ProjectRecord, ProjectStore
from toolrecap_v2.settings import AppSettings
from toolrecap_v2.voice.bootstrap import (
    BootstrapValidationError,
    VoiceRuntimeBootstrap,
    _classify_install_failure,
)
from toolrecap_v2.voice.manager import VoiceError, VoiceModelManager, validate_wav_audio
from toolrecap_v2.voice.runtime import (
    CORE_DEPENDENCIES,
    DEPENDENCY_MANIFEST_VERSION,
    VOICE_ENGINE_VERSION,
    VOICE_MODEL_REVISION,
    VOICE_RUNTIME_SCHEMA,
    VOICE_RUNTIME_VERSION,
    VoiceHealthResult,
    VoiceRuntimeInspector,
    runtime_fingerprint,
)


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x20" * 4800)


def _healthy() -> VoiceHealthResult:
    return VoiceHealthResult(
        runtime_present=True,
        runtime_version_ok=True,
        dependencies_ok=True,
        imports_ok=True,
        engine_initialized=True,
        model_present=True,
        model_valid=True,
        synthesis_test_ok=True,
        audio_validation_ok=True,
        ready=True,
        state="READY",
        human_message="ToolRecap local voice is ready.",
    )


def test_dependency_manifest_is_internally_compatible() -> None:
    assert CORE_DEPENDENCIES["omnivoice"] == "0.2.1"
    assert tuple(int(x) for x in CORE_DEPENDENCIES["transformers"].split(".")[:2]) >= (5, 3)
    assert CORE_DEPENDENCIES["torch"] == CORE_DEPENDENCIES["torchaudio"]
    assert VOICE_RUNTIME_SCHEMA and VOICE_RUNTIME_VERSION and DEPENDENCY_MANIFEST_VERSION


def test_cached_preview_cannot_bypass_runtime_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = VoiceModelManager(cache_dir=tmp_path / "models", subsystem_dir=tmp_path / "runtime", auto_bootstrap=False)
    stale = tmp_path / "stale-preview.wav"
    _write_wav(stale)
    calls: list[str] = []

    monkeypatch.setattr(mgr, "ensure_ready", lambda *a, **k: calls.append("health") or _healthy())

    def synth(_voice: str, _text: str, output: Path, **kwargs):
        calls.append("synthesis")
        _write_wav(output)
        return output

    monkeypatch.setattr(mgr, "_synthesize_production", synth)
    monkeypatch.setattr(mgr, "health_check", lambda *a, **k: _healthy())
    preview = mgr.preview("voicestudio.en.documentarian", "documentary")
    assert calls == ["health", "synthesis"]
    assert preview != stale
    validate_wav_audio(preview)


def test_preview_and_render_use_same_production_synthesis_method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = VoiceModelManager(cache_dir=tmp_path / "models", subsystem_dir=tmp_path / "runtime", auto_bootstrap=False)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(mgr, "ensure_ready", lambda *a, **k: _healthy())
    monkeypatch.setattr(mgr, "health_check", lambda *a, **k: _healthy())

    def synth(voice_id: str, text: str, output: Path, *, style: str, **kwargs):
        calls.append((voice_id, style))
        _write_wav(output)
        return output

    monkeypatch.setattr(mgr, "_synthesize_production", synth)
    mgr.preview("voicestudio.en.documentarian", "documentary")
    mgr.synthesize("voicestudio.en.documentarian", "Production narration", tmp_path / "render.wav", style="documentary")
    assert calls == [
        ("voicestudio.en.documentarian", "documentary"),
        ("voicestudio.en.documentarian", "documentary"),
    ]


def test_dependency_conflict_is_structured() -> None:
    code, message = _classify_install_failure(
        "ERROR: Cannot install packages because these package versions have conflicting dependencies. "
        "ResolutionImpossible"
    )
    assert code == "DEPENDENCY_CONFLICT"
    assert "incompatible" in message


def test_interrupted_install_does_not_replace_working_runtime(tmp_path: Path) -> None:
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "python.exe").write_bytes(b"working-runtime")
    bootstrap = VoiceRuntimeBootstrap(target_dir=target, staging_dir=tmp_path / "stage")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Exception):
        bootstrap.bootstrap(cancel_event=cancel, force=True)
    assert (target / "python.exe").read_bytes() == b"working-runtime"


def test_second_launch_recognizes_managed_runtime_without_reinstall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = tmp_path / "current"
    runtime.mkdir()
    (runtime / "python.exe").write_bytes(b"python")
    (runtime / "voice_runtime.json").write_text(json.dumps({
        "voice_runtime_schema": VOICE_RUNTIME_SCHEMA,
        "voice_runtime_version": VOICE_RUNTIME_VERSION,
        "dependency_manifest_version": DEPENDENCY_MANIFEST_VERSION,
        "runtime_fingerprint": runtime_fingerprint(),
    }), encoding="utf-8")
    cache = tmp_path / "models" / "omnivoice"
    snapshot = cache / "hub" / "models--k2-fsa--OmniVoice" / "snapshots" / VOICE_MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    with (snapshot / "model.safetensors").open("wb") as handle:
        handle.seek(101 * 1024 * 1024)
        handle.write(b"0")
    (cache / "toolrecap_model.json").write_text(json.dumps({
        "model_id": "k2-fsa/OmniVoice",
        "model_revision": VOICE_MODEL_REVISION,
        "runtime_fingerprint": runtime_fingerprint(),
    }), encoding="utf-8")
    versions = dict(CORE_DEPENDENCIES)
    report = json.dumps({"python": "3.11.9", "versions": versions, "cuda": False, "device": "cpu"})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, report, ""))
    health = VoiceRuntimeInspector(runtime, cache).inspect(run_imports=True)
    assert health.dependencies_ok
    assert health.model_valid
    assert health.selected_device == "cpu"


def test_external_voicestudio_environment_never_changes_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    external = tmp_path / "external" / "python.exe"
    external.parent.mkdir()
    external.write_bytes(b"external")
    monkeypatch.setenv("VOICESTUDIO_PYTHON", str(external))
    mgr = VoiceModelManager(cache_dir=tmp_path / "models", subsystem_dir=tmp_path / "managed", detect_official=True, auto_bootstrap=False)
    assert mgr.get_backend_python_executable() is None


def test_voice_preflight_fails_before_media_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    record = ProjectRecord.from_video_path(source, tmp_path / "out")
    store = ProjectStore(tmp_path / "projects.json")
    fake_manager = MagicMock()
    fake_manager.ensure_ready.side_effect = VoiceError("dependency verification failed")
    monkeypatch.setattr("toolrecap_v2.projects.get_voice_manager", lambda: fake_manager)
    monkeypatch.setattr(
        "toolrecap_v2.projects.probe_typed_media",
        lambda *a, **k: pytest.fail("media probing must not start before voice preflight"),
    )
    queue = ProjectQueue([record], store=store, settings=AppSettings())
    with pytest.raises(Exception, match="VOICE_ERROR"):
        queue._process_single_project(record)
    fake_manager.ensure_ready.assert_called_once()
