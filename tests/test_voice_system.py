"""Tests for voice catalog, 12 V1 voices, model manager caching, truthful backend status, and audio synthesis."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from toolrecap_v2.voice.catalog import (
    BUILTIN_VOICES,
    DEFAULT_VOICE_BY_LANGUAGE,
    DEFAULT_VOICE_ID,
    DEFAULT_VOICE_ID_GB,
    DEFAULT_VOICE_ID_US,
    PIPER_COMPATIBILITY_VOICES,
    STYLE_INSTRUCTIONS,
    STYLE_NAMES,
    SUPPORTED_VOICE_STYLES,
    clear_official_runtime_cache,
    detect_official_voicestudio_runtime,
    get_available_voices,
    get_voice_spec,
    get_voice_status,
    is_voice_selectable,
    is_voicestudio_ready,
    migrate_voice_setting,
)
from toolrecap_v2.voice.manager import (
    DownloadCancelled,
    VoiceError,
    VoiceModelManager,
    validate_wav_audio,
)
from toolrecap_v2.voice.audio_preview import AudioPreviewPlayer
from toolrecap_v2.voice.bootstrap import (
    BootstrapAsset,
    BootstrapCancelled,
    BootstrapError,
    BootstrapSecurityError,
    BootstrapValidationError,
    VoiceRuntimeBootstrap,
    VoiceRuntimeBootstrapManifest,
    safe_extract_zip,
)
from toolrecap_v2.voice.omnivoice_adapter import (
    OFFICIAL_VOICE_INSTRUCTS,
    validate_request,
)


def test_builtin_voice_catalog_exact_12_voicestudio() -> None:
    """Invariant 1: Exact 12 V1 voices, engine voicestudio, correct IDs/labels/languages/genders/defaults."""
    assert len(BUILTIN_VOICES) == 12

    # Check defaults
    assert DEFAULT_VOICE_ID == "voicestudio.en.documentarian"
    assert DEFAULT_VOICE_ID_US == "voicestudio.en.documentarian"
    assert DEFAULT_VOICE_ID_GB == "voicestudio.en.commentator"
    assert DEFAULT_VOICE_BY_LANGUAGE["en-US"] == "voicestudio.en.documentarian"
    assert DEFAULT_VOICE_BY_LANGUAGE["en-GB"] == "voicestudio.en.commentator"
    assert DEFAULT_VOICE_ID in BUILTIN_VOICES

    # Exact expected 12 IDs from survey
    expected_us_voices = {
        "voicestudio.en.neighbor": ("Female", "Neighbor — Nữ — Premium Local"),
        "voicestudio.en.companion": ("Female", "Companion — Nữ — Premium Local"),
        "voicestudio.en.teacher": ("Female", "Teacher — Nữ — Premium Local"),
        "voicestudio.en.anchor": ("Male", "Anchor — Nam — Premium Local"),
        "voicestudio.en.documentarian": ("Male", "Documentarian — Nam — Premium Local"),
        "voicestudio.en.promo": ("Male", "Promo — Nam — Premium Local"),
    }
    expected_gb_voices = {
        "voicestudio.en.librarian": ("Female", "Librarian — Nữ — Premium Local"),
        "voicestudio.en.podcaster": ("Female", "Podcaster — Nữ — Premium Local"),
        "voicestudio.en.luxe": ("Female", "Luxe — Nữ — Premium Local"),
        "voicestudio.en.storyteller": ("Male", "Storyteller — Nam — Premium Local"),
        "voicestudio.en.commentator": ("Male", "Commentator — Nam — Premium Local"),
        "voicestudio.en.explainer": ("Male", "Explainer — Nam — Premium Local"),
    }

    # Verify en-US voices
    for vid, (expected_gender, expected_label) in expected_us_voices.items():
        assert vid in BUILTIN_VOICES
        spec = BUILTIN_VOICES[vid]
        assert spec.voice_id == vid
        assert spec.engine == "voicestudio"
        assert spec.language == "en-US"
        assert spec.gender == expected_gender
        assert spec.display_name == expected_label
        assert spec.preview_text
        assert spec.repo_id == "k2-fsa/OmniVoice@c5fdb5c"

    # Verify en-GB voices
    for vid, (expected_gender, expected_label) in expected_gb_voices.items():
        assert vid in BUILTIN_VOICES
        spec = BUILTIN_VOICES[vid]
        assert spec.voice_id == vid
        assert spec.engine == "voicestudio"
        assert spec.language == "en-GB"
        assert spec.gender == expected_gender
        assert spec.display_name == expected_label
        assert spec.preview_text
        assert spec.repo_id == "k2-fsa/OmniVoice@c5fdb5c"


def test_no_piper_in_production_selectable_list() -> None:
    """Invariant 1: Four Piper voices may remain internal compatibility fallback only, NOT in selectable catalog."""
    # 1. BUILTIN_VOICES must contain NO Piper voices
    for vid, spec in BUILTIN_VOICES.items():
        assert not vid.startswith("piper.")
        assert spec.engine == "voicestudio"

    # 2. get_available_voices must contain NO Piper voices
    available = get_available_voices()
    assert len(available) == 12
    for vid, spec in available.items():
        assert not vid.startswith("piper.")
        assert spec.engine == "voicestudio"

    # 3. Compatibility voices exist separately for backwards lookup
    assert len(PIPER_COMPATIBILITY_VOICES) == 4
    assert "piper.en_US-lessac-medium" in PIPER_COMPATIBILITY_VOICES
    assert "piper.en_US-ryan-medium" in PIPER_COMPATIBILITY_VOICES
    assert "piper.en_GB-alba-medium" in PIPER_COMPATIBILITY_VOICES
    assert "piper.en_GB-alan-medium" in PIPER_COMPATIBILITY_VOICES

    # 4. get_voice_spec can still resolve legacy Piper voice for compatibility
    legacy_spec = get_voice_spec("piper.en_US-lessac-medium")
    assert legacy_spec.engine == "piper"
    assert legacy_spec.voice_id == "piper.en_US-lessac-medium"


def test_truthful_backend_readiness_and_status(tmp_path: Path) -> None:
    """Invariant 2: Catalog must not lie that VoiceStudio is installed when absent."""
    subsystem_dir = tmp_path / "voice_subsystem"
    subsystem_dir.mkdir()

    # 1. When adapter is NOT present:
    assert is_voicestudio_ready(subsystem_dir) is False
    assert is_voice_selectable("voicestudio.en.documentarian", subsystem_dir=subsystem_dir) is False

    status = get_voice_status("voicestudio.en.documentarian", subsystem_dir=subsystem_dir)
    assert status["ready"] is False
    assert status["status"] == "NOT_INSTALLED"
    assert "chưa được cài đặt" in status["status_label"]

    # 2. When executable adapter IS present:
    (subsystem_dir / "VoiceStudio.exe").write_bytes(b"MZ_EXE")
    assert is_voicestudio_ready(subsystem_dir) is True
    assert is_voice_selectable("voicestudio.en.documentarian", subsystem_dir=subsystem_dir) is True

    status_ready = get_voice_status("voicestudio.en.documentarian", subsystem_dir=subsystem_dir)
    assert status_ready["ready"] is True
    assert status_ready["status"] == "READY"
    assert "sẵn sàng" in status_ready["status_label"]


def test_truthful_synthesis_rejection_without_adapter(tmp_path: Path) -> None:
    """Invariant 2: Production must reject VoiceStudio synthesis when adapter is missing, not silently Piper."""
    empty_subsystem = tmp_path / "empty_subsystem"
    empty_subsystem.mkdir()
    mgr = VoiceModelManager(cache_dir=tmp_path / "models", subsystem_dir=empty_subsystem)
    out_wav = tmp_path / "synth.wav"

    # In production (allow_mock_synth=False), calling synthesize with voicestudio voice must fail honestly
    with pytest.raises(VoiceError, match="VoiceStudio adapter chưa được cài đặt hoặc không khả dụng"):
        mgr.synthesize("voicestudio.en.documentarian", "Test text", out_wav, allow_mock_synth=False)

    # In test harness (allow_mock_synth=True), test synthesis is allowed
    mgr.synthesize("voicestudio.en.documentarian", "Test text", out_wav, allow_mock_synth=True)
    assert out_wav.is_file()
    validate_wav_audio(out_wav)


def test_piper_migration_logic() -> None:
    """Invariant 2: Existing selected Piper migrates to VoiceStudio only when backend ready; preserves old with warning otherwise."""
    # When backend is NOT ready:
    val, warning = migrate_voice_setting("piper.en_US-lessac-medium", "en-US", backend_ready=False)
    assert val == "piper.en_US-lessac-medium"
    assert warning is not None
    assert "chưa sẵn sàng" in warning

    # When backend IS ready:
    val_us_female, warn1 = migrate_voice_setting("piper.en_US-lessac-medium", "en-US", backend_ready=True)
    assert val_us_female == "voicestudio.en.neighbor"
    assert warn1 is None

    val_us_male, warn2 = migrate_voice_setting("piper.en_US-ryan-medium", "en-US", backend_ready=True)
    assert val_us_male == "voicestudio.en.documentarian"
    assert warn2 is None

    val_gb_female, warn3 = migrate_voice_setting("piper.en_GB-alba-medium", "en-GB", backend_ready=True)
    assert val_gb_female == "voicestudio.en.librarian"
    assert warn3 is None

    val_gb_male, warn4 = migrate_voice_setting("piper.en_GB-alan-medium", "en-GB", backend_ready=True)
    assert val_gb_male == "voicestudio.en.commentator"
    assert warn4 is None

    # Already a VoiceStudio voice -> preserved without warning
    val_vs, warn_vs = migrate_voice_setting("voicestudio.en.anchor", "en-US", backend_ready=True)
    assert val_vs == "voicestudio.en.anchor"
    assert warn_vs is None


def test_voice_styles_validation(tmp_path: Path) -> None:
    """Invariant 4: Supported voice styles exact list from V1; unsupported clearly rejected."""
    expected_styles = {
        "film_recap",
        "storytelling",
        "documentary",
        "crime_thriller",
        "drama",
        "soap_emotional",
        "energetic",
        "neutral",
    }
    assert set(SUPPORTED_VOICE_STYLES) == expected_styles
    for style_id in expected_styles:
        assert style_id in STYLE_NAMES
        assert style_id in STYLE_INSTRUCTIONS

    mgr = VoiceModelManager(cache_dir=tmp_path / "models")
    out_wav = tmp_path / "style_test.wav"

    # Unsupported style -> must raise VoiceError
    with pytest.raises(VoiceError, match="không được hỗ trợ"):
        mgr.synthesize(
            DEFAULT_VOICE_ID,
            "Testing unsupported style",
            out_wav,
            style="non_existent_fake_style",
            allow_mock_synth=True,
        )

    # Supported style -> accepted
    mgr.synthesize(
        DEFAULT_VOICE_ID,
        "Testing supported style",
        out_wav,
        style="film_recap",
        allow_mock_synth=True,
    )
    assert out_wav.is_file()


def test_voice_manager_cache_reuse_and_progress(tmp_path: Path) -> None:
    """Invariant 3: Lazy model download, cache lookup, reuse outside app directory."""
    cache_dir = tmp_path / "models"
    subsystem_dir = tmp_path / "subsystem"
    mgr = VoiceModelManager(cache_dir=cache_dir, subsystem_dir=subsystem_dir, detect_official=False)
    spec = get_voice_spec(DEFAULT_VOICE_ID)

    # 1. Initially not installed
    assert mgr.is_voice_installed(spec.voice_id) is False

    # 2. Simulate installed OmniVoice model weights in shared cache directory
    ov_dir = cache_dir / "omnivoice" / "hub" / "models--k2-fsa--OmniVoice"
    ov_dir.mkdir(parents=True, exist_ok=True)
    (ov_dir / "model.safetensors").write_bytes(b"cached_omnivoice_model_weights")

    assert mgr.is_voice_installed(spec.voice_id) is True

    # 3. Cache reuse: ensure_voice_model reuses cache immediately and reports 100% progress
    progress_records: list[tuple[int, int, float]] = []

    def on_progress(downloaded: int, total: int, percent: float) -> None:
        progress_records.append((downloaded, total, percent))

    res_dir = mgr.ensure_voice_model(spec.voice_id, progress_callback=on_progress)
    assert res_dir == cache_dir / "omnivoice"
    assert len(progress_records) == 1
    assert progress_records[0] == (100, 100, 100.0)

    # 4. Piper compatibility fallback voice uses per-voice folder
    piper_spec = get_voice_spec("piper.en_US-lessac-medium")
    assert mgr.is_voice_installed(piper_spec.voice_id) is False

    p_dir = cache_dir / piper_spec.voice_id
    p_dir.mkdir(parents=True, exist_ok=True)
    for req_f in piper_spec.required_files:
        (p_dir / req_f).write_bytes(b"cached_piper_model_weights")

    assert mgr.is_voice_installed(piper_spec.voice_id) is True

    piper_progress: list[tuple[int, int, float]] = []
    p_res = mgr.ensure_voice_model(
        piper_spec.voice_id,
        progress_callback=lambda d, t, p: piper_progress.append((d, t, p)),
    )
    assert p_res == p_dir
    assert piper_progress == [(100, 100, 100.0)]


def test_validate_wav_audio_rejects_corrupted(tmp_path: Path) -> None:
    bad_wav = tmp_path / "bad.wav"
    bad_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00corrupt")
    with pytest.raises(VoiceError):
        validate_wav_audio(bad_wav)

    nonexistent = tmp_path / "missing.wav"
    with pytest.raises(VoiceError, match="Không tìm thấy"):
        validate_wav_audio(nonexistent)


def test_audio_preview_player(tmp_path: Path) -> None:
    mgr = VoiceModelManager(cache_dir=tmp_path / "models")
    out_wav = tmp_path / "preview.wav"
    mgr.synthesize(DEFAULT_VOICE_ID, "Testing preview audio playback.", out_wav, allow_mock_synth=True)

    player = AudioPreviewPlayer()
    assert player.is_playing is False
    player.play(out_wav)
    player.stop()
    assert player.is_playing is False


def test_exact_12_instructs_match_official_voicestudio() -> None:
    """Invariant: All 12 VoiceStudio designed voices match official archetype instructs."""
    assert len(OFFICIAL_VOICE_INSTRUCTS) == 12
    assert len(BUILTIN_VOICES) == 12

    for voice_id, spec in BUILTIN_VOICES.items():
        assert voice_id in OFFICIAL_VOICE_INSTRUCTS
        assert spec.instruct == OFFICIAL_VOICE_INSTRUCTS[voice_id]
        assert spec.instruct, f"Instruct for {voice_id} cannot be empty"


def test_omnivoice_adapter_validate_request(tmp_path: Path) -> None:
    """Invariant: omnivoice_adapter validates voice_id, style, text, and output path."""
    out_wav = tmp_path / "test.wav"

    # Valid request
    validate_request("voicestudio.en.documentarian", "Valid text", "film_recap", out_wav)

    # Invalid voice ID
    with pytest.raises(ValueError, match="không nằm trong danh sách 12 giọng đọc"):
        validate_request("invalid.voice", "Valid text", "film_recap", out_wav)

    # Invalid style
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        validate_request("voicestudio.en.documentarian", "Valid text", "unsupported_style", out_wav)

    # Empty text
    with pytest.raises(ValueError, match="không được để trống"):
        validate_request("voicestudio.en.documentarian", "   ", "film_recap", out_wav)

    # Non-wav extension
    with pytest.raises(ValueError, match="phải là tệp .wav"):
        validate_request("voicestudio.en.documentarian", "Valid text", "film_recap", tmp_path / "test.mp3")


def test_bootstrap_safe_extract_zip_security(tmp_path: Path) -> None:
    """Invariant: safe_extract_zip rejects zip-slip path traversal attacks."""
    bad_zip = tmp_path / "malicious.zip"
    extract_target = tmp_path / "extracted"
    extract_target.mkdir()

    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../evil_file.txt", b"malicious content")

    with pytest.raises(BootstrapSecurityError, match="Path Traversal"):
        safe_extract_zip(bad_zip, extract_target)


def test_bootstrap_download_asset_verification_and_mismatch(tmp_path: Path) -> None:
    """Invariant: download_asset verifies SHA256 and cleans up temp files on mismatch."""
    staging_dir = tmp_path / "staging"
    bootstrap = VoiceRuntimeBootstrap(staging_dir=staging_dir)

    payload = b"test asset data for sha verification"
    expected_sha = hashlib.sha256(payload).hexdigest()

    asset_valid = BootstrapAsset(
        name="valid_asset.bin",
        url="https://example.com/asset.bin",
        sha256=expected_sha,
        size=len(payload),
    )

    class DummyResponse:
        def __init__(self, data: bytes):
            self._stream = io.BytesIO(data)
            self.headers = {"Content-Length": str(len(data))}

        def read(self, chunk_size: int = 65536) -> bytes:
            return self._stream.read(chunk_size)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    # Success case
    with patch("urllib.request.urlopen", return_value=DummyResponse(payload)):
        dest = staging_dir / "valid_asset.bin"
        res_path = bootstrap.download_asset(asset_valid, dest)
        assert res_path.is_file()
        assert res_path.read_bytes() == payload

    # Mismatch case
    asset_invalid = BootstrapAsset(
        name="invalid_asset.bin",
        url="https://example.com/asset.bin",
        sha256="wronghash00000000000000000000000000000000000000000000000000000000",
        size=len(payload),
    )
    with patch("urllib.request.urlopen", return_value=DummyResponse(payload)):
        dest_inv = staging_dir / "invalid_asset.bin"
        with pytest.raises(BootstrapValidationError, match="không khớp"):
            bootstrap.download_asset(asset_invalid, dest_inv)
        assert not dest_inv.exists()
        assert not (staging_dir / "invalid_asset.bin.part").exists()


def test_bootstrap_cancellation(tmp_path: Path) -> None:
    """Invariant: bootstrap and download_asset respect cancellation event immediately."""
    bootstrap = VoiceRuntimeBootstrap(staging_dir=tmp_path / "staging", target_dir=tmp_path / "target")
    cancel_event = threading.Event()
    cancel_event.set()

    asset = BootstrapAsset(
        name="test.bin",
        url="https://example.com/test.bin",
        sha256="",
        size=100,
    )
    with pytest.raises(BootstrapCancelled):
        bootstrap.download_asset(asset, tmp_path / "out.bin", cancel_event=cancel_event)

    with pytest.raises(BootstrapCancelled):
        bootstrap.bootstrap(cancel_event=cancel_event)


def test_bootstrap_atomic_swap_and_rollback_on_failure(tmp_path: Path) -> None:
    """Invariant: bootstrap rollback preserves original runtime if installation fails during swap."""
    target_dir = tmp_path / "runtime"
    target_dir.mkdir(parents=True)
    orig_py = target_dir / "python.exe"
    orig_py.write_bytes(b"original_preexisting_python_binary")

    staging_dir = tmp_path / "staging"
    bootstrap = VoiceRuntimeBootstrap(target_dir=target_dir, staging_dir=staging_dir)

    def fake_download_asset(asset, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"dummy")
        return destination

    def fake_safe_extract(zip_path, target_d):
        target_d.mkdir(parents=True, exist_ok=True)
        (target_d / "python.exe").write_bytes(b"new_extracted_python")

    mock_run = MagicMock()
    mock_run.side_effect = [
        subprocess.CompletedProcess(args=["python", "get-pip.py"], returncode=0, stdout="", stderr=""),
        subprocess.CompletedProcess(args=["python", "-m", "pip", "torch"], returncode=0, stdout="", stderr=""),
        subprocess.CompletedProcess(args=["python", "-m", "pip", "other"], returncode=0, stdout="", stderr=""),
    ]

    original_copy2 = shutil.copy2
    fail_once = True

    def failing_copy2(src, dst):
        nonlocal fail_once
        if fail_once and "python.exe" in str(dst):
            fail_once = False
            raise OSError("Simulated disk error during atomic swap")
        return original_copy2(src, dst)

    with (
        patch.object(bootstrap, "download_asset", side_effect=fake_download_asset),
        patch("toolrecap_v2.voice.bootstrap.safe_extract_zip", side_effect=fake_safe_extract),
        patch("subprocess.run", mock_run),
        patch("shutil.copy2", side_effect=failing_copy2),
    ):
        with pytest.raises(OSError, match="Simulated disk error during atomic swap"):
            bootstrap.bootstrap(install_packages=True, force=True)

    # Check original runtime restored via rollback
    assert orig_py.is_file()
    assert orig_py.read_bytes() == b"original_preexisting_python_binary"
    assert not staging_dir.exists()


def test_manager_synthesize_subprocess_request_inspection(tmp_path: Path) -> None:
    """Invariant: synthesize invokes backend Python via temp JSON request with exact arguments and shell=False."""
    cache_dir = tmp_path / "models"
    subsystem_dir = tmp_path / "subsystem"
    runtime_py = subsystem_dir / "runtime" / "python.exe"
    runtime_py.parent.mkdir(parents=True)
    runtime_py.write_bytes(b"MZ_FAKE_PYTHON")

    # Shared OmniVoice model cache
    ov_dir = cache_dir / "omnivoice" / "hub" / "models--k2-fsa--OmniVoice"
    ov_dir.mkdir(parents=True, exist_ok=True)

    mgr = VoiceModelManager(cache_dir=cache_dir, subsystem_dir=subsystem_dir, detect_official=False)
    out_wav = tmp_path / "output.wav"

    captured: dict[str, Any] = {}

    def _write_dummy_wav(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00\x20" * 2400)

    class MockPopen:
        def __init__(self, cmd, stdout=None, stderr=None, text=True, shell=False, env=None):
            captured["cmd"] = cmd
            captured["shell"] = shell
            captured["env"] = env
            self.pid = 99999
            self.returncode = 0
            req_idx = cmd.index("--request")
            req_path = Path(cmd[req_idx + 1])
            captured["request_payload"] = json.loads(req_path.read_text(encoding="utf-8"))
            _write_dummy_wav(Path(captured["request_payload"]["output"]))
            self.stderr = io.StringIO('{"stage": "generating", "progress": 50}\n{"stage": "done", "progress": 100}\n')
            self._polled = False

        def poll(self):
            if not self._polled:
                self._polled = True
                return None
            return 0

        def wait(self, timeout=None):
            return 0

    progress_events: list[tuple[int, int, float]] = []

    with patch("subprocess.Popen", MockPopen):
        res = mgr.synthesize(
            "voicestudio.en.documentarian",
            "Inspection test text",
            out_wav,
            style="documentary",
            progress_callback=lambda d, t, p: progress_events.append((d, t, p)),
            allow_mock_synth=False,
        )

    assert res == out_wav
    assert out_wav.is_file()
    assert captured["shell"] is False
    assert captured["cmd"][0] == str(runtime_py)
    assert "--request" in captured["cmd"]

    payload = captured["request_payload"]
    assert payload["voice_id"] == "voicestudio.en.documentarian"
    assert payload["text"] == "Inspection test text"
    assert payload["style"] == "documentary"
    assert payload["instruct"] == OFFICIAL_VOICE_INSTRUCTS["voicestudio.en.documentarian"]
    assert payload["output"] == str(out_wav)
    assert payload["cache_dir"] == str(cache_dir / "omnivoice")

    # Verify isolated environment
    env = captured["env"]
    assert env["HF_HOME"] == str(cache_dir / "omnivoice")
    assert env["HUGGINGFACE_HUB_CACHE"] == str(cache_dir / "omnivoice" / "hub")

    # Verify progress reported from JSON stderr
    assert (50, 100, 50.0) in progress_events
    assert (100, 100, 100.0) in progress_events


def test_manager_synthesize_stderr_sanitization_and_progress(tmp_path: Path) -> None:
    """Invariant: synthesize parses progress from stderr and sanitizes error messages."""
    cache_dir = tmp_path / "models"
    subsystem_dir = tmp_path / "subsystem"
    runtime_py = subsystem_dir / "runtime" / "python.exe"
    runtime_py.parent.mkdir(parents=True)
    runtime_py.write_bytes(b"MZ_FAKE_PYTHON")

    ov_dir = cache_dir / "omnivoice" / "hub" / "models--k2-fsa--OmniVoice"
    ov_dir.mkdir(parents=True, exist_ok=True)

    mgr = VoiceModelManager(cache_dir=cache_dir, subsystem_dir=subsystem_dir, detect_official=False)
    out_wav = tmp_path / "fail.wav"

    class FailingPopen:
        def __init__(self, cmd, stdout=None, stderr=None, text=True, shell=False, env=None):
            self.pid = 88888
            self.returncode = 1
            raw_stderr = (
                '{"stage": "generating", "progress": 42}\n'
                '\x1b[31mCritical GPU error occurred!\x1b[0m\n'
                + "A" * 2000
            )
            self.stderr = io.StringIO(raw_stderr)
            self._polled = False

        def poll(self):
            if not self._polled:
                self._polled = True
                return None
            return 1

        def wait(self, timeout=None):
            return 1

    progress_events: list[tuple[int, int, float]] = []

    with patch("subprocess.Popen", FailingPopen):
        with pytest.raises(VoiceError) as exc_info:
            mgr.synthesize(
                "voicestudio.en.anchor",
                "Failing synthesis test",
                out_wav,
                progress_callback=lambda d, t, p: progress_events.append((d, t, p)),
                allow_mock_synth=False,
            )

    # Progress JSON was parsed
    assert (42, 100, 42.0) in progress_events

    # Stderr was sanitized: no JSON progress line, length truncated, error message included
    err_str = str(exc_info.value)
    assert "Critical GPU error occurred!" in err_str
    assert '{"stage": "generating"' not in err_str
    assert "[truncated]" in err_str


def test_manager_synthesize_cancellation_kills_process_tree(tmp_path: Path) -> None:
    """Invariant: cancellation during synthesis terminates process tree via _kill_process_tree."""
    cache_dir = tmp_path / "models"
    subsystem_dir = tmp_path / "subsystem"
    runtime_py = subsystem_dir / "runtime" / "python.exe"
    runtime_py.parent.mkdir(parents=True)
    runtime_py.write_bytes(b"MZ_FAKE_PYTHON")

    ov_dir = cache_dir / "omnivoice" / "hub" / "models--k2-fsa--OmniVoice"
    ov_dir.mkdir(parents=True, exist_ok=True)

    mgr = VoiceModelManager(cache_dir=cache_dir, subsystem_dir=subsystem_dir, detect_official=False)
    out_wav = tmp_path / "cancel.wav"
    cancel_event = threading.Event()

    killed_pids: list[int] = []

    class HangingPopen:
        def __init__(self, cmd, stdout=None, stderr=None, text=True, shell=False, env=None):
            self.pid = 77777
            self.returncode = None
            self.stderr = io.StringIO("")

        def poll(self):
            cancel_event.set()
            return None

        def wait(self, timeout=None):
            return 0

    with (
        patch("subprocess.Popen", HangingPopen),
        patch("toolrecap_v2.voice.manager._kill_process_tree", side_effect=lambda pid: killed_pids.append(pid)),
    ):
        with pytest.raises(VoiceError, match="Quá trình đọc giọng đã bị hủy"):
            mgr.synthesize(
                "voicestudio.en.anchor",
                "Cancel test",
                out_wav,
                cancel_event=cancel_event,
                allow_mock_synth=False,
            )

    assert 77777 in killed_pids


def test_official_voicestudio_runtime_smoke() -> None:
    """Invariant: official VoiceStudio runtime import smoke runs if detected, skips cleanly if absent."""
    clear_official_runtime_cache()
    official_py = detect_official_voicestudio_runtime()
    if official_py is None:
        pytest.skip("Official VoiceStudio runtime not detected on host machine.")

    assert official_py.is_file()
    res = subprocess.run(
        [
            str(official_py),
            "-c",
            "import torch, torchaudio, omnivoice; from importlib.metadata import version; print(version('omnivoice'))",
        ],
        capture_output=True,
        text=True,
        shell=False,
        timeout=30,
    )
    assert res.returncode == 0, f"Official runtime smoke failed: {res.stderr}"
    assert res.stdout.strip(), "Official runtime returned empty omnivoice version"
