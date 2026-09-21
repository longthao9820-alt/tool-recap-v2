"""Pytest fixtures and test environment setup for ToolRecap V2."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def _project_voice_preflight_test_double(monkeypatch: pytest.MonkeyPatch):
    """Keep non-voice ProjectQueue tests offline after voice preflight became mandatory.

    Individual voice/preflight tests can and do override this patch explicitly.
    """
    import wave

    class ReadyVoiceManager:
        def ensure_ready(self, *args, **kwargs):
            return type("Health", (), {"ready": True, "state": "READY"})()

        def synthesize(self, voice_id, text, output_path, **kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\x00\x20" * 2400)
            return output_path

    manager = ReadyVoiceManager()
    monkeypatch.setattr("toolrecap_v2.projects.get_voice_manager", lambda: manager)
    return manager


# These integration cases assert the superseded production topology
# (Season Connection -> Candidate Discovery/Consolidation/Verification -> Finalizer)
# or feed that topology's pre-Final-JSON fixtures into ProjectQueue. The refactor
# intentionally keeps their underlying modules unit-tested for cache migration and
# compatibility, but the forbidden topology is no longer a valid end-to-end contract.
# Canonical replacements live in test_canonical_final_json.py.
RETIRED_EDITORIAL_PIPELINE_TESTS = {
    "tests/test_finalizer_bounds_r7.py::test_final_plan_cache_and_resume_with_engine",
    "tests/test_finalizer_r8.py::test_final_plan_cache_and_resume_with_engine",
    "tests/test_gateway.py::test_sequential_batch_unaffected",
    "tests/test_renderer_queue.py::test_queue_season_phase_ordering_no_perepisode_finalization",
    "tests/test_renderer_queue.py::test_zero_outputs_completes_with_no_publication_files",
    "tests/test_renderer_queue.py::test_queue_state_recovery_on_cancellation_during_render",
    "tests/test_season_analysis.py::test_single_episode_zero_outputs",
    "tests/test_season_analysis.py::test_single_episode_one_output",
    "tests/test_season_analysis.py::test_single_episode_multiple_outputs_no_quota_slicing",
    "tests/test_season_analysis.py::test_simulated_e01_to_e05_season_connection_prompt",
    "tests/test_season_analysis.py::test_incomplete_coverage_allowed_informs_ai",
    "tests/test_season_analysis.py::test_selective_cache_invalidation_reuses_e01_to_e04",
    "tests/test_season_analysis.py::test_cancellation_during_season_connection",
    "tests/test_season_analysis.py::test_cancellation_during_finalizer",
    "tests/test_season_analysis.py::test_cross_batch_merge_finds_e02_setup_e08_payoff_preserves_supporting_arcs",
    "tests/test_season_analysis.py::test_resume_exact_batch3_failure_reuses_prior_batches_and_evidence",
    "tests/test_season_analysis.py::test_finalizer_fail_then_retry_connection_cache_hit_no_connection_calls",
    "tests/test_season_analysis.py::test_payload_size_protection_and_pre_serialization_reduction_10_episodes",
    "tests/test_targeted_r5.py::test_manifest_rewrite_after_render_paths",
    "tests/test_targeted_r6.py::test_plan_cache_hit_zero_calls_and_invalidation",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    retired = pytest.mark.skip(
        reason="Retired integration contract: application-side editorial pipeline was removed from production."
    )
    for item in items:
        if item.nodeid.replace("\\", "/") in RETIRED_EDITORIAL_PIPELINE_TESTS:
            item.add_marker(retired)

# Ensure Tkinter / Tcl libraries can be found if needed in tests
_py_base = Path(sys.base_prefix)
_tcl_dir = _py_base / "tcl" / "tcl8.6"
_tk_dir = _py_base / "tcl" / "tk8.6"
if _tcl_dir.is_dir():
    os.environ["TCL_LIBRARY"] = _tcl_dir.as_posix()
if _tk_dir.is_dir():
    os.environ["TK_LIBRARY"] = _tk_dir.as_posix()


@pytest.fixture(scope="session")
def tk_root():
    import tkinter as tk
    import gc
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.update_idletasks()
        root.update()
    except Exception:
        pass
    gc.collect()
    try:
        root.destroy()
    except Exception:
        pass
    gc.collect()


@pytest.fixture(scope="session")
def _shared_dummy_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import subprocess
    from toolrecap_v2.gpu import bundled_binary

    base_dir = tmp_path_factory.mktemp("shared_video")
    video_path = base_dir / "base_sample.mp4"
    ffmpeg = bundled_binary("ffmpeg")
    if ffmpeg:
        cmd = [
            str(ffmpeg),
            "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2.0:size=320x240:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2.0",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            str(video_path),
        ]
        subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if not video_path.is_file():
        video_path.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 200)
    return video_path


@pytest.fixture
def dummy_video(tmp_path: Path, _shared_dummy_video: Path) -> Path:
    """Create a minimal real MP4 video file using cached session base for fast tests."""
    import shutil
    video_path = tmp_path / "sample_video.mp4"
    shutil.copyfile(_shared_dummy_video, video_path)
    return video_path
