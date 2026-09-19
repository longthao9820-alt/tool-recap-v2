"""Tests for narration preparation, scene segmentation, and manifest handling."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from toolrecap_v2.narration import (
    NarrationError,
    RecapManifest,
    RecapSegment,
    _parse_srt_content,
    prepare_narration_for_video,
    validate_manifest,
)
from toolrecap_v2.settings import AppSettings


def test_recap_segment_properties() -> None:
    seg = RecapSegment(
        segment_id="seg_01",
        start_ms=1000,
        end_ms=4500,
        narration_text="The story begins here.",
    )
    assert seg.duration_ms == 3500
    assert seg.start_sec == 1.0
    assert seg.end_sec == 4.5
    assert seg.duration_sec == 3.5


def test_validate_manifest_success() -> None:
    manifest = RecapManifest(
        project_id="test",
        source_video="video.mp4",
        recap_mode="FULL_EPISODE",
        recap_language="en-US",
        segments=[
            RecapSegment("s1", 0, 5000, "Scene one narration"),
            RecapSegment("s2", 6000, 12000, "Scene two narration"),
        ],
    )
    # Total duration 15s -> valid
    validate_manifest(manifest, 15.0)


def test_validate_manifest_empty_segments() -> None:
    manifest = RecapManifest(
        project_id="test",
        source_video="video.mp4",
        recap_mode="FULL_EPISODE",
        recap_language="en-US",
        segments=[],
    )
    with pytest.raises(NarrationError, match="không có phân đoạn"):
        validate_manifest(manifest, 10.0)


def test_validate_manifest_invalid_timing() -> None:
    # Negative start
    m1 = RecapManifest(
        project_id="test",
        source_video="video.mp4",
        recap_mode="FULL_EPISODE",
        recap_language="en-US",
        segments=[RecapSegment("s1", -100, 5000, "Narration")],
    )
    with pytest.raises(NarrationError, match="âm"):
        validate_manifest(m1, 10.0)

    # End <= Start
    m2 = RecapManifest(
        project_id="test",
        source_video="video.mp4",
        recap_mode="FULL_EPISODE",
        recap_language="en-US",
        segments=[RecapSegment("s1", 5000, 4000, "Narration")],
    )
    with pytest.raises(NarrationError, match="end_ms <= start_ms"):
        validate_manifest(m2, 10.0)

    # Exceeds video duration
    m3 = RecapManifest(
        project_id="test",
        source_video="video.mp4",
        recap_mode="FULL_EPISODE",
        recap_language="en-US",
        segments=[RecapSegment("s1", 0, 25000, "Narration")],
    )
    with pytest.raises(NarrationError, match="vượt quá độ dài video"):
        validate_manifest(m3, 10.0)


def test_validate_manifest_empty_narration_text() -> None:
    m = RecapManifest(
        project_id="test",
        source_video="video.mp4",
        recap_mode="FULL_EPISODE",
        recap_language="en-US",
        segments=[RecapSegment("s1", 0, 5000, "   ", audio_policy="mute")],
    )
    with pytest.raises(NarrationError, match="thiếu nội dung thuyết minh"):
        validate_manifest(m, 10.0)


def test_manifest_roundtrip_serialization() -> None:
    manifest = RecapManifest(
        project_id="demo_ep01",
        source_video="C:/media/demo_ep01.mp4",
        recap_mode="FULL_EPISODE",
        recap_language="en-US",
        total_source_duration_sec=60.0,
        speech_detected=True,
        segments=[
            RecapSegment("scene_01", 1000, 6000, "Opening hook narration"),
            RecapSegment("scene_02", 15000, 25000, "Climax scene narration"),
        ],
    )
    d = manifest.to_dict()
    assert d["speech_detected"] is True
    loaded = RecapManifest.from_dict(d)
    assert loaded.project_id == manifest.project_id
    assert loaded.recap_language == manifest.recap_language
    assert loaded.speech_detected is True
    assert len(loaded.segments) == 2
    assert loaded.segments[0].narration_text == "Opening hook narration"
    assert loaded.segments[1].end_ms == 25000


def test_prepare_narration_with_companion_json(tmp_path: Path, dummy_video: Path) -> None:
    companion = dummy_video.with_suffix(".json")
    companion_data = {
        "project_id": "companion_proj",
        "source_video": str(dummy_video),
        "recap_mode": "FULL_EPISODE",
        "recap_language": "en-US",
        "segments": [
            {
                "segment_id": "c_01",
                "start_ms": 0,
                "end_ms": 1500,
                "narration_text": "Companion scene narration text.",
            }
        ],
    }
    companion.write_text(json.dumps(companion_data), encoding="utf-8")

    out_dir = tmp_path / "output"
    manifest = prepare_narration_for_video(dummy_video, out_dir)
    assert manifest.project_id == "companion_proj"
    assert len(manifest.segments) == 1
    assert manifest.segments[0].segment_id == "c_01"


def test_srt_parsing() -> None:
    srt_text = """1
00:00:01,000 --> 00:00:03,500
Hello world, detective.

2
00:00:04,200 --> 00:00:06,800
We found the evidence right here.
"""
    parsed = _parse_srt_content(srt_text)
    assert len(parsed) == 2
    assert parsed[0] == (1.0, 3.5, "Hello world, detective.")
    assert parsed[1] == (4.2, 6.8, "We found the evidence right here.")


def test_prepare_narration_with_companion_srt(tmp_path: Path, dummy_video: Path) -> None:
    srt_file = dummy_video.with_suffix(".srt")
    srt_file.write_text(
        "1\n00:00:00,100 --> 00:00:01,200\nAgent Carter discovers the hidden file.\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "output_srt"
    # Offline mode test for extractive dialogue recap
    settings = AppSettings(gateway_enabled=False)
    manifest = prepare_narration_for_video(dummy_video, out_dir, settings=settings)

    assert manifest.speech_detected is True
    assert len(manifest.segments) >= 2
    # Verify narration incorporated actual dialogue words
    found_dialogue = any("Agent Carter discovers" in seg.narration_text for seg in manifest.segments)
    assert found_dialogue, "Kịch bản recap phải tích hợp nội dung thoại thực tế từ phụ đề!"


def test_prepare_narration_no_speech_video(tmp_path: Path, dummy_video: Path) -> None:
    """A silent dummy video has no dialogue; verify explicit no-speech handling in offline mode."""
    out_dir = tmp_path / "output_no_speech"
    logs: list[str] = []

    settings = AppSettings(gateway_enabled=False)
    manifest = prepare_narration_for_video(dummy_video, out_dir, log=logs.append, settings=settings)

    assert manifest.speech_detected is False
    assert manifest.recap_mode == "SCENE_ANALYSIS_NO_SPEECH"
    assert len(manifest.segments) >= 2
    # Ensure segments do not claim dialogue and pass validation
    for seg in manifest.segments:
        assert seg.original_dialogue_text == ""
        assert "No" in seg.narration_text or "Visual" in seg.narration_text or "sequence" in seg.narration_text
    validate_manifest(manifest, manifest.total_source_duration_sec)


def test_prepare_narration_invalid_api_key_raises_error(tmp_path: Path, dummy_video: Path) -> None:
    out_dir = tmp_path / "output_api_err"
    # Configure an invalid API key
    settings = AppSettings(
        transcription_provider="openai",
        api_key="invalid-api-key-1234",
    )
    with pytest.raises(NarrationError, match="OpenAI API key không hợp lệ"):
        prepare_narration_for_video(dummy_video, out_dir, settings=settings)


def test_ensure_local_whisper_model_cold_download_and_warm_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock test: cold download once, warm reuse without redownload."""
    from toolrecap_v2.narration import ensure_local_whisper_model
    import huggingface_hub

    download_calls = []

    def _mock_snapshot_download(repo_id: str, **kwargs):
        download_calls.append((repo_id, kwargs))
        target_dir = Path(kwargs["local_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        # Create minimal valid model files
        (target_dir / "model.bin").write_bytes(b"dummy_model_weights")
        (target_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(target_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _mock_snapshot_download)

    logs: list[str] = []
    # 1. Cold download path
    model_dir = ensure_local_whisper_model(model_name="tiny", cache_dir=tmp_path, log=logs.append)
    assert len(download_calls) == 1
    assert (model_dir / "model.bin").is_file()
    assert (model_dir / "config.json").is_file()
    assert any("Đang tải mô hình" in l for l in logs)

    # 2. Warm reuse path - must NOT call snapshot_download again
    logs.clear()
    model_dir_warm = ensure_local_whisper_model(model_name="tiny", cache_dir=tmp_path, log=logs.append)
    assert len(download_calls) == 1  # Still 1, no second download
    assert model_dir_warm == model_dir
    assert any("bộ nhớ đệm" in l for l in logs)


def test_ensure_local_whisper_model_cancellation_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure cancellation during STT model download cleans up partial cache safely."""
    import threading
    from toolrecap_v2.narration import ensure_local_whisper_model, NarrationError
    import huggingface_hub

    cancel_event = threading.Event()

    def _mock_download_with_cancel(repo_id: str, **kwargs):
        target_dir = Path(kwargs["local_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        # Leave a partial leftover file
        (target_dir / "partial_file.part").write_bytes(b"partial")
        cancel_event.set()
        # If tqdm_class provided, trigger update which checks cancel
        tqdm_cls = kwargs.get("tqdm_class")
        if tqdm_cls:
            bar = tqdm_cls()
            bar.update(1)
        raise RuntimeError("Download interrupted")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _mock_download_with_cancel)

    with pytest.raises(NarrationError, match="bị dừng"):
        ensure_local_whisper_model(model_name="tiny", cache_dir=tmp_path, cancel_event=cancel_event)

    # Partial file must be cleaned up
    model_dir = tmp_path / "tiny"
    assert not (model_dir / "partial_file.part").exists()
