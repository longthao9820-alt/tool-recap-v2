"""Comprehensive test suite for robust multi-source concatenation r9.

Covers:
- Actual fast FFmpeg concat tests (stereo+5.1, mixed audio/no-audio, all no-audio, differing res/fps) under 5 sec total.
- Fast concat compatibility selection and routing to normalized concat.
- Dual diagnostics: ConcatPipelineError preserving both fast and normalized errors when both fail.
- Cancel handling: pre-set cancel ensures no output, cancel during execution cleans up.
- Intermediate validation helper: mock details and malformed files.
- Cut wrapping: PublicationRenderer wraps cut_clip failures in CutClipError with exact metadata.
- Audio and video matrix tests via typed mock probes.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from toolrecap_v2.domain.enums import AudioPolicy
from toolrecap_v2.domain.models import CommentaryOutput, Segment, SourceClip
from toolrecap_v2.media import (
    ConcatPipelineError,
    CutClipError,
    FastConcatError,
    IntermediateValidationError,
    MediaError,
    MediaProbeResult,
    NormalizedConcatError,
    RenderCancelled,
    build_normalization_profile,
    concat_media_clips,
    cut_clip,
    evaluate_fast_concat,
    find_binary,
    probe_typed_media,
    validate_intermediate_output,
)
from toolrecap_v2.renderer import PublicationRenderer
from toolrecap_v2.settings import AppSettings
from tests.helpers_media import create_mock_probe, create_synthetic_test_clip


# ============================================================================
# 1. Actual FFmpeg Concat Tests (4 tests, sub-second each, under 5 sec total)
# ============================================================================

def test_actual_ffmpeg_stereo_and_51_concat(tmp_path: Path):
    """Test concatenating stereo (2ch) and 5.1 (6ch) clips normalizes to 48kHz stereo."""
    c1 = create_synthetic_test_clip(
        tmp_path / "stereo.mp4",
        duration=0.2,
        channels=2,
        channel_layout="stereo",
        sample_rate=48000,
    )
    c2 = create_synthetic_test_clip(
        tmp_path / "surround.mp4",
        duration=0.2,
        channels=6,
        channel_layout="5.1",
        sample_rate=48000,
    )

    out = tmp_path / "out_stereo_51.mp4"
    result = concat_media_clips([c1, c2], out, require_audio=True)

    assert out.is_file()
    assert result.has_video
    assert result.has_audio
    assert result.duration == pytest.approx(0.4, abs=0.15)
    # Output audio must be normalized to stereo (2 channels)
    assert len(result.audio_streams) >= 1
    assert result.audio_streams[0].channels == 2
    assert result.audio_streams[0].sample_rate == 48000


def test_actual_ffmpeg_mixed_audio_and_no_audio(tmp_path: Path):
    """Test concatenating clip with audio and clip without audio synthesizes silent audio for clip 2."""
    c1 = create_synthetic_test_clip(tmp_path / "with_audio.mp4", duration=0.2, has_audio=True)
    c2 = create_synthetic_test_clip(tmp_path / "no_audio.mp4", duration=0.2, has_audio=False)

    out = tmp_path / "out_mixed_audio.mp4"
    result = concat_media_clips([c1, c2], out, require_audio=True)

    assert out.is_file()
    assert result.has_video
    assert result.has_audio
    assert result.duration == pytest.approx(0.4, abs=0.15)
    assert len(result.audio_streams) >= 1
    assert result.audio_streams[0].channels == 2


def test_actual_ffmpeg_all_clips_no_audio(tmp_path: Path):
    """Test concatenating clips where all clips lack audio generates silent stereo when require_audio=True."""
    c1 = create_synthetic_test_clip(tmp_path / "na1.mp4", duration=0.2, has_audio=False)
    c2 = create_synthetic_test_clip(tmp_path / "na2.mp4", duration=0.2, has_audio=False)

    out = tmp_path / "out_all_no_audio.mp4"
    result = concat_media_clips([c1, c2], out, require_audio=True)

    assert out.is_file()
    assert result.has_video
    assert result.has_audio
    assert result.duration == pytest.approx(0.4, abs=0.15)
    assert result.audio_streams[0].channels == 2


def test_actual_ffmpeg_differing_resolution_and_fps(tmp_path: Path):
    """Test concatenating clips with different resolutions and fps scales and normalizes properly."""
    c1 = create_synthetic_test_clip(tmp_path / "c160.mp4", duration=0.2, width=160, height=120, fps=25)
    c2 = create_synthetic_test_clip(tmp_path / "c320.mp4", duration=0.2, width=320, height=240, fps=30)

    out = tmp_path / "out_res_fps.mp4"
    result = concat_media_clips([c1, c2], out, require_audio=True)

    assert out.is_file()
    assert result.has_video
    assert result.has_audio
    # Canvas should match first clip (160x120)
    assert result.width == 160
    assert result.height == 120
    assert result.duration == pytest.approx(0.4, abs=0.15)


def test_actual_ffmpeg_51_side_concat(tmp_path: Path):
    """Test concatenating real 5.1(side) surround clip with stereo normalizes to 48kHz stereo."""
    ffmpeg = find_binary("ffmpeg")
    c1 = tmp_path / "side51.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=0.2:size=160x120:rate=25",
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=0.2",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "ac3",
        "-channel_layout", "5.1(side)",
        "-ac", "6",
        str(c1),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0:
        pytest.skip(f"FFmpeg fixture unsupported for 5.1(side): {proc.stderr}")

    probe1 = probe_typed_media(c1)
    if not probe1.has_audio or not probe1.audio_streams or "5.1(side)" not in probe1.audio_streams[0].channel_layout:
        pytest.skip(f"FFmpeg fixture did not produce 5.1(side) layout: {probe1.audio_streams}")

    c2 = create_synthetic_test_clip(
        tmp_path / "stereo.mp4",
        duration=0.2,
        channels=2,
        channel_layout="stereo",
        sample_rate=48000,
    )
    out = tmp_path / "out_51side_concat.mp4"
    result = concat_media_clips([c1, c2], out, require_audio=True)

    assert out.is_file()
    assert result.has_video
    assert result.has_audio
    assert result.duration == pytest.approx(0.4, abs=0.15)
    assert len(result.audio_streams) >= 1
    assert result.audio_streams[0].channels == 2
    assert result.audio_streams[0].sample_rate == 48000


# ============================================================================
# 2. Fast Concat Selection and Routing
# ============================================================================

def test_fast_concat_compatible_path_executes_demux_copy(tmp_path: Path):
    """Compatible identical clips use the fast demux copy path."""
    c1 = create_synthetic_test_clip(tmp_path / "fast1.mp4", duration=0.2)
    c2 = create_synthetic_test_clip(tmp_path / "fast2.mp4", duration=0.2)

    logs: list[str] = []
    out = tmp_path / "out_fast.mp4"
    result = concat_media_clips([c1, c2], out, log=logs.append)

    assert out.is_file()
    assert result.has_video
    assert any("Fast concat tương thích" in log for log in logs)


def test_fast_concat_incompatible_routes_directly_to_normalized(tmp_path: Path):
    """Clips with different parameters bypass fast concat directly into normalized concat."""
    c1 = create_synthetic_test_clip(tmp_path / "inc1.mp4", duration=0.2, width=160, height=120)
    c2 = create_synthetic_test_clip(tmp_path / "inc2.mp4", duration=0.2, width=320, height=240)

    logs: list[str] = []
    out = tmp_path / "out_inc.mp4"
    result = concat_media_clips([c1, c2], out, log=logs.append)

    assert out.is_file()
    assert result.has_video
    assert any("Fast concat không được áp dụng" in log for log in logs)


# ============================================================================
# 3. Dual Diagnostics Error Preservation (ConcatPipelineError)
# ============================================================================

def test_dual_diagnostics_preserves_fast_and_normalized_errors(tmp_path: Path):
    """When fast concat fails and normalized concat also fails, preserve both error diagnostics."""
    c1 = create_synthetic_test_clip(tmp_path / "d1.mp4", duration=0.2)
    c2 = create_synthetic_test_clip(tmp_path / "d2.mp4", duration=0.2)
    out = tmp_path / "out_dual_err.mp4"

    def mock_run_command(cmd, cancel_event=None, log=None):
        cmd_str = " ".join(str(c) for c in cmd)
        if "-f concat" in cmd_str or "concat_list" in cmd_str:
            raise MediaError("simulated fast demux copy crash: bitstream error")
        if "-filter_complex" in cmd_str:
            raise MediaError("simulated normalized filter crash: out of memory")
        return ""

    with patch("toolrecap_v2.media.run_command", side_effect=mock_run_command):
        with pytest.raises(ConcatPipelineError) as exc_info:
            concat_media_clips([c1, c2], out)

    err = exc_info.value
    assert err.category == "CONCAT_PIPELINE_ERROR"
    assert err.fast_error is not None
    assert "bitstream error" in str(err.fast_error)
    assert err.normalized_error is not None
    assert "out of memory" in str(err.normalized_error)
    assert not out.exists()


# ============================================================================
# 4. Cancellation Handling
# ============================================================================

def test_cancel_preset_ensures_no_output(tmp_path: Path):
    """Pre-set cancellation event immediately raises RenderCancelled without writing output."""
    c1 = create_synthetic_test_clip(tmp_path / "c1.mp4", duration=0.2)
    c2 = create_synthetic_test_clip(tmp_path / "c2.mp4", duration=0.2)
    out = tmp_path / "out_cancelled.mp4"

    cancel_evt = threading.Event()
    cancel_evt.set()

    with pytest.raises(RenderCancelled):
        concat_media_clips([c1, c2], out, cancel=cancel_evt)

    assert not out.exists()
    assert not out.with_suffix(".concat_list.txt").exists()


def test_cancel_during_execution_cleans_partial_output(tmp_path: Path):
    """Cancellation triggered during run_command cleans partial output and raises RenderCancelled."""
    c1 = create_synthetic_test_clip(tmp_path / "c3.mp4", duration=0.2)
    c2 = create_synthetic_test_clip(tmp_path / "c4.mp4", duration=0.2)
    out = tmp_path / "out_cancelled_during.mp4"

    cancel_evt = threading.Event()

    def mock_run_command_cancel(cmd, cancel_event=None, log=None):
        out.write_bytes(b"partial corrupted data")
        cancel_evt.set()
        raise RenderCancelled("Xử lý đã bị dừng theo yêu cầu của người dùng.")

    with patch("toolrecap_v2.media.run_command", side_effect=mock_run_command_cancel):
        with pytest.raises(RenderCancelled):
            concat_media_clips([c1, c2], out, cancel=cancel_evt)

    assert not out.exists()


# ============================================================================
# 5. Intermediate Validation Helper Tests
# ============================================================================

def test_validate_intermediate_valid_details(tmp_path: Path):
    """Valid intermediate probe details pass validation cleanly."""
    valid_file = tmp_path / "valid.mp4"
    valid_file.write_bytes(b"mock valid video data")

    details = {
        "format": {"duration": "10.0", "start_time": "0.0"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "duration": "10.0",
                "nb_read_packets": "240",
                "start_time": "0.0",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "channel_layout": "stereo",
                "sample_rate": 48000,
                "duration": "10.0",
                "nb_read_packets": "400",
                "start_time": "0.0",
            },
        ],
    }

    res = validate_intermediate_output(valid_file, expected_duration=10.0, require_audio=True, details=details)
    assert isinstance(res, MediaProbeResult)
    assert res.duration == 10.0
    assert res.has_video
    assert res.has_audio


def test_validate_intermediate_missing_video_stream(tmp_path: Path):
    """Intermediate validation fails when video stream is missing."""
    f = tmp_path / "no_v.mp4"
    f.write_bytes(b"data")
    details = {
        "format": {"duration": "5.0"},
        "streams": [{"codec_type": "audio", "channels": 2, "channel_layout": "stereo", "sample_rate": 48000}],
    }
    with pytest.raises(IntermediateValidationError, match="không chứa luồng video"):
        validate_intermediate_output(f, details=details)


def test_validate_intermediate_invalid_dimensions(tmp_path: Path):
    """Intermediate validation fails when video dimensions are zero."""
    f = tmp_path / "zero_dim.mp4"
    f.write_bytes(b"data")
    details = {
        "format": {"duration": "5.0"},
        "streams": [{"codec_type": "video", "width": 0, "height": 0}],
    }
    with pytest.raises(IntermediateValidationError, match="Kích thước video intermediate không hợp lệ"):
        validate_intermediate_output(f, details=details)


def test_validate_intermediate_missing_audio_when_required(tmp_path: Path):
    """Intermediate validation fails when require_audio=True but audio stream is missing."""
    f = tmp_path / "no_a.mp4"
    f.write_bytes(b"data")
    details = {
        "format": {"duration": "5.0"},
        "streams": [{"codec_type": "video", "width": 1920, "height": 1080, "nb_read_packets": "100"}],
    }
    with pytest.raises(IntermediateValidationError, match="không có luồng audio"):
        validate_intermediate_output(f, require_audio=True, details=details)


def test_validate_intermediate_unknown_channel_layout(tmp_path: Path):
    """Intermediate validation fails when audio channel_layout is unknown or empty."""
    f = tmp_path / "unknown_layout.mp4"
    f.write_bytes(b"data")
    details = {
        "format": {"duration": "5.0"},
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "nb_read_packets": "100"},
            {"codec_type": "audio", "channels": 2, "channel_layout": "unknown", "sample_rate": 48000},
        ],
    }
    with pytest.raises(IntermediateValidationError, match="channel layout không xác định"):
        validate_intermediate_output(f, require_audio=True, details=details)


def test_validate_intermediate_zero_channels_or_sample_rate(tmp_path: Path):
    """Intermediate validation fails when audio channels or sample rate is zero."""
    f = tmp_path / "zero_ch.mp4"
    f.write_bytes(b"data")
    details_ch = {
        "format": {"duration": "5.0"},
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "nb_read_packets": "100"},
            {"codec_type": "audio", "channels": 0, "channel_layout": "stereo", "sample_rate": 48000},
        ],
    }
    with pytest.raises(IntermediateValidationError, match="số kênh không hợp lệ"):
        validate_intermediate_output(f, require_audio=True, details=details_ch)

    details_sr = {
        "format": {"duration": "5.0"},
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "nb_read_packets": "100"},
            {"codec_type": "audio", "channels": 2, "channel_layout": "stereo", "sample_rate": 0},
        ],
    }
    with pytest.raises(IntermediateValidationError, match="sample rate không hợp lệ"):
        validate_intermediate_output(f, require_audio=True, details=details_sr)


def test_validate_intermediate_negative_start_time(tmp_path: Path):
    """Intermediate validation fails when start time is negative (< -0.05)."""
    f = tmp_path / "neg_st.mp4"
    f.write_bytes(b"data")
    details = {
        "format": {"duration": "5.0", "start_time": "-0.15"},
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "nb_read_packets": "100"},
            {"codec_type": "audio", "channels": 2, "channel_layout": "stereo", "sample_rate": 48000},
        ],
    }
    with pytest.raises(IntermediateValidationError, match="Start time của format bị âm"):
        validate_intermediate_output(f, details=details)


def test_validate_intermediate_duration_mismatch(tmp_path: Path):
    """Intermediate validation fails when duration differs from expected beyond tolerance."""
    f = tmp_path / "dur_mismatch.mp4"
    f.write_bytes(b"data")
    details = {
        "format": {"duration": "10.0"},
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "duration": "10.0", "nb_read_packets": "100"},
            {"codec_type": "audio", "channels": 2, "channel_layout": "stereo", "sample_rate": 48000, "duration": "10.0", "nb_read_packets": "100"},
        ],
    }
    # Expected 15.0s, actual 10.0s, tolerance max(0.75, 0.3) = 0.75s -> fails
    with pytest.raises(IntermediateValidationError, match="Thời lượng intermediate lệch quá mức"):
        validate_intermediate_output(f, expected_duration=15.0, details=details)


def test_validate_intermediate_zero_packets(tmp_path: Path):
    """Intermediate validation fails when packet count is explicitly 0."""
    f = tmp_path / "zero_packets.mp4"
    f.write_bytes(b"data")
    details = {
        "format": {"duration": "5.0"},
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "nb_read_packets": "0"},
            {"codec_type": "audio", "channels": 2, "channel_layout": "stereo", "sample_rate": 48000, "nb_read_packets": "100"},
        ],
    }
    with pytest.raises(IntermediateValidationError, match="Luồng video không có packet nào"):
        validate_intermediate_output(f, details=details)


def test_validate_intermediate_empty_or_missing_file(tmp_path: Path):
    """Intermediate validation fails when file is 0 bytes or does not exist."""
    empty_f = tmp_path / "empty.mp4"
    empty_f.write_bytes(b"")
    with pytest.raises(IntermediateValidationError, match="rỗng"):
        validate_intermediate_output(empty_f)

    missing_f = tmp_path / "non_existent.mp4"
    with pytest.raises(IntermediateValidationError, match="không tồn tại"):
        validate_intermediate_output(missing_f)


# ============================================================================
# 6. Renderer Cut Wrapping (CutClipError)
# ============================================================================

def test_renderer_cut_clip_failure_wrapped_in_cut_clip_error(tmp_path: Path):
    """PublicationRenderer._render_single_output wraps cut_clip failures in CutClipError with exact metadata."""
    clip_file = create_synthetic_test_clip(tmp_path / "src_clip.mp4", duration=0.1)

    out = CommentaryOutput(
        output_id="out_01",
        title="Test Output",
        segments=[
            Segment(
                segment_id="seg_01",
                source_clips=[
                    SourceClip(
                        episode_id="E01",
                        source_video=str(clip_file),
                        start=0.0,
                        end=0.1,
                    )
                ],
                audio_policy=AudioPolicy.ORIGINAL_ONLY.value,
            )
        ],
    )
    renderer = PublicationRenderer(voice_manager=object())
    settings = AppSettings(burn_subtitles=False, use_gpu=False)

    with patch("toolrecap_v2.renderer.cut_clip", side_effect=RuntimeError("simulated disk full during cut")):
        with pytest.raises(CutClipError) as exc_info:
            renderer._render_single_output(
                out=out,
                safe_title="test-output",
                pub_dir=tmp_path / "pub",
                temp_dir=tmp_path / "temp",
                settings=settings,
                voice_id="mock_voice",
                voice_style="default",
                voice_mgr=object(),
                transcript_cues_by_episode={},
                callbacks=None,
                out_idx=1,
                total_outputs=1,
                cancel_evt=None,
                allow_mock_synth=False,
            )

    exc = exc_info.value
    assert exc.category == "CUT_CLIP_ERROR"
    assert exc.clip_index == 1
    assert exc.source_path == str(clip_file.resolve())
    assert isinstance(exc.__cause__, RuntimeError)
    assert "simulated disk full during cut" in str(exc)


def test_real_cut_no_audio_index_produces_video_without_audio(tmp_path: Path):
    """Calling cut_clip with audio_stream_index=None on source with audio outputs video without audio."""
    src = create_synthetic_test_clip(tmp_path / "src_audio.mp4", duration=0.2, has_audio=True)
    dst = tmp_path / "cut_no_audio.mp4"
    cut_clip(src, dst, start_sec=0.0, end_sec=0.2, audio_stream_index=None, use_gpu=False)

    assert dst.is_file()
    probe = probe_typed_media(dst)
    assert probe.has_video is True
    assert probe.has_audio is False
    assert len(probe.audio_streams) == 0


def test_video_only_source_invalid_audio_mapping_succeeds(tmp_path: Path):
    """Calling cut_clip with audio_stream_index on video-only source avoids invalid FFmpeg audio map."""
    src = create_synthetic_test_clip(tmp_path / "src_silent.mp4", duration=0.2, has_audio=False)
    dst = tmp_path / "cut_silent.mp4"
    cut_clip(src, dst, start_sec=0.0, end_sec=0.2, audio_stream_index=0, use_gpu=False)

    assert dst.is_file()
    probe = probe_typed_media(dst)
    assert probe.has_video is True
    assert probe.has_audio is False
    assert len(probe.audio_streams) == 0


# ============================================================================
# 7. Audio & Video Matrix Pure Probe Tests (Section 12 Matrix)
# ============================================================================

def test_matrix_audio_51_sources():
    """Clips with identical 5.1 layout are compatible for fast concat."""
    p1 = create_mock_probe(channels=6, channel_layout="5.1")
    p2 = create_mock_probe(channels=6, channel_layout="5.1")
    decision = evaluate_fast_concat([p1, p2])
    assert decision.compatible


def test_matrix_audio_stereo_plus_51():
    """Clips with stereo and 5.1 layout are incompatible and route to normalized concat."""
    p1 = create_mock_probe(channels=2, channel_layout="stereo")
    p2 = create_mock_probe(channels=6, channel_layout="5.1")
    decision = evaluate_fast_concat([p1, p2])
    assert not decision.compatible
    assert any("channel_layout_mismatch" in d for d in decision.differences)


def test_matrix_audio_mono_plus_stereo():
    """Clips with mono and stereo layout are incompatible."""
    p1 = create_mock_probe(channels=1, channel_layout="mono")
    p2 = create_mock_probe(channels=2, channel_layout="stereo")
    decision = evaluate_fast_concat([p1, p2])
    assert not decision.compatible
    assert any("channels_mismatch" in d for d in decision.differences)


def test_matrix_audio_differing_sample_rates():
    """Clips with 44.1kHz and 48kHz are incompatible."""
    p1 = create_mock_probe(sample_rate=44100)
    p2 = create_mock_probe(sample_rate=48000)
    decision = evaluate_fast_concat([p1, p2])
    assert not decision.compatible
    assert any("sample_rate_mismatch" in d for d in decision.differences)


def test_matrix_audio_differing_codecs():
    """Clips with AAC and E-AC-3 or AC-3 are incompatible."""
    p1 = create_mock_probe(audio_codec="aac")
    p2 = create_mock_probe(audio_codec="eac3")
    decision = evaluate_fast_concat([p1, p2])
    assert not decision.compatible
    assert any("audio_codec_mismatch" in d for d in decision.differences)


def test_matrix_audio_empty_channel_layout():
    """Clips with empty channel layout are rejected from fast concat."""
    p1 = create_mock_probe(channel_layout="")
    p2 = create_mock_probe(channel_layout="stereo")
    decision = evaluate_fast_concat([p1, p2])
    assert not decision.compatible
    assert any("audio_channel_layout_unknown" in d for d in decision.differences)


def test_matrix_video_h264_plus_hevc():
    """Clips with different video codecs (H.264 + HEVC) are incompatible."""
    p1 = create_mock_probe(video_codec="h264")
    p2 = create_mock_probe(video_codec="hevc")
    decision = evaluate_fast_concat([p1, p2])
    assert not decision.compatible
    assert any("video_codec_mismatch" in d for d in decision.differences)


def test_matrix_video_differing_resolutions():
    """Clips with 720p, 1080p, and 4K are incompatible."""
    p1 = create_mock_probe(width=1280, height=720)
    p2 = create_mock_probe(width=1920, height=1080)
    p3 = create_mock_probe(width=3840, height=2160)
    decision = evaluate_fast_concat([p1, p2, p3])
    assert not decision.compatible
    assert any("resolution_mismatch" in d for d in decision.differences)


def test_matrix_video_differing_fps():
    """Clips with different frame rates (23.976 vs 24 vs 25 vs 30) are incompatible."""
    p1 = create_mock_probe(fps=24.0, fps_rational="24/1")
    p2 = create_mock_probe(fps=25.0, fps_rational="25/1")
    decision = evaluate_fast_concat([p1, p2])
    assert not decision.compatible
    assert any("fps_mismatch" in d for d in decision.differences)


def test_matrix_video_differing_sar():
    """Clips with different SAR are incompatible."""
    p1 = create_mock_probe(sar="1:1")
    p2 = create_mock_probe(sar="16:11")
    decision = evaluate_fast_concat([p1, p2])
    assert not decision.compatible
    assert any("sar_mismatch" in d for d in decision.differences)


def test_matrix_video_differing_pix_fmt():
    """Clips with different pixel formats are incompatible."""
    p1 = create_mock_probe(pix_fmt="yuv420p")
    p2 = create_mock_probe(pix_fmt="yuv444p")
    decision = evaluate_fast_concat([p1, p2])
    assert not decision.compatible
    assert any("pix_fmt_mismatch" in d for d in decision.differences)
