"""Comprehensive unit tests for stream policy r9: typed stream metadata, signatures, fast concat evaluation, and normalization profiles."""
from __future__ import annotations

import pytest

from toolrecap_v2.media import (
    AudioSelectionResult,
    AudioStreamInfo,
    FastConcatDecision,
    MediaProbeResult,
    NormalizationProfile,
    StreamSignature,
    SubtitleStreamInfo,
    VideoStreamInfo,
    build_normalization_profile,
    evaluate_fast_concat,
    normalize_sar,
    normalize_time_base,
    parse_rational,
    parse_stream_metadata,
    select_english_audio_stream,
    signature_from_probe,
)


# ============================================================================
# 1. Model Typed Fields, Backward Defaults & Dict Roundtrip
# ============================================================================

def test_video_stream_info_typed_fields_and_defaults():
    v = VideoStreamInfo(
        index=0,
        video_index=0,
        codec="h264",
        width=1920,
        height=1080,
    )
    assert v.fps == 0.0
    assert v.duration == 0.0
    assert v.bitrate == 0
    assert v.fps_rational == ""
    assert v.pix_fmt == ""
    assert v.profile == ""
    assert v.time_base == ""
    assert v.sar == ""
    assert v.SAR == ""
    assert v.start_time == 0.0

    # Set all new fields and verify roundtrip
    v2 = VideoStreamInfo(
        index=0,
        video_index=0,
        codec="h264",
        width=1920,
        height=1080,
        fps=23.976023976,
        fps_rational="24000/1001",
        pix_fmt="yuv420p",
        profile="High",
        time_base="1/1000",
        sar="1:1",
        start_time=0.083,
    )
    assert v2.SAR == "1:1"
    d = v2.to_dict()
    assert d["fps_rational"] == "24000/1001"
    assert d["pix_fmt"] == "yuv420p"
    assert d["profile"] == "High"
    assert d["time_base"] == "1/1000"
    assert d["sar"] == "1:1"
    assert d["start_time"] == 0.083

    restored = VideoStreamInfo.from_dict(d)
    assert restored == v2


def test_audio_stream_info_backward_defaults_and_roundtrip():
    a = AudioStreamInfo(
        index=1,
        audio_index=0,
        codec="aac",
        language="eng",
    )
    # Strict backward default requirement: channels=0, channel_layout=""
    assert a.channels == 0
    assert a.channel_layout == ""
    assert a.sample_rate == 0
    assert a.sample_fmt == ""
    assert a.profile == ""
    assert a.time_base == ""
    assert a.start_time == 0.0

    a2 = AudioStreamInfo(
        index=1,
        audio_index=0,
        codec="aac",
        language="eng",
        channels=2,
        channel_layout="stereo",
        sample_rate=48000,
        sample_fmt="fltp",
        profile="LC",
        time_base="1/48000",
        start_time=0.05,
    )
    d = a2.to_dict()
    assert d["channels"] == 2
    assert d["channel_layout"] == "stereo"
    assert d["sample_rate"] == 48000
    assert d["sample_fmt"] == "fltp"
    assert d["profile"] == "LC"
    assert d["time_base"] == "1/48000"
    assert d["start_time"] == 0.05

    restored = AudioStreamInfo.from_dict(d)
    assert restored == a2


def test_media_probe_result_start_time_and_roundtrip():
    probe = MediaProbeResult(
        path="test.mp4",
        duration=120.0,
        width=1920,
        height=1080,
        has_video=True,
        has_audio=True,
        video_codec="h264",
        audio_codec="aac",
        start_time=0.025,
    )
    d = probe.to_dict()
    assert d["start_time"] == 0.025
    restored = MediaProbeResult.from_dict(d)
    assert restored.start_time == 0.025
    assert restored.duration == 120.0


# ============================================================================
# 2. Pure Parser Helpers (Rational, SAR, Timebase, Stream Metadata)
# ============================================================================

def test_parse_rational_helper():
    assert parse_rational("24/1") == (24.0, "24/1")
    assert parse_rational("24000/1001") == (pytest.approx(23.976023976, 1e-6), "24000/1001")
    assert parse_rational("30000/1001") == (pytest.approx(29.97002997, 1e-6), "30000/1001")
    assert parse_rational("48/2") == (24.0, "24/1")
    assert parse_rational("25") == (25.0, "25/1")
    assert parse_rational("0/0") == (0.0, "")
    assert parse_rational("0") == (0.0, "")
    assert parse_rational("") == (0.0, "")
    assert parse_rational(None) == (0.0, "")
    assert parse_rational("N/A") == (0.0, "")
    assert parse_rational("-24/1") == (0.0, "")


def test_normalize_sar_helper():
    assert normalize_sar("1:1") == "1:1"
    assert normalize_sar("1/1") == "1:1"
    assert normalize_sar("16:11") == "16:11"
    assert normalize_sar("0:1") == ""
    assert normalize_sar("0/1") == ""
    assert normalize_sar("0:0") == ""
    assert normalize_sar("unknown") == ""
    assert normalize_sar("") == ""
    assert normalize_sar(None) == ""
    assert normalize_sar("1.0") == "1:1"


def test_normalize_time_base_helper():
    assert normalize_time_base("1/1000") == "1/1000"
    assert normalize_time_base("1/48000") == "1/48000"
    assert normalize_time_base("2/2000") == "1/1000"
    assert normalize_time_base("0/0") == ""
    assert normalize_time_base("0") == ""
    assert normalize_time_base("") == ""
    assert normalize_time_base(None) == ""


def test_parse_stream_metadata_captures_all_fields():
    raw_streams = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "24000/1001",
            "r_frame_rate": "120000/1001",
            "pix_fmt": "yuv420p",
            "profile": "High",
            "time_base": "1/1000",
            "sample_aspect_ratio": "1:1",
            "start_time": "0.083333",
            "duration": "60.0",
            "bit_rate": "5000000",
            "disposition": {"default": 1},
            "tags": {"title": "Primary Video"},
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "aac",
            "channels": 2,
            "channel_layout": "stereo",
            "sample_rate": "48000",
            "sample_fmt": "fltp",
            "profile": "LC",
            "time_base": "1/48000",
            "start_time": "0.020000",
            "bit_rate": "192000",
            "disposition": {"default": 1},
            "tags": {"language": "eng", "title": "Main Audio"},
        },
    ]

    v_streams, a_streams, s_streams, sel_audio = parse_stream_metadata(raw_streams)

    assert len(v_streams) == 1
    v = v_streams[0]
    assert v.codec == "h264"
    assert v.fps == pytest.approx(23.976023976, 1e-6)
    assert v.fps_rational == "24000/1001"
    assert v.pix_fmt == "yuv420p"
    assert v.profile == "High"
    assert v.time_base == "1/1000"
    assert v.sar == "1:1"
    assert v.SAR == "1:1"
    assert v.start_time == pytest.approx(0.083333, 1e-5)

    assert len(a_streams) == 1
    a = a_streams[0]
    assert a.codec == "aac"
    assert a.channels == 2
    assert a.channel_layout == "stereo"
    assert a.sample_rate == 48000
    assert a.sample_fmt == "fltp"
    assert a.profile == "LC"
    assert a.time_base == "1/48000"
    assert a.start_time == pytest.approx(0.020, 1e-4)


def test_parse_stream_metadata_prefers_nonzero_avg_frame_rate():
    # avg_frame_rate is 24/1, r_frame_rate is 60/1 -> should prefer avg_frame_rate
    raw = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1280,
            "height": 720,
            "avg_frame_rate": "24/1",
            "r_frame_rate": "60/1",
        }
    ]
    v_streams, _, _, _ = parse_stream_metadata(raw)
    assert v_streams[0].fps == 24.0
    assert v_streams[0].fps_rational == "24/1"

    # avg_frame_rate is 0/0, r_frame_rate is 30/1 -> fallback to r_frame_rate
    raw_fallback = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1280,
            "height": 720,
            "avg_frame_rate": "0/0",
            "r_frame_rate": "30/1",
        }
    ]
    v_streams2, _, _, _ = parse_stream_metadata(raw_fallback)
    assert v_streams2[0].fps == 30.0
    assert v_streams2[0].fps_rational == "30/1"


def test_parse_stream_metadata_audio_no_default_stereo():
    """Parser MUST NOT default to stereo or 2 channels when audio channel_layout or channels missing."""
    raw = [
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "aac",
            # No channels, no channel_layout
        }
    ]
    _, a_streams, _, sel_audio = parse_stream_metadata(raw)
    assert len(a_streams) == 1
    assert a_streams[0].channels == 0
    assert a_streams[0].channel_layout == ""
    assert a_streams[0].sample_rate == 0
    # Selection scoring with channels=0 succeeds
    assert sel_audio.selected_stream is not None


# ============================================================================
# 3. StreamSignature and signature_from_probe
# ============================================================================

def test_signature_from_probe_uses_selected_audio_over_primary():
    """signature_from_probe uses primary video + selected audio (not primary_a) because cut maps selected track."""
    v_stream = VideoStreamInfo(
        index=0,
        video_index=0,
        codec="h264",
        width=1920,
        height=1080,
        fps=24.0,
        fps_rational="24/1",
        pix_fmt="yuv420p",
        profile="High",
        time_base="1/1000",
        sar="1:1",
    )
    commentary_track = AudioStreamInfo(
        index=1,
        audio_index=0,
        codec="ac3",
        language="eng",
        channels=6,
        channel_layout="5.1",
        sample_rate=48000,
        sample_fmt="fltp",
        time_base="1/48000",
        is_commentary=True,
    )
    clean_stereo_track = AudioStreamInfo(
        index=2,
        audio_index=1,
        codec="aac",
        language="eng",
        channels=2,
        channel_layout="stereo",
        sample_rate=48000,
        sample_fmt="fltp",
        time_base="1/48000",
        default=True,
        is_commentary=False,
    )

    probe = MediaProbeResult(
        path="sample.mkv",
        duration=100.0,
        width=1920,
        height=1080,
        has_video=True,
        has_audio=True,
        video_codec="h264",
        audio_codec="ac3",
        video_streams=[v_stream],
        audio_streams=[commentary_track, clean_stereo_track],
        selected_audio=AudioSelectionResult(selected_stream=clean_stereo_track),
    )

    sig = signature_from_probe(probe)

    # Video matches primary video
    assert sig.video_codec == "h264"
    assert sig.video_profile == "High"
    assert sig.width == 1920
    assert sig.height == 1080
    assert sig.fps == 24.0
    assert sig.fps_rational == "24/1"
    assert sig.pix_fmt == "yuv420p"
    assert sig.video_time_base == "1/1000"
    assert sig.sar == "1:1"
    assert sig.SAR == "1:1"

    # Audio MUST match clean_stereo_track (selected audio), NOT commentary_track (primary_a)
    assert sig.has_audio is True
    assert sig.audio_codec == "aac"
    assert sig.channels == 2
    assert sig.channel_layout == "stereo"
    assert sig.sample_rate == 48000
    assert sig.sample_fmt == "fltp"
    assert sig.audio_time_base == "1/48000"


def test_signature_from_probe_no_audio():
    probe = MediaProbeResult(
        path="silent.mp4",
        duration=10.0,
        width=1280,
        height=720,
        has_video=True,
        has_audio=False,
        video_codec="h264",
        audio_codec=None,
        video_streams=[
            VideoStreamInfo(
                index=0,
                video_index=0,
                codec="h264",
                width=1280,
                height=720,
                fps=30.0,
                fps_rational="30/1",
                pix_fmt="yuv420p",
                profile="Main",
                time_base="1/30",
                sar="1:1",
            )
        ],
        audio_streams=[],
    )
    sig = signature_from_probe(probe)
    assert sig.has_audio is False
    assert sig.audio_codec == ""
    assert sig.sample_rate == 0
    assert sig.channels == 0
    assert sig.channel_layout == ""


def test_signature_from_probe_dict_input():
    d = {
        "has_video": True,
        "has_audio": True,
        "video_streams": [
            {
                "codec": "hevc",
                "profile": "Main 10",
                "width": 3840,
                "height": 2160,
                "pix_fmt": "yuv420p10le",
                "fps_rational": "60/1",
                "fps": 60.0,
                "time_base": "1/60",
                "sar": "1:1",
            }
        ],
        "audio_streams": [
            {
                "codec": "opus",
                "profile": "",
                "sample_rate": 48000,
                "sample_fmt": "fltp",
                "channels": 2,
                "channel_layout": "stereo",
                "time_base": "1/48000",
            }
        ],
    }
    sig = signature_from_probe(d)
    assert sig.video_codec == "hevc"
    assert sig.video_profile == "Main 10"
    assert sig.width == 3840
    assert sig.height == 2160
    assert sig.has_audio is True
    assert sig.audio_codec == "opus"
    assert sig.sample_rate == 48000


# ============================================================================
# 4. FastConcatDecision and evaluate_fast_concat Strict Rules
# ============================================================================

def _make_probe(
    *,
    path: str = "clip.mp4",
    video_codec: str = "h264",
    profile: str = "High",
    width: int = 1920,
    height: int = 1080,
    pix_fmt: str = "yuv420p",
    fps: float = 24.0,
    fps_rational: str = "24/1",
    video_time_base: str = "1/1000",
    sar: str = "1:1",
    has_audio: bool = True,
    audio_codec: str = "aac",
    channels: int = 2,
    channel_layout: str = "stereo",
    sample_rate: int = 48000,
    sample_fmt: str = "fltp",
    audio_time_base: str = "1/48000",
) -> MediaProbeResult:
    v = VideoStreamInfo(
        index=0,
        video_index=0,
        codec=video_codec,
        width=width,
        height=height,
        fps=fps,
        fps_rational=fps_rational,
        pix_fmt=pix_fmt,
        profile=profile,
        time_base=video_time_base,
        sar=sar,
    )
    a_streams = []
    sel_audio = None
    if has_audio:
        a = AudioStreamInfo(
            index=1,
            audio_index=0,
            codec=audio_codec,
            language="eng",
            channels=channels,
            channel_layout=channel_layout,
            sample_rate=sample_rate,
            sample_fmt=sample_fmt,
            time_base=audio_time_base,
        )
        a_streams.append(a)
        sel_audio = AudioSelectionResult(selected_stream=a)

    return MediaProbeResult(
        path=path,
        duration=10.0,
        width=width,
        height=height,
        has_video=True,
        has_audio=has_audio,
        video_codec=video_codec,
        audio_codec=audio_codec if has_audio else None,
        video_streams=[v],
        audio_streams=a_streams,
        selected_audio=sel_audio,
    )


def test_evaluate_fast_concat_empty_probes():
    decision = evaluate_fast_concat([])
    assert decision.compatible is False
    assert "no_clips" in decision.differences


def test_evaluate_fast_concat_single_clip_fully_known():
    p = _make_probe()
    decision = evaluate_fast_concat([p])
    assert decision.compatible is True
    assert len(decision.differences) == 0


def test_evaluate_fast_concat_identical_clips_compatible():
    p1 = _make_probe(path="clip1.mp4")
    p2 = _make_probe(path="clip2.mkv")  # container format is irrelevant!
    decision = evaluate_fast_concat([p1, p2])
    assert decision.compatible is True
    assert len(decision.differences) == 0


def test_evaluate_fast_concat_rejects_unknown_video_properties():
    # Unknown video profile
    p_unknown_prof = _make_probe(profile="unknown")
    assert evaluate_fast_concat([p_unknown_prof]).compatible is False

    # Empty video profile
    p_empty_prof = _make_probe(profile="")
    assert evaluate_fast_concat([p_empty_prof]).compatible is False

    # Undefined video profile
    p_und_prof = _make_probe(profile="und")
    assert evaluate_fast_concat([p_und_prof]).compatible is False

    # Unknown SAR
    p_unknown_sar = _make_probe(sar="")
    assert evaluate_fast_concat([p_unknown_sar]).compatible is False

    # Invalid SAR 0:1
    p_zero_sar = _make_probe(sar="0:1")
    assert evaluate_fast_concat([p_zero_sar]).compatible is False

    # Unknown pix_fmt
    p_empty_pix = _make_probe(pix_fmt="")
    assert evaluate_fast_concat([p_empty_pix]).compatible is False

    # Unknown fps
    p_zero_fps = _make_probe(fps=0.0, fps_rational="")
    assert evaluate_fast_concat([p_zero_fps]).compatible is False

    # Unknown time_base
    p_empty_tb = _make_probe(video_time_base="")
    assert evaluate_fast_concat([p_empty_tb]).compatible is False


def test_evaluate_fast_concat_rejects_unknown_audio_properties():
    # Unknown audio channel layout
    p_unk_layout = _make_probe(channel_layout="unknown")
    d1 = evaluate_fast_concat([p_unk_layout])
    assert d1.compatible is False
    assert any("audio_channel_layout_unknown" in diff for diff in d1.differences)

    # Empty audio channel layout
    p_empty_layout = _make_probe(channel_layout="")
    d2 = evaluate_fast_concat([p_empty_layout])
    assert d2.compatible is False
    assert any("audio_channel_layout_unknown" in diff for diff in d2.differences)

    # Zero sample rate
    p_zero_sr = _make_probe(sample_rate=0)
    assert evaluate_fast_concat([p_zero_sr]).compatible is False

    # Zero channels
    p_zero_ch = _make_probe(channels=0)
    assert evaluate_fast_concat([p_zero_ch]).compatible is False

    # Empty sample format
    p_empty_fmt = _make_probe(sample_fmt="")
    assert evaluate_fast_concat([p_empty_fmt]).compatible is False


def test_evaluate_fast_concat_video_mismatches():
    base = _make_probe()

    # Codec mismatch
    p_codec = _make_probe(video_codec="hevc")
    assert evaluate_fast_concat([base, p_codec]).compatible is False

    # Profile mismatch
    p_prof = _make_probe(profile="Main")
    assert evaluate_fast_concat([base, p_prof]).compatible is False

    # Resolution mismatch
    p_res = _make_probe(width=1280, height=720)
    assert evaluate_fast_concat([base, p_res]).compatible is False

    # Pix fmt mismatch
    p_fmt = _make_probe(pix_fmt="yuv422p")
    assert evaluate_fast_concat([base, p_fmt]).compatible is False

    # FPS mismatch
    p_fps = _make_probe(fps=30.0, fps_rational="30/1")
    assert evaluate_fast_concat([base, p_fps]).compatible is False

    # Video time base mismatch
    p_tb = _make_probe(video_time_base="1/24")
    assert evaluate_fast_concat([base, p_tb]).compatible is False

    # SAR mismatch
    p_sar = _make_probe(sar="16:11")
    assert evaluate_fast_concat([base, p_sar]).compatible is False


def test_evaluate_fast_concat_audio_mismatches():
    base = _make_probe()

    # Audio presence mismatch: one with audio, one without audio
    p_no_audio = _make_probe(has_audio=False)
    d_presence = evaluate_fast_concat([base, p_no_audio])
    assert d_presence.compatible is False
    assert any("audio_presence_mismatch" in diff for diff in d_presence.differences)

    # Audio codec mismatch
    p_acodec = _make_probe(audio_codec="ac3")
    assert evaluate_fast_concat([base, p_acodec]).compatible is False

    # Sample rate mismatch
    p_sr = _make_probe(sample_rate=44100)
    assert evaluate_fast_concat([base, p_sr]).compatible is False

    # Sample fmt mismatch
    p_sfmt = _make_probe(sample_fmt="s16")
    assert evaluate_fast_concat([base, p_sfmt]).compatible is False

    # Channels mismatch
    p_ch = _make_probe(channels=6, channel_layout="5.1")
    assert evaluate_fast_concat([base, p_ch]).compatible is False

    # Channel layout mismatch
    p_layout = _make_probe(channel_layout="2.0")
    assert evaluate_fast_concat([base, p_layout]).compatible is False

    # Audio time base mismatch
    p_atb = _make_probe(audio_time_base="1/44100")
    assert evaluate_fast_concat([base, p_atb]).compatible is False


def test_evaluate_fast_concat_both_no_audio_compatible():
    """When all clips have no audio, fast concat is compatible if video streams match."""
    p1 = _make_probe(has_audio=False)
    p2 = _make_probe(has_audio=False)
    decision = evaluate_fast_concat([p1, p2])
    assert decision.compatible is True


def test_evaluate_fast_concat_case_insensitivity():
    """Profile, codec, and layout casing differences should normalize smoothly without false negatives."""
    p1 = _make_probe(profile="High", pix_fmt="YUV420P", channel_layout="Stereo")
    p2 = _make_probe(profile="high", pix_fmt="yuv420p", channel_layout="stereo")
    decision = evaluate_fast_concat([p1, p2])
    assert decision.compatible is True


# ============================================================================
# 5. NormalizationProfile and build_normalization_profile
# ============================================================================

def test_normalization_profile_canvas_floor_even_min2():
    # Odd dimensions 1921x1079 -> floor to 1920x1078
    p_odd = _make_probe(width=1921, height=1079)
    prof1 = build_normalization_profile([p_odd])
    assert prof1.width == 1920
    assert prof1.height == 1078

    # Tiny dimension 1x1 -> floor to min 2
    p_tiny = _make_probe(width=1, height=1)
    prof2 = build_normalization_profile([p_tiny])
    assert prof2.width == 2
    assert prof2.height == 2

    # Standard 1920x1080 -> 1920x1080
    p_std = _make_probe(width=1920, height=1080)
    prof3 = build_normalization_profile([p_std])
    assert prof3.width == 1920
    assert prof3.height == 1080


def test_normalization_profile_fps_selection():
    # First clip fps=0, second clip fps=29.97002997 -> picks second clip
    p_zero_fps = _make_probe(fps=0.0, fps_rational="")
    p_valid_fps = _make_probe(fps=29.97002997, fps_rational="30000/1001")
    prof = build_normalization_profile([p_zero_fps, p_valid_fps])
    assert prof.fps == pytest.approx(29.97002997, 1e-6)
    assert prof.fps_rational == "30000/1001"

    # All clips fps=0 -> fallback to 24.0 ('24/1')
    prof_fallback = build_normalization_profile([p_zero_fps])
    assert prof_fallback.fps == 24.0
    assert prof_fallback.fps_rational == "24/1"


def test_normalization_profile_audio_target_and_requires_audio():
    p_with_audio = _make_probe(has_audio=True, sample_rate=44100, channels=6, channel_layout="5.1")
    prof = build_normalization_profile([p_with_audio])

    # Audio always targets standardized 48000 Hz, fltp, stereo, aac
    assert prof.sample_rate == 48000
    assert prof.sample_fmt == "fltp"
    assert prof.channels == 2
    assert prof.channel_layout == "stereo"
    assert prof.audio_codec == "aac"
    assert prof.requires_audio is True

    # Clips with no audio, but default synthesize_audio=True -> requires_audio=True
    p_no_audio = _make_probe(has_audio=False)
    prof_synth = build_normalization_profile([p_no_audio], synthesize_audio=True)
    assert prof_synth.requires_audio is True
    assert prof_synth.synthesize_audio is True

    # Clips with no audio and synthesize_audio=False -> requires_audio=False
    prof_no_synth = build_normalization_profile([p_no_audio], synthesize_audio=False)
    assert prof_no_synth.requires_audio is False
    assert prof_no_synth.synthesize_audio is False


def test_normalization_profile_roundtrip():
    prof = NormalizationProfile(
        width=1280,
        height=720,
        fps=30.0,
        fps_rational="30/1",
        requires_audio=True,
    )
    d = prof.to_dict()
    restored = NormalizationProfile.from_dict(d)
    assert restored == prof
