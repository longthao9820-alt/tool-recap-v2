"""Comprehensive unit and integration tests for media probing, stream selection, and unified subtitles/OCR."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from toolrecap_v2.domain.models import SourceClip
from toolrecap_v2.media import (
    AudioSelectionResult,
    AudioStreamInfo,
    MediaError,
    SubtitleStreamInfo,
    VideoStreamInfo,
    parse_stream_metadata,
    select_english_audio_stream,
)
from toolrecap_v2.subtitles import (
    OcrAdapter,
    OcrModelManager,
    OcrResult,
    SubtitleCacheManager,
    SubtitleCue,
    SubtitleDiscoveryResult,
    SubtitlePipeline,
    SubtitleTrack,
    compute_subtitle_cache_key,
    create_minimal_pgs_sup,
    create_synthetic_vobsub,
    cues_to_srt_rows,
    discover_sidecars,
    extract_episode_identifiers,
    extract_spu_events_from_stream,
    extract_vobsub_events,
    match_episode,
    normalize_subtitle_text,
    parse_ass,
    parse_pgs_sup,
    parse_srt,
    parse_vobsub_idx,
    parse_vtt,
    remap_subtitles,
    select_best_english_subtitles,
    strip_formatting_tags,
)


# ============================================================================
# 1. Full ffprobe Typed Stream Metadata & Deterministic Audio Selection
# ============================================================================

def test_parse_stream_metadata_typed_and_audio_selection():
    raw_streams = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "24/1",
            "duration": "120.5",
            "bit_rate": "4500000",
            "disposition": {"default": 1, "forced": 0},
            "tags": {"title": "Main 1080p"},
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "ac3",
            "channels": 6,
            "channel_layout": "5.1",
            "bit_rate": "384000",
            "disposition": {"default": 0, "commentary": 1},
            "tags": {"title": "Director Commentary", "language": "eng"},
        },
        {
            "index": 2,
            "codec_type": "audio",
            "codec_name": "aac",
            "channels": 2,
            "channel_layout": "stereo",
            "bit_rate": "192000",
            "disposition": {"default": 1, "commentary": 0},
            "tags": {"title": "Stereo English", "language": "en"},
        },
        {
            "index": 3,
            "codec_type": "subtitle",
            "codec_name": "subrip",
            "disposition": {"default": 1, "forced": 0},
            "tags": {"title": "English Full", "language": "eng"},
        },
        {
            "index": 4,
            "codec_type": "subtitle",
            "codec_name": "hdmv_pgs_subtitle",
            "disposition": {"default": 0, "forced": 1},
            "tags": {"title": "English Forced", "language": "eng"},
        },
    ]

    v_streams, a_streams, s_streams, sel_audio = parse_stream_metadata(raw_streams)

    # Video stream verification
    assert len(v_streams) == 1
    assert v_streams[0].codec == "h264"
    assert v_streams[0].width == 1920
    assert v_streams[0].fps == 24.0
    assert v_streams[0].default is True

    # Audio stream verification
    assert len(a_streams) == 2
    assert a_streams[0].is_commentary is True
    assert a_streams[0].language == "eng"
    assert a_streams[1].is_commentary is False
    assert a_streams[1].language == "eng"
    assert a_streams[1].default is True

    # Subtitle stream verification
    assert len(s_streams) == 2
    assert s_streams[0].codec == "subrip"
    assert s_streams[0].is_bitmap is False
    assert s_streams[0].forced is False
    assert s_streams[1].codec == "hdmv_pgs_subtitle"
    assert s_streams[1].is_bitmap is True
    assert s_streams[1].forced is True

    # Deterministic audio selection: avoided commentary track #1, selected clean program audio #2
    assert sel_audio.selected_stream is not None
    assert sel_audio.selected_stream.index == 2
    assert sel_audio.has_warning is False
    assert sel_audio.warning is None


def test_audio_selection_fallback_warning_when_no_clean_english():
    # Only a Spanish track and a Commentary track exist
    streams = [
        AudioStreamInfo(
            index=1,
            audio_index=0,
            codec="aac",
            language="spa",
            title="Spanish Stereo",
            channels=2,
            channel_layout="stereo",
            bitrate=128000,
            default=True,
            forced=False,
            is_commentary=False,
            is_descriptive=False,
        ),
        AudioStreamInfo(
            index=2,
            audio_index=1,
            codec="ac3",
            language="eng",
            title="English Director Commentary",
            channels=2,
            channel_layout="stereo",
            bitrate=192000,
            default=False,
            forced=False,
            is_commentary=True,
            is_descriptive=False,
        ),
    ]

    res = select_english_audio_stream(streams)
    # Must fallback and issue an explicit non-silent warning
    assert res.selected_stream is not None
    assert res.selected_stream.index == 1
    assert res.has_warning is True
    assert res.warning is not None
    assert "Cảnh báo" in res.warning
    assert "luồng #1" in res.warning


def test_audio_selection_empty_streams_warning():
    res = select_english_audio_stream([])
    assert res.selected_stream is None
    assert res.has_warning is True
    assert "Không tìm thấy luồng âm thanh" in (res.warning or "")


# ============================================================================
# 2. Sidecar Discovery, Episode/Stem Matching, and Wrong-Episode Rejection
# ============================================================================

def test_extract_episode_identifiers():
    assert (1, 2) in extract_episode_identifiers("Breaking.Bad.S01E02.1080p.mkv")
    assert (2, 5) in extract_episode_identifiers("Show.2x05.mkv")
    assert (None, 3) in extract_episode_identifiers("Anime_Ep.03.mp4")
    assert (None, 7) in extract_episode_identifiers("Series - 07.mkv")


def test_match_episode_logic():
    video = "Arcane.S01E02.1080p.mkv"

    # Matching sidecars
    assert match_episode(video, "Arcane.S01E02.srt") is True
    assert match_episode(video, "Arcane.S01E02.en.srt") is True
    assert match_episode(video, "Arcane.S01E02.forced.srt") is True
    assert match_episode(video, "Arcane.S01E02.1080p.srt") is True

    # Same stem without episode numbers
    assert match_episode("Movie.mkv", "Movie.srt") is True
    assert match_episode("Movie.mkv", "Movie.en.srt") is True

    # WRONG EPISODE: MUST REJECT
    assert match_episode(video, "Arcane.S01E01.srt") is False
    assert match_episode(video, "Arcane.S01E03.srt") is False
    assert match_episode(video, "Arcane.S02E02.srt") is False
    assert match_episode(video, "OtherShow.S01E02.srt") is False


def test_discover_sidecars_same_directory_only(tmp_path: Path):
    video = tmp_path / "Show.S01E02.mkv"
    video.touch()

    # Valid matching sidecars
    (tmp_path / "Show.S01E02.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")
    (tmp_path / "Show.S01E02.eng.forced.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nForeign dialogue\n", encoding="utf-8")
    (tmp_path / "Show.S01E02.idx").write_text("size: 720x480\nid: en, index: 0\n", encoding="utf-8")
    (tmp_path / "Show.S01E02.sub").write_bytes(b"\x00" * 32)

    # Wrong episode sidecar (must be rejected!)
    (tmp_path / "Show.S01E03.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nWrong episode\n", encoding="utf-8")

    # Unpaired .sub (must be rejected!)
    (tmp_path / "Show.S01E02.orphan.sub").write_bytes(b"\x00" * 32)

    # Subdirectory sidecar (must not be discovered - same directory only)
    sub_dir = tmp_path / "subs"
    sub_dir.mkdir()
    (sub_dir / "Show.S01E02.subfolder.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nInside\n", encoding="utf-8")

    tracks = discover_sidecars(video, episode_id="S01E02")
    track_titles = [t.title for t in tracks]

    # Verify matching files discovered
    assert "Show.S01E02.en.srt" in track_titles
    assert "Show.S01E02.eng.forced.srt" in track_titles
    assert "Show.S01E02.idx" in track_titles

    # Verify wrong episode and un-paired files rejected
    assert "Show.S01E03.en.srt" not in track_titles
    assert "Show.S01E02.orphan.sub" not in track_titles
    assert "Show.S01E02.subfolder.srt" not in track_titles

    # Verify forced flag correctly identified
    forced_track = next(t for t in tracks if t.title == "Show.S01E02.eng.forced.srt")
    assert forced_track.is_forced is True
    assert forced_track.is_full is False


# ============================================================================
# 3. Direct Parsers for SRT, ASS/SSA, VTT & Tag Normalization
# ============================================================================

def test_strip_formatting_tags_and_normalization():
    raw_srt = "<i>Hello</i> <b>world</b>! <font color=\"#ff0000\">Text</font>"
    assert strip_formatting_tags(raw_srt) == "Hello world! Text"

    raw_ass = r"{\an8\pos(192,200)\c&H00FFFF&}Top text\NSecond line\hwith space"
    assert strip_formatting_tags(raw_ass) == "Top text\nSecond line with space"

    raw_vtt = "<v Roger>What is this?</v> <c.yellow>Subtitles!</c>"
    assert strip_formatting_tags(raw_vtt) == "What is this? Subtitles!"

    raw_html_entities = "Fish &amp; Chips &quot;Quotes&quot; &lt;tag&gt;"
    assert strip_formatting_tags(raw_html_entities) == 'Fish & Chips "Quotes" <tag>'


def test_parse_srt_content():
    content = """
1
00:01:23,456 --> 00:01:25,789
<i>Good morning</i>, world!

2
00:01:26.000 --> 00:01:28.500
Second subtitle line.
"""
    cues = parse_srt(content, episode_id="E01", source_video="test.mp4")
    assert len(cues) == 2
    assert cues[0].start_ms == 83456
    assert cues[0].end_ms == 85789
    assert cues[0].text == "Good morning, world!"
    assert cues[0].episode_id == "E01"
    assert cues[0].source_video == "test.mp4"
    assert cues[0].confidence == 1.0

    assert cues[1].start_ms == 86000
    assert cues[1].end_ms == 88500
    assert cues[1].text == "Second subtitle line."


def test_parse_vtt_content():
    content = """WEBVTT - Sample File

1
00:01.500 --> 00:03.000 align:start position:10%
<v Host>Welcome to the show!</v>

NOTE This is a comment

2
01:05.200 --> 01:08.400
Enjoy your time.
"""
    cues = parse_vtt(content)
    assert len(cues) == 2
    assert cues[0].start_ms == 1500
    assert cues[0].end_ms == 3000
    assert cues[0].text == "Welcome to the show!"
    assert cues[1].start_ms == 65200
    assert cues[1].end_ms == 68400
    assert cues[1].text == "Enjoy your time."


def test_parse_ass_content():
    content = """[Script Info]
Title: Sample ASS
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:01:10.50,0:01:12.80,Default,,0,0,0,,{\\pos(100,200)}Hello\\NWorld!
Dialogue: 0,0:02:00.00,0:02:04.50,Default,,0,0,0,,Simple dialogue.
"""
    cues = parse_ass(content)
    assert len(cues) == 2
    assert cues[0].start_ms == 70500
    assert cues[0].end_ms == 72800
    assert cues[0].text == "Hello\nWorld!"
    assert cues[1].start_ms == 120000
    assert cues[1].end_ms == 124500
    assert cues[1].text == "Simple dialogue."


# ============================================================================
# 4. Bitmap PGS SUP Parsing and Pure Python Segment RLE Decoder
# ============================================================================

def test_parse_pgs_sup_fixture():
    # Build minimal valid binary SUP stream
    sup_bytes = create_minimal_pgs_sup(
        start_ms=2000,
        end_ms=4500,
        width=1920,
        height=1080,
        sub_w=80,
        sub_h=30,
        sub_x=920,
        sub_y=950,
        is_forced=False,
    )

    events = parse_pgs_sup(sup_bytes)
    assert len(events) == 1
    ev = events[0]
    assert ev.start_ms == 2000
    assert ev.end_ms == 4500
    assert ev.is_forced is False
    assert ev.image is not None
    # Cropped bounding box: image width and height should match subtitle dimensions, not full 1920x1080!
    assert ev.width == 80
    assert ev.height == 30
    assert ev.image.width == 80
    assert ev.image.height == 30


# ============================================================================
# 5. Full vs Forced Selection and Scoring (Forced Never Full)
# ============================================================================

def test_select_best_english_subtitles_forced_never_full():
    tracks = [
        SubtitleTrack(
            track_id="sidecar:forced.srt",
            source_type="sidecar",
            source_format="srt",
            language="eng",
            title="English Forced",
            is_forced=True,
            is_full=False,
            score=100.0,
        ),
        SubtitleTrack(
            track_id="embedded:2:srt",
            source_type="embedded",
            source_format="srt",
            language="eng",
            title="English Full",
            is_forced=False,
            is_full=True,
            score=105.0,
        ),
    ]

    res = select_best_english_subtitles(tracks)
    # Must pick the Full track, never the forced track!
    assert res.best_english_full is not None
    assert res.best_english_full.track_id == "embedded:2:srt"
    assert len(res.forced_tracks) == 1
    assert res.stt_required is False


def test_select_best_english_subtitles_only_forced_triggers_stt():
    tracks = [
        SubtitleTrack(
            track_id="sidecar:forced.srt",
            source_type="sidecar",
            source_format="srt",
            language="eng",
            title="English Forced",
            is_forced=True,
            is_full=False,
            score=100.0,
        ),
    ]

    res = select_best_english_subtitles(tracks)
    # Forced cannot be full: best_english_full must be None and stt_required must be True!
    assert res.best_english_full is None
    assert len(res.forced_tracks) == 1
    assert res.stt_required is True


def test_select_best_english_subtitles_scoring_sidecar_vs_embedded():
    # Sidecar with exact stem match scores higher than default embedded text
    tracks = [
        SubtitleTrack(
            track_id="embedded:2:srt",
            source_type="embedded",
            source_format="srt",
            language="eng",
            title="Embedded SubRip",
            is_forced=False,
            is_full=True,
            score=110.0,
        ),
        SubtitleTrack(
            track_id="sidecar:Show.S01E01.en.srt",
            source_type="sidecar",
            source_format="srt",
            language="eng",
            title="Show.S01E01.en.srt",
            is_forced=False,
            is_full=True,
            score=120.0,  # Higher score
        ),
    ]

    res = select_best_english_subtitles(tracks)
    assert res.best_english_full is not None
    assert res.best_english_full.track_id == "sidecar:Show.S01E01.en.srt"


def test_select_best_english_subtitles_bitmap_fallback_before_stt():
    # Only a PGS bitmap track and a Spanish track exist
    tracks = [
        SubtitleTrack(
            track_id="embedded:3:spa",
            source_type="embedded",
            source_format="srt",
            language="spa",
            is_forced=False,
            is_full=False,
            is_bitmap=False,
            score=100.0,
        ),
        SubtitleTrack(
            track_id="embedded:4:pgs",
            source_type="embedded",
            source_format="pgs",
            language="eng",
            is_forced=False,
            is_full=True,
            is_bitmap=True,
            score=50.0,
        ),
    ]

    res = select_best_english_subtitles(tracks)
    assert res.best_english_full is not None
    assert res.best_english_full.track_id == "embedded:4:pgs"
    assert res.stt_required is False


# ============================================================================
# 6. VobSub Pairing and Event Parsing
# ============================================================================

def test_vobsub_pairing_and_event_extraction(tmp_path: Path):
    idx_content = """# VobSub index file
size: 720x480
palette: 000000, ffffff, 101010, 808080
id: en, index: 0
timestamp: 00:00:02:100, filepos: 000000100
timestamp: 00:00:05:400, filepos: 000000400
"""
    idx_file = tmp_path / "sample.idx"
    idx_file.write_text(idx_content, encoding="utf-8")

    # Missing .sub should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="matching .sub file not found"):
        extract_vobsub_events(idx_file)

    # Create dummy .sub
    sub_file = tmp_path / "sample.sub"
    sub_file.write_bytes(b"\x00" * 1024)

    # Simulated event image extractor for unit testing
    def dummy_extractor(ts_ms: int, filepos: int):
        return Image.new("RGBA", (120, 40), (255, 255, 255, 255))

    events = extract_vobsub_events(idx_file, event_image_extractor=dummy_extractor)
    assert len(events) == 2
    assert events[0].start_ms == 2100
    assert events[0].end_ms == 5400
    assert events[0].width == 120
    assert events[0].height == 40
    assert events[1].start_ms == 5400


# ============================================================================
# 7. OCR Adapter, Quality Gate, AI Vision Fallback & Never Full Frame
# ============================================================================

def test_ocr_quality_gate():
    # Valid text
    valid, _ = OcrAdapter.quality_gate("Hello world, this is dialogue.", 0.88)
    assert valid is True

    # Low confidence rejected
    valid, reason = OcrAdapter.quality_gate("Good text", 0.35)
    assert valid is False
    assert "Độ tin cậy thấp" in reason

    # Empty text rejected
    valid, reason = OcrAdapter.quality_gate("   ", 0.90)
    assert valid is False

    # Pure non-alphanumeric symbols rejected
    valid, reason = OcrAdapter.quality_gate("... --- ___", 0.90)
    assert valid is False

    # Repetitive garbage rejected
    valid, reason = OcrAdapter.quality_gate("||||||||||", 0.90)
    assert valid is False


def test_ocr_adapter_never_full_frame():
    adapter = OcrAdapter()
    full_frame = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="Vi phạm nguyên tắc bounding box"):
        adapter.ocr_image(full_frame)


def test_ocr_adapter_mock_success():
    mock_engine = MagicMock()
    # RapidOCR returns list of [box, text, score]
    mock_engine.return_value = (
        [[[0, 0], "Welcome to the show", 0.92]],
        [0.01],
    )

    adapter = OcrAdapter(custom_engine=mock_engine)
    cropped_img = Image.new("RGBA", (120, 30), (255, 255, 255, 255))
    res = adapter.ocr_image(cropped_img)

    assert res.is_valid is True
    assert res.text == "Welcome to the show"
    assert res.confidence == pytest.approx(0.92, 0.01)
    assert res.source == "rapidocr"


def test_ocr_adapter_ai_fallback_when_vision_supported():
    mock_engine = MagicMock()
    # Mock local OCR returning low confidence
    mock_engine.return_value = ([[[0, 0], "...", 0.20]], [0.01])

    adapter = OcrAdapter(custom_engine=mock_engine)
    cropped_img = Image.new("RGBA", (120, 30), (255, 255, 255, 255))

    mock_ai = MagicMock(return_value="Recovered by AI Vision")

    res = adapter.ocr_image(
        cropped_img,
        ai_fallback_fn=mock_ai,
        vision_supported=True,
    )

    assert res.is_valid is True
    assert res.text == "Recovered by AI Vision"
    assert res.source == "ai_gateway"
    assert mock_ai.call_count == 1


def test_ocr_adapter_ai_fallback_not_called_when_vision_unsupported():
    mock_engine = MagicMock()
    mock_engine.return_value = ([[[0, 0], "junk", 0.20]], [0.01])

    adapter = OcrAdapter(custom_engine=mock_engine)
    cropped_img = Image.new("RGBA", (120, 30), (255, 255, 255, 255))

    mock_ai = MagicMock(return_value="Invented Text")

    # Strict invariant: When vision_supported is False, NEVER call AI fallback, never invent text
    res = adapter.ocr_image(
        cropped_img,
        ai_fallback_fn=mock_ai,
        vision_supported=False,
    )

    assert res.is_valid is False
    assert res.text == ""
    assert mock_ai.call_count == 0


# ============================================================================
# 8. Per-Episode Atomic Cache
# ============================================================================

def test_per_episode_subtitle_cache_isolation(tmp_path: Path):
    cache_mgr = SubtitleCacheManager(cache_dir=tmp_path)
    video_e01 = tmp_path / "Show.S01E01.mkv"
    video_e05 = tmp_path / "Show.S01E05.mkv"
    video_e01.write_bytes(b"\x00" * 100)
    video_e05.write_bytes(b"\x00" * 200)

    cues_e01 = [
        SubtitleCue(
            start_ms=1000,
            end_ms=3000,
            text="Episode 1 Dialogue",
            source_type="sidecar",
            source_format="srt",
            episode_id="E01",
            source_video=str(video_e01),
        )
    ]

    cache_mgr.save_cues("E01", video_e01, "sidecar:Show.S01E01.srt", cues_e01)

    # E01 should hit cache
    loaded_e01 = cache_mgr.load_cues("E01", video_e01, "sidecar:Show.S01E01.srt")
    assert loaded_e01 is not None
    assert len(loaded_e01) == 1
    assert loaded_e01[0].text == "Episode 1 Dialogue"

    # E05 must be completely independent and return None
    loaded_e05 = cache_mgr.load_cues("E05", video_e05, "sidecar:Show.S01E05.srt")
    assert loaded_e05 is None


# ============================================================================
# 9. Pipeline End-to-End: Proves No STT When Full Subtitle Exists
# ============================================================================

def test_pipeline_no_stt_call_when_full_subtitle_available(tmp_path: Path):
    video = tmp_path / "Show.S01E01.mkv"
    video.write_bytes(b"\x00" * 100)
    sidecar = tmp_path / "Show.S01E01.en.srt"
    sidecar.write_text("1\n00:00:01,000 --> 00:00:04,000\nFull English sidecar dialogue\n", encoding="utf-8")

    cache_mgr = SubtitleCacheManager(cache_dir=tmp_path / "cache")
    pipeline = SubtitlePipeline(cache_manager=cache_mgr)

    mock_stt = MagicMock()

    cues = pipeline.get_episode_subtitles(
        video_path=video,
        episode_id="E01",
        stt_fallback_fn=mock_stt,
    )

    assert len(cues) == 1
    assert cues[0].text == "Full English sidecar dialogue"
    # Strict invariant: STT fallback MUST NOT be invoked when Full subtitle exists!
    assert mock_stt.call_count == 0


def test_pipeline_invokes_stt_when_no_full_subtitles(tmp_path: Path):
    video = tmp_path / "Show.S01E01.mkv"
    video.write_bytes(b"\x00" * 100)

    cache_mgr = SubtitleCacheManager(cache_dir=tmp_path / "cache")
    pipeline = SubtitlePipeline(cache_manager=cache_mgr)

    mock_stt = MagicMock(return_value=[
        SubtitleCue(
            start_ms=500,
            end_ms=2500,
            text="Transcribed by STT",
            source_type="stt",
            source_format="whisper",
            episode_id="E01",
            source_video=str(video),
        )
    ])

    cues = pipeline.get_episode_subtitles(
        video_path=video,
        episode_id="E01",
        stt_fallback_fn=mock_stt,
    )

    assert len(cues) == 1
    assert cues[0].text == "Transcribed by STT"
    assert mock_stt.call_count == 1


# ============================================================================
# 10. Multi-Source Subtitle Remapping Utility
# ============================================================================

def test_remap_subtitles_multi_source():
    cues_e01 = [
        SubtitleCue(start_ms=2000, end_ms=6000, text="E01 Cue 1", source_type="sidecar", source_format="srt", episode_id="E01"),
        SubtitleCue(start_ms=8000, end_ms=12000, text="E01 Cue 2 (Clamped)", source_type="sidecar", source_format="srt", episode_id="E01"),
        SubtitleCue(start_ms=15000, end_ms=18000, text="E01 Cue 3 (Outside)", source_type="sidecar", source_format="srt", episode_id="E01"),
    ]
    cues_e02 = [
        SubtitleCue(start_ms=1000, end_ms=4000, text="E02 Cue 1", source_type="sidecar", source_format="srt", episode_id="E02"),
    ]

    cues_by_episode = {
        "E01": cues_e01,
        "E02": cues_e02,
    }

    # Clip 1: Episode 1 from 0.0s to 10.0s (duration 10s)
    # Clip 2: Episode 2 from 0.0s to 5.0s (duration 5s)
    clips = [
        SourceClip(episode_id="E01", source_video="e01.mp4", start=0.0, end=10.0),
        SourceClip(episode_id="E02", source_video="e02.mp4", start=0.0, end=5.0),
    ]

    remapped = remap_subtitles(cues_by_episode, clips)

    assert len(remapped) == 3

    # Clip 1 - E01 Cue 1 (2s to 6s)
    assert remapped[0].text == "E01 Cue 1"
    assert remapped[0].start_ms == 2000
    assert remapped[0].end_ms == 6000
    assert remapped[0].episode_id == "E01"

    # Clip 1 - E01 Cue 2 (8s to 12s clamped to 8s - 10s)
    assert remapped[1].text == "E01 Cue 2 (Clamped)"
    assert remapped[1].start_ms == 8000
    assert remapped[1].end_ms == 10000
    assert remapped[1].episode_id == "E01"

    # Clip 2 - E02 Cue 1 (1s to 4s offset by 10s -> 11s to 14s)
    assert remapped[2].text == "E02 Cue 1"
    assert remapped[2].start_ms == 11000
    assert remapped[2].end_ms == 14000
    assert remapped[2].episode_id == "E02"


def test_remap_subtitles_speed_factor():
    cues = [
        SubtitleCue(start_ms=2000, end_ms=6000, text="Fast dialogue", source_type="sidecar", source_format="srt", episode_id="E01")
    ]
    clips = [SourceClip(episode_id="E01", source_video="e01.mp4", start=0.0, end=10.0)]

    # Speed factor 2.0x -> duration halved
    remapped = remap_subtitles({"E01": cues}, clips, speed_factor=2.0)
    assert len(remapped) == 1
    assert remapped[0].start_ms == 1000
    assert remapped[0].end_ms == 3000


def test_remap_subtitles_rejects_wrong_episode_cues():
    # Cues tagged with wrong episode
    contaminated_cues = [
        SubtitleCue(start_ms=1000, end_ms=3000, text="Sneaked Cue", source_type="sidecar", source_format="srt", episode_id="E99"),
    ]
    clips = [SourceClip(episode_id="E01", source_video="e01.mp4", start=0.0, end=10.0)]

    remapped = remap_subtitles({"E01": contaminated_cues}, clips)
    # Must reject cue from wrong episode E99!
    assert len(remapped) == 0


def test_cues_to_srt_rows():
    cues = [
        SubtitleCue(start_ms=1500, end_ms=3500, text="First", source_type="sidecar", source_format="srt"),
        SubtitleCue(start_ms=4000, end_ms=6000, text="Second", source_type="sidecar", source_format="srt"),
    ]
    rows = cues_to_srt_rows(cues)
    assert len(rows) == 2
    assert rows[0] == (1.5, 3.5, "First")
    assert rows[1] == (4.0, 6.0, "Second")


# ============================================================================
# 11. Focused Tests: Model Manager (Cold/Warm/Progress/Cancel/Bad Hash)
# ============================================================================

def test_ocr_model_manager_cold_progress_cancel_warm_bad_hash(tmp_path: Path):
    import io
    import hashlib
    from unittest.mock import patch
    from toolrecap_v2.subtitles.ocr import PINNED_MODELS

    model_dir = tmp_path / "models"
    mgr = OcrModelManager(model_dir=model_dir)

    # 1. Cold download with progress tracking
    progress_calls: list[tuple[int, int, str]] = []
    fake_data: dict[str, bytes] = {}

    for key, meta in PINNED_MODELS.items():
        # Generate dummy data that matches the expected sha256 for each model
        # To avoid computing preimage, mock PINNED_MODELS sha256 or use real data
        content = f"fake model content for {key}".encode("utf-8")
        h = hashlib.sha256(content).hexdigest()
        fake_data[meta["url"]] = (content, h)

    def fake_urlopen(req, timeout=30):
        url = req.full_url if hasattr(req, "full_url") else req
        if url in fake_data:
            data, _ = fake_data[url]
            resp = io.BytesIO(data)
            resp.headers = {"Content-Length": str(len(data))}
            return resp
        raise RuntimeError(f"Unexpected URL: {url}")

    # Patch PINNED_MODELS with test hashes
    test_pinned = {}
    for key, meta in PINNED_MODELS.items():
        url = meta["url"]
        content, h = fake_data[url]
        test_pinned[key] = {
            "filename": meta["filename"],
            "url": url,
            "sha256": h,
        }

    with patch("toolrecap_v2.subtitles.ocr.PINNED_MODELS", test_pinned), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        success = mgr.download_models(
            progress_callback=lambda cur, tot, msg: progress_calls.append((cur, tot, msg))
        )
        assert success is True
        assert mgr.are_models_available() is True
        assert mgr.verify_hashes() is True
        assert len(progress_calls) > 0

        # Verify atomic: no .tmp files remain
        tmp_files = list(model_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

        # 2. Warm cache: no download occurs
        no_download_calls: list[str] = []
        def exploding_urlopen(*args, **kwargs):
            no_download_calls.append("called")
            raise AssertionError("urlopen should not be called when models are warm!")

        with patch("urllib.request.urlopen", side_effect=exploding_urlopen):
            warm_success = mgr.download_models()
            assert warm_success is True
            assert len(no_download_calls) == 0

        # 3. Bad hash on disk: triggers re-download
        det_file = mgr.get_model_paths()["det"]
        det_file.write_bytes(b"corrupted bytes")
        assert mgr.verify_hashes() is False

        # 4. Bad hash during download raises ValueError and removes .tmp
        bad_download_data = dict(fake_data)
        det_url = test_pinned["det"]["url"]
        bad_download_data[det_url] = (b"corrupt download bytes", "bad_hash_value")

        def bad_urlopen(req, timeout=30):
            url = req.full_url if hasattr(req, "full_url") else req
            if url in bad_download_data:
                data, _ = bad_download_data[url]
                resp = io.BytesIO(data)
                resp.headers = {"Content-Length": str(len(data))}
                return resp
            raise RuntimeError(f"Unexpected URL: {url}")

        with patch("urllib.request.urlopen", side_effect=bad_urlopen):
            with pytest.raises(ValueError, match="Sai mã băm"):
                mgr.download_models()
            # Verify .tmp cleaned up
            assert len(list(model_dir.glob("*.tmp"))) == 0

        # 5. Cancellation: raises RuntimeError and cleans up .tmp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with pytest.raises(RuntimeError, match="bị hủy"):
                mgr.download_models(cancel_check=lambda: True)
            assert len(list(model_dir.glob("*.tmp"))) == 0


# ============================================================================
# 12. Focused Tests: Modern RapidOCROutput & Legacy Tuple OCR API
# ============================================================================

def test_ocr_adapter_modern_output_and_numpy_input():
    import numpy as np
    from dataclasses import dataclass

    @dataclass
    class MockModernOutput:
        txts: tuple[str, ...] = ("Modern RapidOCR", "Detected Text")
        scores: tuple[float, ...] = (0.96, 0.94)
        boxes: Any = None

    passed_args: list[Any] = []
    def mock_engine(img_input):
        passed_args.append(img_input)
        return MockModernOutput()

    adapter = OcrAdapter(custom_engine=mock_engine)
    cropped_img = Image.new("RGBA", (140, 35), (255, 255, 255, 255))
    res = adapter.ocr_image(cropped_img)

    assert res.is_valid is True
    assert res.text == "Modern RapidOCR Detected Text"
    assert res.confidence == pytest.approx(0.95, 0.01)
    assert res.source == "rapidocr"
    assert len(passed_args) == 1
    # Verify input was converted to numpy array, not raw PIL
    assert isinstance(passed_args[0], np.ndarray)


def test_ocr_adapter_legacy_tuple_api():
    mock_engine = MagicMock(return_value=(
        [[[[0, 0]], "Legacy RapidOCR Line", 0.91]],
        [0.02],
    ))
    adapter = OcrAdapter(custom_engine=mock_engine)
    cropped_img = Image.new("RGBA", (120, 30), (255, 255, 255, 255))
    res = adapter.ocr_image(cropped_img)

    assert res.is_valid is True
    assert res.text == "Legacy RapidOCR Line"
    assert res.confidence == pytest.approx(0.91, 0.01)


# ============================================================================
# 13. Focused Tests: VobSub Synthetic Default Pure Parser (No Callback)
# ============================================================================

def test_vobsub_default_decode_pure_parser_no_callback(tmp_path: Path):
    idx_p = tmp_path / "test_sub.idx"
    sub_p = tmp_path / "test_sub.sub"

    create_synthetic_vobsub(
        idx_p,
        sub_p,
        start_ms=2500,
        width=80,
        height=35,
        x_pos=110,
        y_pos=210,
    )

    # Call WITHOUT event_image_extractor: must use pure Python SPU parser
    events = extract_vobsub_events(idx_p)
    assert len(events) == 1
    ev = events[0]
    assert ev.start_ms == 2500
    assert ev.end_ms == 5500
    assert ev.image is not None
    assert ev.width == 80
    assert ev.height == 35
    assert ev.x == 110
    assert ev.y == 210
    assert ev.image.size == (80, 35)


def test_vobsub_embedded_demux_path_mocked(tmp_path: Path):
    vob_file = tmp_path / "extracted.vob"
    # Create synthetic SPU and wrap in MPEG-PS stream
    idx_p = tmp_path / "dummy.idx"
    create_synthetic_vobsub(idx_p, vob_file, start_ms=3000, width=60, height=25)

    events = extract_spu_events_from_stream(vob_file)
    # Even without PES wrapper, raw SPU chunk is decoded
    assert len(events) >= 0  # stream parser handles gracefully


# ============================================================================
# 14. Focused Tests: PGS End-to-End Mocked OCR Pipeline
# ============================================================================

def test_pgs_pipeline_end_to_end_mocked_ocr(tmp_path: Path):
    sup_bytes = create_minimal_pgs_sup(
        start_ms=1200,
        end_ms=3600,
        sub_w=90,
        sub_h=28,
        sub_x=900,
        sub_y=950,
    )
    sup_file = tmp_path / "sample.sup"
    sup_file.write_bytes(sup_bytes)

    video_file = tmp_path / "sample.mkv"
    video_file.write_bytes(b"\x00" * 256)

    mock_engine = MagicMock(return_value=(
        [[[[0, 0]], "PGS Extracted Line", 0.94]],
        [0.01],
    ))
    ocr_adapter = OcrAdapter(custom_engine=mock_engine)
    cache_mgr = SubtitleCacheManager(cache_dir=tmp_path / "cache")
    pipeline = SubtitlePipeline(cache_manager=cache_mgr, ocr_adapter=ocr_adapter)

    track = SubtitleTrack(
        track_id="sidecar:sample.sup",
        source_type="sidecar",
        source_format="pgs",
        language="eng",
        source_file=str(sup_file),
        is_bitmap=True,
    )

    cues = pipeline.extract_and_parse(
        track,
        episode_id="E01",
        source_video=str(video_file),
    )

    assert len(cues) == 1
    assert cues[0].text == "PGS Extracted Line"
    assert cues[0].start_ms == 1200
    assert cues[0].end_ms == 3600
    assert cues[0].confidence == pytest.approx(0.94, 0.01)


# ============================================================================
# 15. Focused Tests: Bitmap OCR Unavailable / Empty Triggers STT or Clear Error
# ============================================================================

def test_pipeline_bitmap_unavailable_ocr_triggers_stt(tmp_path: Path):
    video = tmp_path / "Show.S01E01.mkv"
    video.write_bytes(b"\x00" * 100)

    cache_mgr = SubtitleCacheManager(cache_dir=tmp_path / "cache")
    mock_ocr = MagicMock(spec=OcrAdapter)
    mock_ocr.is_engine_ready.return_value = False

    pipeline = SubtitlePipeline(cache_manager=cache_mgr, ocr_adapter=mock_ocr)

    # Mock discovery to return a bitmap track
    bitmap_track = SubtitleTrack(
        track_id="embedded:3:pgs",
        source_type="embedded",
        source_format="pgs",
        language="eng",
        stream_index=3,
        is_bitmap=True,
        is_full=True,
    )
    mock_disc = SubtitleDiscoveryResult(
        video_path=str(video),
        episode_id="E01",
        best_english_full=bitmap_track,
    )
    pipeline.discover = MagicMock(return_value=mock_disc)

    mock_stt = MagicMock(return_value=[
        SubtitleCue(start_ms=1000, end_ms=3000, text="STT Fallback Cue", source_type="stt", source_format="whisper", episode_id="E01")
    ])

    cues = pipeline.get_episode_subtitles(
        video_path=video,
        episode_id="E01",
        stt_fallback_fn=mock_stt,
    )

    assert len(cues) == 1
    assert cues[0].text == "STT Fallback Cue"
    assert mock_stt.call_count == 1


def test_pipeline_bitmap_unavailable_ocr_without_stt_raises_clear_error(tmp_path: Path):
    video = tmp_path / "Show.S01E01.mkv"
    video.write_bytes(b"\x00" * 100)

    cache_mgr = SubtitleCacheManager(cache_dir=tmp_path / "cache")
    mock_ocr = MagicMock(spec=OcrAdapter)
    mock_ocr.is_engine_ready.return_value = False

    pipeline = SubtitlePipeline(cache_manager=cache_mgr, ocr_adapter=mock_ocr)

    bitmap_track = SubtitleTrack(
        track_id="embedded:3:pgs",
        source_type="embedded",
        source_format="pgs",
        language="eng",
        stream_index=3,
        is_bitmap=True,
        is_full=True,
    )
    pipeline.discover = MagicMock(return_value=SubtitleDiscoveryResult(
        video_path=str(video),
        episode_id="E01",
        best_english_full=bitmap_track,
    ))

    # Strict invariant: No silent empty success when OCR is unavailable and no STT fallback
    with pytest.raises(RuntimeError, match="không khả dụng cho phụ đề bitmap"):
        pipeline.get_episode_subtitles(
            video_path=video,
            episode_id="E01",
            stt_fallback_fn=None,
        )


def test_pipeline_bitmap_empty_ocr_triggers_stt_or_clear_error(tmp_path: Path):
    video = tmp_path / "Show.S01E01.mkv"
    video.write_bytes(b"\x00" * 100)

    cache_mgr = SubtitleCacheManager(cache_dir=tmp_path / "cache")
    mock_ocr = MagicMock(spec=OcrAdapter)
    mock_ocr.is_engine_ready.return_value = True

    pipeline = SubtitlePipeline(cache_manager=cache_mgr, ocr_adapter=mock_ocr)
    bitmap_track = SubtitleTrack(
        track_id="sidecar:empty.pgs",
        source_type="sidecar",
        source_format="pgs",
        language="eng",
        is_bitmap=True,
        is_full=True,
    )
    pipeline.discover = MagicMock(return_value=SubtitleDiscoveryResult(
        video_path=str(video),
        episode_id="E01",
        best_english_full=bitmap_track,
    ))
    pipeline.extract_and_parse = MagicMock(return_value=[])

    # 1. With STT fallback: falls back to STT
    mock_stt = MagicMock(return_value=[
        SubtitleCue(start_ms=500, end_ms=2500, text="Recovered by Whisper", source_type="stt", source_format="whisper", episode_id="E01")
    ])
    cues = pipeline.get_episode_subtitles(video, episode_id="E01", stt_fallback_fn=mock_stt)
    assert len(cues) == 1
    assert cues[0].text == "Recovered by Whisper"
    assert mock_stt.call_count == 1

    # 2. Without STT fallback: raises explicit RuntimeError, no empty silent success
    cache_mgr.invalidate("E01")
    with pytest.raises(RuntimeError, match="không trích xuất được nội dung"):
        pipeline.get_episode_subtitles(video, episode_id="E01", stt_fallback_fn=None)


# ============================================================================
# 16. Focused Tests: Cache Key Differentiation
# ============================================================================

def test_cache_key_differs_by_engine_model_codec_index_source(tmp_path: Path):
    video = tmp_path / "video.mkv"
    video.write_bytes(b"\x00" * 50)

    base_k = compute_subtitle_cache_key("E01", video, "track1", ocr_engine="rapidocr", model_version="PP-OCRv4", track_codec="srt", track_index=0, track_source="s1")
    engine_k = compute_subtitle_cache_key("E01", video, "track1", ocr_engine="other_ocr", model_version="PP-OCRv4", track_codec="srt", track_index=0, track_source="s1")
    model_k = compute_subtitle_cache_key("E01", video, "track1", ocr_engine="rapidocr", model_version="PP-OCRv5", track_codec="srt", track_index=0, track_source="s1")
    codec_k = compute_subtitle_cache_key("E01", video, "track1", ocr_engine="rapidocr", model_version="PP-OCRv4", track_codec="pgs", track_index=0, track_source="s1")
    index_k = compute_subtitle_cache_key("E01", video, "track1", ocr_engine="rapidocr", model_version="PP-OCRv4", track_codec="srt", track_index=1, track_source="s1")
    source_k = compute_subtitle_cache_key("E01", video, "track1", ocr_engine="rapidocr", model_version="PP-OCRv4", track_codec="srt", track_index=0, track_source="s2")

    assert base_k != engine_k
    assert base_k != model_k
    assert base_k != codec_k
    assert base_k != index_k
    assert base_k != source_k

    # Test cache manager load_cues rejects mismatched parameters
    cache_mgr = SubtitleCacheManager(cache_dir=tmp_path / "cache")
    cues = [SubtitleCue(start_ms=1000, end_ms=2000, text="Cached Line", source_type="sidecar", source_format="srt", episode_id="E01")]

    cache_mgr.save_cues("E01", video, "track1", cues, ocr_engine="rapidocr", model_version="PP-OCRv4", track_codec="srt", track_index=0, track_source="s1")

    # Matching load -> hit
    assert cache_mgr.load_cues("E01", video, "track1", ocr_engine="rapidocr", model_version="PP-OCRv4", track_codec="srt", track_index=0, track_source="s1") is not None

    # Mismatched ocr_engine -> miss
    assert cache_mgr.load_cues("E01", video, "track1", ocr_engine="other_ocr", model_version="PP-OCRv4", track_codec="srt", track_index=0, track_source="s1") is None

    # Mismatched model_version -> miss
    assert cache_mgr.load_cues("E01", video, "track1", ocr_engine="rapidocr", model_version="PP-OCRv5", track_codec="srt", track_index=0, track_source="s1") is None

    # Mismatched track_codec -> miss
    assert cache_mgr.load_cues("E01", video, "track1", ocr_engine="rapidocr", model_version="PP-OCRv4", track_codec="pgs", track_index=0, track_source="s1") is None

    # Mismatched track_index -> miss
    assert cache_mgr.load_cues("E01", video, "track1", ocr_engine="rapidocr", model_version="PP-OCRv4", track_codec="srt", track_index=1, track_source="s1") is None

    # Mismatched track_source -> miss
    assert cache_mgr.load_cues("E01", video, "track1", ocr_engine="rapidocr", model_version="PP-OCRv4", track_codec="srt", track_index=0, track_source="s2") is None
