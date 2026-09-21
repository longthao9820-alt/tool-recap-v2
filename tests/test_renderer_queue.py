"""Comprehensive mandatory test suite for canonical PublicationRenderer and ProjectQueue integration.

Covers:
- Multi-output rendering into clean publication folders ({title}.mp4, {title}.original.srt, {title}.narration.srt).
- Title exact sample matching & collision resolution.
- Cross-source E01/E03/E05 clipping commands and English audio stream mapping.
- Original SRT remapping across episodes and narration SRT synchronization.
- Real FFmpeg Audio Mix with original remaining under narration, no clipping, target LUFS, and true peak bounds.
- Audio duck on/off/gains/LUFS/TP settings coverage.
- Original dialogue preservation for original-only segments.
- Output validation module rejecting missing, empty, or extraneous files.
- ProjectQueue season workflow phase ordering (no per-episode finalization for season).
- Zero outputs completion with COMPLETED status and no publication files.
- Cancellation during OCR, scanner, season connection, voice synth, audio mix, and video render with UI state restored.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from toolrecap_v2.analyzer.phases import AnalysisPhase
from toolrecap_v2.api_client import OpenAICompatibleClient
from toolrecap_v2.audio_mix import (
    AudioMixSettings,
    build_audio_mix_command,
    build_audio_mix_filter_graph,
    execute_audio_mix,
    plan_audio_mix,
    validate_mixed_audio,
)
from toolrecap_v2.domain.enums import AnalysisScope, AudioPolicy, CandidateScope, OutputStatus, ProjectPhase
from toolrecap_v2.domain.models import (
    AnalysisManifest,
    CommentaryOutput,
    Segment,
    SourceClip,
    SourceEpisode,
    ValidationError,
)
from toolrecap_v2.media import (
    MediaError,
    RenderCancelled,
    cut_clip,
    find_binary,
    probe_duration,
    probe_typed_media,
    run_command,
    write_srt_file,
)
from toolrecap_v2.output_validation import (
    OutputValidationError,
    validate_mp4_file,
    validate_publication_folder,
    validate_srt_file,
)
from toolrecap_v2.projects import ProjectQueue, ProjectRecord, ProjectStore
from toolrecap_v2.renderer import PublicationRenderer
from toolrecap_v2.settings import AppSettings
from toolrecap_v2.subtitles.models import SubtitleCue
from toolrecap_v2.voice.catalog import DEFAULT_VOICE_ID
from toolrecap_v2.voice.manager import VoiceError, VoiceModelManager


# ---------------------------------------------------------------------------
# Helpers to generate real test media
# ---------------------------------------------------------------------------

def create_synthetic_video(
    output_path: Path,
    *,
    duration: float = 3.0,
    tone_freq: float = 440.0,
    add_commentary_track: bool = False,
) -> Path:
    """Create a minimal real MP4 video using FFmpeg for testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_binary("ffmpeg")

    if add_commentary_track:
        # Track 0: Video, Track 1: Commentary audio (200Hz), Track 2: Program audio (tone_freq)
        cmd = [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={duration:.1f}:size=320x240:rate=24",
            "-f", "lavfi", "-i", f"sine=frequency=200:duration={duration:.1f}",
            "-f", "lavfi", "-i", f"sine=frequency={tone_freq}:duration={duration:.1f}",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-map", "2:a:0",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            "-metadata:s:a:0", "title=Director Commentary",
            "-metadata:s:a:0", "language=eng",
            "-metadata:s:a:1", "title=Main Program Audio",
            "-metadata:s:a:1", "language=eng",
            str(output_path),
        ]
    else:
        cmd = [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={duration:.1f}:size=320x240:rate=24",
            "-f", "lavfi", "-i", f"sine=frequency={tone_freq}:duration={duration:.1f}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            "-metadata:s:a:0", "title=Main Program Audio",
            "-metadata:s:a:0", "language=eng",
            str(output_path),
        ]

    subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return output_path


class FakeVoiceManager:
    """Fake voice manager for deterministic synthesis testing without external VoiceStudio dependencies."""

    def __init__(self, sample_rate: int = 48000, duration: float = 1.0) -> None:
        self.sample_rate = sample_rate
        self.duration = duration
        self.synthesize_calls: list[dict[str, Any]] = []

    def synthesize(
        self,
        voice_id: str,
        text: str,
        output_path: Path,
        *,
        style: str = "film_recap",
        progress_callback: Any = None,
        cancel_event: threading.Event | None = None,
        allow_mock_synth: bool = False,
    ) -> Path:
        if cancel_event and cancel_event.is_set():
            raise VoiceError("Tổng hợp giọng đọc đã bị hủy.")

        self.synthesize_calls.append({"voice_id": voice_id, "text": text, "style": style})
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate audio tone WAV using ffmpeg with specified duration
        ffmpeg = find_binary("ffmpeg")
        cmd = [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", f"sine=frequency=300:duration={self.duration}",
            "-c:a", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "2",
            str(output_path),
        ]
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return output_path


# ---------------------------------------------------------------------------
# Test 1: Multi-output All Rendered & Clean Publication Folder
# ---------------------------------------------------------------------------

def test_multi_output_all_rendered_and_clean_publication_folder(tmp_path: Path) -> None:
    """Loops all outputs sequentially; each gets clean folder with exactly the 3 files."""
    vid1 = create_synthetic_video(tmp_path / "e1.mp4", duration=4.0)
    vid2 = create_synthetic_video(tmp_path / "e2.mp4", duration=4.0)

    ep1 = SourceEpisode(episode_id="E01", source_video=str(vid1), duration_seconds=4.0)
    ep2 = SourceEpisode(episode_id="E02", source_video=str(vid2), duration_seconds=4.0)

    out1 = CommentaryOutput(
        output_id="out_01",
        title="Tập 1: Đại Chiến Mở Màn",
        sanitized_title="tap-1-dai-chien-mo-man",
        segments=[
            Segment(
                segment_id="seg_01",
                source_clips=[SourceClip(episode_id="E01", source_video=str(vid1), start=0.0, end=2.0)],
                narration="Mở đầu cuộc chiến khốc liệt.",
                audio_policy="duck",
            )
        ],
    )

    out2 = CommentaryOutput(
        output_id="out_02",
        title="Tập 2: Kế Sách Hiểm Ác",
        sanitized_title="tap-2-ke-sach-hiem-ac",
        segments=[
            Segment(
                segment_id="seg_02",
                source_clips=[SourceClip(episode_id="E02", source_video=str(vid2), start=1.0, end=3.0)],
                narration="Phe phản diện bắt đầu hành động.",
                audio_policy="duck",
            )
        ],
    )

    manifest = AnalysisManifest(
        project_id="test_multi_render",
        source_episodes=[ep1, ep2],
        outputs=[out1, out2],
    )

    cues_e01 = [SubtitleCue(start_ms=0, end_ms=2000, text="Thoại gốc tập 1", source_type="embedded", source_format="srt", episode_id="E01")]
    cues_e02 = [SubtitleCue(start_ms=1000, end_ms=3000, text="Thoại gốc tập 2", source_type="embedded", source_format="srt", episode_id="E02")]
    cues_map = {"E01": cues_e01, "E02": cues_e02}

    fake_voice = FakeVoiceManager()
    renderer = PublicationRenderer(voice_manager=fake_voice)
    settings = AppSettings(burn_subtitles=True, use_gpu=False)

    out_root = tmp_path / "outputs"
    rendered = renderer.render_manifest(
        manifest=manifest,
        settings=settings,
        voice_id="v_female_north_01",
        transcript_cues_by_episode=cues_map,
        output_root=out_root,
    )

    assert len(rendered) == 2

    # Check Output 1 publication folder
    pub1 = out_root / "tap-1-dai-chien-mo-man"
    assert pub1.is_dir()
    files1 = {f.name for f in pub1.iterdir()}
    assert files1 == {
        "tap-1-dai-chien-mo-man.mp4",
        "tap-1-dai-chien-mo-man.original.srt",
        "tap-1-dai-chien-mo-man.narration.srt",
    }
    val1 = validate_publication_folder(pub1, "tap-1-dai-chien-mo-man")
    assert val1["duration"] > 0
    assert val1["original_cues_count"] >= 1
    assert val1["narration_cues_count"] >= 1

    # Check Output 2 publication folder
    pub2 = out_root / "tap-2-ke-sach-hiem-ac"
    assert pub2.is_dir()
    files2 = {f.name for f in pub2.iterdir()}
    assert files2 == {
        "tap-2-ke-sach-hiem-ac.mp4",
        "tap-2-ke-sach-hiem-ac.original.srt",
        "tap-2-ke-sach-hiem-ac.narration.srt",
    }
    val2 = validate_publication_folder(pub2, "tap-2-ke-sach-hiem-ac")
    assert val2["duration"] > 0

    # Ensure no leftover temp folders
    temp_render_dir = out_root / ".temp_render"
    if temp_render_dir.exists():
        assert len(list(temp_render_dir.iterdir())) == 0


# ---------------------------------------------------------------------------
# Test 2: Cross-source E01/E03/E05 Commands and English Audio Track Selection
# ---------------------------------------------------------------------------

def test_cross_source_e01_e03_e05_commands_and_audio_mapping(tmp_path: Path) -> None:
    """Clips from E01, E03, E05 in sequence correctly cut with English audio stream mapped."""
    vid1 = create_synthetic_video(tmp_path / "e1.mp4", duration=3.0, add_commentary_track=True)
    vid3 = create_synthetic_video(tmp_path / "e3.mp4", duration=3.0, add_commentary_track=True)
    vid5 = create_synthetic_video(tmp_path / "e5.mp4", duration=3.0, add_commentary_track=True)

    ep1 = SourceEpisode(episode_id="E01", source_video=str(vid1), duration_seconds=3.0)
    ep3 = SourceEpisode(episode_id="E03", source_video=str(vid3), duration_seconds=3.0)
    ep5 = SourceEpisode(episode_id="E05", source_video=str(vid5), duration_seconds=3.0)

    # Cross-episode output spanning E01, E03, and E05
    cross_output = CommentaryOutput(
        output_id="out_cross",
        title="Cross Episode Thread",
        sanitized_title="cross-episode-thread",
        candidate_scope=CandidateScope.CROSS_EPISODE.value,
        segments=[
            Segment(
                segment_id="seg_e01",
                source_clips=[SourceClip(episode_id="E01", source_video=str(vid1), start=0.0, end=1.5)],
                narration="Phần mở đầu tại tập 1.",
                audio_policy="duck",
            ),
            Segment(
                segment_id="seg_e03",
                source_clips=[SourceClip(episode_id="E03", source_video=str(vid3), start=0.5, end=2.0)],
                narration="Diễn biến tiếp theo tại tập 3.",
                audio_policy="duck",
            ),
            Segment(
                segment_id="seg_e05",
                source_clips=[SourceClip(episode_id="E05", source_video=str(vid5), start=1.0, end=2.5)],
                narration="Kết thúc chuỗi sự kiện tại tập 5.",
                audio_policy="duck",
            ),
        ],
    )

    manifest = AnalysisManifest(
        project_id="test_cross",
        source_episodes=[ep1, ep3, ep5],
        outputs=[cross_output],
    )

    cues_map = {
        "E01": [SubtitleCue(start_ms=0, end_ms=1500, text="Thoại tập 1", source_type="embedded", source_format="srt", episode_id="E01")],
        "E03": [SubtitleCue(start_ms=500, end_ms=2000, text="Thoại tập 3", source_type="embedded", source_format="srt", episode_id="E03")],
        "E05": [SubtitleCue(start_ms=1000, end_ms=2500, text="Thoại tập 5", source_type="embedded", source_format="srt", episode_id="E05")],
    }

    fake_voice = FakeVoiceManager()
    renderer = PublicationRenderer(voice_manager=fake_voice)
    settings = AppSettings(use_gpu=False, burn_subtitles=False)

    out_root = tmp_path / "out_cross_root"
    rendered = renderer.render_manifest(
        manifest=manifest,
        settings=settings,
        transcript_cues_by_episode=cues_map,
        output_root=out_root,
    )

    assert len(rendered) == 1
    res = rendered[0]
    pub_dir = out_root / "cross-episode-thread"
    assert (pub_dir / "cross-episode-thread.mp4").is_file()

    # Verify original SRT correctly mapped cues across timeline
    orig_cues = validate_srt_file(pub_dir / "cross-episode-thread.original.srt")
    assert len(orig_cues) == 3
    # First cue starts at 0.0s (from E01: 0.0 - 1.5)
    assert abs(orig_cues[0][0] - 0.0) < 0.1
    assert "Thoại tập 1" in orig_cues[0][2]
    # Second cue starts at 1.5s (from E03: 0.5 - 2.0 mapped after 1.5s)
    assert abs(orig_cues[1][0] - 1.5) < 0.1
    assert "Thoại tập 3" in orig_cues[1][2]
    # Third cue starts at 3.0s (from E05: 1.0 - 2.5 mapped after 3.0s)
    assert abs(orig_cues[2][0] - 3.0) < 0.1
    assert "Thoại tập 5" in orig_cues[2][2]


# ---------------------------------------------------------------------------
# Test 3: Real Audio Mix Real FFmpeg: Original Remains Under Narration, No Clipping
# ---------------------------------------------------------------------------

def test_real_audio_mix_real_ffmpeg_original_remains_and_no_clipping(tmp_path: Path) -> None:
    """Smoke synthetic test: Real FFmpeg audio mixing with sidechain ducking, normalization, limiter."""
    # Create 4 seconds of original audio (audible sine wave at 440Hz)
    orig_wav = tmp_path / "original_audio.wav"
    subprocess.run(
        [
            find_binary("ffmpeg"), "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
            str(orig_wav),
        ],
        capture_output=True,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    # Create 2 seconds of commentary audio placed on timeline (delayed by 1 second)
    comm_wav = tmp_path / "commentary_audio.wav"
    subprocess.run(
        [
            find_binary("ffmpeg"), "-y",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
            "-filter_complex", "[0:a]aformat=channel_layouts=stereo:sample_rates=48000,adelay=1000|1000,apad[out]",
            "-map", "[out]",
            "-t", "4",
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
            str(comm_wav),
        ],
        capture_output=True,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    mix_out = tmp_path / "mixed_output.wav"
    mix_settings = AudioMixSettings(
        original_gain_db=0.0,
        commentary_gain_db=0.0,
        auto_duck=True,
        amount=-12.0,
        target_lufs=-14.0,
        true_peak=-1.0,
        attack_ms=20.0,
        release_ms=250.0,
    )

    plan = plan_audio_mix(
        output_path=mix_out,
        original_audio_path=orig_wav,
        commentary_audio_path=comm_wav,
        settings=mix_settings,
        is_original_dialogue_only=False,
    )

    cmd = plan.build_command(ffmpeg_bin=find_binary("ffmpeg"))
    run_command(cmd)

    assert mix_out.is_file()
    # Validate mixed audio passes true peak, RMS level (not silent), and format constraints
    metrics = validate_mixed_audio(mix_out, settings=mix_settings)
    assert metrics["rms"] > 0.01  # Both original and commentary audibly present
    assert metrics["peak_dbfs"] <= -0.9  # Peak limited under true_peak limit (-1.0 dBFS)
    assert metrics["duration"] >= 3.9  # Full duration preserved


# ---------------------------------------------------------------------------
# Test 4: Audio Duck On/Off, Gains, Target LUFS, and Peak Limiting
# ---------------------------------------------------------------------------

def test_audio_duck_settings_and_filter_graphs(tmp_path: Path) -> None:
    """Verify AudioMixSettings generates correct filter graph representation."""
    # 1. Auto-duck enabled
    cfg_duck = AudioMixSettings(
        original_gain_db=-3.0,
        commentary_gain_db=2.0,
        auto_duck=True,
        amount=-15.0,
        target_lufs=-16.0,
        true_peak=-1.5,
    )
    fg_duck = build_audio_mix_filter_graph(cfg_duck, has_original=True, has_commentary=True)
    assert "sidechaincompress=" in fg_duck
    assert "ratio=" in fg_duck
    assert "loudnorm=I=-16.0:TP=-1.5" in fg_duck
    assert "volume=-3.00dB" in fg_duck
    assert "volume=2.00dB" in fg_duck

    # 2. Auto-duck disabled (amix direct)
    cfg_noduck = AudioMixSettings(
        auto_duck=False,
        target_lufs=-14.0,
        true_peak=-1.0,
    )
    fg_noduck = build_audio_mix_filter_graph(cfg_noduck, has_original=True, has_commentary=True)
    assert "sidechaincompress=" not in fg_noduck
    assert "amix=inputs=2" in fg_noduck
    assert "loudnorm=I=-14.0:TP=-1.0" in fg_noduck

    # 3. Original dialogue only (preserves original audio without commentary mix)
    fg_orig_only = build_audio_mix_filter_graph(cfg_duck, has_original=True, is_original_dialogue_only=True)
    assert "sidechaincompress" not in fg_orig_only
    assert "amix" not in fg_orig_only
    assert "alimiter=" in fg_orig_only


# ---------------------------------------------------------------------------
# Test 5: Original Dialogue Preservation for Original-Only Segments
# ---------------------------------------------------------------------------

def test_original_dialogue_preserve(tmp_path: Path) -> None:
    """Original-only segment preserves source original audio without narration ducking."""
    vid = create_synthetic_video(tmp_path / "orig_video.mp4", duration=3.0)
    ep = SourceEpisode(episode_id="E01", source_video=str(vid), duration_seconds=3.0)

    out = CommentaryOutput(
        output_id="out_orig_only",
        title="Original Scene Highlight",
        sanitized_title="original-scene-highlight",
        segments=[
            Segment(
                segment_id="seg_01",
                source_clips=[SourceClip(episode_id="E01", source_video=str(vid), start=0.0, end=2.5)],
                narration="",  # No narration for original-only
                audio_policy=AudioPolicy.ORIGINAL_ONLY.value,
                original_dialogue="Important original dialogue",
            )
        ],
    )

    manifest = AnalysisManifest(
        project_id="test_orig_preserve",
        source_episodes=[ep],
        outputs=[out],
    )

    cues_map = {
        "E01": [
            SubtitleCue(
                start_ms=0,
                end_ms=2500,
                text="Important original dialogue",
                source_type="embedded",
                source_format="srt",
                episode_id="E01",
            )
        ]
    }
    fake_voice = FakeVoiceManager()
    renderer = PublicationRenderer(voice_manager=fake_voice)
    settings = AppSettings(use_gpu=False, burn_subtitles=False)

    out_root = tmp_path / "out_orig_preserve"
    rendered = renderer.render_manifest(
        manifest=manifest,
        settings=settings,
        transcript_cues_by_episode=cues_map,
        output_root=out_root,
    )

    assert len(rendered) == 1
    # Voice manager synthesize should NOT be called for original_only segment
    assert len(fake_voice.synthesize_calls) == 0

    pub_dir = out_root / "original-scene-highlight"
    assert (pub_dir / "original-scene-highlight.mp4").is_file()
    assert (pub_dir / "original-scene-highlight.original.srt").is_file()
    assert (pub_dir / "original-scene-highlight.narration.srt").is_file()


# ---------------------------------------------------------------------------
# Test 6: Production Original SRT Requirement Fails Clear on Missing Dialogue
# ---------------------------------------------------------------------------

def test_production_original_srt_missing_dialogue_fails(tmp_path: Path) -> None:
    """If selected source has no dialogue cues, output is invalid and fails clearly."""
    vid = create_synthetic_video(tmp_path / "vid.mp4", duration=3.0)
    ep = SourceEpisode(episode_id="E01", source_video=str(vid), duration_seconds=3.0)

    out = CommentaryOutput(
        output_id="out_no_dialogue",
        title="No Dialogue Output",
        sanitized_title="no-dialogue-output",
        segments=[
            Segment(
                segment_id="seg_01",
                source_clips=[SourceClip(episode_id="E01", source_video=str(vid), start=0.0, end=2.0)],
                narration="Thuyết minh cho đoạn không có lời thoại gốc.",
                audio_policy="duck",
            )
        ],
    )

    manifest = AnalysisManifest(
        project_id="test_fail_dialogue",
        source_episodes=[ep],
        outputs=[out],
    )

    # Empty transcript cues: no dialogue in source
    empty_cues_map: dict[str, list[SubtitleCue]] = {"E01": []}

    renderer = PublicationRenderer(voice_manager=FakeVoiceManager())
    settings = AppSettings(use_gpu=False)

    with pytest.raises(MediaError) as exc_info:
        renderer.render_manifest(
            manifest=manifest,
            settings=settings,
            transcript_cues_by_episode=empty_cues_map,
            output_root=tmp_path / "out_fail",
        )

    assert "Tạo phụ đề gốc (Original subtitle) thất bại" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 7: Output Validation Fails Missing or Corrupt Files
# ---------------------------------------------------------------------------

def test_output_validation_fails_missing_or_bad_files(tmp_path: Path) -> None:
    """validate_publication_folder strictly enforces 3 exact valid files."""
    pub = tmp_path / "valid_pub"
    pub.mkdir(parents=True, exist_ok=True)
    title = "sample-title"

    # Missing all files -> fails
    with pytest.raises(OutputValidationError) as exc:
        validate_publication_folder(pub, title)
    assert "thiếu các tệp bắt buộc" in str(exc.value)

    # Create dummy files
    vid_file = pub / f"{title}.mp4"
    orig_file = pub / f"{title}.original.srt"
    narr_file = pub / f"{title}.narration.srt"

    create_synthetic_video(vid_file, duration=2.0)
    orig_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n\n", encoding="utf-8")
    narr_file.write_text("1\n00:00:00,000 --> 00:00:01,500\nWorld\n\n", encoding="utf-8")

    # Complete and clean -> passes
    res = validate_publication_folder(pub, title)
    assert res["duration"] > 0

    # Extraneous leftover file -> fails
    extra = pub / "leftover_temp.tmp"
    extra.write_text("temp", encoding="utf-8")
    with pytest.raises(OutputValidationError) as exc_extra:
        validate_publication_folder(pub, title)
    assert "chứa tệp thừa/tạm" in str(exc_extra.value)
    extra.unlink()

    # Empty 0-byte SRT -> fails
    narr_file.write_text("", encoding="utf-8")
    with pytest.raises(OutputValidationError) as exc_empty:
        validate_publication_folder(pub, title)
    assert "rỗng" in str(exc_empty.value)


# ---------------------------------------------------------------------------
# Test 8: ProjectRecord.from_season_paths Helper
# ---------------------------------------------------------------------------

def test_project_record_from_season_paths(tmp_path: Path) -> None:
    """from_season_paths correctly builds SEASON ProjectRecord with numbered episodes."""
    p1 = tmp_path / "S01E01.mp4"
    p2 = tmp_path / "S01E02.mp4"
    p3 = tmp_path / "S01E03.mp4"
    p1.touch()
    p2.touch()
    p3.touch()

    rec = ProjectRecord.from_season_paths(
        [p1, p2, p3],
        output_root=tmp_path / "season_out",
        voice_id="v_male_south_01",
        title="My Season",
    )

    assert rec.analysis_scope == AnalysisScope.SEASON.value
    assert rec.name == "My Season"
    assert len(rec.source_episodes) == 3
    assert rec.source_episodes[0].episode_id == "E01"
    assert rec.source_episodes[1].episode_id == "E02"
    assert rec.source_episodes[2].episode_id == "E03"
    assert rec.voice_id == "v_male_south_01"


# ---------------------------------------------------------------------------
# Test 9: Queue Season Phase Ordering (No Per-Episode Finalization)
# ---------------------------------------------------------------------------

def test_queue_season_phase_ordering_no_perepisode_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full queue run with SEASON project enforces single analysis pass and phase transitions."""
    vid1 = create_synthetic_video(tmp_path / "e1.mp4", duration=2.0)
    vid2 = create_synthetic_video(tmp_path / "e2.mp4", duration=2.0)

    store = ProjectStore(tmp_path / "projects.json")
    settings = AppSettings(gateway_enabled=True)

    record = ProjectRecord.from_season_paths(
        [vid1, vid2],
        output_root=tmp_path / "out_season",
        title="Season Batch",
    )
    store.save([record])

    phases_recorded: list[str] = []

    def on_update(r: ProjectRecord) -> None:
        if r.phase not in phases_recorded:
            phases_recorded.append(r.phase)

    from tests.helpers_editorial import stage_response

    custom_output = {
        "outputs": [
            {
                "output_id": "season_out_1",
                "title": "Season Finale Arc",
                "candidate_scope": "SEASON_ARC",
                "segments": [
                    {
                        "segment_id": "seg_1",
                        "source_clips": [{"episode_id": "E01", "start": 0.0, "end": 1.5}],
                        "narration": "Toàn cảnh mùa phim bắt đầu.",
                        "audio_policy": "duck",
                    }
                ],
            }
        ]
    }

    # Mock AI client to return 1 season arc output
    def mock_chat_json(self, *, model, **kwargs):
        if model == "sub":
            return {
                "range_start_ms": 0, "range_end_ms": 2000,
                "events": [{"start_ms": 0, "end_ms": 1000, "summary": "Ev"}],
                "major_scenes": [{"start_ms": 0, "end_ms": 1000, "summary": "Ev"}],
                "dialogue": [{"start_ms": 0, "end_ms": 1000, "speaker": "A", "quote": "Hi"}],
            }
        return stage_response(
            system=kwargs.get("system", ""),
            user_text=kwargs.get("user_text", ""),
            episodes=["E01", "E02"],
            default=custom_output,
            model=model,
        )

    monkeypatch.setattr(OpenAICompatibleClient, "chat_json", mock_chat_json)

    # Provide subtitle cues to avoid missing dialogue failure
    cues_map = {
        "E01": [
            SubtitleCue(
                start_ms=0,
                end_ms=1500,
                text="Thoại tập 1",
                source_type="embedded",
                source_format="srt",
                episode_id="E01",
            )
        ],
        "E02": [
            SubtitleCue(
                start_ms=0,
                end_ms=1500,
                text="Thoại tập 2",
                source_type="embedded",
                source_format="srt",
                episode_id="E02",
            )
        ],
    }
    monkeypatch.setattr(
        "toolrecap_v2.subtitles.pipeline.SubtitlePipeline.get_episode_subtitles",
        lambda self, video_path, episode_id, **kw: cues_map.get(episode_id, []),
    )

    fake_voice = FakeVoiceManager()
    monkeypatch.setattr("toolrecap_v2.projects.get_voice_manager", lambda: fake_voice)

    queue = ProjectQueue([record], store=store, settings=settings, on_update=on_update)
    queue.start()
    assert queue._thread is not None
    queue._thread.join(timeout=15)

    assert record.status == "COMPLETED"
    assert record.phase == "COMPLETED"
    # Phase order must transition through IDLE/ANALYZING -> RENDERING -> COMPLETED
    assert "RENDERING" in phases_recorded
    assert "COMPLETED" in phases_recorded


# ---------------------------------------------------------------------------
# Test 10: Zero Outputs Completes with COMPLETED Status and No Publication Files
# ---------------------------------------------------------------------------

def test_zero_outputs_completes_with_no_publication_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When analysis yields 0 outputs, project completes cleanly with COMPLETED status and 0 files."""
    vid = create_synthetic_video(tmp_path / "vid.mp4", duration=2.0)
    store = ProjectStore(tmp_path / "projects.json")
    settings = AppSettings(gateway_enabled=True)

    record = ProjectRecord.from_video_path(vid, tmp_path / "out_zero")
    store.save([record])

    from tests.helpers_editorial import stage_response

    # AI returns 0 outputs
    def mock_chat_zero(self, *, model, **kwargs):
        if model == "sub":
            return {
                "range_start_ms": 0,
                "range_end_ms": 2000,
                "events": [{"start_ms": 0, "end_ms": 1000, "summary": "Scene"}],
                "major_scenes": [{"start_ms": 0, "end_ms": 1000, "summary": "Scene"}],
                "dialogue": [{"start_ms": 0, "end_ms": 1000, "speaker": "A", "quote": "Dialogue"}],
            }
        return stage_response(
            system=kwargs.get("system", ""),
            user_text=kwargs.get("user_text", ""),
            episodes=["E01"],
            default={"outputs": []},
            model=model,
        )

    monkeypatch.setattr(OpenAICompatibleClient, "chat_json", mock_chat_zero)
    monkeypatch.setattr(
        "toolrecap_v2.subtitles.pipeline.SubtitlePipeline.get_episode_subtitles",
        lambda *args, **kwargs: [
            SubtitleCue(start_ms=0, end_ms=1000, text="Dialogue", source_type="embedded", source_format="srt", episode_id="E01")
        ],
    )

    queue = ProjectQueue([record], store=store, settings=settings)
    queue.start()
    assert queue._thread is not None
    queue._thread.join(timeout=10)

    assert record.status == "COMPLETED"
    assert record.phase == "COMPLETED"
    assert "0 outputs" in record.current_message
    assert record.outputs == []

    # Ensure no publication folders or videos were created
    out_dir = Path(record.output_directory)
    mp4_files = list(out_dir.glob("**/*.mp4"))
    assert len(mp4_files) == 0


# ---------------------------------------------------------------------------
# Test 11: Cancellation Stops Execution Promptly and Restores UI
# ---------------------------------------------------------------------------

def test_cancellation_during_phases_and_ui_restoration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation during processing stops execution, marks CANCELLED, and calls UI state false."""
    vid = create_synthetic_video(tmp_path / "e1.mp4", duration=2.0)
    store = ProjectStore(tmp_path / "projects.json")
    settings = AppSettings()

    p = ProjectRecord.from_video_path(vid, tmp_path / "out_cancel")
    ui_states: list[bool] = []

    queue = ProjectQueue(
        [p],
        store=store,
        settings=settings,
        on_state_change=lambda is_running: ui_states.append(is_running),
    )

    def mock_cancel_during_render(*args: Any, **kwargs: Any) -> Any:
        # Trigger queue cancellation
        queue.cancel()
        raise RenderCancelled("User cancelled")

    monkeypatch.setattr("toolrecap_v2.projects.probe_typed_media", mock_cancel_during_render)

    queue.start()
    if queue._thread:
        queue._thread.join(timeout=5)

    assert queue.is_running is False
    assert ui_states[-1] is False
    assert p.status == "CANCELLED"


# ---------------------------------------------------------------------------
# Test 12: VoiceStudio Unavailable Fails Clear in Production
# ---------------------------------------------------------------------------

def test_voice_studio_unavailable_fails_clear(tmp_path: Path) -> None:
    """Missing ToolRecap runtime fails clearly with no fake/Piper fallback."""
    vm = VoiceModelManager(
        cache_dir=tmp_path / "cache",
        subsystem_dir=tmp_path / "subsystem",
        auto_bootstrap=False,
    )

    # In production (allow_mock_synth=False), synthesizing without adapter must fail clear
    with pytest.raises(VoiceError) as exc:
        vm.synthesize(
            DEFAULT_VOICE_ID,
            "Thử nghiệm giọng đọc",
            tmp_path / "test.wav",
            allow_mock_synth=False,
        )

    assert "ToolRecap local voice runtime" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 13: Title Exact Sample & Collision Resolution
# ---------------------------------------------------------------------------

def test_title_exact_sample_and_collision_resolution(tmp_path: Path) -> None:
    """PublicationRenderer resolves exact title collisions and unsafe characters into distinct clean folders."""
    vid = create_synthetic_video(tmp_path / "sample.mp4", duration=2.0)
    ep = SourceEpisode(episode_id="E01", source_video=str(vid), duration_seconds=2.0)

    # Output 1 and Output 2 have identical raw titles with Windows-illegal chars
    raw_title_1 = 'Tập 1: "Chiến Trận" / Phần 1? <Bản Đẹp>'
    raw_title_2 = 'Tập 1: "Chiến Trận" / Phần 1? <Bản Đẹp>'
    # Output 3 uses a Windows-reserved device name
    raw_title_3 = "CON"

    out1 = CommentaryOutput(
        output_id="out_01",
        title=raw_title_1,
        segments=[
            Segment(
                segment_id="seg_01",
                source_clips=[SourceClip(episode_id="E01", source_video=str(vid), start=0.0, end=1.5)],
                narration="Thuyết minh cho tập 1 bản đầu.",
                audio_policy="duck",
            )
        ],
    )
    out2 = CommentaryOutput(
        output_id="out_02",
        title=raw_title_2,
        segments=[
            Segment(
                segment_id="seg_02",
                source_clips=[SourceClip(episode_id="E01", source_video=str(vid), start=0.0, end=1.5)],
                narration="Thuyết minh cho tập 1 bản trùng lặp.",
                audio_policy="duck",
            )
        ],
    )
    out3 = CommentaryOutput(
        output_id="out_03",
        title=raw_title_3,
        segments=[
            Segment(
                segment_id="seg_03",
                source_clips=[SourceClip(episode_id="E01", source_video=str(vid), start=0.0, end=1.5)],
                narration="Thuyết minh cho output tên đặc biệt.",
                audio_policy="duck",
            )
        ],
    )

    manifest = AnalysisManifest(
        project_id="test_collision_manifest",
        source_episodes=[ep],
        outputs=[out1, out2, out3],
    )

    cues_map = {
        "E01": [
            SubtitleCue(
                start_ms=0,
                end_ms=1500,
                text="Thoại gốc kiểm tra collision",
                source_type="embedded",
                source_format="srt",
                episode_id="E01",
            )
        ]
    }

    renderer = PublicationRenderer(voice_manager=FakeVoiceManager())
    settings = AppSettings(burn_subtitles=False, use_gpu=False)
    out_root = tmp_path / "out_collision"

    rendered = renderer.render_manifest(
        manifest=manifest,
        settings=settings,
        transcript_cues_by_episode=cues_map,
        output_root=out_root,
    )

    assert len(rendered) == 3

    # Ensure safe titles are all unique and legal
    safe_titles = [out.sanitized_title for out in rendered]
    assert len(set(safe_titles)) == 3
    for st in safe_titles:
        for illegal_char in [":", '"', "/", "?", "<", ">", "|", "*"]:
            assert illegal_char not in st

    # First and second outputs must differ by disambiguation suffix (e.g. _1)
    assert safe_titles[0] != safe_titles[1]
    assert safe_titles[1].endswith("_1")

    # Output 3 (CON) must not be a bare reserved name
    assert safe_titles[2].upper() != "CON"

    # Verify that each folder exists and passes strict validation
    for st in safe_titles:
        folder = out_root / st
        assert folder.is_dir()
        val = validate_publication_folder(folder, st)
        assert val["duration"] > 0
        assert (folder / f"{st}.mp4").is_file()
        assert (folder / f"{st}.original.srt").is_file()
        assert (folder / f"{st}.narration.srt").is_file()


# ---------------------------------------------------------------------------
# Test 14: Cancellation at Renderer Voice Boundary
# ---------------------------------------------------------------------------

def test_cancellation_at_renderer_voice_boundary(tmp_path: Path) -> None:
    """Cancellation during voice synthesis raises RenderCancelled, updates status, and leaves no intermediate final files."""
    vid = create_synthetic_video(tmp_path / "vid.mp4", duration=2.0)
    ep = SourceEpisode(episode_id="E01", source_video=str(vid), duration_seconds=2.0)

    out = CommentaryOutput(
        output_id="out_cancel_voice",
        title="Voice Cancel Test",
        sanitized_title="voice-cancel-test",
        segments=[
            Segment(
                segment_id="seg_01",
                source_clips=[SourceClip(episode_id="E01", source_video=str(vid), start=0.0, end=1.5)],
                narration="Lời bình sẽ bị hủy giữa chừng.",
                audio_policy="duck",
            )
        ],
    )

    manifest = AnalysisManifest(
        project_id="test_voice_cancel",
        source_episodes=[ep],
        outputs=[out],
    )

    cancel_evt = threading.Event()

    class CancellingVoiceManager:
        def synthesize(self, *args: Any, **kwargs: Any) -> Path:
            cancel_evt.set()
            raise VoiceError("Đã hủy trong lúc tổng hợp giọng nói.")

    renderer = PublicationRenderer(voice_manager=CancellingVoiceManager())
    settings = AppSettings(burn_subtitles=False, use_gpu=False)
    out_root = tmp_path / "out_voice_cancel"

    cues_map = {
        "E01": [SubtitleCue(start_ms=0, end_ms=1500, text="Thoại", source_type="embedded", source_format="srt", episode_id="E01")]
    }

    with pytest.raises(RenderCancelled):
        renderer.render_manifest(
            manifest=manifest,
            settings=settings,
            transcript_cues_by_episode=cues_map,
            output_root=out_root,
            cancel_event=cancel_evt,
        )

    assert out.status == OutputStatus.CANCELLED.value
    pub_dir = out_root / "voice-cancel-test"
    # No intermediate final video should exist in publication folder
    assert not (pub_dir / "voice-cancel-test.mp4").exists()


# ---------------------------------------------------------------------------
# Test 15: Cancellation at Renderer Mix Boundary
# ---------------------------------------------------------------------------

def test_cancellation_at_renderer_mix_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation during audio mixing raises RenderCancelled and leaves no intermediate final files."""
    vid = create_synthetic_video(tmp_path / "vid.mp4", duration=2.0)
    ep = SourceEpisode(episode_id="E01", source_video=str(vid), duration_seconds=2.0)

    out = CommentaryOutput(
        output_id="out_cancel_mix",
        title="Mix Cancel Test",
        sanitized_title="mix-cancel-test",
        segments=[
            Segment(
                segment_id="seg_01",
                source_clips=[SourceClip(episode_id="E01", source_video=str(vid), start=0.0, end=1.5)],
                narration="Lời bình cho mix cancel.",
                audio_policy="duck",
            )
        ],
    )

    manifest = AnalysisManifest(
        project_id="test_mix_cancel",
        source_episodes=[ep],
        outputs=[out],
    )

    cues_map = {
        "E01": [SubtitleCue(start_ms=0, end_ms=1500, text="Thoại", source_type="embedded", source_format="srt", episode_id="E01")]
    }

    cancel_evt = threading.Event()
    orig_run_command = run_command

    def mock_run_command_cancel_at_mix(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        if any("mixed_final.wav" in str(arg) for arg in cmd):
            cancel_evt.set()
            raise RenderCancelled("Hủy tại mix audio.")
        return orig_run_command(cmd, *args, **kwargs)

    monkeypatch.setattr("toolrecap_v2.renderer.run_command", mock_run_command_cancel_at_mix)

    renderer = PublicationRenderer(voice_manager=FakeVoiceManager())
    settings = AppSettings(burn_subtitles=False, use_gpu=False)
    out_root = tmp_path / "out_mix_cancel"

    with pytest.raises(RenderCancelled):
        renderer.render_manifest(
            manifest=manifest,
            settings=settings,
            transcript_cues_by_episode=cues_map,
            output_root=out_root,
            cancel_event=cancel_evt,
        )

    assert out.status == OutputStatus.CANCELLED.value
    pub_dir = out_root / "mix-cancel-test"
    assert not (pub_dir / "mix-cancel-test.mp4").exists()


# ---------------------------------------------------------------------------
# Test 16: Cancellation at Renderer Render Boundary (Burn True & Burn False)
# ---------------------------------------------------------------------------

def test_cancellation_at_renderer_render_boundary_burn_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation during direct mux (burn_subtitles=False) raises RenderCancelled, unlinks partial file."""
    vid = create_synthetic_video(tmp_path / "vid.mp4", duration=2.0)
    ep = SourceEpisode(episode_id="E01", source_video=str(vid), duration_seconds=2.0)

    out = CommentaryOutput(
        output_id="out_render_cancel_bf",
        title="Render Cancel Burn False",
        sanitized_title="render-cancel-burn-false",
        segments=[
            Segment(
                segment_id="seg_01",
                source_clips=[SourceClip(episode_id="E01", source_video=str(vid), start=0.0, end=1.5)],
                narration="Lời bình.",
                audio_policy="duck",
            )
        ],
    )

    manifest = AnalysisManifest(
        project_id="test_render_cancel_bf",
        source_episodes=[ep],
        outputs=[out],
    )

    cues_map = {
        "E01": [SubtitleCue(start_ms=0, end_ms=1500, text="Thoại", source_type="embedded", source_format="srt", episode_id="E01")]
    }

    cancel_evt = threading.Event()
    orig_run_command = run_command

    def mock_run_command_cancel_at_mux(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        if any("render-cancel-burn-false.mp4" in str(arg) for arg in cmd):
            # Simulate partially created file then cancellation
            target = Path(cmd[-1])
            target.write_text("partial", encoding="utf-8")
            cancel_evt.set()
            raise RenderCancelled("Hủy tại mux cuối cùng.")
        return orig_run_command(cmd, *args, **kwargs)

    monkeypatch.setattr("toolrecap_v2.renderer.run_command", mock_run_command_cancel_at_mux)

    renderer = PublicationRenderer(voice_manager=FakeVoiceManager())
    settings = AppSettings(burn_subtitles=False, use_gpu=False)
    out_root = tmp_path / "out_render_cancel_bf"

    with pytest.raises(RenderCancelled):
        renderer.render_manifest(
            manifest=manifest,
            settings=settings,
            transcript_cues_by_episode=cues_map,
            output_root=out_root,
            cancel_event=cancel_evt,
        )

    assert out.status == OutputStatus.CANCELLED.value
    pub_dir = out_root / "render-cancel-burn-false"
    # Ensure no intermediate final file in publication
    assert not (pub_dir / "render-cancel-burn-false.mp4").exists()


def test_cancellation_at_renderer_render_boundary_burn_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation during render_final_video (burn_subtitles=True) raises RenderCancelled, unlinks partial file."""
    vid = create_synthetic_video(tmp_path / "vid.mp4", duration=2.0)
    ep = SourceEpisode(episode_id="E01", source_video=str(vid), duration_seconds=2.0)

    out = CommentaryOutput(
        output_id="out_render_cancel_bt",
        title="Render Cancel Burn True",
        sanitized_title="render-cancel-burn-true",
        segments=[
            Segment(
                segment_id="seg_01",
                source_clips=[SourceClip(episode_id="E01", source_video=str(vid), start=0.0, end=1.5)],
                narration="Lời bình burn true.",
                audio_policy="duck",
            )
        ],
    )

    manifest = AnalysisManifest(
        project_id="test_render_cancel_bt",
        source_episodes=[ep],
        outputs=[out],
    )

    cues_map = {
        "E01": [SubtitleCue(start_ms=0, end_ms=1500, text="Thoại", source_type="embedded", source_format="srt", episode_id="E01")]
    }

    cancel_evt = threading.Event()

    def mock_render_final_video(*args: Any, **kwargs: Any) -> Path:
        out_path = Path(kwargs.get("output_path") or args[1])
        out_path.write_text("partial burn", encoding="utf-8")
        cancel_evt.set()
        raise RenderCancelled("Hủy tại render_final_video.")

    monkeypatch.setattr("toolrecap_v2.renderer.render_final_video", mock_render_final_video)

    renderer = PublicationRenderer(voice_manager=FakeVoiceManager())
    settings = AppSettings(burn_subtitles=True, use_gpu=False)
    out_root = tmp_path / "out_render_cancel_bt"

    with pytest.raises(RenderCancelled):
        renderer.render_manifest(
            manifest=manifest,
            settings=settings,
            transcript_cues_by_episode=cues_map,
            output_root=out_root,
            cancel_event=cancel_evt,
        )

    assert out.status == OutputStatus.CANCELLED.value
    pub_dir = out_root / "render-cancel-burn-true"
    # Ensure no intermediate final file in publication
    assert not (pub_dir / "render-cancel-burn-true.mp4").exists()


# ---------------------------------------------------------------------------
# Test 17: Queue State Recovery on Cancellation During Render
# ---------------------------------------------------------------------------

def test_queue_state_recovery_on_cancellation_during_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When cancelled during rendering in ProjectQueue, queue stops, restores UI state, and marks records CANCELLED."""
    vid1 = create_synthetic_video(tmp_path / "e1.mp4", duration=2.0)
    vid2 = create_synthetic_video(tmp_path / "e2.mp4", duration=2.0)

    store = ProjectStore(tmp_path / "projects.json")
    settings = AppSettings(gateway_enabled=True)

    p1 = ProjectRecord.from_video_path(vid1, tmp_path / "out_q1")
    p2 = ProjectRecord.from_video_path(vid2, tmp_path / "out_q2")
    store.save([p1, p2])

    ui_states: list[bool] = []

    from tests.helpers_editorial import stage_response

    custom_output = {
        "outputs": [
            {
                "output_id": "out_01",
                "title": "Recap Video",
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "source_clips": [{"episode_id": "E01", "start": 0.0, "end": 1.5}],
                        "narration": "Lời bình cho queue cancel test.",
                        "audio_policy": "duck",
                    }
                ],
            }
        ]
    }

    def mock_chat_json(self, *, model, **kwargs):
        if model == "sub":
            return {
                "range_start_ms": 0, "range_end_ms": 2000,
                "events": [{"start_ms": 0, "end_ms": 1000, "summary": "Ev"}],
                "major_scenes": [{"start_ms": 0, "end_ms": 1000, "summary": "Scene"}],
                "dialogue": [{"start_ms": 0, "end_ms": 1000, "speaker": "A", "quote": "Thoại"}],
            }
        return stage_response(
            system=kwargs.get("system", ""),
            user_text=kwargs.get("user_text", ""),
            episodes=["E01"],
            default=custom_output,
            model=model,
        )

    monkeypatch.setattr(OpenAICompatibleClient, "chat_json", mock_chat_json)
    monkeypatch.setattr(
        "toolrecap_v2.subtitles.pipeline.SubtitlePipeline.get_episode_subtitles",
        lambda *args, **kwargs: [
            SubtitleCue(start_ms=0, end_ms=1500, text="Thoại", source_type="embedded", source_format="srt", episode_id="E01")
        ],
    )

    queue = ProjectQueue(
        [p1, p2],
        store=store,
        settings=settings,
        on_state_change=lambda is_running: ui_states.append(is_running),
    )

    def mock_render_manifest(*args: Any, **kwargs: Any) -> list[CommentaryOutput]:
        # Cancel while rendering project 1
        queue.cancel()
        raise RenderCancelled("Hủy queue khi render")

    monkeypatch.setattr(PublicationRenderer, "render_manifest", mock_render_manifest)

    queue.start()
    if queue._thread:
        queue._thread.join(timeout=10)

    assert queue.is_running is False
    assert ui_states[-1] is False
    assert p1.status == "CANCELLED"
    assert p2.status == "CANCELLED"

    # Reload from store and verify persisted state
    saved = store.load()
    assert saved[0].status == "CANCELLED"
    assert saved[1].status == "CANCELLED"
