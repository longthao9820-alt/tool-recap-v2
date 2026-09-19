"""Tests for domain schema, dataclasses, enums, validation, and JSON serialization."""
from __future__ import annotations

import json
import pytest

from toolrecap_v2.domain import (
    AnalysisManifest,
    AnalysisScope,
    AudioPolicy,
    CandidateScope,
    CommentaryOutput,
    EpisodeEvidence,
    MediaSelection,
    OutputStatus,
    ProjectPhase,
    Segment,
    SourceClip,
    SourceEpisode,
    ValidationError,
)


def test_enum_exact_values() -> None:
    # AnalysisScope
    assert AnalysisScope.SINGLE_EPISODE.value == "SINGLE_EPISODE"
    assert AnalysisScope.SEASON.value == "SEASON"

    # CandidateScope exact values
    assert CandidateScope.SINGLE_SCENE.value == "SINGLE_SCENE"
    assert CandidateScope.SINGLE_EPISODE.value == "SINGLE_EPISODE"
    assert CandidateScope.CROSS_EPISODE.value == "CROSS_EPISODE"
    assert CandidateScope.SEASON_ARC.value == "SEASON_ARC"

    # AudioPolicy
    assert AudioPolicy.MUTE.value == "mute"
    assert AudioPolicy.DUCK.value == "duck"
    assert AudioPolicy.KEEP.value == "keep"
    assert AudioPolicy.ORIGINAL_ONLY.value == "original_only"

    # ProjectPhase & OutputStatus
    assert ProjectPhase.IDLE.value == "IDLE"
    assert OutputStatus.WAITING.value == "WAITING"


def test_dataclasses_roundtrip_serialization() -> None:
    media = MediaSelection(
        video_path="video.mp4",
        audio_track=1,
        subtitle_track=2,
        start_seconds=10.0,
        end_seconds=100.0,
        extra_info={"lang": "en"},
    )
    media_dict = media.to_dict()
    media_restored = MediaSelection.from_dict(media_dict)
    assert media_restored == media

    ep1 = SourceEpisode(
        episode_id="E01",
        source_video="C:/media/e01.mp4",
        season_number=1,
        episode_number=1,
        title="Pilot",
        duration_seconds=3000.0,
        media_selection=media,
    )
    ep1_dict = ep1.to_dict()
    ep1_restored = SourceEpisode.from_dict(ep1_dict)
    assert ep1_restored.episode_id == ep1.episode_id
    assert ep1_restored.media_selection is not None
    assert ep1_restored.media_selection.video_path == "video.mp4"

    clip = SourceClip(
        episode_id="E01",
        source_video="C:/media/e01.mp4",
        start=50.0,
        end=75.0,
    )
    assert clip.duration == 25.0
    clip_dict = clip.to_dict()
    assert SourceClip.from_dict(clip_dict) == clip

    seg = Segment(
        segment_id="seg_1",
        source_clips=[clip],
        original_dialogue="Say my name.",
        narration="He asks for his reputation to be acknowledged.",
        audio_policy=AudioPolicy.DUCK.value,
    )
    seg_dict = seg.to_dict()
    seg_restored = Segment.from_dict(seg_dict)
    assert seg_restored.segment_id == "seg_1"
    assert len(seg_restored.source_clips) == 1
    assert seg_restored.source_clips[0].start == 50.0

    output = CommentaryOutput(
        output_id="out_1",
        title="Walter White: The Turn? [Deep Dive]",
        candidate_scope=CandidateScope.SINGLE_EPISODE.value,
        segments=[seg],
        status=OutputStatus.WAITING.value,
        publication_video_path="C:/out/video.mp4",
        publication_original_srt_path="C:/out/orig.srt",
        publication_narration_srt_path="C:/out/narr.srt",
    )
    # Check aliases
    assert output.video_path == "C:/out/video.mp4"
    assert output.original_srt_path == "C:/out/orig.srt"
    assert output.narration_srt_path == "C:/out/narr.srt"

    out_dict = output.to_dict()
    out_restored = CommentaryOutput.from_dict(out_dict)
    assert out_restored.title == output.title
    assert out_restored.segments[0].segment_id == "seg_1"

    manifest = AnalysisManifest(
        project_id="proj_1",
        analysis_scope=AnalysisScope.SINGLE_EPISODE.value,
        source_episodes=[ep1],
        outputs=[output],
        created_at="2026-09-19T16:00:00Z",
    )
    manifest.validate()

    json_str = manifest.to_json()
    manifest_restored = AnalysisManifest.from_json(json_str)
    assert manifest_restored.project_id == "proj_1"
    assert len(manifest_restored.outputs) == 1
    assert manifest_restored.outputs[0].title == "Walter White: The Turn? [Deep Dive]"
    # AI publication title preserved
    assert manifest_restored.outputs[0].title == output.title
    assert manifest_restored.outputs[0].sanitized_title != ""


def test_empty_outputs_valid() -> None:
    # 0 candidates found -> manifest should be completely valid
    ep = SourceEpisode(episode_id="E01", source_video="C:/media/e01.mp4", duration_seconds=600.0)
    manifest = AnalysisManifest(
        project_id="proj_empty",
        analysis_scope=AnalysisScope.SINGLE_EPISODE.value,
        source_episodes=[ep],
        outputs=[],
    )
    manifest.validate()
    assert len(manifest.outputs) == 0


def test_multiple_outputs_unlimited() -> None:
    # Multiple outputs (0, 1, 5) without artificial quota or slicing
    ep = SourceEpisode(episode_id="E01", source_video="C:/media/e01.mp4", duration_seconds=600.0)
    outputs = [
        CommentaryOutput(
            output_id=f"out_{i}",
            title=f"Analysis Part {i}",
            candidate_scope=CandidateScope.SINGLE_EPISODE.value,
            segments=[
                Segment(
                    segment_id=f"seg_{i}",
                    source_clips=[SourceClip(episode_id="E01", source_video="C:/media/e01.mp4", start=float(i * 10), end=float(i * 10 + 5))],
                )
            ],
        )
        for i in range(5)
    ]

    manifest = AnalysisManifest(
        project_id="proj_multi",
        source_episodes=[ep],
        outputs=outputs,
    )
    manifest.validate()
    assert len(manifest.outputs) == 5
    # Unique sanitized titles
    sanitized = [out.sanitized_title for out in manifest.outputs]
    assert len(set(sanitized)) == 5


def test_validation_rejects_hallucinated_episode_id() -> None:
    ep = SourceEpisode(episode_id="E01", source_video="C:/media/e01.mp4", duration_seconds=600.0)
    # Clip points to non-existent E99
    bad_clip = SourceClip(episode_id="E99", source_video="C:/media/e01.mp4", start=0.0, end=10.0)
    out = CommentaryOutput(
        output_id="out_1",
        title="Title 1",
        segments=[Segment(segment_id="s1", source_clips=[bad_clip])],
    )
    manifest = AnalysisManifest(
        project_id="proj_hallucinate",
        source_episodes=[ep],
        outputs=[out],
    )
    with pytest.raises(ValidationError, match="Invalid episode_id 'E99'"):
        manifest.validate()


def test_validation_rejects_hallucinated_source_file() -> None:
    ep = SourceEpisode(episode_id="E01", source_video="C:/media/e01.mp4", duration_seconds=600.0)
    # Clip points to wrong file
    bad_clip = SourceClip(episode_id="E01", source_video="C:/media/wrong.mp4", start=0.0, end=10.0)
    out = CommentaryOutput(
        output_id="out_1",
        title="Title 1",
        segments=[Segment(segment_id="s1", source_clips=[bad_clip])],
    )
    manifest = AnalysisManifest(
        project_id="proj_bad_file",
        source_episodes=[ep],
        outputs=[out],
    )
    with pytest.raises(ValidationError, match="does not match episode 'E01' file"):
        manifest.validate()


def test_validation_rejects_negative_start() -> None:
    ep = SourceEpisode(episode_id="E01", source_video="C:/media/e01.mp4", duration_seconds=600.0)
    bad_clip = SourceClip(episode_id="E01", source_video="C:/media/e01.mp4", start=-5.0, end=10.0)
    out = CommentaryOutput(
        output_id="out_1",
        title="Title 1",
        segments=[Segment(segment_id="s1", source_clips=[bad_clip])],
    )
    manifest = AnalysisManifest(
        project_id="proj_neg_start",
        source_episodes=[ep],
        outputs=[out],
    )
    with pytest.raises(ValidationError, match="Invalid clip start"):
        manifest.validate()


def test_validation_rejects_end_less_than_start() -> None:
    ep = SourceEpisode(episode_id="E01", source_video="C:/media/e01.mp4", duration_seconds=600.0)
    bad_clip = SourceClip(episode_id="E01", source_video="C:/media/e01.mp4", start=20.0, end=10.0)
    out = CommentaryOutput(
        output_id="out_1",
        title="Title 1",
        segments=[Segment(segment_id="s1", source_clips=[bad_clip])],
    )
    manifest = AnalysisManifest(
        project_id="proj_end_less",
        source_episodes=[ep],
        outputs=[out],
    )
    with pytest.raises(ValidationError, match="greater than start"):
        manifest.validate()


def test_validation_rejects_end_exceeding_duration() -> None:
    ep = SourceEpisode(episode_id="E01", source_video="C:/media/e01.mp4", duration_seconds=100.0)
    bad_clip = SourceClip(episode_id="E01", source_video="C:/media/e01.mp4", start=50.0, end=150.0)
    out = CommentaryOutput(
        output_id="out_1",
        title="Title 1",
        segments=[Segment(segment_id="s1", source_clips=[bad_clip])],
    )
    manifest = AnalysisManifest(
        project_id="proj_exceed",
        source_episodes=[ep],
        outputs=[out],
    )
    with pytest.raises(ValidationError, match="exceeds episode 'E01' duration"):
        manifest.validate()


def test_validation_rejects_duplicate_output_ids() -> None:
    ep = SourceEpisode(episode_id="E01", source_video="C:/media/e01.mp4", duration_seconds=600.0)
    out1 = CommentaryOutput(output_id="out_dup", title="Title A")
    out2 = CommentaryOutput(output_id="out_dup", title="Title B")
    manifest = AnalysisManifest(
        project_id="proj_dup_ids",
        source_episodes=[ep],
        outputs=[out1, out2],
    )
    with pytest.raises(ValidationError, match="Duplicate output_id"):
        manifest.validate()


def test_episode_evidence_coverage_and_missing_reasons() -> None:
    evidence = EpisodeEvidence(
        episode_id="E01",
        source_video="C:/media/e01.mp4",
        duration_seconds=1800.0,
        coverage={"ratio": 0.95, "covered_intervals": [[0.0, 1710.0]]},
        missing_reasons=["Credits skipped (1710s-1800s)"],
        source_mtime=1726750000.0,
        source_size=104857600,
        data={"scenes": [{"start": 0, "end": 100}]},
    )
    d = evidence.to_dict()
    restored = EpisodeEvidence.from_dict(d)
    assert restored.episode_id == "E01"
    assert restored.coverage["ratio"] == 0.95
    assert restored.missing_reasons == ["Credits skipped (1710s-1800s)"]
    assert restored.source_size == 104857600
