"""Real local-voice + multi-output renderer acceptance smoke for release builds."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolrecap_v2.domain.models import AnalysisManifest, CommentaryOutput, Segment, SourceClip, SourceEpisode
from toolrecap_v2.media import find_binary, run_command
from toolrecap_v2.paths import default_data_directory
from toolrecap_v2.renderer import PublicationRenderer
from toolrecap_v2.settings import AppSettings
from toolrecap_v2.subtitles.models import SubtitleCue
from toolrecap_v2.voice.manager import get_voice_manager


def create_source(path: Path, color: str, frequency: int) -> None:
    run_command(
        [
            find_binary("ffmpeg"), "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=640x360:r=24:d=5",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000:duration=5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
        ]
    )


def main() -> int:
    root = default_data_directory() / "smoke_test" / "voice_render_e2e_v060"
    if root.exists():
        shutil.rmtree(root)
    source_dir = root / "sources"
    output_dir = root / "outputs"
    source_dir.mkdir(parents=True)
    first = source_dir / "E01.mp4"
    second = source_dir / "E02.mp4"
    create_source(first, "blue", 440)
    create_source(second, "green", 660)

    episodes = [
        SourceEpisode("E01", str(first), duration_seconds=5.0, title="Synthetic Episode 1"),
        SourceEpisode("E02", str(second), duration_seconds=5.0, title="Synthetic Episode 2"),
    ]
    outputs = [
        CommentaryOutput(
            output_id="voice_e2e_01", title="Voice E2E Output One", file_name="voice-e2e-output-one.mp4",
            segments=[Segment(
                segment_id="s01", narration="A calm blue scene begins.", audio_policy="duck",
                source_clips=[SourceClip("E01", str(first), 0.0, 5.0)],
            )],
        ),
        CommentaryOutput(
            output_id="voice_e2e_02", title="Voice E2E Output Two", file_name="voice-e2e-output-two.mp4",
            segments=[Segment(
                segment_id="s01", narration="A bright green scene follows.", audio_policy="duck",
                source_clips=[SourceClip("E02", str(second), 0.0, 5.0)],
            )],
        ),
    ]
    manifest = AnalysisManifest(
        project_id="voice-render-e2e-v060", analysis_scope="SEASON", source_episodes=episodes, outputs=outputs,
    )
    manifest.validate()
    cues = {
        "E01": [SubtitleCue(0, 4500, "Original dialogue one", "sidecar", "srt", episode_id="E01")],
        "E02": [SubtitleCue(0, 4500, "Original dialogue two", "sidecar", "srt", episode_id="E02")],
    }
    settings = AppSettings(
        output_dir=str(output_dir), use_gpu=False, burn_subtitles=False,
        voice_id="voicestudio.en.documentarian", voice_style="documentary",
    )
    rendered = PublicationRenderer(voice_manager=get_voice_manager()).render_manifest(
        manifest, settings, transcript_cues_by_episode=cues, output_root=output_dir,
    )
    if len(rendered) != 2 or any(out.status != "COMPLETED" for out in rendered):
        raise RuntimeError("Real voice multi-output E2E did not complete both outputs.")
    for output in rendered:
        print(output.output_id, output.publication_video_path)
    print("VOICE_RENDER_E2E_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
