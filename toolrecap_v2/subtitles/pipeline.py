"""Unified subtitle extraction pipeline: discovery, parsing, bitmap OCR, caching, and STT fallback."""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .cache import SubtitleCacheManager
from .discovery import build_embedded_tracks, discover_sidecars, select_best_english_subtitles
from .models import (
    MediaProbeResult,
    SubtitleCue,
    SubtitleDiscoveryResult,
    SubtitleTrack,
)
from .ocr import OcrAdapter
from .parsers import parse_ass, parse_srt, parse_vtt
from .pgs import parse_pgs_sup
from .vobsub import extract_spu_events_from_stream, extract_vobsub_events


SttFallbackFn = Callable[[str, str], list[SubtitleCue]]  # (source_video, episode_id) -> cues
FfmpegExtractFn = Callable[[str, int, Path], Path]  # (video_path, stream_index, output_path) -> output_path


def default_ffmpeg_extract_text(video_path: str, stream_index: int, output_path: Path) -> Path:
    """Extract embedded text subtitle stream using FFmpeg."""
    from ..media import find_binary, run_command

    cmd = [
        find_binary("ffmpeg"),
        "-y",
        "-i", str(video_path),
        "-map", f"0:{stream_index}",
        "-c:s", "srt",
        str(output_path),
    ]
    run_command(cmd)
    return output_path


def default_ffmpeg_demux_sup(video_path: str, stream_index: int, output_path: Path) -> Path:
    """Demux embedded PGS SUP subtitle stream using FFmpeg."""
    from ..media import find_binary, run_command

    cmd = [
        find_binary("ffmpeg"),
        "-y",
        "-i", str(video_path),
        "-map", f"0:{stream_index}",
        "-c:s", "copy",
        "-f", "sup",
        str(output_path),
    ]
    run_command(cmd)
    return output_path


def default_ffmpeg_demux_vobsub(video_path: str, stream_index: int, output_path: Path) -> Path:
    """Demux embedded DVD subtitle stream using FFmpeg into MPEG-2 PS stream."""
    from ..media import find_binary, run_command

    cmd = [
        find_binary("ffmpeg"),
        "-y",
        "-i", str(video_path),
        "-map", f"0:{stream_index}",
        "-c:s", "copy",
        "-f", "dvd",
        str(output_path),
    ]
    run_command(cmd)
    return output_path


class SubtitlePipeline:
    """High-level subtitle processing pipeline."""

    def __init__(
        self,
        cache_manager: SubtitleCacheManager | None = None,
        ocr_adapter: OcrAdapter | None = None,
        ffmpeg_text_extractor: FfmpegExtractFn | None = None,
        ffmpeg_sup_demuxer: FfmpegExtractFn | None = None,
        ffmpeg_vobsub_demuxer: FfmpegExtractFn | None = None,
    ) -> None:
        self.cache_manager = cache_manager or SubtitleCacheManager()
        self.ocr_adapter = ocr_adapter or OcrAdapter()
        self.extract_text = ffmpeg_text_extractor or default_ffmpeg_extract_text
        self.demux_sup = ffmpeg_sup_demuxer or default_ffmpeg_demux_sup
        self.demux_vobsub = ffmpeg_vobsub_demuxer or default_ffmpeg_demux_vobsub

    def discover(
        self,
        video_path: str | Path,
        episode_id: str = "",
        probe_result: MediaProbeResult | None = None,
    ) -> SubtitleDiscoveryResult:
        """Discover sidecar and embedded subtitle tracks and evaluate best English Full track."""
        v_path = Path(video_path).resolve()
        v_str = str(v_path)

        all_tracks: list[SubtitleTrack] = []

        # 1. Discover sidecars in same directory
        sidecars = discover_sidecars(v_path, episode_id=episode_id)
        all_tracks.extend(sidecars)

        # 2. Add embedded tracks if probe result provided
        if probe_result is not None:
            embedded = build_embedded_tracks(probe_result.subtitle_streams, v_str)
            all_tracks.extend(embedded)

        return select_best_english_subtitles(all_tracks, video_path=v_str, episode_id=episode_id)

    def extract_and_parse(
        self,
        track: SubtitleTrack,
        episode_id: str,
        source_video: str,
        *,
        ai_vision_fallback: Callable[[Image.Image], str | None] | None = None,
        vision_supported: bool = False,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[SubtitleCue]:
        """Extract and parse/OCR cues for a selected track."""
        # 1. Sidecar text formats
        if track.source_type == "sidecar" and not track.is_bitmap:
            assert track.source_file is not None
            p = Path(track.source_file)
            fmt = track.source_format.lower()
            if fmt == "srt":
                return parse_srt(p, source_file=track.source_file, language=track.language, episode_id=episode_id, source_video=source_video)
            elif fmt in ("ass", "ssa"):
                return parse_ass(p, source_file=track.source_file, language=track.language, episode_id=episode_id, source_video=source_video)
            elif fmt == "vtt":
                return parse_vtt(p, source_file=track.source_file, language=track.language, episode_id=episode_id, source_video=source_video)

        # 2. Embedded text format
        if track.source_type == "embedded" and not track.is_bitmap:
            assert track.stream_index is not None
            with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                self.extract_text(source_video, track.stream_index, tmp_path)
                return parse_srt(
                    tmp_path,
                    source_type="embedded",
                    stream_index=track.stream_index,
                    source_file=source_video,
                    language=track.language,
                    episode_id=episode_id,
                    source_video=source_video,
                )
            finally:
                tmp_path.unlink(missing_ok=True)

        # 3. Bitmap format: PGS SUP
        if track.source_format == "pgs":
            if track.source_type == "embedded":
                assert track.stream_index is not None
                with tempfile.NamedTemporaryFile(suffix=".sup", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    self.demux_sup(source_video, track.stream_index, tmp_path)
                    events = parse_pgs_sup(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
            else:
                assert track.source_file is not None
                events = parse_pgs_sup(Path(track.source_file))

            cues: list[SubtitleCue] = []
            for event_index, ev in enumerate(events, start=1):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("OCR phụ đề đã bị hủy.")
                if progress_callback:
                    progress_callback(f"OCR PGS {event_index}/{len(events)}")
                if not ev.image:
                    continue
                ocr_res = self.ocr_adapter.ocr_image(
                    ev.image,
                    ai_fallback_fn=ai_vision_fallback,
                    vision_supported=vision_supported,
                    cancel_check=(cancel_event.is_set if cancel_event else None),
                )
                if ocr_res.is_valid and ocr_res.text:
                    cues.append(
                        SubtitleCue(
                            start_ms=ev.start_ms,
                            end_ms=ev.end_ms,
                            text=ocr_res.text,
                            source_type=track.source_type,
                            source_format="pgs",
                            stream_index=track.stream_index,
                            source_file=track.source_file or source_video,
                            language=track.language,
                            confidence=ocr_res.confidence,
                            episode_id=episode_id,
                            source_video=source_video,
                        )
                    )
            return cues

        # 4. Bitmap format: VobSub
        if track.source_format == "vobsub":
            if track.source_type == "embedded":
                assert track.stream_index is not None
                with tempfile.NamedTemporaryFile(suffix=".vob", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    self.demux_vobsub(source_video, track.stream_index, tmp_path)
                    events = extract_spu_events_from_stream(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
            else:
                assert track.source_file is not None
                idx_path = Path(track.source_file)
                events = extract_vobsub_events(idx_path, language=track.language)

            cues = []
            for event_index, v_ev in enumerate(events, start=1):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("OCR phụ đề đã bị hủy.")
                if progress_callback:
                    progress_callback(f"OCR VobSub {event_index}/{len(events)}")
                if not v_ev.image:
                    continue
                ocr_res = self.ocr_adapter.ocr_image(
                    v_ev.image,
                    ai_fallback_fn=ai_vision_fallback,
                    vision_supported=vision_supported,
                    cancel_check=(cancel_event.is_set if cancel_event else None),
                )
                if ocr_res.is_valid and ocr_res.text:
                    cues.append(
                        SubtitleCue(
                            start_ms=v_ev.start_ms,
                            end_ms=v_ev.end_ms,
                            text=ocr_res.text,
                            source_type=track.source_type,
                            source_format="vobsub",
                            stream_index=track.stream_index,
                            source_file=track.source_file or source_video,
                            language=track.language,
                            confidence=ocr_res.confidence,
                            episode_id=episode_id,
                            source_video=source_video,
                        )
                    )
            return cues

        return []

    def get_episode_subtitles(
        self,
        video_path: str | Path,
        episode_id: str,
        probe_result: MediaProbeResult | None = None,
        *,
        stt_fallback_fn: SttFallbackFn | None = None,
        ai_vision_fallback: Callable[[Image.Image], str | None] | None = None,
        vision_supported: bool = False,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[SubtitleCue]:
        """End-to-end subtitle resolution for an episode:

        1. Discover sidecars and embedded tracks.
        2. If usable Full English track found:
           - If bitmap and OCR is unavailable: fallback to STT or raise user-facing error.
           - Check atomic cache. If hit, return cached cues.
           - Extract / parse / OCR track.
           - If OCR produced no cues: fallback to STT or raise user-facing error.
           - Save to cache.
           - Return cues.
        3. If NO usable Full English track found:
           - Check STT cache. If hit, return cached cues.
           - If stt_fallback_fn is provided, invoke STT fallback interface.
           - Save STT cues to cache and return.
        """
        v_path = Path(video_path).resolve()
        v_str = str(v_path)
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Xử lý phụ đề đã bị hủy.")

        discovery = self.discover(v_path, episode_id=episode_id, probe_result=probe_result)

        if discovery.best_english_full is not None:
            track = discovery.best_english_full

            # If bitmap track, check OCR availability before proceeding
            if track.is_bitmap and not self.ocr_adapter.is_engine_ready():
                # Primary: When RapidOCR package installed but model files missing,
                # attempt downloading pinned models before deciding unavailable/STT.
                is_pkg = False
                if hasattr(self.ocr_adapter, "is_package_installed"):
                    try:
                        is_pkg = bool(self.ocr_adapter.is_package_installed())
                    except Exception:
                        is_pkg = False

                if is_pkg and hasattr(self.ocr_adapter, "model_manager") and self.ocr_adapter.model_manager is not None:
                    def _adapt_progress(downloaded: int, total_bytes: int, msg: str) -> None:
                        if progress_callback:
                            if total_bytes > 0:
                                mb_down = downloaded / (1024 * 1024)
                                mb_total = total_bytes / (1024 * 1024)
                                pct = int((downloaded / total_bytes) * 100)
                                progress_callback(
                                    f"Tải mô hình OCR: {pct}% ({mb_down:.1f}/{mb_total:.1f} MB)"
                                )
                            else:
                                progress_callback(f"Tải mô hình OCR: {msg}")

                    try:
                        self.ocr_adapter.model_manager.download_models(
                            progress_callback=_adapt_progress,
                            cancel_check=(cancel_event.is_set if cancel_event else None),
                        )
                    except Exception:
                        if cancel_event and cancel_event.is_set():
                            raise RuntimeError("OCR phụ đề đã bị hủy.")

                if not self.ocr_adapter.is_engine_ready():
                    if stt_fallback_fn is not None:
                        stt_track_id = "stt:whisper"
                        cached_stt = self.cache_manager.load_cues(episode_id, v_str, stt_track_id)
                        if cached_stt is not None:
                            return cached_stt
                        stt_cues = stt_fallback_fn(v_str, episode_id)
                        self.cache_manager.save_cues(episode_id, v_str, stt_track_id, stt_cues)
                        return stt_cues
                    raise RuntimeError(
                        f"Mô hình hoặc engine OCR không khả dụng cho phụ đề bitmap ({track.source_format}) và không có STT fallback."
                    )

            # Check per-episode cache with track details
            cached = self.cache_manager.load_cues(
                episode_id,
                v_str,
                track.track_id,
                track_codec=track.source_format,
                track_index=track.stream_index,
                track_source=track.source_file or track.track_id,
            )
            if cached is not None:
                return cached

            # Extract and parse/OCR
            cues = self.extract_and_parse(
                track,
                episode_id=episode_id,
                source_video=v_str,
                ai_vision_fallback=ai_vision_fallback,
                vision_supported=vision_supported,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
            )

            # Strict: If bitmap track yielded no valid text after OCR, do not claim empty silent success
            if track.is_bitmap and not cues:
                if stt_fallback_fn is not None:
                    stt_track_id = "stt:whisper"
                    cached_stt = self.cache_manager.load_cues(episode_id, v_str, stt_track_id)
                    if cached_stt is not None:
                        return cached_stt
                    stt_cues = stt_fallback_fn(v_str, episode_id)
                    self.cache_manager.save_cues(episode_id, v_str, stt_track_id, stt_cues)
                    return stt_cues
                raise RuntimeError(
                    f"OCR phụ đề bitmap ({track.source_format}) không trích xuất được nội dung và không có STT fallback."
                )

            self.cache_manager.save_cues(
                episode_id,
                v_str,
                track.track_id,
                cues,
                track_codec=track.source_format,
                track_index=track.stream_index,
                track_source=track.source_file or track.track_id,
            )
            return cues

        # No usable Full English subtitles: STT Fallback required
        stt_track_id = "stt:whisper"
        cached_stt = self.cache_manager.load_cues(episode_id, v_str, stt_track_id)
        if cached_stt is not None:
            return cached_stt

        if stt_fallback_fn is not None:
            stt_cues = stt_fallback_fn(v_str, episode_id)
            self.cache_manager.save_cues(episode_id, v_str, stt_track_id, stt_cues)
            return stt_cues

        return []
