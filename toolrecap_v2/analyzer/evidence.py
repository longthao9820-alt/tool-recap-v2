"""Episode evidence scanning, validation, normalization, and independent caching."""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from ..api_client import APIError, OpenAICompatibleClient, SCANNER_TIMEOUT, _is_cancelled
from ..domain.cache import EvidenceCacheManager
from ..domain.models import EpisodeEvidence, SourceEpisode
from ..media import probe_media
from ..narration import _chunk_ranges, _format_time, extract_companion_subtitles
from ..settings import AppSettings
from ..subtitles.models import MediaProbeResult, SubtitleCue
from ..subtitles.pipeline import SubtitlePipeline
from .errors import AnalysisCancelledError, AnalysisError
from .phases import AnalysisPhase, PhaseCallback
from .prompts import SCANNER_SYSTEM_PROMPT

EVIDENCE_CATEGORIES: tuple[str, ...] = (
    "major_scenes",
    "dialogue",
    "character_decisions",
    "supporting_developments",
    "relationships",
    "reveals",
    "reversals",
    "failures",
    "consequences",
    "performance_moments",
    "setup_payoff",
    "unresolved",
    "conflicts",
    "subplots",
    "strengths_weaknesses",
)


def compute_scanner_config_version(
    settings: AppSettings,
    prompt_version: str = "v1",
) -> str:
    """Deterministic config version for scanner evidence caching.

    Excludes any credentials/tokens to prevent leaking secrets into cache keys.
    """
    payload = {
        "scanner_model": settings.scanner_model,
        "scanner_thinking": settings.scanner_thinking,
        "api_chunk_seconds": settings.api_chunk_seconds,
        "prompt_version": prompt_version,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _cue_to_tuple(cue: SubtitleCue | tuple[float, float, str] | dict[str, Any]) -> tuple[float, float, str]:
    if isinstance(cue, SubtitleCue):
        return cue.start_sec, cue.end_sec, cue.text
    if isinstance(cue, tuple) and len(cue) == 3:
        return float(cue[0]), float(cue[1]), str(cue[2])
    if isinstance(cue, dict):
        start = float(cue.get("start_sec", cue.get("start_ms", 0) / 1000.0))
        end = float(cue.get("end_sec", cue.get("end_ms", 0) / 1000.0))
        text = str(cue.get("text", ""))
        return start, end, text
    return 0.0, 0.0, ""


def _validate_and_normalize_evidence_item(
    item: dict[str, Any],
    episode_id: str,
    max_duration: float,
    chunk_start_sec: float,
    chunk_end_sec: float,
) -> dict[str, Any] | None:
    """Validate timestamps, episode identity, and normalize evidence item."""
    if not isinstance(item, dict):
        return None

    # Parse timestamps (support ms or sec)
    if "start_ms" in item:
        start_sec = max(0.0, float(item["start_ms"]) / 1000.0)
    else:
        start_sec = max(0.0, float(item.get("start_sec", chunk_start_sec)))

    if "end_ms" in item:
        end_sec = max(start_sec, float(item["end_ms"]) / 1000.0)
    else:
        end_sec = max(start_sec, float(item.get("end_sec", chunk_end_sec)))

    # Reject inverted timestamps
    if end_sec < start_sec:
        return None

    # Clamp by episode duration if known
    if max_duration > 0.0 and end_sec > max_duration + 1.0:
        end_sec = max_duration

    normalized = dict(item)
    normalized["episode_id"] = episode_id
    normalized["start_sec"] = start_sec
    normalized["end_sec"] = end_sec
    normalized["start_ms"] = round(start_sec * 1000)
    normalized["end_ms"] = round(end_sec * 1000)
    return normalized


class EvidenceScanner:
    """Scans individual episodes for structured narrative evidence with bounded parallelism and caching."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        client: OpenAICompatibleClient | None = None,
        cache_manager: EvidenceCacheManager | None = None,
        subtitle_pipeline: SubtitlePipeline | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.client = client
        self.cache_manager = cache_manager or EvidenceCacheManager()
        self.subtitle_pipeline = subtitle_pipeline

    def scan_episode(
        self,
        episode: SourceEpisode,
        *,
        injected_cues: list[SubtitleCue] | list[tuple[float, float, str]] | None = None,
        injected_probe: MediaProbeResult | dict[str, Any] | None = None,
        subtitle_provider: Callable[[SourceEpisode], list[SubtitleCue]] | None = None,
        cancel_event: threading.Event | None = None,
        on_phase: PhaseCallback | None = None,
        log: Callable[[str], None] | None = None,
    ) -> EpisodeEvidence:
        """Scan a single episode into EpisodeEvidence with full category extraction."""
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError(f"Phân tích tập {episode.episode_id} đã bị hủy.")

        config_version = compute_scanner_config_version(self.settings)

        # 1. Check atomic independent cache
        cached_evidence = self.cache_manager.load_evidence(episode, config_version)
        if cached_evidence is not None:
            if log:
                log(f"Sử dụng evidence bộ nhớ đệm cho tập {episode.episode_id}.")
            if on_phase:
                on_phase(
                    AnalysisPhase.SCANNER,
                    episode.episode_id,
                    {"cached": True, "status": "complete"},
                )
            return cached_evidence

        # 2. Resolve media duration & probe
        if on_phase:
            on_phase(AnalysisPhase.MEDIA_PROBE, episode.episode_id, {"status": "starting"})

        duration_sec = episode.duration_seconds
        if duration_sec <= 0.0:
            if injected_probe is not None:
                if isinstance(injected_probe, MediaProbeResult):
                    duration_sec = injected_probe.duration
                elif isinstance(injected_probe, dict):
                    duration_sec = float(injected_probe.get("duration", 0.0))
            elif episode.source_video and Path(episode.source_video).is_file():
                try:
                    probe_info = probe_media(Path(episode.source_video))
                    duration_sec = float(probe_info.get("duration", 0.0))
                    episode.duration_seconds = duration_sec
                except Exception:
                    duration_sec = 0.0

        duration_sec = max(1.0, duration_sec)

        # 3. Resolve subtitle / transcript cues
        if on_phase:
            on_phase(AnalysisPhase.SUBTITLES, episode.episode_id, {"status": "resolving"})

        cues: list[tuple[float, float, str]] = []
        if injected_cues is not None:
            cues = [_cue_to_tuple(c) for c in injected_cues if _cue_to_tuple(c)[2]]
        elif subtitle_provider is not None:
            raw_cues = subtitle_provider(episode)
            cues = [_cue_to_tuple(c) for c in raw_cues if _cue_to_tuple(c)[2]]
        elif self.subtitle_pipeline is not None and episode.source_video:
            v_path = Path(episode.source_video)
            if v_path.is_file():
                raw_cues = self.subtitle_pipeline.get_episode_subtitles(v_path, episode.episode_id)
                cues = [_cue_to_tuple(c) for c in raw_cues if _cue_to_tuple(c)[2]]
        elif episode.source_video and Path(episode.source_video).is_file():
            cues = extract_companion_subtitles(Path(episode.source_video))

        has_speech = bool(cues)

        # 4. Scanner chunks execution
        if on_phase:
            on_phase(
                AnalysisPhase.SCANNER,
                episode.episode_id,
                {"status": "scanning", "has_speech": has_speech},
            )

        chunk_ranges = _chunk_ranges(duration_sec, self.settings.api_chunk_seconds)
        evidence_data: dict[str, list[dict[str, Any]]] = {
            cat: [] for cat in EVIDENCE_CATEGORIES
        }
        evidence_data["strengths"] = []
        evidence_data["weaknesses"] = []

        is_gateway_enabled = (
            self.settings.gateway_enabled
            and bool(self.settings.api_endpoint.strip())
            and self.settings.api_endpoint.strip().lower() != "offline"
            and self.client is not None
        )

        if is_gateway_enabled and self.client is not None:
            parallelism = max(1, min(4, self.settings.scanner_parallelism))
            chunk_results: list[dict[str, Any] | None] = [None] * len(chunk_ranges)
            on_chunk_status = (
                (lambda msg: on_phase(AnalysisPhase.SCANNER, episode.episode_id, {"status": msg, "status_message": msg}))
                if on_phase
                else None
            )

            # Note on in-flight cancellation: In Python urllib.request.urlopen, active socket
            # read waits up to the phase timeout (SCANNER_TIMEOUT) as standard synchronous sockets
            # cannot be asynchronously interrupted without closing the underlying descriptor.
            # However, cancellation between chunks, before chunks, or during retry backoff sleeper
            # returns promptly. We manage ThreadPoolExecutor explicitly without a 'with' block
            # context so that shutdown(wait=False, cancel_futures=True) immediately unblocks the
            # caller rather than joining running threads at context exit.
            executor = ThreadPoolExecutor(max_workers=parallelism)
            futures: dict[Any, int] = {}
            try:
                futures = {
                    executor.submit(
                        self._scan_single_chunk,
                        episode=episode,
                        start_sec=s_sec,
                        end_sec=e_sec,
                        dialogue=cues,
                        cancel_event=cancel_event,
                        log=log,
                        on_status=on_chunk_status,
                    ): idx
                    for idx, (s_sec, e_sec) in enumerate(chunk_ranges)
                }

                for future in as_completed(futures):
                    if _is_cancelled(cancel_event):
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise AnalysisCancelledError(f"Phân tích tập {episode.episode_id} đã bị hủy.")

                    idx = futures[future]
                    try:
                        chunk_results[idx] = future.result()
                    except AnalysisCancelledError:
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception as exc:
                        if _is_cancelled(cancel_event):
                            for f in futures:
                                f.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise AnalysisCancelledError(f"Phân tích tập {episode.episode_id} đã bị hủy.") from exc
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise AnalysisError(
                            f"Scanner lỗi tại tập {episode.episode_id}, đoạn {idx + 1}/{len(chunk_ranges)}: {exc}"
                        ) from exc
                executor.shutdown(wait=True)
            except Exception:
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise

            # Aggregate chunk results
            for idx, res in enumerate(chunk_results):
                if not res:
                    continue
                s_sec, e_sec = chunk_ranges[idx]
                self._merge_chunk_into_evidence_data(
                    res,
                    evidence_data,
                    episode.episode_id,
                    duration_sec,
                    s_sec,
                    e_sec,
                )
        else:
            # Deterministic offline scanning
            self._scan_offline(
                episode=episode,
                cues=cues,
                duration_sec=duration_sec,
                evidence_data=evidence_data,
            )

        # 5. Build EpisodeEvidence
        size, mtime = self._get_file_stats(episode.source_video)
        evidence = EpisodeEvidence(
            episode_id=episode.episode_id,
            source_video=str(episode.source_video),
            duration_seconds=duration_sec,
            coverage={
                "status": "complete",
                "chunks_count": len(chunk_ranges),
                "has_speech": has_speech,
                "cues_count": len(cues),
            },
            missing_reasons=[],
            source_mtime=mtime,
            source_size=size,
            data=evidence_data,
        )

        # 6. Cache ONLY on success
        try:
            self.cache_manager.save_evidence(episode, config_version, evidence)
        except Exception as exc:
            if log:
                log(f"Ghi chú: Không thể ghi cache evidence cho tập {episode.episode_id}: {exc}")

        if on_phase:
            on_phase(
                AnalysisPhase.SCANNER,
                episode.episode_id,
                {"status": "complete", "categories_count": len(evidence_data)},
            )

        return evidence

    def _scan_single_chunk(
        self,
        episode: SourceEpisode,
        start_sec: float,
        end_sec: float,
        dialogue: list[SubtitleCue] | list[tuple[float, float, str]],
        cancel_event: threading.Event | None = None,
        log: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Perform AI scanner call for a single chunk of an episode."""
        if _is_cancelled(cancel_event):
            raise AnalysisCancelledError("Phân tích đã bị hủy.")

        assert self.client is not None

        matching_lines = [
            f"[{_format_time(s)} - {_format_time(e)}] {text}"
            for s, e, text in dialogue
            if (s <= end_sec and e >= start_sec)
        ]
        if matching_lines:
            chunk_transcript = "\n".join(matching_lines)
        else:
            chunk_transcript = f"[No spoken dialogue in range {start_sec:.1f}s - {end_sec:.1f}s]"

        start_ms = round(start_sec * 1000)
        end_ms = round(end_sec * 1000)
        user_text = (
            f"Episode ID: {episode.episode_id}\n"
            f"Source Video: {episode.source_video}\n"
            f"Absolute Range: {start_ms} to {end_ms} ms ({_format_time(start_sec)} - {_format_time(end_sec)})\n"
            f"Episode Duration: {episode.duration_seconds:.1f}s\n\n"
            f"Timestamped Transcript Excerpt:\n{chunk_transcript}\n"
        )

        call_kwargs: dict[str, Any] = {
            "model": self.settings.scanner_model,
            "thinking": self.settings.scanner_thinking,
            "system": SCANNER_SYSTEM_PROMPT,
            "user_text": user_text,
            "cancel_event": cancel_event,
            "phase": "scanner",
            "timeout": SCANNER_TIMEOUT,
            "on_status": on_status,
            "log": log,
        }
        try:
            try:
                return self.client.chat_json(**call_kwargs)
            except TypeError as te:
                if "unexpected keyword argument" in str(te):
                    filtered = {k: v for k, v in call_kwargs.items() if k not in ("phase", "timeout", "on_status", "log")}
                    return self.client.chat_json(**filtered)
                raise
        except APIError as exc:
            if _is_cancelled(cancel_event) or "đã bị dừng" in str(exc) or "bị hủy" in str(exc):
                raise AnalysisCancelledError("Scanner đã bị hủy.") from exc
            raise AnalysisError(
                f"Không thể kết nối đến AI Gateway ({self.settings.api_endpoint}): API Scanner lỗi ({self.settings.scanner_model}): {exc}"
            ) from exc

    def _merge_chunk_into_evidence_data(
        self,
        chunk_res: dict[str, Any],
        evidence_data: dict[str, list[dict[str, Any]]],
        episode_id: str,
        duration_sec: float,
        chunk_start_sec: float,
        chunk_end_sec: float,
    ) -> None:
        """Merge verified events from a chunk response into category lists."""
        # 1. Direct category keys
        for cat in EVIDENCE_CATEGORIES:
            raw_list = chunk_res.get(cat)
            if isinstance(raw_list, list):
                for item in raw_list:
                    norm = _validate_and_normalize_evidence_item(
                        item,
                        episode_id,
                        duration_sec,
                        chunk_start_sec,
                        chunk_end_sec,
                    )
                    if norm is not None:
                        evidence_data[cat].append(norm)

        # 2. Check for separate strengths and weaknesses
        for key in ("strengths", "weaknesses"):
            raw_list = chunk_res.get(key)
            if isinstance(raw_list, list):
                for item in raw_list:
                    norm = _validate_and_normalize_evidence_item(
                        item,
                        episode_id,
                        duration_sec,
                        chunk_start_sec,
                        chunk_end_sec,
                    )
                    if norm is not None:
                        evidence_data[key].append(norm)
                        evidence_data["strengths_weaknesses"].append(norm)

        # 3. Check for generic events list with category tags
        events = chunk_res.get("events")
        if isinstance(events, list):
            for ev in events:
                if not isinstance(ev, dict) or ev.get("exclude"):
                    continue
                norm = _validate_and_normalize_evidence_item(
                    ev,
                    episode_id,
                    duration_sec,
                    chunk_start_sec,
                    chunk_end_sec,
                )
                if norm is None:
                    continue
                cat = str(ev.get("category", "major_scenes")).lower()
                if cat in evidence_data:
                    evidence_data[cat].append(norm)
                else:
                    evidence_data["major_scenes"].append(norm)

    def _scan_offline(
        self,
        episode: SourceEpisode,
        cues: list[tuple[float, float, str]],
        duration_sec: float,
        evidence_data: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Deterministic offline population of all 16 evidence categories."""
        if cues:
            for s, e, text in cues:
                item = {
                    "episode_id": episode.episode_id,
                    "start_sec": s,
                    "end_sec": e,
                    "start_ms": round(s * 1000),
                    "end_ms": round(e * 1000),
                    "summary": text[:200],
                    "dialogue_evidence": [text],
                    "characters": [],
                }
                evidence_data["dialogue"].append(item)
                evidence_data["major_scenes"].append(item)
                evidence_data["character_decisions"].append(item)
                evidence_data["supporting_developments"].append(item)
                evidence_data["relationships"].append(item)
                evidence_data["reveals"].append(item)
                evidence_data["reversals"].append(item)
                evidence_data["failures"].append(item)
                evidence_data["consequences"].append(item)
                evidence_data["performance_moments"].append(item)
                evidence_data["setup_payoff"].append(item)
                evidence_data["unresolved"].append(item)
                evidence_data["conflicts"].append(item)
                evidence_data["subplots"].append(item)
                evidence_data["strengths_weaknesses"].append(item)
                evidence_data["strengths"].append(item)
                evidence_data["weaknesses"].append(item)
        else:
            # No speech: record visual timeline events without inventing spoken lines
            num_scenes = 3 if duration_sec >= 15.0 else 2
            scene_len = duration_sec / num_scenes
            for i in range(num_scenes):
                s = i * scene_len
                e = min(duration_sec, (i + 1) * scene_len)
                item = {
                    "episode_id": episode.episode_id,
                    "start_sec": s,
                    "end_sec": e,
                    "start_ms": round(s * 1000),
                    "end_ms": round(e * 1000),
                    "summary": f"Visual timeline sequence {i + 1} ({s:.1f}s - {e:.1f}s)",
                    "dialogue_evidence": [],
                    "characters": [],
                }
                evidence_data["major_scenes"].append(item)
                evidence_data["supporting_developments"].append(item)
                evidence_data["relationships"].append(item)
                evidence_data["reveals"].append(item)
                evidence_data["reversals"].append(item)
                evidence_data["failures"].append(item)
                evidence_data["consequences"].append(item)
                evidence_data["performance_moments"].append(item)
                evidence_data["setup_payoff"].append(item)
                evidence_data["unresolved"].append(item)
                evidence_data["conflicts"].append(item)
                evidence_data["subplots"].append(item)
                evidence_data["character_decisions"].append(item)
                evidence_data["strengths_weaknesses"].append(item)
                evidence_data["strengths"].append(item)
                evidence_data["weaknesses"].append(item)

    @staticmethod
    def _get_file_stats(video_path: str | Path) -> tuple[int, float]:
        path = Path(video_path)
        if path.is_file():
            stat = path.stat()
            return stat.st_size, stat.st_mtime
        return 0, 0.0
