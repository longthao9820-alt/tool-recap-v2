"""Canonical Analysis Engine orchestrating episode evidence, season connection, and final candidate plans."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..api_client import OpenAICompatibleClient
from ..domain.cache import EvidenceCacheManager, HierarchyCacheManager
from ..domain.enums import AnalysisScope
from ..domain.models import AnalysisManifest, CommentaryOutput, EpisodeEvidence, SourceEpisode
from ..paths import default_data_directory
from ..settings import AppSettings
from ..subtitles.models import MediaProbeResult, SubtitleCue
from ..subtitles.pipeline import SubtitlePipeline
from .connection import SeasonConnectionResult, SeasonConnector
from .errors import AnalysisCancelledError, AnalysisError, CoverageIncompleteError
from .evidence import EvidenceScanner
from .finalizer import CandidateFinalizer
from .phases import AnalysisPhase, PhaseCallback


def compute_final_plan_cache_key(
    evidence_map: dict[str, EpisodeEvidence],
    settings: AppSettings,
    scope: str,
    language: str | None = None,
    mode: str | None = None,
    content_type: str | None = None,
    rights: str | None = None,
    recap_prompt: str | None = None,
    legacy_wrapper: bool = False,
    connection_key: str | None = None,
) -> str:
    """Compute deterministic cache key for final CommentaryOutput plans.

    Keyed by hashes of all episode evidence payloads + finalizer config + connection identity.
    Excludes all API credentials or secrets.
    """
    sorted_items = sorted(evidence_map.items(), key=lambda x: x[0])
    ev_hashes: list[str] = []
    for ep_id, ev in sorted_items:
        ev_raw = json.dumps(ev.to_dict(), sort_keys=True)
        h = hashlib.sha256(ev_raw.encode("utf-8")).hexdigest()[:16]
        ev_hashes.append(f"{ep_id}:{h}")

    payload = {
        "scope": scope,
        "evidence_hashes": ev_hashes,
        "finalizer_model": settings.finalizer_model,
        "finalizer_thinking": settings.finalizer_thinking,
        "recap_prompt": recap_prompt if recap_prompt is not None else settings.recap_prompt,
        "recap_language": language or getattr(settings, "recap_language", "en-US"),
        "recap_mode": mode or getattr(settings, "recap_mode", "MAIN_STORIES"),
        "content_type": content_type or getattr(settings, "content_type", "US_TV_SHOW"),
        "rights": rights or getattr(settings, "source_rights_status", "UNVERIFIED"),
        "legacy_wrapper": legacy_wrapper,
        "connection_key": connection_key or "",
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class AnalysisEngine:
    """Canonical analyzer engine for single episode and season commentary generation."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        client: OpenAICompatibleClient | None = None,
        cache_manager: EvidenceCacheManager | None = None,
        subtitle_pipeline: SubtitlePipeline | None = None,
        hierarchy_cache: HierarchyCacheManager | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.client = client
        self.cache_manager = cache_manager or EvidenceCacheManager()
        if hierarchy_cache is not None:
            self.hierarchy_cache = hierarchy_cache
        else:
            hierarchy_dir = self.cache_manager.cache_dir.parent / "season_hierarchy"
            self.hierarchy_cache = HierarchyCacheManager(base_dir=hierarchy_dir)
        self.subtitle_pipeline = subtitle_pipeline
        self.scanner = EvidenceScanner(
            settings=self.settings,
            client=self.client,
            cache_manager=self.cache_manager,
            subtitle_pipeline=self.subtitle_pipeline,
        )
        self.connector = SeasonConnector(
            settings=self.settings,
            client=self.client,
            hierarchy_cache=self.hierarchy_cache,
        )
        self.finalizer = CandidateFinalizer(settings=self.settings, client=self.client)

    def analyze(
        self,
        project_id: str,
        episodes: list[SourceEpisode],
        *,
        scope: AnalysisScope | str = AnalysisScope.SINGLE_EPISODE,
        allow_incomplete: bool = False,
        injected_transcripts: dict[str, list[SubtitleCue] | list[tuple[float, float, str]]] | None = None,
        injected_probes: dict[str, MediaProbeResult | dict[str, Any]] | None = None,
        subtitle_provider: Callable[[SourceEpisode], list[SubtitleCue]] | None = None,
        cancel_event: threading.Event | None = None,
        on_phase: PhaseCallback | None = None,
        log: Callable[[str], None] | None = None,
        use_final_plan_cache: bool = True,
        legacy_wrapper: bool = False,
    ) -> AnalysisManifest:
        """Execute end-to-end analysis workflow returning a strictly validated AnalysisManifest."""
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("Quy trình phân tích đã bị hủy.")

        if not episodes:
            raise AnalysisError("Không có tập phim (source_episodes) nào để phân tích.")

        scope_str = scope.value if isinstance(scope, AnalysisScope) else str(scope)
        transcripts = injected_transcripts or {}
        probes = injected_probes or {}

        # 1. Episode Evidence Phase: Scan all episodes
        evidence_map: dict[str, EpisodeEvidence] = {}
        for ep in episodes:
            if cancel_event and cancel_event.is_set():
                raise AnalysisCancelledError("Quy trình phân tích đã bị hủy.")

            try:
                ev = self.scanner.scan_episode(
                    ep,
                    injected_cues=transcripts.get(ep.episode_id),
                    injected_probe=probes.get(ep.episode_id),
                    subtitle_provider=subtitle_provider,
                    cancel_event=cancel_event,
                    on_phase=on_phase,
                    log=log,
                )
                evidence_map[ep.episode_id] = ev
            except AnalysisCancelledError:
                raise
            except Exception as exc:
                if scope_str == AnalysisScope.SINGLE_EPISODE.value:
                    raise
                if scope_str == AnalysisScope.SEASON.value and not allow_incomplete:
                    raise CoverageIncompleteError(
                        f"Quá trình quét evidence thất bại tại tập {ep.episode_id}: {exc}. "
                        f"Set allow_incomplete=True to proceed with partial season analysis.",
                        missing_episodes=[ep.episode_id],
                    ) from exc
                if log:
                    log(f"Cảnh báo: Lỗi khi quét tập {ep.episode_id}: {exc}")

        # 2. Check Plan Cache (optional)
        plan_cache_key = ""
        if use_final_plan_cache and evidence_map:
            conn_key = None
            if scope_str == AnalysisScope.SEASON.value:
                conn_key = self.connector.compute_connection_key(episodes, evidence_map)
            plan_cache_key = compute_final_plan_cache_key(
                evidence_map,
                self.settings,
                scope_str,
                language=self.settings.recap_language,
                mode=self.settings.recap_mode,
                content_type=self.settings.content_type,
                rights=self.settings.source_rights_status,
                recap_prompt=self.settings.recap_prompt,
                legacy_wrapper=legacy_wrapper,
                connection_key=conn_key,
            )
            cached_manifest = self._load_final_plan_cache(plan_cache_key, project_id)
            if cached_manifest is not None:
                if log:
                    log(f"Sử dụng kết quả kế hoạch bình luận đã lưu trong bộ nhớ đệm ({len(cached_manifest.outputs)} outputs).")
                if on_phase:
                    on_phase(AnalysisPhase.OUTPUT_PLAN_READY, project_id, {"cached": True, "count": len(cached_manifest.outputs)})
                return cached_manifest

        # 3. Execution by scope
        outputs: list[CommentaryOutput] = []

        if scope_str == AnalysisScope.SINGLE_EPISODE.value:
            ep = episodes[0]
            ev = evidence_map.get(ep.episode_id)
            if ev is None:
                raise AnalysisError(f"Không có evidence cho tập {ep.episode_id}.")

            outputs = self.finalizer.finalize_single(
                ep,
                ev,
                cancel_event=cancel_event,
                on_phase=on_phase,
                log=log,
                legacy_wrapper=legacy_wrapper,
            )
        elif scope_str == AnalysisScope.SEASON.value:
            # Explicit Barrier: All episodes must complete evidence phase before season connection starts!
            if on_phase:
                on_phase(
                    AnalysisPhase.SEASON_BARRIER,
                    "season",
                    {
                        "total_episodes": len(episodes),
                        "scanned_episodes": len(evidence_map),
                    },
                )

            # Season Connection Pass
            connection_result = self.connector.connect_season(
                episodes=episodes,
                evidence_map=evidence_map,
                allow_incomplete=allow_incomplete,
                cancel_event=cancel_event,
                on_phase=on_phase,
                log=log,
            )

            # Candidate Mining & Finalization
            outputs = self.finalizer.finalize_season(
                episodes=episodes,
                evidence_map=evidence_map,
                connection_result=connection_result,
                cancel_event=cancel_event,
                on_phase=on_phase,
                log=log,
                legacy_wrapper=legacy_wrapper,
            )
        else:
            raise AnalysisError(f"Phạm vi phân tích không được hỗ trợ: {scope_str}")

        # 4. Construct & Validate AnalysisManifest
        manifest = AnalysisManifest(
            project_id=project_id,
            analysis_scope=scope_str,
            source_episodes=episodes,
            outputs=outputs,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            recap_language=self.settings.recap_language,
            recap_mode=self.settings.recap_mode,
            content_type=self.settings.content_type,
            source_rights_status=self.settings.source_rights_status,
        )
        manifest.validate()

        # 5. Save Plan Cache
        if use_final_plan_cache and plan_cache_key:
            try:
                self._save_final_plan_cache(plan_cache_key, manifest)
            except Exception as exc:
                if log:
                    log(f"Ghi chú: Không thể lưu plan cache: {exc}")

        if on_phase:
            on_phase(
                AnalysisPhase.OUTPUT_PLAN_READY,
                project_id,
                {"output_count": len(manifest.outputs)},
            )

        return manifest

    @property
    def plan_cache_dir(self) -> Path:
        cache_dir = self.cache_manager.cache_dir.parent / "analysis_plans"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _load_final_plan_cache(self, cache_key: str, project_id: str) -> AnalysisManifest | None:
        cache_file = self.plan_cache_dir / f"{cache_key}.json"
        if not cache_file.is_file():
            return None
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            raw["project_id"] = project_id
            return AnalysisManifest.from_dict(raw, validate=True)
        except Exception:
            return None

    def _save_final_plan_cache(self, cache_key: str, manifest: AnalysisManifest) -> None:
        cache_dir = self.plan_cache_dir
        cache_file = cache_dir / f"{cache_key}.json"
        tmp_file = cache_file.with_suffix(f".tmp.{threading.get_ident()}")
        tmp_file.write_text(manifest.to_json(indent=2), encoding="utf-8")
        tmp_file.replace(cache_file)


def run_analysis(
    project_id: str,
    episodes: list[SourceEpisode],
    *,
    scope: AnalysisScope | str = AnalysisScope.SINGLE_EPISODE,
    settings: AppSettings | None = None,
    client: OpenAICompatibleClient | None = None,
    cache_manager: EvidenceCacheManager | None = None,
    subtitle_pipeline: SubtitlePipeline | None = None,
    hierarchy_cache: HierarchyCacheManager | None = None,
    allow_incomplete: bool = False,
    injected_transcripts: dict[str, list[SubtitleCue] | list[tuple[float, float, str]]] | None = None,
    injected_probes: dict[str, MediaProbeResult | dict[str, Any]] | None = None,
    subtitle_provider: Callable[[SourceEpisode], list[SubtitleCue]] | None = None,
    cancel_event: threading.Event | None = None,
    on_phase: PhaseCallback | None = None,
    log: Callable[[str], None] | None = None,
    legacy_wrapper: bool = False,
) -> AnalysisManifest:
    """Convenience functional API for running single or season analysis."""
    engine = AnalysisEngine(
        settings=settings,
        client=client,
        cache_manager=cache_manager,
        subtitle_pipeline=subtitle_pipeline,
        hierarchy_cache=hierarchy_cache,
    )
    return engine.analyze(
        project_id=project_id,
        episodes=episodes,
        scope=scope,
        allow_incomplete=allow_incomplete,
        injected_transcripts=injected_transcripts,
        injected_probes=injected_probes,
        subtitle_provider=subtitle_provider,
        cancel_event=cancel_event,
        on_phase=on_phase,
        log=log,
        legacy_wrapper=legacy_wrapper,
    )
