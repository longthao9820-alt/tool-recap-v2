"""Canonical Source -> Scanner -> Finalizer -> Final JSON analysis engine."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..api_client import OpenAICompatibleClient
from ..domain.cache import EvidenceCacheManager, HierarchyCacheManager
from ..domain.enums import AnalysisScope, CandidateScope
from ..domain.models import (
    AnalysisManifest,
    CommentaryOutput,
    EpisodeEvidence,
    PipelineHealth,
    SourceEpisode,
    build_compact_summary,
)
from ..domain.policy import EditorialPolicy, OutputDirective
from ..paths import default_data_directory
from ..settings import AppSettings
from ..subtitles.models import MediaProbeResult, SubtitleCue
from ..subtitles.pipeline import SubtitlePipeline
from .candidates import CandidateConsolidator, CandidateDiscoverer, CandidateVerifier
from .candidates.consolidation import CONSOLIDATION_ALGO_VERSION
from .candidates.discovery import DISCOVERY_ALGO_VERSION
from .candidates.verifier import VERIFIER_ALGO_VERSION
from .connection import HIERARCHY_ALGO_VERSION, SeasonConnectionResult, SeasonConnector
from .errors import AnalysisCancelledError, AnalysisError, CoverageIncompleteError
from .evidence import EvidenceScanner
from .finalizer import CandidateFinalizer
from .final_json import CanonicalProjectFinalizer
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
    *,
    policy: EditorialPolicy | None = None,
    output_directive: OutputDirective | None = None,
    policy_hash: str | None = None,
    output_directive_hash: str | None = None,
) -> str:
    """Compute the Final JSON cache key from its actual upstream dependencies.

    Source/Scanner evidence, Finalizer role configuration, the raw Recap Prompt, and
    renderer contract metadata invalidate this layer. Voice and render settings do not.
    Legacy parameters remain accepted so existing callers and cache migrations are safe.
    """
    sorted_items = sorted(evidence_map.items(), key=lambda x: x[0])
    ev_hashes: list[str] = []
    for ep_id, ev in sorted_items:
        ev_raw = json.dumps(ev.to_dict(), sort_keys=True)
        h = hashlib.sha256(ev_raw.encode("utf-8")).hexdigest()[:16]
        ev_hashes.append(f"{ep_id}:{h}")

    effective_prompt = recap_prompt if recap_prompt is not None else getattr(settings, "recap_prompt", "")
    if policy is not None:
        eff_policy_hash = policy_hash or policy.policy_hash or policy.compute_policy_hash()
        eff_output_hash = output_directive_hash or policy.output_directive.directive_hash or policy.output_directive.compute_hash()
    elif policy_hash or output_directive_hash:
        eff_policy_hash = policy_hash or ""
        eff_output_hash = output_directive_hash or ""
    elif output_directive is not None:
        eff_policy_hash = ""
        eff_output_hash = output_directive.directive_hash or output_directive.compute_hash()
    elif effective_prompt:
        pol = EditorialPolicy.from_prompt(effective_prompt)
        eff_policy_hash = pol.policy_hash or pol.compute_policy_hash()
        eff_output_hash = pol.output_directive.directive_hash or pol.output_directive.compute_hash()
    else:
        pol = EditorialPolicy.from_prompt("")
        eff_policy_hash = pol.policy_hash or pol.compute_policy_hash()
        eff_output_hash = pol.output_directive.directive_hash or pol.output_directive.compute_hash()

    payload = {
        "analysis_plan_version": "final-json-v1",
        "scope": scope,
        "evidence_hashes": ev_hashes,
        "scanner_model": settings.scanner_model,
        "scanner_thinking": settings.scanner_thinking,
        "scanner_vision": bool(settings.scanner_supports_vision),
        "scanner_chunk_seconds": int(settings.api_chunk_seconds),
        "finalizer_model": settings.finalizer_model,
        "finalizer_thinking": settings.finalizer_thinking,
        "recap_prompt": effective_prompt,
        "recap_language": language or getattr(settings, "recap_language", "en-US"),
        "recap_mode": mode or getattr(settings, "recap_mode", "MAIN_STORIES"),
        "content_type": content_type or getattr(settings, "content_type", "US_TV_SHOW"),
        "rights": rights or getattr(settings, "source_rights_status", "UNVERIFIED"),
        "final_json_schema": "1.0",
        # Compatibility for callers that explicitly pass a pre-parsed policy;
        # the canonical UI path keys directly on the raw Recap Prompt above.
        "explicit_policy_hash": str(eff_policy_hash) if policy is not None or policy_hash else "",
        "explicit_output_directive_hash": str(eff_output_hash) if policy is not None or output_directive_hash else "",
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
        *,
        policy: EditorialPolicy | None = None,
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

        if policy is not None:
            self.policy = policy
        elif self.settings.recap_prompt:
            self.policy = EditorialPolicy.from_prompt(self.settings.recap_prompt)
        else:
            self.policy = EditorialPolicy.from_prompt("")

        self.scanner = EvidenceScanner(
            settings=self.settings,
            client=self.client,
            cache_manager=self.cache_manager,
            subtitle_pipeline=self.subtitle_pipeline,
            # Scanner performs broad source observation only. The raw Recap Prompt
            # belongs to the Finalizer and must not become an application-side
            # editorial filter at this stage.
            policy=EditorialPolicy.from_prompt(""),
        )
        # Retain the caller-supplied policy as diagnostic metadata without applying
        # its editorial directives to Scanner requests.
        self.scanner.policy = self.policy
        self.connector = SeasonConnector(
            settings=self.settings,
            client=self.client,
            hierarchy_cache=self.hierarchy_cache,
            policy=self.policy,
        )
        self.discoverer = CandidateDiscoverer(
            settings=self.settings, client=self.client, policy=self.policy, cache=self.hierarchy_cache
        )
        self.consolidator = CandidateConsolidator(
            settings=self.settings, client=self.client, policy=self.policy, cache=self.hierarchy_cache
        )
        self.verifier = CandidateVerifier(
            settings=self.settings, client=self.client, cache=self.hierarchy_cache, policy=self.policy
        )
        self.finalizer = CandidateFinalizer(
            settings=self.settings,
            client=self.client,
            hierarchy_cache=self.hierarchy_cache,
            policy=self.policy,
        )
        self.project_finalizer = (
            CanonicalProjectFinalizer(self.settings, self.client) if self.client is not None else None
        )

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
        policy: EditorialPolicy | None = None,
    ) -> AnalysisManifest:
        """Execute end-to-end analysis workflow returning a strictly validated AnalysisManifest."""
        if cancel_event and cancel_event.is_set():
            raise AnalysisCancelledError("Quy trình phân tích đã bị hủy.")

        if not episodes:
            raise AnalysisError("Không có tập phim (source_episodes) nào để phân tích.")

        scope_str = scope.value if isinstance(scope, AnalysisScope) else str(scope)
        transcripts = injected_transcripts or {}
        probes = injected_probes or {}

        effective_policy = policy if policy is not None else self.policy
        if policy is not None and policy is not self.policy:
            self.policy = policy
            self.scanner.policy = policy
            self.connector.policy = policy
            self.connector.connection_directive = policy.connection_directive
            self.discoverer.policy = policy
            self.consolidator.policy = policy
            self.verifier.policy = policy
            self.finalizer.policy = policy
            self.finalizer.output_directive = policy.output_directive

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
                policy=effective_policy,
            )
            cached_manifest = self._load_final_plan_cache(plan_cache_key, project_id)
            if cached_manifest is not None:
                if log:
                    log(f"Sử dụng kết quả kế hoạch bình luận đã lưu trong bộ nhớ đệm ({len(cached_manifest.outputs)} outputs).")
                if on_phase:
                    on_phase(AnalysisPhase.OUTPUT_PLAN_READY, project_id, {"cached": True, "count": len(cached_manifest.outputs)})
                return cached_manifest

        # 3. Canonical editorial boundary: Scanner observations -> configured Finalizer
        # -> one technically validated Final JSON. Legacy editorial helpers remain
        # import-compatible but are intentionally absent from this production path.
        if scope_str not in (AnalysisScope.SINGLE_EPISODE.value, AnalysisScope.SEASON.value):
            raise AnalysisError(f"Phạm vi phân tích không được hỗ trợ: {scope_str}")
        if on_phase and scope_str == AnalysisScope.SEASON.value:
            on_phase(AnalysisPhase.SEASON_BARRIER, "season", {
                "total_episodes": len(episodes), "scanned_episodes": len(evidence_map)
            })

        gateway_enabled = (
            self.settings.gateway_enabled
            and bool(self.settings.api_endpoint.strip())
            and self.settings.api_endpoint.strip().lower() != "offline"
            and self.client is not None
        )
        if not gateway_enabled:
            # Preserve deterministic offline behavior; no unavailable AI stages are invoked.
            outputs = []
            if scope_str == AnalysisScope.SEASON.value:
                outputs = self.finalizer.finalize_season(
                    episodes=episodes,
                    evidence_map=evidence_map,
                    connection_result=SeasonConnectionResult(),
                    cancel_event=cancel_event,
                    on_phase=on_phase,
                    log=log,
                    legacy_wrapper=legacy_wrapper,
                )
            else:
                for ep in episodes:
                    ev = evidence_map.get(ep.episode_id)
                    if ev is not None:
                        outputs.extend(self.finalizer.finalize_single(
                            ep, ev, cancel_event=cancel_event, on_phase=on_phase,
                            log=log, legacy_wrapper=legacy_wrapper,
                        ))
            verification = self.verifier.verify(
                scope_id=project_id,
                health=PipelineHealth(
                    coverage_ledgers={key: ev.coverage for key, ev in evidence_map.items()},
                    discovered_count=len(outputs),
                    consolidated_candidates=[],
                    finalizer_attempted=True,
                    finalizer_completed=True,
                    finalizer_results=[out.to_dict() for out in outputs],
                    total_evidence_count=sum(
                        len(items) for ev in evidence_map.values() for items in ev.data.values() if isinstance(items, list)
                    ),
                ),
                candidates=[],
                cancellation_token=cancel_event,
                phase_callback=on_phase,
            ) if not outputs else self.verifier.verify(
                scope_id=project_id,
                health=PipelineHealth(discovered_count=len(outputs), finalizer_attempted=True, finalizer_completed=True,
                                      finalizer_results=[out.to_dict() for out in outputs]),
                candidates=[], cancellation_token=cancel_event,
            )
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
                zero_output_reason=(str(verification.reason) if not outputs else None),
                zero_output_status=("OFFLINE_ZERO" if not outputs else None),
                verification=verification.to_dict(),
            )
            manifest.validate()
            if on_phase:
                on_phase(AnalysisPhase.OUTPUT_PLAN_READY, project_id, {"output_count": len(outputs)})
            return manifest

        if self.project_finalizer is None:
            raise AnalysisError("AI Gateway Finalizer client is not configured.")
        manifest = self.project_finalizer.finalize(
            project_id=project_id,
            episodes=episodes,
            evidence_map=evidence_map,
            scope=scope_str,
            cancel_event=cancel_event,
            on_phase=on_phase,
            log=log,
            # ProjectQueue verifies files before analysis. Direct unit/API callers may
            # provide synthetic source identities, so require disk existence whenever
            # this is a real prepared project rather than a synthetic contract test.
            require_source_files=all(Path(ep.source_video).is_file() for ep in episodes),
        )
        manifest.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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
            if not isinstance(raw, dict) or "outputs" not in raw or not isinstance(raw.get("outputs"), list):
                return None
            if not raw["outputs"] and raw.get("zero_output_status") != "VALID_EMPTY_OUTPUT":
                # Never let a legacy parser failure or incomplete cache masquerade
                # as an explicit editorial zero-output result.
                return None
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
    policy: EditorialPolicy | None = None,
) -> AnalysisManifest:
    """Convenience functional API for running single or season analysis."""
    engine = AnalysisEngine(
        settings=settings,
        client=client,
        cache_manager=cache_manager,
        subtitle_pipeline=subtitle_pipeline,
        hierarchy_cache=hierarchy_cache,
        policy=policy,
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
        policy=policy,
    )
