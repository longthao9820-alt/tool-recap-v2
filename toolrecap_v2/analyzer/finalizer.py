"""Candidate mining and script finalization for single episode and season commentary outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Callable

from ..api_client import (
    APIError,
    FINALIZER_TIMEOUT,
    OpenAICompatibleClient,
    _is_cancelled,
    estimate_request_size,
)
from ..domain.cache import (
    HierarchyCacheManager,
    compute_finalizer_cache_key,
    validate_finalizer_cache_data,
)
from ..domain.enums import AnalysisScope, AudioPolicy, CandidateScope, CompactionLevel, OutputStatus
from ..domain.models import (
    CommentaryOutput,
    EpisodeEvidence,
    Segment,
    SourceClip,
    SourceEpisode,
    ValidationError,
    build_compact_summary,
)
from ..domain.policy import (
    EditorialPolicy,
    OutputDirective,
    format_output_directive,
)
from ..domain.title import resolve_unique_titles, sanitize_title
from ..settings import AppSettings
from .connection import (
    HARD_PAYLOAD_CEILING,
    HIERARCHY_ALGO_VERSION,
    TARGET_PAYLOAD_CEILING,
    CandidateProposal,
    SeasonConnectionResult,
    check_payload_size,
    compact_merge_result,
)
from .errors import AnalysisCancelledError, AnalysisError
from .phases import AnalysisPhase, PhaseCallback
from .prompts import FINALIZER_SYSTEM_PROMPT as _BASE_FINALIZER_SYSTEM_PROMPT

FINALIZER_SYSTEM_PROMPT = """You are the lead editor and writer for a video recap pipeline.
Transform the candidate narrative proposals and evidence into polished commentary outputs.

Requirements:
1. OUTPUTS LIST: Return an "outputs" array containing 0, 1, or multiple commentary outputs. Empty list [] is valid if no candidate meets editorial quality.
2. TITLE PRESERVATION: Provide clear, compelling, unique titles for each output.
3. SOURCE CANDIDATE ID (MANDATORY): Each output MUST specify "source_candidate_id" (or "candidate_id") indicating the exact proposal_id from the candidate proposals that this output finalizes.
4. SEGMENTS & CLIPS: Each output consists of sequential segments.
   Every segment must define:
   - segment_id: Unique string (e.g. "seg_01").
   - source_clips: Array of clips from the source video(s). Each clip MUST define:
     - episode_id: Must match an actual source episode ID from the evidence.
     - source_video: Path or filename of the episode source video.
     - start: Timestamp in seconds (float or int >= 0.0).
     - end: Timestamp in seconds (float or int > start).
   - narration: Engaging commentary script for this sequence. Non-empty for commentary segments; empty only for original_only segments.
   - audio_policy: Audio policy intent for this sequence ("mix", "duck", "keep", "original_only"). Audio mix settings govern mix balance; original source audio is preserved under narration.
   - original_dialogue: Dialogue excerpt if applicable.
5. CROSS-EPISODE CLIPPING: An output with candidate_scope "CROSS_EPISODE" or "SEASON_ARC" can include source clips from different episodes in sequence.
6. STRICT TIMELINE BOUNDS: Clip timestamps must stay strictly within the source episode's duration (0 <= start < end <= duration). Do not invent timecodes or episodes.
7. NARRATION FIT: Select enough footage for every commentary segment so spoken narration fits naturally without truncation or overlap. Keep narration concise when evidence supports only a short clip.

Return JSON only:
{
  "outputs": [
    {
      "output_id": "out_01",
      "source_candidate_id": "prop_01",
      "title": "Clear and Engaging Title",
      "candidate_scope": "CROSS_EPISODE",
      "segments": [
        {
          "segment_id": "seg_01",
          "source_clips": [
            {
              "episode_id": "E01",
              "source_video": "e01.mp4",
              "start": 10.0,
              "end": 35.0
            }
          ],
          "narration": "Narration text for this segment...",
          "audio_policy": "duck",
          "original_dialogue": ""
        }
      ]
    }
  ]
}
""".strip()

MAX_PAYLOAD_BYTES = HARD_PAYLOAD_CEILING


@dataclass
class FinalizationResult:
    outputs: list[CommentaryOutput] = field(default_factory=list)
    attempted_candidates: list[str] = field(default_factory=list)
    accepted_candidate_ids: list[str] = field(default_factory=list)
    rejected_candidate_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    completed: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_candidate_ids)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_candidate_ids)

    @property
    def attempted_count(self) -> int:
        return len(self.attempted_candidates)

    @property
    def outputs_count(self) -> int:
        return len(self.outputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": [o.to_dict() if hasattr(o, "to_dict") else o for o in self.outputs],
            "attempted_candidates": list(self.attempted_candidates),
            "accepted_candidate_ids": list(self.accepted_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "errors": list(self.errors),
            "completed": self.completed,
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinalizationResult:
        if not isinstance(data, dict):
            return cls()
        raw_outs = data.get("outputs", [])
        outs = [
            o if isinstance(o, CommentaryOutput) else CommentaryOutput.from_dict(o)
            for o in raw_outs
            if isinstance(o, (dict, CommentaryOutput))
        ]
        return cls(
            outputs=outs,
            attempted_candidates=[str(c) for c in data.get("attempted_candidates", [])],
            accepted_candidate_ids=[str(c) for c in data.get("accepted_candidate_ids", [])],
            rejected_candidate_ids=[str(c) for c in data.get("rejected_candidate_ids", [])],
            errors=[str(e) for e in data.get("errors", [])],
            completed=bool(data.get("completed", False)),
            diagnostics=dict(data.get("diagnostics", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> FinalizationResult:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValidationError("JSON root must be an object.")
        return cls.from_dict(data)


FinalizerResult = FinalizationResult


@dataclass
class FinalizerGroupPlan:
    group_id: str
    user_text: str
    estimated_bytes: int
    proposals: list[dict[str, Any]]


def format_finalizer_user_text(
    episodes: list[SourceEpisode],
    connection_payload: dict[str, Any],
    settings: AppSettings | None = None,
    scope_hint: str | None = None,
    is_complete: bool = True,
    missing_episodes: list[str] | None = None,
    recap_prompt: str | OutputDirective = "",
    *,
    output_directive: OutputDirective | str | None = None,
) -> str:
    """Format finalizer prompt without raw evidence categories or transcripts."""
    episodes_summary = [
        {
            "episode_id": ep.episode_id,
            "source_video": ep.source_video,
            "duration_seconds": ep.duration_seconds,
            "title": ep.title,
        }
        for ep in episodes
    ]

    coverage_notice = ""
    if not is_complete and missing_episodes:
        coverage_notice = (
            f"\nCRITICAL COVERAGE NOTICE:\n"
            f"Missing episodes: {missing_episodes}.\n"
            f"This is a PARTIAL season analysis. All outputs and clips MUST ONLY reference available episodes.\n"
        )

    scope_restriction = ""
    if scope_hint == CandidateScope.SINGLE_EPISODE.value or scope_hint == "SINGLE_EPISODE":
        scope_restriction = (
            f"\nSCOPE RESTRICTION: Candidate scope MUST be SINGLE_EPISODE. "
            f"All commentary outputs and source clips MUST ONLY reference episode {episodes[0].episode_id}. "
            f"Do not produce cross-episode or season-arc outputs.\n"
        )

    if output_directive is not None:
        if isinstance(output_directive, OutputDirective):
            effective_instructions = format_output_directive(output_directive)
        else:
            effective_instructions = str(output_directive)
    elif isinstance(recap_prompt, OutputDirective):
        effective_instructions = format_output_directive(recap_prompt)
    elif recap_prompt and str(recap_prompt).strip():
        policy = EditorialPolicy.from_prompt(str(recap_prompt))
        formatted = format_output_directive(policy.output_directive)
        if formatted and formatted != "Standard video recap commentary and output finalization.":
            effective_instructions = formatted
        else:
            effective_instructions = str(recap_prompt).strip()
    elif settings is not None and getattr(settings, "recap_prompt", None) and str(settings.recap_prompt).strip():
        raw_p = str(settings.recap_prompt).strip()
        policy = EditorialPolicy.from_prompt(raw_p)
        effective_instructions = format_output_directive(policy.output_directive)
    else:
        effective_instructions = "Standard video recap commentary and output finalization."

    if not effective_instructions.strip():
        effective_instructions = "Standard video recap commentary and output finalization."

    lang = getattr(settings, "recap_language", "en-US") if settings else "en-US"
    mode = getattr(settings, "recap_mode", "MAIN_STORIES") if settings else "MAIN_STORIES"
    content_type = getattr(settings, "content_type", "US_TV_SHOW") if settings else "US_TV_SHOW"
    rights = getattr(settings, "source_rights_status", "UNVERIFIED") if settings else "UNVERIFIED"

    target_label = "episode" if (scope_hint == "SINGLE_EPISODE" or len(episodes) == 1) else "season"
    return (
        f"Finalize commentary outputs for the {target_label}.\n"
        f"{coverage_notice}"
        f"{scope_restriction}\n"
        f"CRITICAL REQUIREMENT: Each commentary output in 'outputs' MUST specify 'source_candidate_id' matching the proposal_id of the candidate proposal being finalized.\n"
        f"Recap instructions:\n{effective_instructions}\n"
        f"Recap Language: {lang}\n"
        f"Recap Mode: {mode}\n"
        f"Content Type: {content_type}\n"
        f"Rights Status: {rights}\n\n"
        f"Source Episodes:\n{json.dumps(episodes_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Season Connection Analysis & Candidate Proposals:\n"
        f"{json.dumps(connection_payload, ensure_ascii=False, indent=2)}\n"
    )


def estimate_finalizer_request_size(
    model: str,
    user_text: str,
    thinking: str = "auto",
    max_tokens: int = 32_000,
) -> int:
    """Estimate exact request size for finalizer payload."""
    return estimate_request_size(
        model=model,
        system=FINALIZER_SYSTEM_PROMPT,
        user_text=user_text,
        thinking=thinking,
        max_tokens=max_tokens,
        variant=0,
    )


def cap_long_fields(data: Any, max_len: int = 120, safe_keys: set[str] | None = None) -> Any:
    """Recursively cap oversized string fields in nested structures, preserving identifiers."""
    if safe_keys is None:
        safe_keys = {"proposal_id", "thread_id", "episode_id", "candidate_scope", "status"}
    if isinstance(data, dict):
        res: dict[str, Any] = {}
        for k, v in data.items():
            if k in safe_keys:
                res[k] = v
            elif isinstance(v, str) and len(v) > max_len:
                res[k] = v[:max_len]
            else:
                res[k] = cap_long_fields(v, max_len, safe_keys)
        return res
    elif isinstance(data, list):
        return [cap_long_fields(item, max_len, safe_keys) for item in data]
    return data


def deterministic_cap_proposal_entry(prop: dict[str, Any], cap: int = 80) -> dict[str, Any]:
    """Deterministically cap proposal string descriptions while keeping IDs, episodes, characters, title safe."""
    p_copy = dict(prop)
    if "editorial_reason" in p_copy:
        p_copy["editorial_reason"] = str(p_copy["editorial_reason"])[:cap]
    if "description" in p_copy:
        p_copy["description"] = str(p_copy["description"])[:cap]
    if "title" in p_copy:
        p_copy["title"] = str(p_copy["title"])[:max(cap, 120)]
    if "characters" in p_copy and isinstance(p_copy["characters"], list):
        p_copy["characters"] = [str(c)[:cap] for c in p_copy["characters"][:5]]
    return cap_long_fields(p_copy, max_len=cap)


def deterministic_cap_finalizer_dict(raw: dict[str, Any], cap: int = 80) -> dict[str, Any]:
    """Deterministically cap string fields in connection dictionary."""
    props = raw.get("candidate_proposals", [])
    links = raw.get("cross_episode_links", [])
    arcs = raw.get("supporting_character_arcs", [])
    rej = raw.get("rejected_or_merged", [])

    out_props = [deterministic_cap_proposal_entry(p, cap=cap) if isinstance(p, dict) else p for p in props]
    out_links = []
    for l in links:
        l_copy = dict(l)
        if "theme" in l_copy:
            l_copy["theme"] = str(l_copy["theme"])[:cap]
        if "summary" in l_copy:
            l_copy["summary"] = str(l_copy["summary"])[:cap]
        out_links.append(cap_long_fields(l_copy, max_len=cap))

    out_arcs = []
    for a in arcs:
        a_copy = dict(a)
        if "character" in a_copy:
            a_copy["character"] = str(a_copy["character"])[:cap]
        if "name" in a_copy:
            a_copy["name"] = str(a_copy["name"])[:cap]
        if "arc_summary" in a_copy:
            a_copy["arc_summary"] = str(a_copy["arc_summary"])[:cap]
        out_arcs.append(cap_long_fields(a_copy, max_len=cap))

    return {
        "cross_episode_links": out_links,
        "candidate_proposals": out_props,
        "supporting_character_arcs": out_arcs,
        "rejected_or_merged": rej[:10] if cap >= 80 else [],
    }


def plan_finalizer_groups(
    episodes: list[SourceEpisode],
    connection_result: SeasonConnectionResult | dict[str, Any],
    settings: AppSettings,
    *,
    scope_hint: str | None = None,
    is_complete: bool = True,
    missing_episodes: list[str] | None = None,
    target_ceiling: int = TARGET_PAYLOAD_CEILING,
    hard_ceiling: int = HARD_PAYLOAD_CEILING,
    output_directive: OutputDirective | str | None = None,
) -> list[FinalizerGroupPlan]:
    """Plan bounded finalizer groups progressively compacting and adaptively partitioning proposals."""
    if output_directive is None and getattr(settings, "recap_prompt", None):
        p = EditorialPolicy.from_prompt(settings.recap_prompt)
        output_directive = p.output_directive

    # 1. Check baseline envelope
    base_text = format_finalizer_user_text(
        episodes,
        {"cross_episode_links": [], "candidate_proposals": [], "supporting_character_arcs": [], "rejected_or_merged": []},
        settings,
        scope_hint=scope_hint,
        is_complete=is_complete,
        missing_episodes=missing_episodes,
        output_directive=output_directive,
    )
    base_size = estimate_finalizer_request_size(settings.finalizer_model, base_text, settings.finalizer_thinking)
    if base_size > target_ceiling:
        raise AnalysisError(
            f"Baseline finalizer envelope ({base_size} bytes) exceeds target ceiling {target_ceiling} bytes. "
            f"Request cannot be constructed."
        )

    conn_dict = connection_result.to_dict() if isinstance(connection_result, SeasonConnectionResult) else dict(connection_result)
    raw_props = [p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in conn_dict.get("candidate_proposals", [])]
    conn_dict["candidate_proposals"] = raw_props

    # 2. Try FULL compaction on the full connection
    full_compacted = compact_merge_result(conn_dict, CompactionLevel.FULL)
    full_text = format_finalizer_user_text(
        episodes,
        full_compacted,
        settings,
        scope_hint=scope_hint,
        is_complete=is_complete,
        missing_episodes=missing_episodes,
        output_directive=output_directive,
    )
    full_size = estimate_finalizer_request_size(settings.finalizer_model, full_text, settings.finalizer_thinking)
    if full_size <= target_ceiling:
        return [
            FinalizerGroupPlan(
                group_id="group_01",
                user_text=full_text,
                estimated_bytes=full_size,
                proposals=full_compacted.get("candidate_proposals", []),
            )
        ]

    # 3. If FULL doesn't fit, partition proposals into bounded groups
    proposals = [p for p in raw_props if p.get("status") not in ("rejected", "merged")]
    if not proposals:
        proposals = raw_props

    all_links = conn_dict.get("cross_episode_links", [])
    all_arcs = conn_dict.get("supporting_character_arcs", [])

    if not proposals:
        for lvl in (CompactionLevel.TRIMMED, CompactionLevel.PRIORITY, CompactionLevel.SKELETON):
            c = compact_merge_result(conn_dict, lvl)
            t = format_finalizer_user_text(
                episodes,
                c,
                settings,
                scope_hint=scope_hint,
                is_complete=is_complete,
                missing_episodes=missing_episodes,
                output_directive=output_directive,
            )
            s = estimate_finalizer_request_size(settings.finalizer_model, t, settings.finalizer_thinking)
            if s <= target_ceiling:
                return [FinalizerGroupPlan(group_id="group_01", user_text=t, estimated_bytes=s, proposals=[])]
        raise AnalysisError("Connection result exceeds ceiling even without proposals.")

    def _build_group_payload(props: list[dict[str, Any]], cap: int | None = None) -> tuple[dict[str, Any], str, int]:
        grp_episodes = {ep for pr in props for ep in pr.get("episodes", [])}
        grp_characters = {c for pr in props for c in pr.get("characters", [])}
        grp_links = [l for l in all_links if any(ep in grp_episodes for ep in l.get("episodes", []))]
        grp_arcs = [a for a in all_arcs if a.get("character") in grp_characters or any(ep in grp_episodes for ep in a.get("episodes", []))]
        raw_grp = {
            "cross_episode_links": grp_links,
            "candidate_proposals": props,
            "supporting_character_arcs": grp_arcs,
            "rejected_or_merged": [],
        }
        if cap is not None:
            skel = compact_merge_result(raw_grp, CompactionLevel.SKELETON)
            payload = deterministic_cap_finalizer_dict(skel, cap=cap)
        else:
            payload = raw_grp

        t = format_finalizer_user_text(
            episodes,
            payload,
            settings,
            scope_hint=scope_hint,
            is_complete=is_complete,
            missing_episodes=missing_episodes,
            output_directive=output_directive,
        )
        s = estimate_finalizer_request_size(settings.finalizer_model, t, settings.finalizer_thinking)
        return payload, t, s

    groups: list[FinalizerGroupPlan] = []
    curr_props: list[dict[str, Any]] = []

    def _cap_single_proposal_if_needed(p: dict[str, Any]) -> dict[str, Any]:
        _, _, s = _build_group_payload([p])
        if s <= target_ceiling:
            return p
        for c in (120, 80, 40, 0):
            capped_p = deterministic_cap_proposal_entry(p, cap=c)
            _, _, s = _build_group_payload([capped_p])
            if s <= target_ceiling:
                return capped_p
        capped_p = deterministic_cap_proposal_entry(p, cap=0)
        p_payload = {
            "cross_episode_links": [],
            "candidate_proposals": [capped_p],
            "supporting_character_arcs": [],
            "rejected_or_merged": [],
        }
        t = format_finalizer_user_text(
            episodes,
            p_payload,
            settings,
            scope_hint=scope_hint,
            is_complete=is_complete,
            missing_episodes=missing_episodes,
            output_directive=output_directive,
        )
        s = estimate_finalizer_request_size(settings.finalizer_model, t, settings.finalizer_thinking)
        if s <= target_ceiling:
            return capped_p
        raise AnalysisError(
            f"Candidate proposal '{p.get('proposal_id', 'unknown')}' indivisible and exceeds target ceiling {target_ceiling} bytes ({s} bytes)."
        )

    for p in proposals:
        effective_p = _cap_single_proposal_if_needed(p)
        test_props = curr_props + [effective_p]
        _, test_txt, test_size = _build_group_payload(test_props)
        if test_size <= target_ceiling:
            curr_props.append(effective_p)
        else:
            if curr_props:
                grp_payload, grp_txt, grp_size = _build_group_payload(curr_props)
                groups.append(
                    FinalizerGroupPlan(
                        group_id=f"group_{len(groups)+1:02d}",
                        user_text=grp_txt,
                        estimated_bytes=grp_size,
                        proposals=grp_payload.get("candidate_proposals", []),
                    )
                )
            curr_props = [effective_p]

    if curr_props:
        grp_payload, grp_txt, grp_size = _build_group_payload(curr_props)
        groups.append(
            FinalizerGroupPlan(
                group_id=f"group_{len(groups)+1:02d}",
                user_text=grp_txt,
                estimated_bytes=grp_size,
                proposals=grp_payload.get("candidate_proposals", []),
            )
        )

    return groups


class CandidateFinalizer:
    """Mines candidate proposals and finalizes CommentaryOutput plans."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        client: OpenAICompatibleClient | None = None,
        hierarchy_cache: HierarchyCacheManager | None = None,
        *,
        policy: EditorialPolicy | None = None,
        output_directive: OutputDirective | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.client = client
        self.hierarchy_cache = hierarchy_cache

        if policy is not None:
            self.policy = policy
            self.output_directive = policy.output_directive
        elif output_directive is not None:
            self.output_directive = output_directive
            self.policy = None
        elif self.settings.recap_prompt:
            self.policy = EditorialPolicy.from_prompt(self.settings.recap_prompt)
            self.output_directive = self.policy.output_directive
        else:
            self.policy = EditorialPolicy.from_prompt("")
            self.output_directive = self.policy.output_directive

    def finalize_single(
        self,
        episode: SourceEpisode,
        evidence: EpisodeEvidence,
        *,
        cancel_event: threading.Event | None = None,
        on_phase: PhaseCallback | None = None,
        log: Callable[[str], None] | None = None,
        legacy_wrapper: bool = False,
        output_directive: OutputDirective | None = None,
    ) -> list[CommentaryOutput]:
        """Finalize commentary outputs for a single episode without sending raw evidence."""
        if _is_cancelled(cancel_event):
            raise AnalysisCancelledError("Finalizer đã bị hủy.")

        if on_phase:
            on_phase(AnalysisPhase.SEASON_MINING, episode.episode_id, {"status": "finalizing_single"})

        is_gateway_enabled = (
            self.settings.gateway_enabled
            and bool(self.settings.api_endpoint.strip())
            and self.settings.api_endpoint.strip().lower() != "offline"
            and self.client is not None
        )

        if not is_gateway_enabled:
            if legacy_wrapper:
                return self._finalize_single_offline(episode, evidence)
            return []

        # Direct callers already completed episode scanning. Build a grounded compact
        # candidate locally instead of repeating Season Connection. AnalysisEngine uses
        # the canonical connector pass before calling finalize_from_connection.
        compact = build_compact_summary(episode, evidence)
        characters = sorted({character for item in compact.items for character in item.characters})
        evidence_digest = " | ".join(item.summary for item in compact.items)
        connection_result = SeasonConnectionResult(
            cross_episode_links=[],
            candidate_proposals=[
                CandidateProposal(
                    proposal_id=f"single_{hashlib.sha256(compact.canonical_hash().encode('utf-8')).hexdigest()[:16]}",
                    title=episode.title or episode.episode_id,
                    candidate_scope=CandidateScope.SINGLE_EPISODE.value,
                    episodes=[episode.episode_id],
                    characters=characters,
                    editorial_reason=evidence_digest,
                    description=evidence_digest,
                    status="keep",
                )
            ] if compact.items else [],
            supporting_character_arcs=[],
            rejected_or_merged=[],
            is_complete=True,
            missing_episodes=[],
        )
        return self.finalize_from_connection(
            episodes=[episode],
            connection_result=connection_result,
            cancel_event=cancel_event,
            on_phase=on_phase,
            log=log,
            legacy_wrapper=legacy_wrapper,
            scope_hint=CandidateScope.SINGLE_EPISODE.value,
            hierarchy_cache=self.hierarchy_cache,
            output_directive=output_directive or self.output_directive,
        )

    def finalize_from_connection(
        self,
        episodes: list[SourceEpisode],
        connection_result: SeasonConnectionResult,
        *,
        cancel_event: threading.Event | None = None,
        on_phase: PhaseCallback | None = None,
        log: Callable[[str], None] | None = None,
        legacy_wrapper: bool = False,
        scope_hint: str | None = None,
        hierarchy_cache: HierarchyCacheManager | None = None,
        output_directive: OutputDirective | None = None,
    ) -> list[CommentaryOutput]:
        """Finalize commentary outputs from SeasonConnectionResult with bounded adaptive grouping."""
        if _is_cancelled(cancel_event):
            raise AnalysisCancelledError("Finalizer đã bị hủy.")

        is_gateway_enabled = (
            self.settings.gateway_enabled
            and bool(self.settings.api_endpoint.strip())
            and self.settings.api_endpoint.strip().lower() != "offline"
            and self.client is not None
        )

        if not is_gateway_enabled:
            if legacy_wrapper:
                return self._finalize_season_offline(episodes, {}, connection_result)
            return []

        effective_cache = hierarchy_cache if hierarchy_cache is not None else self.hierarchy_cache
        effective_output_dir = output_directive if output_directive is not None else self.output_directive
        phase_scope = episodes[0].episode_id if (scope_hint == CandidateScope.SINGLE_EPISODE.value and len(episodes) == 1) else "season"
        if on_phase:
            on_phase(AnalysisPhase.SEASON_MINING, phase_scope, {"status": "finalizing_from_connection"})

        groups = plan_finalizer_groups(
            episodes=episodes,
            connection_result=connection_result,
            settings=self.settings,
            scope_hint=scope_hint,
            is_complete=connection_result.is_complete,
            missing_episodes=connection_result.missing_episodes,
            output_directive=effective_output_dir,
        )

        all_outputs: list[CommentaryOutput] = []
        seen_output_ids: set[str] = set()

        for group_idx, group in enumerate(groups, start=1):
            if _is_cancelled(cancel_event):
                raise AnalysisCancelledError("Finalizer đã bị hủy.")

            check_payload_size(group.user_text, max_bytes=HARD_PAYLOAD_CEILING, context=group.group_id)

            raw_result = None
            group_cache_key = ""
            if effective_cache is not None:
                payload_hash = hashlib.sha256(group.user_text.encode("utf-8")).hexdigest()
                prompt_ver = hashlib.sha256(FINALIZER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
                out_dir_hash = (
                    effective_output_dir.directive_hash or effective_output_dir.compute_hash()
                    if effective_output_dir is not None
                    else ""
                )
                group_cache_key = compute_finalizer_cache_key(
                    group_payload_hash=payload_hash,
                    model=self.settings.finalizer_model,
                    thinking=self.settings.finalizer_thinking,
                    prompt_version=prompt_ver,
                    scope=scope_hint or "SEASON",
                    algo=HIERARCHY_ALGO_VERSION,
                    recap_prompt=self.settings.recap_prompt or "",
                    recap_language=self.settings.recap_language or "",
                    recap_mode=self.settings.recap_mode or "",
                    content_type=self.settings.content_type or "",
                    source_rights_status=self.settings.source_rights_status or "",
                    voice_style=self.settings.voice_style or "",
                    output_directive_hash=out_dir_hash,
                )
                cached_res, meta = effective_cache.load_finalizer_result(group.group_id, group_cache_key)
                if meta.get("hit") and cached_res is not None:
                    raw_result = cached_res
                    if log:
                        log(f"Sử dụng kết quả finalizer đã lưu trong bộ nhớ đệm cho nhóm {group.group_id}.")
                    if on_phase:
                        on_phase(
                            AnalysisPhase.SEASON_MINING,
                            phase_scope,
                            {"status": "cached", "group": group.group_id, "cached": True},
                        )

            if raw_result is None:
                def _status_cb(msg: str) -> None:
                    if on_phase:
                        on_phase(
                            AnalysisPhase.SEASON_MINING,
                            phase_scope,
                            {"status": msg, "status_message": msg, "group": group.group_id},
                        )

                call_kwargs: dict[str, Any] = {
                    "model": self.settings.finalizer_model,
                    "thinking": self.settings.finalizer_thinking,
                    "system": FINALIZER_SYSTEM_PROMPT,
                    "user_text": group.user_text,
                    "cancel_event": cancel_event,
                    "phase": "finalizer",
                    "timeout": FINALIZER_TIMEOUT,
                    "on_status": _status_cb,
                    "log": log,
                }
                try:
                    try:
                        raw_result = self.client.chat_json(**call_kwargs)
                    except TypeError as te:
                        if "unexpected keyword argument" in str(te):
                            filtered = {k: v for k, v in call_kwargs.items() if k not in ("phase", "timeout", "on_status", "log")}
                            raw_result = self.client.chat_json(**filtered)
                        else:
                            raise
                except AnalysisCancelledError:
                    raise
                except APIError as exc:
                    if _is_cancelled(cancel_event) or "đã bị dừng" in str(exc) or "bị hủy" in str(exc):
                        raise AnalysisCancelledError("Finalizer đã bị hủy.") from exc
                    raise AnalysisError(
                        f"Không thể kết nối đến AI Gateway ({self.settings.api_endpoint}): Finalizer nhóm {group.group_id} lỗi sau các lần thử: {exc}. "
                        f"Dữ liệu evidence và liên kết đã được lưu an toàn trong bộ nhớ đệm (preserved work)."
                    ) from exc

                to_cache = raw_result
                if isinstance(raw_result, dict) and "outputs" not in raw_result and "segments" in raw_result:
                    to_cache = {"outputs": [raw_result]}

                if effective_cache is not None and group_cache_key and validate_finalizer_cache_data(to_cache):
                    try:
                        effective_cache.save_finalizer_result(group.group_id, group_cache_key, to_cache)
                    except Exception as exc:
                        if log:
                            log(f"Cảnh báo: Không thể lưu cache cho nhóm finalizer {group.group_id}: {exc}")

                if _is_cancelled(cancel_event):
                    raise AnalysisCancelledError("Finalizer đã bị hủy.")

            group_outputs = self.parse_outputs(raw_result, episodes, scope_hint=scope_hint)
            for out in group_outputs:
                base_id = out.output_id
                unique_id = base_id
                suffix = 1
                while unique_id in seen_output_ids:
                    unique_id = f"{base_id}_{suffix}"
                    suffix += 1
                out.output_id = unique_id
                seen_output_ids.add(unique_id)
                all_outputs.append(out)

        # Title deduplication across all outputs
        raw_titles = [out.title or out.output_id for out in all_outputs]
        unique_titles = resolve_unique_titles(raw_titles, max_length=120)
        for out, san_title in zip(all_outputs, unique_titles):
            out.sanitized_title = san_title

        return all_outputs

    def finalize_season(
        self,
        episodes: list[SourceEpisode],
        evidence_map: dict[str, EpisodeEvidence],
        connection_result: SeasonConnectionResult,
        *,
        cancel_event: threading.Event | None = None,
        on_phase: PhaseCallback | None = None,
        log: Callable[[str], None] | None = None,
        legacy_wrapper: bool = False,
        scope_hint: str | None = None,
        output_directive: OutputDirective | None = None,
    ) -> list[CommentaryOutput]:
        """Finalize commentary outputs for a full season with cross-episode candidates."""
        is_gateway_enabled = (
            self.settings.gateway_enabled
            and bool(self.settings.api_endpoint.strip())
            and self.settings.api_endpoint.strip().lower() != "offline"
            and self.client is not None
        )

        if not is_gateway_enabled:
            if legacy_wrapper:
                return self._finalize_season_offline(episodes, evidence_map, connection_result)
            return []

        return self.finalize_from_connection(
            episodes=episodes,
            connection_result=connection_result,
            cancel_event=cancel_event,
            on_phase=on_phase,
            log=log,
            legacy_wrapper=legacy_wrapper,
            scope_hint=scope_hint,
            hierarchy_cache=self.hierarchy_cache,
            output_directive=output_directive or self.output_directive,
        )

    def finalize_candidates(
        self,
        candidates: list[CandidateProposal | dict[str, Any]],
        source_episodes: list[SourceEpisode],
        scope_type: AnalysisScope | CandidateScope | str = CandidateScope.CROSS_EPISODE.value,
        topology: SeasonConnectionResult | dict[str, Any] | None = None,
        *,
        cancel_event: threading.Event | None = None,
        on_phase: PhaseCallback | None = None,
        log: Callable[[str], None] | None = None,
        legacy_wrapper: bool = False,
        hierarchy_cache: HierarchyCacheManager | None = None,
        output_directive: OutputDirective | None = None,
        return_diagnostics: bool = True,
    ) -> FinalizationResult | list[CommentaryOutput]:
        """Finalize commentary outputs strictly for consolidated candidates without rediscovery."""
        if _is_cancelled(cancel_event):
            raise AnalysisCancelledError("Finalizer đã bị hủy.")

        if hasattr(scope_type, "value"):
            resolved_scope = str(scope_type.value)
        else:
            resolved_scope = str(scope_type) if scope_type else CandidateScope.CROSS_EPISODE.value

        attempted_candidate_ids: list[str] = []
        for idx, c in enumerate(candidates, start=1):
            cid = ""
            if hasattr(c, "proposal_id") and c.proposal_id:
                cid = str(c.proposal_id).strip()
            elif hasattr(c, "candidate_id") and c.candidate_id:
                cid = str(c.candidate_id).strip()
            elif isinstance(c, dict):
                cid = str(c.get("proposal_id") or c.get("candidate_id") or "").strip()
            if not cid:
                cid = f"cand_{idx:02d}"
            attempted_candidate_ids.append(cid)

        if not candidates:
            res = FinalizationResult(
                outputs=[],
                attempted_candidates=[],
                accepted_candidate_ids=[],
                rejected_candidate_ids=[],
                errors=[],
                completed=True,
                diagnostics={"zero_candidates": True},
            )
            return res if return_diagnostics else res.outputs

        converted_candidates: list[CandidateProposal] = []
        for cid, c in zip(attempted_candidate_ids, candidates):
            if isinstance(c, CandidateProposal):
                cp = CandidateProposal.from_dict(c.to_dict())
                cp.proposal_id = cid
                if cp.status in ("rejected", "merged"):
                    cp.status = "keep"
                converted_candidates.append(cp)
            elif isinstance(c, dict):
                d = dict(c)
                d["proposal_id"] = cid
                if d.get("status") in ("rejected", "merged"):
                    d["status"] = "keep"
                converted_candidates.append(CandidateProposal.from_dict(d))
            elif hasattr(c, "to_dict"):
                d = c.to_dict()
                d["proposal_id"] = cid
                if d.get("status") in ("rejected", "merged"):
                    d["status"] = "keep"
                converted_candidates.append(CandidateProposal.from_dict(d))
            else:
                converted_candidates.append(
                    CandidateProposal(
                        proposal_id=cid,
                        title=getattr(c, "title", cid),
                        status="keep",
                    )
                )

        if isinstance(topology, SeasonConnectionResult):
            conn_result = SeasonConnectionResult(
                cross_episode_links=list(topology.cross_episode_links),
                candidate_proposals=converted_candidates,
                supporting_character_arcs=list(topology.supporting_character_arcs),
                rejected_or_merged=list(topology.rejected_or_merged),
                is_complete=topology.is_complete,
                missing_episodes=list(topology.missing_episodes),
            )
        elif isinstance(topology, dict):
            conn_result = SeasonConnectionResult(
                cross_episode_links=list(topology.get("cross_episode_links", [])),
                candidate_proposals=converted_candidates,
                supporting_character_arcs=list(topology.get("supporting_character_arcs", [])),
                rejected_or_merged=list(topology.get("rejected_or_merged", [])),
                is_complete=bool(topology.get("is_complete", True)),
                missing_episodes=list(topology.get("missing_episodes", [])),
            )
        else:
            conn_result = SeasonConnectionResult(
                cross_episode_links=[],
                candidate_proposals=converted_candidates,
                supporting_character_arcs=[],
                rejected_or_merged=[],
                is_complete=True,
                missing_episodes=[],
            )

        is_gateway_enabled = (
            self.settings.gateway_enabled
            and bool(self.settings.api_endpoint.strip())
            and self.settings.api_endpoint.strip().lower() != "offline"
            and self.client is not None
        )

        if not is_gateway_enabled:
            if legacy_wrapper:
                offline_outs = self._finalize_season_offline(source_episodes, {}, conn_result)
                for idx, out in enumerate(offline_outs):
                    if idx < len(attempted_candidate_ids):
                        out.source_candidate_id = attempted_candidate_ids[idx]
                        out.candidate_id = attempted_candidate_ids[idx]
                accepted_ids = [out.source_candidate_id for out in offline_outs if getattr(out, "source_candidate_id", "")]
                res = FinalizationResult(
                    outputs=offline_outs,
                    attempted_candidates=attempted_candidate_ids,
                    accepted_candidate_ids=accepted_ids,
                    rejected_candidate_ids=[cid for cid in attempted_candidate_ids if cid not in set(accepted_ids)],
                    errors=[],
                    completed=True,
                    diagnostics={"offline": True},
                )
                return res if return_diagnostics else res.outputs
            res = FinalizationResult(
                outputs=[],
                attempted_candidates=attempted_candidate_ids,
                accepted_candidate_ids=[],
                rejected_candidate_ids=list(attempted_candidate_ids),
                errors=["Gateway disabled or client offline."],
                completed=True,
                diagnostics={"offline_disabled": True},
            )
            return res if return_diagnostics else res.outputs

        effective_cache = hierarchy_cache if hierarchy_cache is not None else self.hierarchy_cache
        effective_output_dir = output_directive if output_directive is not None else self.output_directive
        phase_scope = source_episodes[0].episode_id if (resolved_scope == CandidateScope.SINGLE_EPISODE.value and len(source_episodes) == 1) else "season"

        if on_phase:
            on_phase(AnalysisPhase.SEASON_MINING, phase_scope, {"status": "finalizing_candidates"})

        groups = plan_finalizer_groups(
            episodes=source_episodes,
            connection_result=conn_result,
            settings=self.settings,
            scope_hint=resolved_scope,
            is_complete=conn_result.is_complete,
            missing_episodes=conn_result.missing_episodes,
            output_directive=effective_output_dir,
        )

        all_accepted_outputs: list[CommentaryOutput] = []
        accepted_candidate_id_set: set[str] = set()
        seen_output_ids: set[str] = set()
        errors: list[str] = []

        out_dir_hash = (
            effective_output_dir.directive_hash or effective_output_dir.compute_hash()
            if effective_output_dir is not None
            else ""
        )

        for group in groups:
            if _is_cancelled(cancel_event):
                raise AnalysisCancelledError("Finalizer đã bị hủy.")

            check_payload_size(group.user_text, max_bytes=HARD_PAYLOAD_CEILING, context=group.group_id)

            group_cids = [
                str(p.get("proposal_id", "") or p.get("candidate_id", "")).strip()
                for p in group.proposals
            ]
            group_cids = [c for c in group_cids if c]

            raw_result = None
            group_cache_key = ""
            if effective_cache is not None:
                payload_hash = hashlib.sha256(group.user_text.encode("utf-8")).hexdigest()
                prompt_ver = hashlib.sha256(FINALIZER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
                group_cand_hash = hashlib.sha256(",".join(sorted(group_cids)).encode("utf-8")).hexdigest()[:16]
                group_cache_key = compute_finalizer_cache_key(
                    group_payload_hash=payload_hash,
                    model=self.settings.finalizer_model,
                    thinking=self.settings.finalizer_thinking,
                    prompt_version=prompt_ver,
                    scope=resolved_scope or "SEASON",
                    algo=HIERARCHY_ALGO_VERSION,
                    recap_prompt=self.settings.recap_prompt or "",
                    recap_language=self.settings.recap_language or "",
                    recap_mode=self.settings.recap_mode or "",
                    content_type=self.settings.content_type or "",
                    source_rights_status=self.settings.source_rights_status or "",
                    voice_style=self.settings.voice_style or "",
                    recap_settings={"candidate_ids": sorted(group_cids), "candidates_hash": group_cand_hash} if group_cids else None,
                    output_directive_hash=out_dir_hash,
                )
                cached_res, meta = effective_cache.load_finalizer_result(group.group_id, group_cache_key)
                if meta.get("hit") and cached_res is not None:
                    raw_result = cached_res
                    if log:
                        log(f"Sử dụng kết quả finalizer đã lưu trong bộ nhớ đệm cho nhóm {group.group_id}.")
                    if on_phase:
                        on_phase(
                            AnalysisPhase.SEASON_MINING,
                            phase_scope,
                            {"status": "cached", "group": group.group_id, "cached": True},
                        )

            if raw_result is None:
                def _status_cb(msg: str) -> None:
                    if on_phase:
                        on_phase(
                            AnalysisPhase.SEASON_MINING,
                            phase_scope,
                            {"status": msg, "status_message": msg, "group": group.group_id},
                        )

                call_kwargs: dict[str, Any] = {
                    "model": self.settings.finalizer_model,
                    "thinking": self.settings.finalizer_thinking,
                    "system": FINALIZER_SYSTEM_PROMPT,
                    "user_text": group.user_text,
                    "cancel_event": cancel_event,
                    "phase": "finalizer",
                    "timeout": FINALIZER_TIMEOUT,
                    "on_status": _status_cb,
                    "log": log,
                }

                def _call_ai() -> dict[str, Any]:
                    try:
                        return self.client.chat_json(**call_kwargs)
                    except TypeError as te:
                        if "unexpected keyword argument" in str(te):
                            filtered = {k: v for k, v in call_kwargs.items() if k not in ("phase", "timeout", "on_status", "log")}
                            return self.client.chat_json(**filtered)
                        raise

                def _is_well_formed(res: Any) -> bool:
                    if not isinstance(res, dict):
                        return False
                    return "outputs" in res or "segments" in res

                try:
                    raw_result = _call_ai()
                except AnalysisCancelledError:
                    raise
                except APIError as exc:
                    if _is_cancelled(cancel_event) or "đã bị dừng" in str(exc) or "bị hủy" in str(exc):
                        raise AnalysisCancelledError("Finalizer đã bị hủy.") from exc
                    raise AnalysisError(
                        f"Không thể kết nối đến AI Gateway ({self.settings.api_endpoint}): Finalizer nhóm {group.group_id} lỗi sau các lần thử: {exc}. "
                        f"Dữ liệu evidence và liên kết đã được lưu an toàn trong bộ nhớ đệm (preserved work)."
                    ) from exc

                # Task 4: Parser malformed shared retry then raises
                if not _is_well_formed(raw_result):
                    try:
                        raw_result = _call_ai()
                    except Exception:
                        pass
                    if not _is_well_formed(raw_result):
                        raise AnalysisError(
                            f"Phản hồi finalizer từ AI cho nhóm {group.group_id} sai cấu trúc schema (malformed parser response)."
                        )

                to_cache = raw_result
                if isinstance(raw_result, dict) and "outputs" not in raw_result and "segments" in raw_result:
                    to_cache = {"outputs": [raw_result]}

                if effective_cache is not None and group_cache_key and validate_finalizer_cache_data(to_cache):
                    try:
                        effective_cache.save_finalizer_result(group.group_id, group_cache_key, to_cache)
                    except Exception as exc:
                        if log:
                            log(f"Cảnh báo: Không thể lưu cache cho nhóm finalizer {group.group_id}: {exc}")

                if _is_cancelled(cancel_event):
                    raise AnalysisCancelledError("Finalizer đã bị hủy.")

            raw_outputs = raw_result.get("outputs")
            if raw_outputs is None and "segments" in raw_result:
                raw_outputs = [raw_result]
            if not isinstance(raw_outputs, list):
                raw_outputs = []

            for idx, out_dict in enumerate(raw_outputs, start=1):
                if not isinstance(out_dict, dict):
                    continue

                out_id = str(out_dict.get("output_id", f"output_{idx:02d}"))
                raw_cid = str(
                    out_dict.get("source_candidate_id")
                    or out_dict.get("candidate_id")
                    or out_dict.get("proposal_id")
                    or ""
                ).strip()

                matched_cid = ""
                if not raw_cid:
                    if len(group_cids) == 1:
                        matched_cid = group_cids[0]
                    else:
                        title_lower = str(out_dict.get("title", "")).strip().lower()
                        title_matched = None
                        for p in group.proposals:
                            p_title = str(p.get("title", "")).strip().lower()
                            if title_lower and p_title and (title_lower == p_title or title_lower in p_title or p_title in title_lower):
                                title_matched = str(p.get("proposal_id", "") or p.get("candidate_id", "")).strip()
                                break
                        if title_matched:
                            matched_cid = title_matched
                        else:
                            errors.append(
                                f"Output '{out_id}' in group '{group.group_id}' omitted source_candidate_id "
                                f"with multiple candidates ({', '.join(group_cids)})."
                            )
                            continue
                else:
                    if raw_cid in group_cids or raw_cid in attempted_candidate_ids:
                        matched_cid = raw_cid
                    else:
                        errors.append(
                            f"Output '{out_id}' in group '{group.group_id}' references unknown candidate_id '{raw_cid}'."
                        )
                        continue

                try:
                    parsed_list = self.parse_outputs(
                        {"outputs": [out_dict]},
                        source_episodes,
                        scope_hint=None,
                        raise_errors=True,
                    )
                    if parsed_list:
                        parsed_out = parsed_list[0]
                        parsed_out.source_candidate_id = matched_cid
                        parsed_out.candidate_id = matched_cid
                        all_accepted_outputs.append(parsed_out)
                        accepted_candidate_id_set.add(matched_cid)
                except ValidationError as ve:
                    errors.append(f"Output '{out_id}' (candidate '{matched_cid}') rejected by schema: {ve}")

        for out in all_accepted_outputs:
            base_id = out.output_id
            unique_id = base_id
            suffix = 1
            while unique_id in seen_output_ids:
                unique_id = f"{base_id}_{suffix}"
                suffix += 1
            out.output_id = unique_id
            seen_output_ids.add(unique_id)

        raw_titles = [out.title or out.output_id for out in all_accepted_outputs]
        unique_titles = resolve_unique_titles(raw_titles, max_length=120)
        for out, san_title in zip(all_accepted_outputs, unique_titles):
            out.sanitized_title = san_title

        accepted_candidate_ids = [cid for cid in attempted_candidate_ids if cid in accepted_candidate_id_set]
        rejected_candidate_ids = [cid for cid in attempted_candidate_ids if cid not in accepted_candidate_id_set]
        completed = True

        final_res = FinalizationResult(
            outputs=all_accepted_outputs,
            attempted_candidates=attempted_candidate_ids,
            accepted_candidate_ids=accepted_candidate_ids,
            rejected_candidate_ids=rejected_candidate_ids,
            errors=errors,
            completed=completed,
            diagnostics={
                "groups_count": len(groups),
                "total_attempted": len(attempted_candidate_ids),
                "total_accepted": len(accepted_candidate_id_set),
                "total_rejected": len(rejected_candidate_ids),
            },
        )

        if not return_diagnostics:
            return final_res.outputs
        return final_res

    def parse_outputs(
        self,
        raw: dict[str, Any],
        episodes: list[SourceEpisode],
        scope_hint: str | None = None,
        *,
        raise_errors: bool = True,
        errors: list[str] | None = None,
    ) -> list[CommentaryOutput]:
        """Parse AI response into CommentaryOutput objects with strict validation.

        - Parses ALL items in outputs[] (no artificial slicing or quota).
        - Preserves titles and deterministically sanitizes them.
        - Validates source clips against episode sources and durations.
        - Rejects hallucinated episode IDs or source file mismatches.
        - Supports cross-episode clipping (e.g. E01, E03, E05).
        - Empty outputs [] is valid.
        """
        episode_map: dict[str, SourceEpisode] = {ep.episode_id: ep for ep in episodes}

        # Check for outputs list
        raw_outputs = raw.get("outputs")
        if raw_outputs is None:
            # Fallback if AI returned single output format with 'segments'
            if "segments" in raw:
                raw_outputs = [raw]
            else:
                raw_outputs = []

        if not isinstance(raw_outputs, list):
            raw_outputs = []

        # Empty outputs is valid
        if not raw_outputs:
            return []

        parsed_outputs: list[CommentaryOutput] = []

        for i, out_dict in enumerate(raw_outputs, start=1):
            if not isinstance(out_dict, dict):
                continue

            try:
                output_id = str(out_dict.get("output_id", f"output_{i:02d}"))
                title = str(out_dict.get("title", f"Recap Output {i}"))
                scope_val = str(out_dict.get("candidate_scope", CandidateScope.SINGLE_EPISODE.value))
                if scope_val not in {s.value for s in CandidateScope}:
                    scope_val = CandidateScope.SINGLE_EPISODE.value
                if scope_hint:
                    scope_val = scope_hint

                cid = str(
                    out_dict.get("source_candidate_id")
                    or out_dict.get("candidate_id")
                    or out_dict.get("proposal_id")
                    or ""
                ).strip()

                segments_raw = out_dict.get("segments", [])
                if not isinstance(segments_raw, list):
                    segments_raw = []

                segments: list[Segment] = []
                for j, s_dict in enumerate(segments_raw, start=1):
                    if not isinstance(s_dict, dict):
                        continue

                    segment_id = str(s_dict.get("segment_id", f"seg_{j:02d}"))
                    narration = str(s_dict.get("narration", "") or s_dict.get("narration_text", "")).strip()
                    audio_policy = str(s_dict.get("audio_policy", AudioPolicy.DUCK.value)).strip().lower()
                    if audio_policy not in {p.value for p in AudioPolicy}:
                        audio_policy = AudioPolicy.DUCK.value

                    orig_dialogue = str(
                        s_dict.get("original_dialogue", "") or s_dict.get("original_dialogue_text", "")
                    ).strip()

                    # Parse source clips
                    clips_raw = s_dict.get("source_clips", [])
                    if not isinstance(clips_raw, list):
                        clips_raw = []

                    # Fallback if segment defined start/end directly
                    if not clips_raw:
                        start_s = float(s_dict.get("start", s_dict.get("start_ms", 0) / 1000.0))
                        end_s = float(s_dict.get("end", s_dict.get("end_ms", start_s + 5.0) / 1000.0))
                        ep_id = str(s_dict.get("episode_id", episodes[0].episode_id))
                        clips_raw = [
                            {
                                "episode_id": ep_id,
                                "source_video": episode_map.get(ep_id, episodes[0]).source_video,
                                "start": start_s,
                                "end": end_s,
                            }
                        ]

                    clips: list[SourceClip] = []
                    for c_dict in clips_raw:
                        if not isinstance(c_dict, dict):
                            continue

                        ep_id = str(c_dict.get("episode_id", "")).strip()
                        if ep_id not in episode_map:
                            raise ValidationError(
                                f"Invalid episode_id '{ep_id}' in source clip: not found in source episodes."
                            )

                        if scope_hint == CandidateScope.SINGLE_EPISODE.value and ep_id != episodes[0].episode_id:
                            raise ValidationError(
                                f"Invalid episode_id '{ep_id}' in single episode source clip: must be '{episodes[0].episode_id}'."
                            )

                        ep = episode_map[ep_id]
                        src_vid = str(c_dict.get("source_video", "") or ep.source_video).strip()

                        # Validate source file matches
                        if src_vid and ep.source_video:
                            if src_vid != ep.source_video and Path(src_vid).name != Path(ep.source_video).name:
                                raise ValidationError(
                                    f"Source clip file '{src_vid}' does not match episode '{ep.episode_id}' file '{ep.source_video}'."
                                )

                        # Timestamps (seconds)
                        start_sec = float(c_dict.get("start", c_dict.get("start_ms", 0) / 1000.0))
                        end_sec = float(c_dict.get("end", c_dict.get("end_ms", start_sec + 5.0) / 1000.0))

                        if start_sec < 0.0:
                            raise ValidationError(
                                f"Invalid clip start timestamp {start_sec}: must be >= 0."
                            )
                        if end_sec <= start_sec:
                            raise ValidationError(
                                f"Invalid clip end timestamp {end_sec}: must be greater than start ({start_sec})."
                            )
                        if ep.duration_seconds > 0.0 and end_sec > ep.duration_seconds + 0.5:
                            raise ValidationError(
                                f"Clip end {end_sec}s exceeds episode '{ep.episode_id}' duration {ep.duration_seconds}s."
                            )

                        clips.append(
                            SourceClip(
                                episode_id=ep_id,
                                source_video=src_vid or ep.source_video,
                                start=start_sec,
                                end=end_sec,
                            )
                        )

                    if audio_policy in (AudioPolicy.MUTE.value, "mute"):
                        # Normalize "mute" policy from AI to mix/duck behavior so original audio is preserved
                        audio_policy = AudioPolicy.DUCK.value

                    if audio_policy != AudioPolicy.ORIGINAL_ONLY.value and not narration:
                        raise ValidationError(
                            f"Phân đoạn '{segment_id}' là phân đoạn thuyết minh nhưng thiếu nội dung narration."
                        )

                    segments.append(
                        Segment(
                            segment_id=segment_id,
                            source_clips=clips,
                            original_dialogue=orig_dialogue,
                            narration=narration,
                            audio_policy=audio_policy,
                        )
                    )

                out_obj = CommentaryOutput(
                    output_id=output_id,
                    title=title,
                    sanitized_title="",
                    candidate_scope=scope_val,
                    segments=segments,
                    status=OutputStatus.WAITING.value,
                )
                out_obj.source_candidate_id = cid
                out_obj.candidate_id = cid
                parsed_outputs.append(out_obj)
            except ValidationError as ve:
                if raise_errors:
                    raise
                if errors is not None:
                    errors.append(str(ve))
                continue

        # Deterministically sanitize and resolve unique titles
        raw_titles = [out.title or out.output_id for out in parsed_outputs]
        unique_titles = resolve_unique_titles(raw_titles, max_length=120)
        for out, san_title in zip(parsed_outputs, unique_titles):
            out.sanitized_title = san_title

        return parsed_outputs

    def _finalize_single_offline(
        self,
        episode: SourceEpisode,
        evidence: EpisodeEvidence,
    ) -> list[CommentaryOutput]:
        """Deterministic offline generation for single episode."""
        dur = max(3.0, episode.duration_seconds)
        clips = [
            SourceClip(
                episode_id=episode.episode_id,
                source_video=episode.source_video,
                start=0.0,
                end=dur,
            )
        ]
        segment = Segment(
            segment_id="seg_01",
            source_clips=clips,
            original_dialogue="",
            narration=f"A complete recap of {episode.title or episode.episode_id}.",
            audio_policy=AudioPolicy.MUTE.value,
        )
        title = episode.title or f"Recap {episode.episode_id}"
        return [
            CommentaryOutput(
                output_id="out_01",
                title=title,
                sanitized_title=sanitize_title(title),
                candidate_scope=CandidateScope.SINGLE_EPISODE.value,
                segments=[segment],
                status=OutputStatus.WAITING.value,
            )
        ]

    def _finalize_season_offline(
        self,
        episodes: list[SourceEpisode],
        evidence_map: dict[str, EpisodeEvidence],
        connection_result: SeasonConnectionResult,
    ) -> list[CommentaryOutput]:
        """Deterministic offline generation for season with multiple candidate outputs."""
        outputs: list[CommentaryOutput] = []

        # 1. Full season arc output
        season_segments: list[Segment] = []
        for i, ep in enumerate(episodes, start=1):
            if not evidence_map or ep.episode_id in evidence_map:
                dur = max(1.0, ep.duration_seconds)
                clip_dur = min(30.0, dur)
                season_segments.append(
                    Segment(
                        segment_id=f"seg_{i:02d}",
                        source_clips=[
                            SourceClip(
                                episode_id=ep.episode_id,
                                source_video=ep.source_video,
                                start=0.0,
                                end=clip_dur,
                            )
                        ],
                        narration=f"Key events unfolding in {ep.title or ep.episode_id}.",
                        audio_policy=AudioPolicy.MUTE.value,
                    )
                )

        if season_segments:
            outputs.append(
                CommentaryOutput(
                    output_id="out_season_arc",
                    title="Season Comprehensive Arc",
                    sanitized_title=sanitize_title("Season Comprehensive Arc"),
                    candidate_scope=CandidateScope.SEASON_ARC.value,
                    segments=season_segments,
                    status=OutputStatus.WAITING.value,
                )
            )

        # 2. Supporting character arc output
        if len(episodes) >= 2:
            supporting_segments: list[Segment] = []
            for j, ep in enumerate(episodes[:3], start=1):
                dur = max(1.0, ep.duration_seconds)
                supporting_segments.append(
                    Segment(
                        segment_id=f"seg_supp_{j:02d}",
                        source_clips=[
                            SourceClip(
                                episode_id=ep.episode_id,
                                source_video=ep.source_video,
                                start=0.0,
                                end=min(15.0, dur),
                            )
                        ],
                        narration=f"Supporting character developments in {ep.episode_id}.",
                        audio_policy=AudioPolicy.MUTE.value,
                    )
                )
            outputs.append(
                CommentaryOutput(
                    output_id="out_supporting_arc",
                    title="Supporting Character Arc",
                    sanitized_title=sanitize_title("Supporting Character Arc"),
                    candidate_scope=CandidateScope.CROSS_EPISODE.value,
                    segments=supporting_segments,
                    status=OutputStatus.WAITING.value,
                )
            )

        return outputs


SeasonFinalizer = CandidateFinalizer


def finalize_candidates(
    candidates: list[CandidateProposal | dict[str, Any]],
    source_episodes: list[SourceEpisode],
    scope_type: AnalysisScope | CandidateScope | str = CandidateScope.CROSS_EPISODE.value,
    topology: SeasonConnectionResult | dict[str, Any] | None = None,
    settings: AppSettings | None = None,
    client: OpenAICompatibleClient | None = None,
    hierarchy_cache: HierarchyCacheManager | None = None,
    *,
    cancel_event: threading.Event | None = None,
    on_phase: PhaseCallback | None = None,
    log: Callable[[str], None] | None = None,
    legacy_wrapper: bool = False,
    output_directive: OutputDirective | None = None,
    return_diagnostics: bool = True,
) -> FinalizationResult | list[CommentaryOutput]:
    """Convenience module-level wrapper to finalize commentary outputs for candidates."""
    finalizer = CandidateFinalizer(
        settings=settings,
        client=client,
        hierarchy_cache=hierarchy_cache,
        output_directive=output_directive,
    )
    return finalizer.finalize_candidates(
        candidates=candidates,
        source_episodes=source_episodes,
        scope_type=scope_type,
        topology=topology,
        cancel_event=cancel_event,
        on_phase=on_phase,
        log=log,
        legacy_wrapper=legacy_wrapper,
        hierarchy_cache=hierarchy_cache,
        output_directive=output_directive,
        return_diagnostics=return_diagnostics,
    )


__all__ = [
    "HARD_PAYLOAD_CEILING",
    "TARGET_PAYLOAD_CEILING",
    "CandidateFinalizer",
    "FinalizationResult",
    "FinalizerGroupPlan",
    "FinalizerResult",
    "SeasonFinalizer",
    "cap_long_fields",
    "deterministic_cap_finalizer_dict",
    "deterministic_cap_proposal_entry",
    "estimate_finalizer_request_size",
    "finalize_candidates",
    "format_finalizer_user_text",
    "plan_finalizer_groups",
]
