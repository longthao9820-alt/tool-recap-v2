"""Narrative candidate consolidation module for ToolRecap V2.

Consolidates, merges cross-episode duplicates, preserves distinct character theses,
and eliminates weak/duplicate proposals while preserving source integrity.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ...api_client import (
    APIError,
    OpenAICompatibleClient,
    estimate_request_size,
)
from ...domain.cache import (
    CONSOLIDATION_SCHEMA_VERSION,
    HierarchyCacheManager,
    compute_consolidation_cache_key,
    validate_consolidation_cache_data,
)
from ...domain.enums import (
    CandidateScope,
    CandidateStatus,
    CompactionLevel,
    ConsolidationAction,
    ConsolidationReason,
)
from ...domain.models import (
    CandidateProposal,
    CandidateSourceRange,
    ConsolidatedCandidateSet,
    ConsolidationDecision,
)
from ...domain.policy import (
    CandidateDirective,
    EditorialPolicy,
)
from ...settings import AppSettings
from ..errors import AnalysisCancelledError, AnalysisError
from ..phases import AnalysisPhase, PhaseCallback
from ..prompts import CANDIDATE_CONSOLIDATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

HARD_PAYLOAD_CEILING: int = 500_000
TARGET_PAYLOAD_CEILING: int = 480_000
CONSOLIDATION_ALGO_VERSION: str = "v1"
CONSOLIDATION_PROMPT_VERSION: str = "v1"
MAX_CANDIDATES_PER_NODE: int = 10


def _tokenize(text: str) -> set[str]:
    """Extract normalized lowercase word tokens from text."""
    return set(re.findall(r"\w+", text.lower()))


def compute_candidate_similarity(c1: CandidateProposal, c2: CandidateProposal) -> float:
    """Compute semantic and grounding similarity between two candidates.

    Considers:
    - Title and central thesis token Jaccard similarity
    - Overlap tags similarity
    - Supporting evidence and source range evidence ref overlap
    - Same primary character with shared episode or thesis overlap
    - Same coherent arc indicators
    """
    # Thesis tokens
    t1 = _tokenize(f"{c1.title} {c1.central_thesis} {c1.subject}")
    t2 = _tokenize(f"{c2.title} {c2.central_thesis} {c2.subject}")
    thesis_jaccard = (len(t1 & t2) / len(t1 | t2)) if (t1 | t2) else 0.0

    # Overlap tags
    tags1 = {t.strip().lower() for t in c1.overlap_tags if t.strip()}
    tags2 = {t.strip().lower() for t in c2.overlap_tags if t.strip()}
    tags_jaccard = (len(tags1 & tags2) / len(tags1 | tags2)) if (tags1 | tags2) else 0.0

    # Evidence refs
    refs1 = set(c1.supporting_evidence) | {r.evidence_ref for r in c1.source_ranges if r.evidence_ref}
    refs2 = set(c2.supporting_evidence) | {r.evidence_ref for r in c2.source_ranges if r.evidence_ref}
    refs_jaccard = (len(refs1 & refs2) / len(refs1 | refs2)) if (refs1 | refs2) else 0.0

    # Character match
    p1 = c1.primary_character.strip().lower()
    p2 = c2.primary_character.strip().lower()
    same_primary = bool(p1 and p2 and p1 == p2)

    chars1 = {c.strip().lower() for c in c1.characters if c.strip()}
    chars2 = {c.strip().lower() for c in c2.characters if c.strip()}
    chars_jaccard = (len(chars1 & chars2) / len(chars1 | chars2)) if (chars1 | chars2) else 0.0

    # Episode relationship
    eps1 = set(c1.episodes)
    eps2 = set(c2.episodes)
    shared_eps = bool(eps1 & eps2)

    # Weighted similarity score
    score = (thesis_jaccard * 0.40) + (tags_jaccard * 0.25) + (refs_jaccard * 0.20) + (chars_jaccard * 0.15)

    # Same character boost for same arc or shared episodes
    if same_primary:
        if thesis_jaccard >= 0.35:
            score = max(score, 0.70)
        elif shared_eps and thesis_jaccard >= 0.2:
            score = max(score, 0.55)
        elif tags_jaccard >= 0.3:
            score = max(score, 0.50)

    # Evidence ref direct overlap is very strong indicator
    if refs1 and refs2 and (refs1 & refs2):
        score = max(score, 0.65)

    return min(1.0, max(0.0, score))


def pre_group_candidates(
    candidates: Sequence[CandidateProposal],
    threshold: float = 0.25,
) -> list[list[CandidateProposal]]:
    """Deterministically pre-group candidates by overlap tags, evidence, thesis similarity, and character arcs.

    Uses connected components so all ambiguously related candidates end up in the same group.
    Singletons remain in individual groups.
    """
    n = len(candidates)
    if n <= 1:
        return [list(candidates)] if n == 1 else []

    # Build adjacency list
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            c1 = candidates[i]
            c2 = candidates[j]
            sim = compute_candidate_similarity(c1, c2)
            if sim >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # Find connected components
    visited: set[int] = set()
    groups: list[list[CandidateProposal]] = []

    # Sort indices deterministically by proposal_id
    sorted_indices = sorted(range(n), key=lambda idx: candidates[idx].proposal_id)

    for i in sorted_indices:
        if i in visited:
            continue
        component: list[int] = []
        queue = [i]
        visited.add(i)
        while queue:
            curr = queue.pop(0)
            component.append(curr)
            for neighbor in sorted(adj[curr]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Sort members of the group deterministically
        component.sort(key=lambda idx: candidates[idx].proposal_id)
        groups.append([candidates[idx] for idx in component])

    return groups


def compact_candidate_payload(
    cand: CandidateProposal,
    level: CompactionLevel = CompactionLevel.FULL,
) -> dict[str, Any]:
    """Deterministically compact candidate representation according to CompactionLevel."""
    if level == CompactionLevel.FULL:
        return cand.to_dict()

    if level == CompactionLevel.TRIMMED:
        return {
            "proposal_id": cand.proposal_id,
            "title": cand.title,
            "candidate_scope": cand.candidate_scope,
            "episodes": list(cand.episodes),
            "characters": list(cand.characters),
            "primary_character": cand.primary_character,
            "supporting_characters": list(cand.supporting_characters),
            "subject": cand.subject,
            "central_thesis": cand.central_thesis,
            "source_ranges": [r.to_dict() for r in cand.source_ranges[:3]],
            "setup": cand.setup[:120] if cand.setup else "",
            "payoff": cand.payoff[:120] if cand.payoff else "",
            "observed_facts": list(cand.observed_facts[:3]),
            "supporting_evidence": list(cand.supporting_evidence[:5]),
            "why": cand.why[:150] if cand.why else "",
            "hooks": list(cand.hooks[:2]),
            "overlap_tags": list(cand.overlap_tags),
            "confidence": float(cand.confidence),
            "status": cand.status,
        }

    if level == CompactionLevel.PRIORITY:
        return {
            "proposal_id": cand.proposal_id,
            "title": cand.title,
            "candidate_scope": cand.candidate_scope,
            "episodes": list(cand.episodes),
            "primary_character": cand.primary_character,
            "supporting_characters": list(cand.supporting_characters[:2]),
            "central_thesis": cand.central_thesis,
            "source_ranges": [r.to_dict() for r in cand.source_ranges[:2]],
            "supporting_evidence": list(cand.supporting_evidence[:3]),
            "why": cand.why[:100] if cand.why else "",
            "overlap_tags": list(cand.overlap_tags[:3]),
            "confidence": float(cand.confidence),
            "status": cand.status,
        }

    # SKELETON
    return {
        "proposal_id": cand.proposal_id,
        "title": cand.title[:80],
        "candidate_scope": cand.candidate_scope,
        "episodes": list(cand.episodes),
        "primary_character": cand.primary_character[:50],
        "central_thesis": cand.central_thesis[:120],
        "supporting_evidence": list(cand.supporting_evidence[:2]),
        "overlap_tags": list(cand.overlap_tags[:2]),
        "confidence": float(cand.confidence),
        "status": cand.status,
    }


def compute_candidates_payload_hash(
    candidates: Sequence[CandidateProposal],
    compaction_level: CompactionLevel = CompactionLevel.FULL,
) -> str:
    """Compute deterministic SHA256 hash of input candidates payload."""
    serialized = [
        compact_candidate_payload(c, compaction_level)
        for c in sorted(candidates, key=lambda x: x.proposal_id)
    ]
    raw = json.dumps(serialized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def estimate_consolidation_request_size(
    model: str,
    user_text: str,
    thinking: str = "auto",
    max_tokens: int = 32_000,
) -> int:
    """Estimate token/byte request size for consolidation prompt."""
    return estimate_request_size(
        model=model,
        system=CANDIDATE_CONSOLIDATION_SYSTEM_PROMPT,
        user_text=user_text,
        thinking=thinking,
        max_tokens=max_tokens,
        variant=0,
    )


def format_consolidation_node_user_text(
    node_id: str,
    candidates: Sequence[CandidateProposal],
    candidate_directive: CandidateDirective | str | None = None,
    compaction_level: CompactionLevel = CompactionLevel.FULL,
) -> str:
    """Format user prompt text for candidate consolidation node."""
    lines: list[str] = [
        f"=== CANDIDATE CONSOLIDATION NODE: {node_id} ===",
        f"Total Input Proposals: {len(candidates)}",
        "",
    ]

    if candidate_directive:
        cd_str = str(candidate_directive).strip()
        if cd_str:
            lines.append("EDITORIAL CANDIDATE DIRECTIVE:")
            lines.append(cd_str)
            lines.append("")

    lines.append("INPUT PROPOSALS TO CONSOLIDATE:")
    serialized = [compact_candidate_payload(c, compaction_level) for c in candidates]
    lines.append(json.dumps(serialized, indent=2, ensure_ascii=False))
    lines.append("")

    lines.append("MANDATES:")
    lines.append("1. Account for EVERY input proposal_id in 'decisions'. Missing IDs are strictly prohibited.")
    lines.append("2. Allowed actions: 'KEEP', 'MERGE' (with 'target_id'), 'REJECT' (with explicit 'reason_code' and 'reason').")
    lines.append("3. Cross-episode duplicates of the same arc must be merged, combining evidence and source ranges.")
    lines.append("4. Distinct same-character theses MUST stay separate ('distinct_thesis').")
    lines.append("5. Supporting characters must receive equal consideration. Retain strong supporting arcs.")
    lines.append("6. Return JSON with 'consolidated_candidates' and 'decisions'.")

    return "\n".join(lines)


def validate_consolidation_response_schema(
    raw: Any,
    *,
    raise_error: bool = False,
) -> bool:
    """Validate AI response schema for candidate consolidation.

    Requirements:
    - Must be a dictionary.
    - Must contain 'consolidated_candidates' (or 'candidates') as a list of dicts.
    - Must contain 'decisions' as a list of dicts.
    """
    if not isinstance(raw, dict):
        if raise_error:
            raise AnalysisError(f"Phản hồi candidate consolidation phải là dict, nhận được {type(raw).__name__}.")
        return False

    cands = None
    for key in ("consolidated_candidates", "candidates"):
        if key in raw:
            cands = raw[key]
            break
    if cands is None or not isinstance(cands, list):
        if raise_error:
            raise AnalysisError("Phản hồi candidate consolidation thiếu danh sách 'consolidated_candidates'.")
        return False

    for idx, item in enumerate(cands):
        if not isinstance(item, dict):
            if raise_error:
                raise AnalysisError(f"Phần tử {idx} trong 'consolidated_candidates' không phải dict.")
            return False

    decs = raw.get("decisions")
    if decs is None or not isinstance(decs, list):
        if raise_error:
            raise AnalysisError("Phản hồi candidate consolidation thiếu danh sách 'decisions'.")
        return False

    for idx, dec in enumerate(decs):
        if not isinstance(dec, dict):
            if raise_error:
                raise AnalysisError(f"Phần tử {idx} trong 'decisions' không phải dict.")
            return False

    return True


def evaluate_candidate_quality(candidate: CandidateProposal) -> tuple[bool, str]:
    """Check whether candidate has sufficient grounding and evidence.

    Returns (is_valid, reason_code).
    """
    if candidate.confidence < 0.35:
        return False, ConsolidationReason.WEAK_EVIDENCE.value

    has_evidence = bool(candidate.source_ranges or candidate.supporting_evidence or candidate.observed_facts)
    if not has_evidence:
        return False, ConsolidationReason.WEAK_EVIDENCE.value

    if not candidate.episodes:
        return False, ConsolidationReason.INVALID_GROUNDING.value

    if not candidate.central_thesis.strip() and not candidate.title.strip():
        return False, ConsolidationReason.INVALID_GROUNDING.value

    return True, ""


def reconcile_and_validate_decisions(
    input_candidates: Sequence[CandidateProposal],
    ai_consolidated: list[CandidateProposal],
    ai_decisions: list[ConsolidationDecision],
    logger_ref: logging.Logger | None = None,
) -> ConsolidatedCandidateSet:
    """Strictly validate and reconcile AI consolidation output.

    Guarantees:
    1. Every input candidate ID is accounted for exactly once in decisions.
    2. Missing decisions default to KEEP conservatively with CONSERVATIVE_KEEP.
    3. AI cannot invent source refs: merged candidates only union existing evidence/ranges.
    4. Distinct same-character theses stay separate.
    5. Candidates for same coherent arc split across episodes merge cross/season with union.
    6. Reject reasons are explicit and validated.
    """
    log = logger_ref or logger
    input_map: dict[str, CandidateProposal] = {c.proposal_id: c for c in input_candidates if c.proposal_id}

    # 1. Deduplicate AI decisions by candidate_id
    decision_map: dict[str, ConsolidationDecision] = {}
    for dec in ai_decisions:
        cid = dec.candidate_id.strip()
        if not cid:
            continue
        if cid in input_map and cid not in decision_map:
            decision_map[cid] = dec

    # 2. Account for all input IDs: Missing decisions default KEEP conservatively
    for cid, cand in input_map.items():
        if cid not in decision_map:
            log.warning(f"Input candidate '{cid}' missing decision in AI output; defaulting to KEEP conservatively.")
            decision_map[cid] = ConsolidationDecision(
                candidate_id=cid,
                action=ConsolidationAction.KEEP.value,
                reason_code=ConsolidationReason.CONSERVATIVE_KEEP.value,
                reason="Missing decision in AI response, defaulted to KEEP conservatively.",
                target_id="",
            )

    # 3. Validate reject decisions: reject reason must be explicit
    for cid, dec in decision_map.items():
        if dec.action == ConsolidationAction.REJECT.value:
            if not dec.reason_code:
                dec.reason_code = ConsolidationReason.WEAK_EVIDENCE.value
            if not dec.reason:
                dec.reason = "Candidate rejected due to weak evidence or invalid grounding."

    # 4. Check distinct same-character theses: ensure they stay separate
    # If AI erroneously merged two candidates with distinct theses, split them back to KEEP
    for cid, dec in list(decision_map.items()):
        if dec.action == ConsolidationAction.MERGE.value and dec.target_id:
            orig = input_map.get(cid)
            target = input_map.get(dec.target_id)
            if orig and target and orig.primary_character and target.primary_character:
                if orig.primary_character.strip().lower() == target.primary_character.strip().lower():
                    # Check thesis similarity
                    t1 = _tokenize(orig.central_thesis + " " + orig.title)
                    t2 = _tokenize(target.central_thesis + " " + target.title)
                    sim = (len(t1 & t2) / len(t1 | t2)) if (t1 | t2) else 0.0
                    # If theses are clearly distinct (sim < 0.20), keep separate
                    if sim < 0.20 and orig.central_thesis.strip() != target.central_thesis.strip():
                        log.info(
                            f"Separating distinct theses for character '{orig.primary_character}': "
                            f"'{orig.proposal_id}' and '{target.proposal_id}' stay separate."
                        )
                        dec.action = ConsolidationAction.KEEP.value
                        dec.reason_code = ConsolidationReason.DISTINCT_THESIS.value
                        dec.reason = f"Distinct character thesis for {orig.primary_character} kept separate."
                        dec.target_id = ""

    # 5. Build surviving candidates list and validate source refs / ranges
    surviving_candidates: list[CandidateProposal] = []
    seen_survivor_ids: set[str] = set()

    # Map target_id -> list of merged candidate IDs
    merge_groups: dict[str, list[str]] = {}
    for cid, dec in decision_map.items():
        if dec.action == ConsolidationAction.MERGE.value and dec.target_id:
            merge_groups.setdefault(dec.target_id, []).append(cid)

    # First, process AI consolidated candidates
    for cand in ai_consolidated:
        cid = cand.proposal_id.strip()
        # Find constituent inputs
        constituent_ids = set()
        if cid in input_map:
            constituent_ids.add(cid)
        if cid in merge_groups:
            constituent_ids.update(merge_groups[cid])

        # If not known in input_map and not target of any merge, check if it matches a candidate
        if not constituent_ids:
            # AI might have synthesized a new ID; match by title / primary_character
            for inp_id, inp_c in input_map.items():
                if inp_c.title.strip().lower() == cand.title.strip().lower():
                    constituent_ids.add(inp_id)
                    break

        if not constituent_ids:
            # Unaccounted / fabricated candidate: reject fabrication
            log.warning(f"Dropping fabricated candidate '{cand.title}' (id={cand.proposal_id}) with no input grounding.")
            continue

        constituents = [input_map[inp_id] for inp_id in constituent_ids if inp_id in input_map]
        if not constituents:
            continue

        # Check if constituent was rejected
        all_rejected = all(
            decision_map.get(inp_id, ConsolidationDecision()).action == ConsolidationAction.REJECT.value
            for inp_id in constituent_ids
        )
        if all_rejected:
            continue

        # Ground source ranges and evidence: AI cannot invent source refs
        allowed_ranges: list[CandidateSourceRange] = []
        allowed_evidence: set[str] = set()
        allowed_episodes: set[str] = set()
        allowed_characters: set[str] = set()
        allowed_facts: set[str] = set()
        allowed_hooks: set[str] = set()

        for c in constituents:
            allowed_ranges.extend(c.source_ranges)
            allowed_evidence.update(c.supporting_evidence)
            allowed_episodes.update(c.episodes)
            allowed_characters.update(c.characters)
            if c.primary_character:
                allowed_characters.add(c.primary_character)
            allowed_characters.update(c.supporting_characters)
            allowed_facts.update(c.observed_facts)
            allowed_hooks.update(c.hooks)

        # Filter candidate source ranges against allowed ranges
        valid_ranges: list[CandidateSourceRange] = []
        for r in cand.source_ranges:
            if not r.episode_id:
                continue
            matched = any(
                ar.episode_id == r.episode_id and abs(ar.start_seconds - r.start_seconds) <= 2.0
                for ar in allowed_ranges
            )
            if matched or (r.evidence_ref and r.evidence_ref in allowed_evidence):
                valid_ranges.append(r)

        # Ensure union of all constituent ranges is preserved
        existing_range_keys = {(r.episode_id, round(r.start_seconds, 1), round(r.end_seconds, 1)) for r in valid_ranges}
        for ar in allowed_ranges:
            key = (ar.episode_id, round(ar.start_seconds, 1), round(ar.end_seconds, 1))
            if key not in existing_range_keys:
                valid_ranges.append(copy.deepcopy(ar))
                existing_range_keys.add(key)

        valid_ranges.sort(key=lambda r: (r.episode_id, r.start_seconds, r.end_seconds))

        # Union supporting evidence
        final_evidence = sorted(allowed_evidence | set(cand.supporting_evidence & allowed_evidence if isinstance(cand.supporting_evidence, set) else [e for e in cand.supporting_evidence if e in allowed_evidence]))
        if not final_evidence and allowed_evidence:
            final_evidence = sorted(allowed_evidence)

        # Update candidate with strictly grounded fields
        cand.source_ranges = valid_ranges
        cand.supporting_evidence = final_evidence
        cand.episodes = sorted(allowed_episodes | set(cand.episodes if cand.episodes else []))
        cand.characters = sorted(allowed_characters | set(cand.characters if cand.characters else []))
        if not cand.primary_character and constituents:
            cand.primary_character = constituents[0].primary_character
        if not cand.supporting_characters:
            cand.supporting_characters = sorted(set(cand.characters) - {cand.primary_character})

        # Scope auto-correction
        if len(cand.episodes) > 1:
            if cand.candidate_scope in (CandidateScope.SINGLE_EPISODE.value, CandidateScope.SINGLE_SCENE.value):
                cand.candidate_scope = CandidateScope.CROSS_EPISODE.value
        elif len(cand.episodes) == 1 and cand.candidate_scope in (CandidateScope.CROSS_EPISODE.value, CandidateScope.SEASON_ARC.value):
            cand.candidate_scope = CandidateScope.SINGLE_EPISODE.value

        cand.status = CandidateStatus.KEEP.value
        target_cid = cand.proposal_id or constituents[0].proposal_id
        cand.proposal_id = target_cid

        if target_cid not in seen_survivor_ids:
            surviving_candidates.append(cand)
            seen_survivor_ids.add(target_cid)

    # 6. Ensure any input candidate with decision KEEP is present in survivors
    for cid, dec in decision_map.items():
        if dec.action == ConsolidationAction.KEEP.value and cid not in seen_survivor_ids:
            orig = input_map.get(cid)
            if orig is not None:
                # Quality check
                is_valid, _ = evaluate_candidate_quality(orig)
                if is_valid:
                    orig_copy = copy.deepcopy(orig)
                    orig_copy.status = CandidateStatus.KEEP.value
                    surviving_candidates.append(orig_copy)
                    seen_survivor_ids.add(cid)
                else:
                    dec.action = ConsolidationAction.REJECT.value
                    dec.reason_code = ConsolidationReason.WEAK_EVIDENCE.value
                    dec.reason = "Candidate rejected due to insufficient grounding evidence."

    # 7. Deterministically sort surviving candidates and decisions
    surviving_candidates.sort(key=lambda c: c.proposal_id)
    final_decisions = [decision_map[cid] for cid in sorted(decision_map.keys())]

    return ConsolidatedCandidateSet(
        candidates=surviving_candidates,
        decisions=final_decisions,
        schema_version=CONSOLIDATION_SCHEMA_VERSION,
        metadata={
            "total_inputs": len(input_candidates),
            "surviving_count": len(surviving_candidates),
        },
    )


class CandidateConsolidator:
    """Narrative candidate consolidator for single-episode and full-season analysis.

    Provides:
    - Deterministic pre-grouping by overlap tags, evidence, thesis token similarity.
    - AI-bounded hierarchical consolidation nodes for ambiguous groups.
    - Adaptive payload scaling with field compaction and recursive splitting.
    - "No singleton loop" guarantee.
    - Atomic content-addressed node and root caching.
    - Strict input candidate ID accounting and source ref integrity.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        client: Any = None,
        policy: EditorialPolicy | None = None,
        cache: HierarchyCacheManager | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.policy = policy or EditorialPolicy()
        self.cache = cache or HierarchyCacheManager()
        self.logger = logger

        if client is not None:
            self.client = client
        else:
            self.client = OpenAICompatibleClient(
                endpoint=self.settings.api_endpoint,
                api_key=self.settings.api_key,
                timeout=getattr(self.settings, "request_timeout", 120),
            )

    def _consolidate_single_node(
        self,
        node_id: str,
        candidates: list[CandidateProposal],
        *,
        model: str,
        thinking: str,
        cancellation_token: threading.Event | None = None,
        phase_callback: PhaseCallback | None = None,
    ) -> ConsolidatedCandidateSet:
        """Consolidate a bounded group of candidates in a single AI call (or cache hit)."""
        # Base case: "no singleton loop"
        if len(candidates) == 0:
            return ConsolidatedCandidateSet()

        if len(candidates) == 1:
            cand = candidates[0]
            is_valid, rcode = evaluate_candidate_quality(cand)
            if is_valid:
                cand_copy = copy.deepcopy(cand)
                cand_copy.status = CandidateStatus.KEEP.value
                dec = ConsolidationDecision(
                    candidate_id=cand.proposal_id,
                    action=ConsolidationAction.KEEP.value,
                    reason_code=ConsolidationReason.DISTINCT_THESIS.value,
                    reason="Single distinct candidate retained with verified evidence.",
                    target_id="",
                )
                return ConsolidatedCandidateSet(candidates=[cand_copy], decisions=[dec])
            else:
                dec = ConsolidationDecision(
                    candidate_id=cand.proposal_id,
                    action=ConsolidationAction.REJECT.value,
                    reason_code=rcode or ConsolidationReason.WEAK_EVIDENCE.value,
                    reason="Candidate has insufficient grounding or low confidence.",
                    target_id="",
                )
                return ConsolidatedCandidateSet(candidates=[], decisions=[dec])

        # Check payload size across compaction levels
        cand_directive = self.policy.candidate_directive if self.policy else CandidateDirective()
        dir_hash = cand_directive.hash() if hasattr(cand_directive, "hash") else hashlib.sha256(str(cand_directive).encode("utf-8")).hexdigest()

        selected_compaction = CompactionLevel.FULL
        selected_text = ""
        target_ceiling = getattr(self.settings, "target_payload_ceiling", TARGET_PAYLOAD_CEILING)

        for comp_level in (CompactionLevel.FULL, CompactionLevel.TRIMMED, CompactionLevel.PRIORITY, CompactionLevel.SKELETON):
            text = format_consolidation_node_user_text(node_id, candidates, cand_directive, comp_level)
            est = estimate_consolidation_request_size(model, text, thinking)
            if est <= target_ceiling:
                selected_compaction = comp_level
                selected_text = text
                break

        # If even SKELETON exceeds target ceiling or candidates > MAX_CANDIDATES_PER_NODE, recursively split
        if not selected_text or len(candidates) > MAX_CANDIDATES_PER_NODE:
            mid = len(candidates) // 2
            left_res = self._consolidate_single_node(
                f"{node_id}_sub1",
                candidates[:mid],
                model=model,
                thinking=thinking,
                cancellation_token=cancellation_token,
                phase_callback=phase_callback,
            )
            right_res = self._consolidate_single_node(
                f"{node_id}_sub2",
                candidates[mid:],
                model=model,
                thinking=thinking,
                cancellation_token=cancellation_token,
                phase_callback=phase_callback,
            )
            combined_survivors = left_res.candidates + right_res.candidates
            combined_decisions = left_res.decisions + right_res.decisions

            # If combined survivors > 1 and reduced, run bounded pass on survivors
            if len(combined_survivors) > 1 and len(combined_survivors) < len(candidates):
                return self._consolidate_single_node(
                    f"{node_id}_merge",
                    combined_survivors,
                    model=model,
                    thinking=thinking,
                    cancellation_token=cancellation_token,
                    phase_callback=phase_callback,
                )
            return ConsolidatedCandidateSet(candidates=combined_survivors, decisions=combined_decisions)

        # Compute cache key
        payload_hash = compute_candidates_payload_hash(candidates, selected_compaction)
        cache_key = compute_consolidation_cache_key(
            node_id=node_id,
            candidate_directive_hash=dir_hash,
            input_payload_hash=payload_hash,
            model=model,
            thinking=thinking,
            prompt_version=CONSOLIDATION_PROMPT_VERSION,
            algo=CONSOLIDATION_ALGO_VERSION,
            compaction_level=selected_compaction.value,
        )

        # Cancellation check before cache / AI
        if cancellation_token and cancellation_token.is_set():
            raise AnalysisCancelledError("Candidate consolidation cancelled.")

        # Cache lookup
        cached_data, meta = self.cache.load_consolidation_result(node_id, cache_key)
        if meta.get("hit") and cached_data is not None:
            self.logger.info(f"Loaded consolidation node {node_id} from cache.")
            raw_cands = cached_data.get("consolidated_candidates", cached_data.get("candidates", []))
            raw_decs = cached_data.get("decisions", [])
            ai_cands = [CandidateProposal.from_dict(c) for c in raw_cands if isinstance(c, dict)]
            ai_decs = [ConsolidationDecision.from_dict(d) for d in raw_decs if isinstance(d, dict)]
            result = reconcile_and_validate_decisions(candidates, ai_cands, ai_decs, self.logger)

            if phase_callback:
                phase_callback(
                    AnalysisPhase.CANDIDATE_CONSOLIDATION,
                    f"Tải kết quả hợp nhất ứng viên {node_id} từ cache",
                    {
                        "merged": result.merged_count,
                        "rejected": result.rejected_count,
                        "eligible": result.eligible_count,
                        "cache": True,
                    },
                )
            return result

        # Call AI client
        raw_resp = self.client.chat_json(
            model=model,
            system=CANDIDATE_CONSOLIDATION_SYSTEM_PROMPT,
            user_text=selected_text,
            thinking=thinking,
            cancel_event=cancellation_token,
        )

        # Validate schema before caching
        validate_consolidation_response_schema(raw_resp, raise_error=True)

        # Save valid before cancel
        self.cache.save_consolidation_result(
            node_id=node_id,
            cache_key=cache_key,
            data=raw_resp,
            compaction_level=selected_compaction.value,
        )

        # Check cancellation right after save
        if cancellation_token and cancellation_token.is_set():
            raise AnalysisCancelledError("Candidate consolidation cancelled after node save.")

        raw_cands = raw_resp.get("consolidated_candidates", raw_resp.get("candidates", []))
        raw_decs = raw_resp.get("decisions", [])
        ai_cands = [CandidateProposal.from_dict(c) for c in raw_cands if isinstance(c, dict)]
        ai_decs = [ConsolidationDecision.from_dict(d) for d in raw_decs if isinstance(d, dict)]

        result = reconcile_and_validate_decisions(candidates, ai_cands, ai_decs, self.logger)

        if phase_callback:
            phase_callback(
                AnalysisPhase.CANDIDATE_CONSOLIDATION,
                f"Đã hợp nhất ứng viên cho node {node_id}",
                {
                    "merged": result.merged_count,
                    "rejected": result.rejected_count,
                    "eligible": result.eligible_count,
                    "cache": False,
                },
            )

        return result

    def consolidate(
        self,
        candidates: Sequence[CandidateProposal | dict[str, Any]],
        *,
        phase_callback: PhaseCallback | None = None,
        cancellation_token: threading.Event | None = None,
        model: str | None = None,
        thinking: str | None = None,
    ) -> ConsolidatedCandidateSet:
        """Consolidate all discovered candidates project-wide.

        Consumes all discovered candidates + policy.
        1. Normalizes input candidates.
        2. Pre-groups deterministically by similarity / overlap.
        3. Consolidates ambiguous groups using bounded AI nodes.
        4. Runs hierarchical global pass comparing surviving representatives.
        5. Returns ConsolidatedCandidateSet with complete accounting and no silent loss.
        """
        # Convert to CandidateProposal objects
        cand_list: list[CandidateProposal] = []
        for c in candidates:
            if isinstance(c, CandidateProposal):
                cand_list.append(c)
            elif isinstance(c, dict):
                cand_list.append(CandidateProposal.from_dict(c))

        if not cand_list:
            return ConsolidatedCandidateSet()

        effective_model = (
            model
            or getattr(self.settings, "season_connection_model", None)
            or getattr(self.settings, "evidence_scanner_model", "gemini-2.5-flash")
        )
        effective_thinking = thinking or getattr(self.settings, "season_connection_thinking", "auto")

        # Step 1: Pre-group candidates
        initial_groups = pre_group_candidates(cand_list, threshold=0.25)
        self.logger.info(f"Pre-grouped {len(cand_list)} candidates into {len(initial_groups)} groups.")

        group_results: list[ConsolidatedCandidateSet] = []
        for idx, group in enumerate(initial_groups):
            if cancellation_token and cancellation_token.is_set():
                raise AnalysisCancelledError("Candidate consolidation cancelled.")

            node_id = f"group_{idx + 1}"
            res = self._consolidate_single_node(
                node_id=node_id,
                candidates=group,
                model=effective_model,
                thinking=effective_thinking,
                cancellation_token=cancellation_token,
                phase_callback=phase_callback,
            )
            group_results.append(res)

        # Step 2: Combine Stage 1 survivors and decisions
        all_stage1_survivors: list[CandidateProposal] = []
        all_decisions_map: dict[str, ConsolidationDecision] = {}

        for res in group_results:
            all_stage1_survivors.extend(res.candidates)
            for d in res.decisions:
                all_decisions_map[d.candidate_id] = d

        # Step 3: Hierarchical Global Pass
        # If there was only 1 group, initial pass already compared all candidates against each other
        if len(initial_groups) <= 1 or len(all_stage1_survivors) <= 1:
            final_candidates = all_stage1_survivors
        else:
            self.logger.info(
                f"Starting hierarchical global pass on {len(all_stage1_survivors)} surviving candidates."
            )
            global_res = self._consolidate_single_node(
                node_id="global_pass",
                candidates=all_stage1_survivors,
                model=effective_model,
                thinking=effective_thinking,
                cancellation_token=cancellation_token,
                phase_callback=phase_callback,
            )
            final_candidates = global_res.candidates

            # Update decision mappings for survivors
            for d in global_res.decisions:
                all_decisions_map[d.candidate_id] = d

            # Chain any indirect merges (e.g. if A merged into B, and B merged into C, A's target becomes C)
            target_chain: dict[str, str] = {
                d.candidate_id: d.target_id
                for d in all_decisions_map.values()
                if d.action == ConsolidationAction.MERGE.value and d.target_id
            }
            for cid, dec in all_decisions_map.items():
                if dec.action == ConsolidationAction.MERGE.value and dec.target_id:
                    curr_target = dec.target_id
                    visited_targets = {cid}
                    while curr_target in target_chain and target_chain[curr_target] not in visited_targets:
                        visited_targets.add(curr_target)
                        curr_target = target_chain[curr_target]
                    dec.target_id = curr_target

        # Reconcile all original inputs against final survivors and decisions
        final_set = reconcile_and_validate_decisions(
            input_candidates=cand_list,
            ai_consolidated=final_candidates,
            ai_decisions=list(all_decisions_map.values()),
            logger_ref=self.logger,
        )

        if phase_callback:
            phase_callback(
                AnalysisPhase.CANDIDATE_CONSOLIDATION,
                f"Hoàn thành hợp nhất ứng viên: {final_set.merged_count} hợp nhất, "
                f"{final_set.rejected_count} loại bỏ, {final_set.eligible_count} hợp lệ.",
                {
                    "merged": final_set.merged_count,
                    "rejected": final_set.rejected_count,
                    "eligible": final_set.eligible_count,
                    "cache": False,
                },
            )

        return final_set


def consolidate_candidates(
    candidates: Sequence[CandidateProposal | dict[str, Any]],
    *,
    settings: AppSettings | None = None,
    client: Any = None,
    policy: EditorialPolicy | None = None,
    cache: HierarchyCacheManager | None = None,
    phase_callback: PhaseCallback | None = None,
    cancellation_token: threading.Event | None = None,
    model: str | None = None,
    thinking: str | None = None,
) -> ConsolidatedCandidateSet:
    """Functional entrypoint for candidate consolidation."""
    consolidator = CandidateConsolidator(settings=settings, client=client, policy=policy, cache=cache)
    return consolidator.consolidate(
        candidates=candidates,
        phase_callback=phase_callback,
        cancellation_token=cancellation_token,
        model=model,
        thinking=thinking,
    )
