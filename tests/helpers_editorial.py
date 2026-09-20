"""Editorial pipeline test mock helper.

Provides stage_response() to dispatch schema-valid mock responses across:
- Evidence Scanner
- Season Connection
- Candidate Discovery
- Candidate Consolidation
- Zero-Output / Coverage Verification
- Candidate Finalizer
"""
from __future__ import annotations

import json
import re
from typing import Any, Sequence

from toolrecap_v2.analyzer.prompts import (
    CANDIDATE_CONSOLIDATION_SYSTEM_PROMPT,
    CANDIDATE_DISCOVERY_SYSTEM_PROMPT,
    FINALIZER_SYSTEM_PROMPT,
    SCANNER_SYSTEM_PROMPT,
    SEASON_CONNECTION_SYSTEM_PROMPT,
    ZERO_OUTPUT_VERIFICATION_SYSTEM_PROMPT,
)


def stage_response(
    system: str = "",
    user_text: str = "",
    episodes: Sequence[str] | Sequence[Any] | None = None,
    default: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Derive a schema-valid response for the current pipeline stage.

    Preserves custom output passed via `default`.
    """
    sys_lower = (system or "").lower().strip()

    # Determine episode list
    ep_list: list[str] = []
    if episodes:
        for ep in episodes:
            if hasattr(ep, "episode_id"):
                ep_list.append(str(ep.episode_id))
            elif isinstance(ep, dict) and "episode_id" in ep:
                ep_list.append(str(ep["episode_id"]))
            elif isinstance(ep, str) and ep.strip():
                ep_list.append(ep.strip())
    if not ep_list and user_text:
        found = re.findall(r"\bE\d+\b", user_text)
        if found:
            ep_list = list(dict.fromkeys(found))
    if not ep_list:
        ep_list = ["E01"]

    # 1. SCANNER STAGE
    if (
        "expert narrative evidence scanner" in sys_lower
        or "targeted evidence gap scanner" in sys_lower
        or (kwargs.get("model") == "sub" and "lead editor" not in sys_lower and "discoverer" not in sys_lower)
    ):
        if isinstance(default, dict) and any(k in default for k in ("major_scenes", "events", "dialogue")):
            return default
        return {
            "episode_id": ep_list[0],
            "range_start_ms": 0,
            "range_end_ms": 2000,
            "events": [{"start_ms": 0, "end_ms": 1000, "summary": "Ev"}],
            "major_scenes": [{"start_ms": 0, "end_ms": 1000, "summary": "Ev"}],
            "dialogue": [{"start_ms": 0, "end_ms": 1000, "speaker": "A", "quote": "Hi"}],
        }

    # 2. ZERO-OUTPUT / LOW-COVERAGE VERIFIER STAGE
    if (
        "zero-output" in sys_lower
        or "low-coverage verifier" in sys_lower
        or "confirm_no_eligible" in sys_lower
        or sys_lower == ZERO_OUTPUT_VERIFICATION_SYSTEM_PROMPT.lower()
    ):
        if isinstance(default, dict) and "confirm_no_eligible" in default:
            return default
        # Zero confirms by default
        return {
            "recovered_candidates": [],
            "confirm_no_eligible": True,
            "rationale": "Audit confirmed zero eligible candidates based on grounded source evidence.",
        }

    # 3. SEASON CONNECTION STAGE
    if (
        "season narrative architect" in sys_lower
        or "batch connection analyst" in sys_lower
        or "cross-batch merge synthesizer" in sys_lower
        or sys_lower == SEASON_CONNECTION_SYSTEM_PROMPT.lower()
    ):
        if isinstance(default, dict) and any(k in default for k in ("cross_episode_links", "candidate_proposals")):
            return default
        return {
            "cross_episode_links": [{
                "thread_id": "thread_main", "theme": "Narrative continuity",
                "episodes": ep_list, "summary": "Grounded cross-episode thread.",
            }] if len(ep_list) > 1 else [],
            "candidate_proposals": [],
            "supporting_character_arcs": [],
            "rejected_or_merged": [],
        }

    # Helper to extract candidates from default["outputs"]
    def _candidates_from_custom_output(outs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cands: list[dict[str, Any]] = []
        for idx, out in enumerate(outs):
            cid = out.get("output_id", f"cand_{idx + 1}")
            title = out.get("title", f"Candidate {idx + 1}")
            scope = out.get("candidate_scope", "SEASON_ARC" if len(ep_list) > 1 else "SINGLE_EPISODE")
            ranges: list[dict[str, Any]] = []
            cand_eps: list[str] = []
            for seg in out.get("segments", []):
                for clip in seg.get("source_clips", []):
                    c_ep = clip.get("episode_id", ep_list[0])
                    if c_ep not in cand_eps:
                        cand_eps.append(c_ep)
                    s_sec = float(clip.get("start", 0.0))
                    e_sec = float(clip.get("end", 1.0))
                    ranges.append({
                        "episode_id": c_ep,
                        "start_seconds": s_sec,
                        "end_seconds": e_sec,
                        "ref": f"ref_{c_ep}_{int(s_sec)}",
                    })
            if not cand_eps:
                cand_eps = list(ep_list)
            if not ranges:
                ranges = [
                    {"episode_id": ep, "start_seconds": 0.0, "end_seconds": 1.0, "ref": f"ref_{ep}"}
                    for ep in cand_eps
                ]
            cands.append({
                "proposal_id": cid,
                "title": title,
                "candidate_scope": scope,
                "episodes": cand_eps,
                "characters": ["Alice"],
                "primary_character": "Alice",
                "supporting_characters": [],
                "subject": "Main Subject",
                "central_thesis": f"Thesis for {title}",
                "source_ranges": ranges,
                "supporting_evidence": [r["ref"] for r in ranges],
                "observed_facts": [f"Observed action in {title}"],
                "status": "keep",
                "confidence": 0.95,
                "editorial_reason": f"Compelling narrative arc for {title}",
                "description": f"Description for {title}",
            })
        return cands

    # 4. CANDIDATE DISCOVERY STAGE
    if (
        "candidate discoverer" in sys_lower
        or "discovered_candidates" in sys_lower
        or sys_lower == CANDIDATE_DISCOVERY_SYSTEM_PROMPT.lower()
    ):
        if isinstance(default, dict) and "discovered_candidates" in default:
            return default
        if isinstance(default, dict) and "outputs" in default:
            outs = default["outputs"]
            if not outs:
                # Zero outputs -> zero discovery candidates
                return {"discovered_candidates": []}
            return {"discovered_candidates": _candidates_from_custom_output(outs)}

        # Default discovery candidates with source episode default E01 and range
        ranges = [
            {"episode_id": ep, "start_seconds": 0.0, "end_seconds": 1.0, "ref": f"ref_{ep}"}
            for ep in ep_list
        ]
        return {
            "discovered_candidates": [
                {
                    "proposal_id": "cand_01",
                    "title": "Default Candidate",
                    "candidate_scope": "SEASON_ARC" if len(ep_list) > 1 else "SINGLE_EPISODE",
                    "episodes": ep_list,
                    "characters": ["Alice"],
                    "primary_character": "Alice",
                    "supporting_characters": [],
                    "subject": "Main Story",
                    "central_thesis": "Story progression",
                    "source_ranges": ranges,
                    "supporting_evidence": [r["ref"] for r in ranges],
                    "observed_facts": ["Story begins"],
                    "status": "keep",
                    "confidence": 0.95,
                    "editorial_reason": "Central narrative thread",
                    "description": "Default candidate description",
                }
            ]
        }

    # 5. CANDIDATE CONSOLIDATION STAGE
    if (
        "candidate consolidator" in sys_lower
        or "consolidated_candidates" in sys_lower
        or sys_lower == CANDIDATE_CONSOLIDATION_SYSTEM_PROMPT.lower()
    ):
        if isinstance(default, dict) and "consolidated_candidates" in default and "decisions" in default:
            return default

        # Try parsing discovery candidates from user_text if feasible
        parsed_candidates: list[dict[str, Any]] | None = None
        if user_text and "INPUT PROPOSALS TO CONSOLIDATE:" in user_text:
            try:
                after = user_text.split("INPUT PROPOSALS TO CONSOLIDATE:", 1)[1]
                start_idx = after.find("[")
                if start_idx != -1:
                    end_idx = after.find("\nMANDATES:", start_idx)
                    json_str = after[start_idx:end_idx].strip() if end_idx != -1 else after[start_idx:].strip()
                    r_bracket = json_str.rfind("]")
                    if r_bracket != -1:
                        json_str = json_str[: r_bracket + 1]
                    loaded = json.loads(json_str)
                    if isinstance(loaded, list):
                        parsed_candidates = loaded
            except Exception:
                parsed_candidates = None

        if parsed_candidates is not None:
            decisions = [
                {
                    "candidate_id": c.get("proposal_id", f"cand_{i+1}"),
                    "action": "KEEP",
                    "reason_code": "distinct_thesis",
                    "reason": "Retain distinct candidate thesis.",
                    "target_id": "",
                }
                for i, c in enumerate(parsed_candidates)
            ]
            return {
                "consolidated_candidates": parsed_candidates,
                "decisions": decisions,
            }

        # Otherwise return same (from default or discovery derivation)
        if isinstance(default, dict) and "outputs" in default:
            outs = default["outputs"]
            if not outs:
                return {"consolidated_candidates": [], "decisions": []}
            derived = _candidates_from_custom_output(outs)
            decisions = [
                {
                    "candidate_id": c.get("proposal_id", f"cand_{i+1}"),
                    "action": "KEEP",
                    "reason_code": "distinct_thesis",
                    "reason": "Retain distinct candidate thesis.",
                    "target_id": "",
                }
                for i, c in enumerate(derived)
            ]
            return {
                "consolidated_candidates": derived,
                "decisions": decisions,
            }

        # Fallback same
        disc = stage_response(CANDIDATE_DISCOVERY_SYSTEM_PROMPT, user_text, episodes=ep_list, default=default)
        cands = disc.get("discovered_candidates", [])
        decisions = [
            {
                "candidate_id": c.get("proposal_id", f"cand_{i+1}"),
                "action": "KEEP",
                "reason_code": "distinct_thesis",
                "reason": "Retain distinct candidate thesis.",
                "target_id": "",
            }
            for i, c in enumerate(cands)
        ]
        return {
            "consolidated_candidates": cands,
            "decisions": decisions,
        }

    # 6. FINALIZER STAGE (and fallback)
    # Keep custom output passed to helper
    if isinstance(default, dict) and "outputs" in default:
        return default
    if isinstance(default, list):
        return {"outputs": default}
    if isinstance(default, dict) and "output_id" in default:
        return {"outputs": [default]}

    # Default fallback finalizer response
    return {
        "outputs": [
            {
                "output_id": "out_01",
                "title": "Recap Video",
                "candidate_scope": "SEASON_ARC" if len(ep_list) > 1 else "SINGLE_EPISODE",
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "source_clips": [
                            {"episode_id": ep, "start": 0.0, "end": 1.0}
                            for ep in ep_list
                        ],
                        "narration": "Recap commentary for narrative sequence.",
                        "audio_policy": "duck",
                    }
                ],
            }
        ]
    }
