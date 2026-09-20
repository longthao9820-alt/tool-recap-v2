"""Editorial policy domain models, directive derivation, and offline prompt taxonomy."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .enums import AudioPolicy

# Exact 24 objective categories required by contract
OBJECTIVE_CATEGORIES: tuple[str, ...] = (
    "major_scene",
    "conflict",
    "confrontation",
    "important_dialogue",
    "character_decision",
    "supporting_development",
    "relationship_change",
    "reveal",
    "reversal",
    "failed_plan",
    "consequence",
    "performance",
    "setup",
    "payoff",
    "unresolved",
    "contradiction",
    "moral/strategic dilemma",
    "visual_storytelling",
    "recurring behavior",
    "subplot",
    "power_shift",
    "reaction",
    "strength/weakness",
    "counter_evidence",
)

# Existing 15 schema categories in analyzer/evidence.py
BASE_SCHEMA_CATEGORIES: tuple[str, ...] = (
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

# Mapping from 24 objective categories to existing 15 schema categories
CATEGORY_MAPPING: dict[str, str] = {
    "major_scene": "major_scenes",
    "conflict": "conflicts",
    "confrontation": "conflicts",
    "important_dialogue": "dialogue",
    "character_decision": "character_decisions",
    "supporting_development": "supporting_developments",
    "relationship_change": "relationships",
    "reveal": "reveals",
    "reversal": "reversals",
    "failed_plan": "failures",
    "consequence": "consequences",
    "performance": "performance_moments",
    "setup": "setup_payoff",
    "payoff": "setup_payoff",
    "unresolved": "unresolved",
    "contradiction": "conflicts",
    "moral/strategic dilemma": "character_decisions",
    "visual_storytelling": "major_scenes",
    "recurring behavior": "supporting_developments",
    "subplot": "subplots",
    "power_shift": "relationships",
    "reaction": "performance_moments",
    "strength/weakness": "strengths_weaknesses",
    "counter_evidence": "conflicts",
}

# Regex patterns for detecting objective categories in clauses
CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "major_scene": re.compile(r"\b(major[\s_-]?scenes?)\b", re.IGNORECASE),
    "conflict": re.compile(r"\b(conflicts?)\b", re.IGNORECASE),
    "confrontation": re.compile(r"\b(confrontations?)\b", re.IGNORECASE),
    "important_dialogue": re.compile(
        r"\b((important|significant|key)[\s_-]?dialogues?|quotes?)\b", re.IGNORECASE
    ),
    "character_decision": re.compile(
        r"\b(character[\s_-]?decisions?|pivotal[\s_-]?choices?)\b", re.IGNORECASE
    ),
    "supporting_development": re.compile(
        r"\b(supporting[\s_-]?(character|development)s?|secondary[\s_-]?characters?)\b",
        re.IGNORECASE,
    ),
    "relationship_change": re.compile(
        r"\b(relationship[\s_-]?(changes?|dynamics?)|alliances?|rivalr(y|ies))\b",
        re.IGNORECASE,
    ),
    "reveal": re.compile(r"\b(reveals?|revelations?|secrets?[\s_-]?exposed)\b", re.IGNORECASE),
    "reversal": re.compile(r"\b(reversals?|plot[\s_-]?twists?)\b", re.IGNORECASE),
    "failed_plan": re.compile(
        r"\b(failed[\s_-]?plans?|fallen[\s_-]?plans?|tactical[\s_-]?blunders?|failures?)\b",
        re.IGNORECASE,
    ),
    "consequence": re.compile(r"\b(consequences?|fallout)\b", re.IGNORECASE),
    "performance": re.compile(
        r"\b(performance[\s_-]?moments?|standout[\s_-]?interactions?|dramatic[\s_-]?scenes?)\b",
        re.IGNORECASE,
    ),
    "setup": re.compile(r"\b(setups?|narrative[\s_-]?plants?|foreshadowing)\b", re.IGNORECASE),
    "payoff": re.compile(r"\b(payoffs?)\b", re.IGNORECASE),
    "unresolved": re.compile(
        r"\b(unresolved|cliffhangers?|lingering[\s_-]?questions?|mysteries)\b",
        re.IGNORECASE,
    ),
    "contradiction": re.compile(
        r"\b(contradictions?|inconsistenc(y|ies))\b", re.IGNORECASE
    ),
    "moral/strategic dilemma": re.compile(
        r"\b(moral[\s_-]?(and|or|/)?[\s_-]?strategic[\s_-]?dilemmas?|dilemmas?)\b",
        re.IGNORECASE,
    ),
    "visual_storytelling": re.compile(
        r"\b(visual[\s_-]?storytelling|visual[\s_-]?cues?|visual[\s_-]?details?)\b",
        re.IGNORECASE,
    ),
    "recurring behavior": re.compile(
        r"\b(recurring[\s_-]?behaviors?|habits?|patterns?[\s_-]?of[\s_-]?behavior)\b",
        re.IGNORECASE,
    ),
    "subplot": re.compile(r"\b(subplots?|secondary[\s_-]?storylines?)\b", re.IGNORECASE),
    "power_shift": re.compile(r"\b(power[\s_-]?shifts?|shifts?[\s_-]?in[\s_-]?power)\b", re.IGNORECASE),
    "reaction": re.compile(r"\b(reactions?|emotional[\s_-]?reactions?)\b", re.IGNORECASE),
    "strength/weakness": re.compile(
        r"\b(strengths?[\s_-]?(and|or|/)?[\s_-]?weakness(es)?|vulnerabilit(y|ies))\b",
        re.IGNORECASE,
    ),
    "counter_evidence": re.compile(
        r"\b(counter[\s_-]?evidence|counter[\s_-]?claims?)\b", re.IGNORECASE
    ),
}

# Publication, Audio, File, Render patterns (STRICTLY EXCLUDED FROM SCANNER)
# Notice: word 'video' alone is NOT in this pattern to avoid dropping scene-review clauses.
RE_AUDIO_FILE_PUB = re.compile(
    r"("
    r"\.(mp3|wav|aac|flac|ogg|m4a)\b|"
    r"\b(background[\s_-]?music|bgm|audio[\s_-]?(file|track|gain|volume|mix)|"
    r"commentary[\s_-]?gain|auto[\s_-]?duck|ducking|loudness|lufs|true[\s_-]?peak|"
    r"voice[\s_-]?id|tts|dubbing|voiceover|narration[\s_-]?voice|speech[\s_-]?synthesis)\b|"
    r"\b(mute[\s_-]?audio|duck[\s_-]?audio|keep[\s_-]?audio|original[\s_-]?only)\b|"
    r"\.(srt|vtt|ass)\b|\b(export[\s_-]?srt|generate[\s_-]?srt|burn[\s_-]?subtitles|subtitle[\s_-]?files?)\b|"
    r"\.(mp4|mkv|mov|avi|webm)\b|\b(\d+p\b|\d+fps\b|bitrates?|codecs?|h\.?264|hevc|"
    r"render[\s_-]?(queue|settings|video|output)|output[\s_-]?(dir|directory))\b|"
    r"\b(youtube|tiktok|instagram|facebook|vimeo|channel[\s_-]?names?|video[\s_-]?tags|"
    r"seo[\s_-]?descriptions?|upload[\s_-]?to|publish[\s_-]?to)\b|"
    r"\b(narration[\s_-]?scripts?|final[\s_-]?commentary|script[\s_-]?format|"
    r"target[\s_-]?duration|recap[\s_-]?length|recap[\s_-]?language|recap[\s_-]?mode|voice[\s_-]?style)\b"
    r")",
    re.IGNORECASE,
)

RE_COVERAGE = re.compile(
    r"\b(episodes?(\s+\d+)+|season[\s_-]?coverage|all[\s_-]?episodes|skip[\s_-]?episodes?|"
    r"only[\s_-]?episodes?|missing[\s_-]?episodes?|exclude[\s_-]?episodes?|min[\s_-]?episodes?|max[\s_-]?episodes?)\b",
    re.IGNORECASE,
)

RE_VALIDATION = re.compile(
    r"\b(hallucinat\w*|strict(ly)?[\s_-]?within|timestamps?|grounded|fact[\s_-]?check\w*|no[\s_-]?invent\w*)\b",
    re.IGNORECASE,
)

RE_CONNECTION = re.compile(
    r"\b(cross[\s_-]?episodes?|season[\s_-]?wide|overarching|causal[\s_-]?threads?|"
    r"connect(ion)?s?[\s_-]?across[\s_-]?episodes|batch[\s_-]?connections?|thread[\s_-]?id)\b",
    re.IGNORECASE,
)

RE_CANDIDATE = re.compile(
    r"\b(candidates?(\s+proposals?)?|single[\s_-]?scene|single[\s_-]?episode|season[\s_-]?arc|"
    r"strong[\s_-]?keep|weak[\s_-]?reject|duplicate[\s_-]?merge|quotas?|proposal[\s_-]?id)\b",
    re.IGNORECASE,
)

RE_EVIDENCE_TERMS = re.compile(
    r"\b(evidence|dialogues?|scenes?|transcripts?|quotes?|clues?|actions?|story[\s_-]?beats?|"
    r"characters?|events?|motives?|timeline)\b",
    re.IGNORECASE,
)

RE_OUTPUT_TERMS = re.compile(
    r"\b(output|script|commentary|narrator|tone|style|audience|summary|pacing|presentation)\b",
    re.IGNORECASE,
)


def normalize_prompt(prompt: str) -> str:
    """Normalize raw prompt by standardizing line endings and stripping trailing whitespace."""
    if not prompt:
        return ""
    text = prompt.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


@dataclass
class ScannerDirective:
    """Directive guiding evidence scanning stage. Base broad categories always active."""

    requested_categories: list[str] = field(default_factory=list)
    schema_categories: list[str] = field(default_factory=list)
    focus_clauses: list[str] = field(default_factory=list)
    custom_rules: list[str] = field(default_factory=list)
    directive_hash: str = ""

    def __post_init__(self) -> None:
        if not self.directive_hash:
            self.directive_hash = self.compute_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_categories": sorted(list(dict.fromkeys(self.requested_categories))),
            "schema_categories": sorted(list(dict.fromkeys(self.schema_categories))),
            "focus_clauses": list(self.focus_clauses),
            "custom_rules": list(self.custom_rules),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def format_directive(self, max_bytes: int = 4096) -> str:
        parts: list[str] = []
        if self.requested_categories:
            parts.append(f"Editorial Priority Categories: {', '.join(sorted(set(self.requested_categories)))}")
        if self.schema_categories:
            parts.append(f"Mapped Evidence Categories: {', '.join(sorted(set(self.schema_categories)))}")
        if self.focus_clauses:
            parts.append("Editorial Focus Instructions:\n" + "\n".join(f"- {c}" for c in self.focus_clauses))
        if self.custom_rules:
            parts.append("Custom Rules:\n" + "\n".join(f"- {r}" for r in self.custom_rules))

        if not parts:
            return ""

        header = "EDITORIAL EVIDENCE DIRECTIVE (Base broad extraction remains active across all categories):\n"
        full_text = header + "\n\n".join(parts)
        if len(full_text.encode("utf-8")) <= max_bytes:
            return full_text

        # Bounded truncation if exceeding max_bytes
        truncated_clauses = list(self.focus_clauses)
        truncated_rules = list(self.custom_rules)
        while truncated_rules and len(full_text.encode("utf-8")) > max_bytes:
            truncated_rules.pop()
            p = []
            if self.requested_categories:
                p.append(f"Editorial Priority Categories: {', '.join(sorted(set(self.requested_categories)))}")
            if self.schema_categories:
                p.append(f"Mapped Evidence Categories: {', '.join(sorted(set(self.schema_categories)))}")
            if truncated_clauses:
                p.append("Editorial Focus Instructions:\n" + "\n".join(f"- {c}" for c in truncated_clauses))
            if truncated_rules:
                p.append("Custom Rules:\n" + "\n".join(f"- {r}" for r in truncated_rules))
            full_text = header + "\n\n".join(p)

        while truncated_clauses and len(full_text.encode("utf-8")) > max_bytes:
            truncated_clauses.pop()
            p = []
            if self.requested_categories:
                p.append(f"Editorial Priority Categories: {', '.join(sorted(set(self.requested_categories)))}")
            if self.schema_categories:
                p.append(f"Mapped Evidence Categories: {', '.join(sorted(set(self.schema_categories)))}")
            if truncated_clauses:
                p.append("Editorial Focus Instructions:\n" + "\n".join(f"- {c}" for c in truncated_clauses))
            full_text = header + "\n\n".join(p)

        if len(full_text.encode("utf-8")) > max_bytes:
            return full_text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        return full_text


@dataclass
class CoverageDirective:
    """Directive guiding episode coverage requirements."""

    required_episodes: list[str] = field(default_factory=list)
    excluded_episodes: list[str] = field(default_factory=list)
    min_episodes: int | None = None
    max_episodes: int | None = None
    coverage_rules: list[str] = field(default_factory=list)
    directive_hash: str = ""

    def __post_init__(self) -> None:
        if not self.directive_hash:
            self.directive_hash = self.compute_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_episodes": sorted(list(dict.fromkeys(self.required_episodes))),
            "excluded_episodes": sorted(list(dict.fromkeys(self.excluded_episodes))),
            "min_episodes": self.min_episodes,
            "max_episodes": self.max_episodes,
            "coverage_rules": list(self.coverage_rules),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def format_directive(self, max_bytes: int = 4096) -> str:
        parts: list[str] = []
        if self.required_episodes:
            parts.append(f"Required episodes: {', '.join(self.required_episodes)}")
        if self.excluded_episodes:
            parts.append(f"Excluded episodes: {', '.join(self.excluded_episodes)}")
        if self.min_episodes is not None:
            parts.append(f"Minimum episodes: {self.min_episodes}")
        if self.max_episodes is not None:
            parts.append(f"Maximum episodes: {self.max_episodes}")
        if self.coverage_rules:
            parts.append("Coverage rules:\n" + "\n".join(f"- {r}" for r in self.coverage_rules))

        if not parts:
            return ""
        text = "\n\n".join(parts)
        return text[:max_bytes]


@dataclass
class ConnectionDirective:
    """Directive guiding cross-episode connection pass."""

    focus_characters: list[str] = field(default_factory=list)
    focus_arcs: list[str] = field(default_factory=list)
    cross_episode_rules: list[str] = field(default_factory=list)
    editorial_clauses: list[str] = field(default_factory=list)
    directive_hash: str = ""

    def __post_init__(self) -> None:
        if not self.directive_hash:
            self.directive_hash = self.compute_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus_characters": sorted(list(dict.fromkeys(self.focus_characters))),
            "focus_arcs": sorted(list(dict.fromkeys(self.focus_arcs))),
            "cross_episode_rules": list(self.cross_episode_rules),
            "editorial_clauses": list(self.editorial_clauses),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def format_directive(self, max_bytes: int = 4096) -> str:
        parts: list[str] = []
        if self.focus_characters:
            parts.append(f"Focus characters: {', '.join(self.focus_characters)}")
        if self.focus_arcs:
            parts.append(f"Focus arcs: {', '.join(self.focus_arcs)}")
        if self.cross_episode_rules:
            parts.append("Cross-episode connection rules:\n" + "\n".join(f"- {r}" for r in self.cross_episode_rules))
        if self.editorial_clauses:
            parts.append("Editorial guidance:\n" + "\n".join(f"- {c}" for c in self.editorial_clauses))

        if not parts:
            return "Standard season narrative connection and cross-episode analysis."
        text = "\n\n".join(parts)
        if len(text.encode("utf-8")) <= max_bytes:
            return text
        return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


@dataclass
class CandidateDirective:
    """Directive guiding candidate proposal selection and scope prioritization."""

    allowed_scopes: list[str] = field(default_factory=list)
    priority_themes: list[str] = field(default_factory=list)
    candidate_rules: list[str] = field(default_factory=list)
    editorial_clauses: list[str] = field(default_factory=list)
    directive_hash: str = ""

    def __post_init__(self) -> None:
        if not self.directive_hash:
            self.directive_hash = self.compute_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_scopes": sorted(list(dict.fromkeys(self.allowed_scopes))),
            "priority_themes": sorted(list(dict.fromkeys(self.priority_themes))),
            "candidate_rules": list(self.candidate_rules),
            "editorial_clauses": list(self.editorial_clauses),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def format_directive(self, max_bytes: int = 4096) -> str:
        parts: list[str] = []
        if self.allowed_scopes:
            parts.append(f"Allowed scopes: {', '.join(self.allowed_scopes)}")
        if self.priority_themes:
            parts.append(f"Priority themes: {', '.join(self.priority_themes)}")
        if self.candidate_rules:
            parts.append("Candidate rules:\n" + "\n".join(f"- {r}" for r in self.candidate_rules))
        if self.editorial_clauses:
            parts.append("Editorial guidance:\n" + "\n".join(f"- {c}" for c in self.editorial_clauses))

        if not parts:
            return ""
        text = "\n\n".join(parts)
        return text[:max_bytes]


@dataclass
class OutputDirective:
    """Directive guiding final commentary narration, audio mixing, and file export formatting."""

    narration_style: str = ""
    target_duration: str = ""
    tone: str = ""
    output_rules: list[str] = field(default_factory=list)
    audio_rules: list[str] = field(default_factory=list)
    file_rules: list[str] = field(default_factory=list)
    directive_hash: str = ""

    def __post_init__(self) -> None:
        if not self.directive_hash:
            self.directive_hash = self.compute_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "narration_style": self.narration_style,
            "target_duration": self.target_duration,
            "tone": self.tone,
            "output_rules": list(self.output_rules),
            "audio_rules": list(self.audio_rules),
            "file_rules": list(self.file_rules),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def format_directive(self, max_bytes: int = 4096) -> str:
        parts: list[str] = []
        if self.narration_style:
            parts.append(f"Narration style: {self.narration_style}")
        if self.target_duration:
            parts.append(f"Target duration: {self.target_duration}")
        if self.tone:
            parts.append(f"Tone: {self.tone}")
        if self.output_rules:
            parts.append("Output rules:\n" + "\n".join(f"- {r}" for r in self.output_rules))
        if self.audio_rules:
            parts.append("Audio rules:\n" + "\n".join(f"- {r}" for r in self.audio_rules))
        if self.file_rules:
            parts.append("File/Export rules:\n" + "\n".join(f"- {r}" for r in self.file_rules))

        if not parts:
            return "Standard video recap commentary and output finalization."
        text = "\n\n".join(parts)
        if len(text.encode("utf-8")) <= max_bytes:
            return text
        return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


FinalizerDirective = OutputDirective


@dataclass
class ValidationDirective:
    """Directive guiding hallucination checks, factual grounding, and timestamp bounds."""

    strict_timestamps: bool = True
    no_hallucination: bool = True
    validation_rules: list[str] = field(default_factory=list)
    directive_hash: str = ""

    def __post_init__(self) -> None:
        if not self.directive_hash:
            self.directive_hash = self.compute_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "strict_timestamps": self.strict_timestamps,
            "no_hallucination": self.no_hallucination,
            "validation_rules": list(self.validation_rules),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def format_directive(self, max_bytes: int = 4096) -> str:
        parts: list[str] = []
        if self.strict_timestamps:
            parts.append("Strict timestamp bounds enforced.")
        if self.no_hallucination:
            parts.append("Strict grounding: do not hallucinate dialogue or events.")
        if self.validation_rules:
            parts.append("Validation rules:\n" + "\n".join(f"- {r}" for r in self.validation_rules))

        if not parts:
            return ""
        text = "\n\n".join(parts)
        return text[:max_bytes]


@dataclass
class EditorialPolicy:
    """Deterministic offline derived editorial policy governing pipeline stages."""

    version: str = "v1"
    raw_prompt: str = ""
    raw_prompt_hash: str = ""
    policy_hash: str = ""
    scanner_directive: ScannerDirective = field(default_factory=ScannerDirective)
    coverage_directive: CoverageDirective = field(default_factory=CoverageDirective)
    connection_directive: ConnectionDirective = field(default_factory=ConnectionDirective)
    candidate_directive: CandidateDirective = field(default_factory=CandidateDirective)
    output_directive: OutputDirective = field(default_factory=OutputDirective)
    validation_directive: ValidationDirective = field(default_factory=ValidationDirective)

    def __post_init__(self) -> None:
        if not self.policy_hash:
            self.policy_hash = self.compute_policy_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "raw_prompt_hash": self.raw_prompt_hash,
            "scanner_directive": self.scanner_directive.to_dict(),
            "coverage_directive": self.coverage_directive.to_dict(),
            "connection_directive": self.connection_directive.to_dict(),
            "candidate_directive": self.candidate_directive.to_dict(),
            "output_directive": self.output_directive.to_dict(),
            "validation_directive": self.validation_directive.to_dict(),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def compute_policy_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_prompt(cls, raw_prompt: str, version: str = "v1") -> EditorialPolicy:
        """Derive EditorialPolicy deterministically offline via clause classifier taxonomy."""
        normalized = normalize_prompt(raw_prompt)
        raw_prompt_hash = (
            hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""
        )

        if not normalized:
            scanner_dir = ScannerDirective()
            coverage_dir = CoverageDirective()
            conn_dir = ConnectionDirective()
            cand_dir = CandidateDirective()
            output_dir = OutputDirective()
            val_dir = ValidationDirective()
            policy = cls(
                version=version,
                raw_prompt="",
                raw_prompt_hash="",
                scanner_directive=scanner_dir,
                coverage_directive=coverage_dir,
                connection_directive=conn_dir,
                candidate_directive=cand_dir,
                output_directive=output_dir,
                validation_directive=val_dir,
            )
            policy.policy_hash = policy.compute_policy_hash()
            return policy

        # Split normalized prompt into lines and sentence clauses
        raw_lines = normalized.split("\n")
        clauses: list[str] = []
        for line in raw_lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            # Strip bullet list markers
            cleaned = re.sub(r"^[\s*\->#\d.)]+", "", line_str).strip()
            if not cleaned:
                continue
            # Split sentences if line contains multiple distinct statements
            sentence_parts = re.split(r"(?<=[.!?;])\s+", cleaned)
            for part in sentence_parts:
                p_clean = part.strip()
                if p_clean:
                    clauses.append(p_clean)

        requested_cats: list[str] = []
        schema_cats: list[str] = []
        scanner_focus: list[str] = []
        scanner_rules: list[str] = []

        coverage_rules: list[str] = []
        conn_rules: list[str] = []
        cand_rules: list[str] = []
        editorial_clauses: list[str] = []

        output_rules: list[str] = []
        audio_rules: list[str] = []
        file_rules: list[str] = []
        val_rules: list[str] = []

        for clause in clauses:
            # 1. Check Publication, Audio, File, Render (CRITICAL: EXCLUDED FROM SCANNER)
            if RE_AUDIO_FILE_PUB.search(clause):
                if re.search(r"(\.(mp3|wav|aac|flac|ogg|m4a)\b|audio|music|duck|mute|gain|voice|tts|loudness|lufs)", clause, re.I):
                    audio_rules.append(clause)
                elif re.search(r"(\.(srt|vtt|ass)\b|subtitle)", clause, re.I):
                    file_rules.append(clause)
                elif re.search(r"(\.(mp4|mkv|mov|avi|webm)\b|render|\d+p|\d+fps|bitrate|codec|output_dir)", clause, re.I):
                    file_rules.append(clause)
                else:
                    output_rules.append(clause)
                # MUST NOT LEAK TO SCANNER
                continue

            # 2. Validation rules
            if RE_VALIDATION.search(clause):
                val_rules.append(clause)
                continue

            # 3. Coverage rules
            if RE_COVERAGE.search(clause):
                coverage_rules.append(clause)
                continue

            # 4. Connection rules
            is_connection = bool(RE_CONNECTION.search(clause))
            if is_connection:
                conn_rules.append(clause)

            # 5. Candidate rules
            is_candidate = bool(RE_CANDIDATE.search(clause))
            if is_candidate:
                cand_rules.append(clause)

            # 6. Objective category matching for evidence
            matched_cats: list[str] = []
            for cat, pattern in CATEGORY_PATTERNS.items():
                if pattern.search(clause):
                    matched_cats.append(cat)
                    if cat not in requested_cats:
                        requested_cats.append(cat)
                    mapped = CATEGORY_MAPPING[cat]
                    if mapped not in schema_cats:
                        schema_cats.append(mapped)

            if matched_cats:
                scanner_focus.append(clause)
                if not is_candidate and not is_connection:
                    editorial_clauses.append(clause)
                continue

            # 7. General evidence-focused terms
            if RE_EVIDENCE_TERMS.search(clause):
                scanner_focus.append(clause)
                if not is_candidate and not is_connection:
                    editorial_clauses.append(clause)
                continue

            # 8. If already categorized as connection or candidate, don't leak as unknown
            if is_connection or is_candidate:
                continue

            # 9. Output-specific terms route to finalizer
            if RE_OUTPUT_TERMS.search(clause):
                output_rules.append(clause)
                continue

            # 10. Unknown purely editorial narrative clause: route to evidence + candidate + connection
            scanner_focus.append(clause)
            editorial_clauses.append(clause)

        scanner_dir = ScannerDirective(
            requested_categories=requested_cats,
            schema_categories=schema_cats,
            focus_clauses=scanner_focus,
            custom_rules=scanner_rules,
        )
        coverage_dir = CoverageDirective(coverage_rules=coverage_rules)
        conn_dir = ConnectionDirective(
            cross_episode_rules=conn_rules,
            editorial_clauses=editorial_clauses,
        )
        cand_dir = CandidateDirective(
            candidate_rules=cand_rules,
            editorial_clauses=editorial_clauses,
        )
        output_dir = OutputDirective(
            output_rules=output_rules,
            audio_rules=audio_rules,
            file_rules=file_rules,
        )
        val_dir = ValidationDirective(validation_rules=val_rules)

        policy = cls(
            version=version,
            raw_prompt=normalized,
            raw_prompt_hash=raw_prompt_hash,
            scanner_directive=scanner_dir,
            coverage_directive=coverage_dir,
            connection_directive=conn_dir,
            candidate_directive=cand_dir,
            output_directive=output_dir,
            validation_directive=val_dir,
        )
        policy.policy_hash = policy.compute_policy_hash()
        return policy


def format_evidence_directive(
    directive: ScannerDirective | EditorialPolicy | None,
    max_bytes: int = 4096,
) -> str:
    """Format compact evidence directive for scanner prompt."""
    if directive is None:
        return ""
    if isinstance(directive, EditorialPolicy):
        return directive.scanner_directive.format_directive(max_bytes=max_bytes)
    return directive.format_directive(max_bytes=max_bytes)


def format_connection_directive(
    directive: ConnectionDirective | EditorialPolicy | str | None,
    max_bytes: int = 4096,
) -> str:
    """Format compact connection directive for batch/merge prompts."""
    if directive is None:
        return "Standard season narrative connection and cross-episode analysis."
    if isinstance(directive, str):
        if not directive.strip():
            return "Standard season narrative connection and cross-episode analysis."
        return directive.strip()[:max_bytes]
    if isinstance(directive, EditorialPolicy):
        return directive.connection_directive.format_directive(max_bytes=max_bytes)
    return directive.format_directive(max_bytes=max_bytes)


def format_output_directive(
    directive: OutputDirective | EditorialPolicy | str | None,
    max_bytes: int = 4096,
) -> str:
    """Format compact output directive for finalizer prompts."""
    if directive is None:
        return "Standard video recap commentary and output finalization."
    if isinstance(directive, str):
        if not directive.strip():
            return "Standard video recap commentary and output finalization."
        return directive.strip()[:max_bytes]
    if isinstance(directive, EditorialPolicy):
        return directive.output_directive.format_directive(max_bytes=max_bytes)
    return directive.format_directive(max_bytes=max_bytes)
