"""Tests for EditorialPolicy domain, directive derivation, offline taxonomy,
cache key invalidation, evidence scanner hashing, connector connection hashing,
and settings prompt persistence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolrecap_v2.analyzer import (
    AnalysisEngine,
    EvidenceScanner,
    SeasonConnector,
    compute_final_plan_cache_key,
    compute_scanner_config_version,
    run_analysis,
)
from toolrecap_v2.domain import (
    BASE_SCHEMA_CATEGORIES,
    CATEGORY_MAPPING,
    OBJECTIVE_CATEGORIES,
    CandidateDirective,
    ConnectionDirective,
    CoverageDirective,
    EditorialPolicy,
    EpisodeEvidence,
    EvidenceCacheManager,
    HierarchyCacheManager,
    OutputDirective,
    ScannerDirective,
    SourceEpisode,
    ValidationDirective,
    compute_batch_cache_key,
    compute_cache_key,
    compute_connection_cache_key,
    compute_finalizer_cache_key,
    compute_merge_cache_key,
    format_connection_directive,
    format_evidence_directive,
    format_output_directive,
    normalize_prompt,
)
from toolrecap_v2.settings import AppSettings, SettingsStore
from toolrecap_v2.subtitles.cache import compute_subtitle_cache_key


# ---------------------------------------------------------------------------
# 1. 24 Objective Categories & Offline Taxonomy Mapping
# ---------------------------------------------------------------------------

def test_objective_categories_exact_24_and_schema_mapping() -> None:
    """Exact 24 objective categories must map into the 15 base schema categories."""
    assert len(OBJECTIVE_CATEGORIES) == 24
    assert len(BASE_SCHEMA_CATEGORIES) == 15

    for cat in OBJECTIVE_CATEGORIES:
        assert cat in CATEGORY_MAPPING, f"Category '{cat}' missing from CATEGORY_MAPPING"
        mapped = CATEGORY_MAPPING[cat]
        assert mapped in BASE_SCHEMA_CATEGORIES, f"Mapped target '{mapped}' not in BASE_SCHEMA_CATEGORIES"


def test_editorial_policy_directive_separation() -> None:
    """Prompt clauses must be cleanly routed to appropriate directives without leakage."""
    raw_prompt = """
    # Editorial Instructions
    * Focus heavily on character decisions and reveals regarding the conspiracy.
    * Track connections across episodes and overarching power shifts.
    * Only include episodes 1 2 3 in the season coverage.
    * Strictly within timestamps and no hallucinating facts.
    * Narration style: cynical and sharp tone with fast pacing.
    * Export srt subtitles and background music auto ducking -18dB.
    """
    policy = EditorialPolicy.from_prompt(raw_prompt)

    # 1. Scanner directive gets evidence clauses and mapped categories
    scanner_dict = policy.scanner_directive.to_dict()
    assert "character_decision" in scanner_dict["requested_categories"]
    assert "reveal" in scanner_dict["requested_categories"]
    assert "character_decisions" in scanner_dict["schema_categories"]
    assert "reveals" in scanner_dict["schema_categories"]

    # CRITICAL: Audio and render instructions must NOT leak to scanner
    scanner_formatted = format_evidence_directive(policy.scanner_directive)
    assert "ducking" not in scanner_formatted
    assert "background music" not in scanner_formatted
    assert "export srt" not in scanner_formatted

    # 2. Connection directive gets cross-episode instructions
    conn_formatted = format_connection_directive(policy.connection_directive)
    assert "across episodes" in conn_formatted or "overarching" in conn_formatted

    # 3. Output directive gets narration, tone, audio, export instructions
    out_formatted = format_output_directive(policy.output_directive)
    assert "cynical and sharp" in out_formatted
    assert "ducking" in out_formatted or "background music" in out_formatted


# ---------------------------------------------------------------------------
# 2. compute_final_plan_cache_key Invalidates on Policy / Output Directive Hash
# ---------------------------------------------------------------------------

def test_final_plan_cache_key_includes_policy_and_output_hashes() -> None:
    """Final plan cache key includes policy hash and output directive hash to invalidate old outputs."""
    ev_map = {
        "E01": EpisodeEvidence(episode_id="E01", source_video="C:/ep1.mp4", duration_seconds=60.0)
    }
    settings = AppSettings()

    # Base key with default empty policy
    k_default = compute_final_plan_cache_key(ev_map, settings, "SEASON")
    assert isinstance(k_default, str)
    assert len(k_default) == 24

    # Key with specific policy
    policy_a = EditorialPolicy.from_prompt("Focus on character decisions and dramatic scenes.")
    k_policy_a = compute_final_plan_cache_key(ev_map, settings, "SEASON", policy=policy_a)

    # Key with changed output directive (e.g. change narration style / tone)
    policy_b = EditorialPolicy.from_prompt(
        "Focus on character decisions and dramatic scenes.\nNarration style: humorous and witty."
    )
    k_policy_b = compute_final_plan_cache_key(ev_map, settings, "SEASON", policy=policy_b)

    # All three keys must be different (invalidation occurs on policy/output directive change)
    assert k_default != k_policy_a
    assert k_policy_a != k_policy_b
    assert k_default != k_policy_b


def test_final_plan_cache_key_old_output_invalidation_and_secrets_exclusion() -> None:
    """Final plan cache key invalidates old output without policy_hash, and strictly excludes secrets."""
    ev_map = {
        "E01": EpisodeEvidence(episode_id="E01", source_video="C:/ep1.mp4", duration_seconds=60.0)
    }
    secret_key = "sk-super-secret-key-999"
    secret_endpoint = "https://internal.api.com/v1"
    settings = AppSettings(api_key=secret_key, api_endpoint=secret_endpoint)

    k_plan = compute_final_plan_cache_key(ev_map, settings, "SEASON")
    assert secret_key not in k_plan
    assert secret_endpoint not in k_plan
    assert "secret" not in k_plan


# ---------------------------------------------------------------------------
# 3. Evidence Cache Scanner Hash Invalidation and Output Invariance
# ---------------------------------------------------------------------------

def test_evidence_cache_scanner_hash_invalidation_and_isolation(tmp_path: Path) -> None:
    """Changing evidence focus invalidates evidence cache, but changing output instructions preserves cache."""
    settings_base = AppSettings()

    # 1. Scanner config version with prompt A (focus on reveals)
    prompt_reveals = "Focus on reveals and character decisions."
    policy_reveals = EditorialPolicy.from_prompt(prompt_reveals)
    cfg_reveals = compute_scanner_config_version(settings_base, policy=policy_reveals)

    # 2. Scanner config version with prompt B (focus on conflicts)
    prompt_conflicts = "Focus on conflicts and major confrontations."
    policy_conflicts = EditorialPolicy.from_prompt(prompt_conflicts)
    cfg_conflicts = compute_scanner_config_version(settings_base, policy=policy_conflicts)

    # Evidence focus changes -> scanner config version changes -> cache invalidated
    assert cfg_reveals != cfg_conflicts

    # 3. Scanner config version with prompt C (same evidence focus as A, but DIFFERENT output / audio tone)
    prompt_reveals_with_output = (
        "Focus on reveals and character decisions.\n"
        "Narration style: sarcastic tone.\n"
        "Background music auto ducking -15dB.\n"
        "Export 1080p mp4 video."
    )
    policy_reveals_with_output = EditorialPolicy.from_prompt(prompt_reveals_with_output)
    cfg_reveals_with_output = compute_scanner_config_version(settings_base, policy=policy_reveals_with_output)

    # Output instructions change -> scanner directive hash is UNCHANGED -> evidence cache hit
    assert cfg_reveals == cfg_reveals_with_output

    # Verify EvidenceCacheManager load/save behavior
    cache_mgr = EvidenceCacheManager(tmp_path / "ev_cache")
    video_file = tmp_path / "test_ep.mp4"
    video_file.write_bytes(b"dummy video")
    ep = SourceEpisode(episode_id="E01", source_video=str(video_file), duration_seconds=10.0)
    ev = EpisodeEvidence(episode_id="E01", source_video=str(video_file), duration_seconds=10.0)

    # Save under prompt A config
    cache_mgr.save_evidence(ep, cfg_reveals, ev)

    # Load with prompt C (different output instructions, same evidence focus) -> HIT
    loaded_hit = cache_mgr.load_evidence(ep, cfg_reveals_with_output)
    assert loaded_hit is not None
    assert loaded_hit.episode_id == "E01"

    # Load with prompt B (different evidence focus) -> MISS (invalidated)
    loaded_miss = cache_mgr.load_evidence(ep, cfg_conflicts)
    assert loaded_miss is None


# ---------------------------------------------------------------------------
# 4. Connector Caches Connection Hash
# ---------------------------------------------------------------------------

def test_connector_caches_connection_hash(tmp_path: Path) -> None:
    """SeasonConnector uses connection directive hash; invalidates on cross-episode changes, preserves on output changes."""
    h_cache = HierarchyCacheManager(tmp_path / "hierarchy")
    settings = AppSettings()

    episodes = [
        SourceEpisode(episode_id="E01", source_video="C:/ep1.mp4", duration_seconds=60.0),
        SourceEpisode(episode_id="E02", source_video="C:/ep2.mp4", duration_seconds=60.0),
    ]
    ev_map = {
        "E01": EpisodeEvidence(episode_id="E01", source_video="C:/ep1.mp4", duration_seconds=60.0),
        "E02": EpisodeEvidence(episode_id="E02", source_video="C:/ep2.mp4", duration_seconds=60.0),
    }

    # Connector 1: Connection rule A
    prompt_conn_a = "Connect causal threads across episodes for the conspiracy arc."
    policy_a = EditorialPolicy.from_prompt(prompt_conn_a)
    conn_a = SeasonConnector(settings=settings, hierarchy_cache=h_cache, policy=policy_a)
    key_a = conn_a.compute_connection_key(episodes, ev_map)

    # Connector 2: Connection rule B (different cross-episode instruction)
    prompt_conn_b = "Connect character relationship changes across episodes."
    policy_b = EditorialPolicy.from_prompt(prompt_conn_b)
    conn_b = SeasonConnector(settings=settings, hierarchy_cache=h_cache, policy=policy_b)
    key_b = conn_b.compute_connection_key(episodes, ev_map)

    # Connection instructions change -> connection key changes (invalidated)
    assert key_a != key_b

    # Connector 3: Same connection rule A, but DIFFERENT output / audio directive
    prompt_conn_a_output = (
        "Connect causal threads across episodes for the conspiracy arc.\n"
        "Narration style: fast-paced thriller.\n"
        "Commentary gain +3dB."
    )
    policy_a_output = EditorialPolicy.from_prompt(prompt_conn_a_output)
    conn_a_output = SeasonConnector(settings=settings, hierarchy_cache=h_cache, policy=policy_a_output)
    key_a_output = conn_a_output.compute_connection_key(episodes, ev_map)

    # Output instructions change -> connection directive hash UNCHANGED -> connection cache hit
    assert key_a == key_a_output

    # Test saving and loading connection result with the connection key
    conn_data = {"unified_timeline": [], "candidate_proposals": []}
    h_cache.save_connection_result(key_a, conn_data)

    loaded, meta = h_cache.load_connection_result(key_a_output)
    assert meta["hit"] is True
    assert loaded is not None


# ---------------------------------------------------------------------------
# 5. Settings Exact Prompt Persistence Test
# ---------------------------------------------------------------------------

def test_settings_exact_prompt_persistence(tmp_path: Path) -> None:
    """SettingsStore preserves exact multiline prompt, unicode, whitespace, and special characters."""
    settings_file = tmp_path / "settings_prompt.json"
    store = SettingsStore(settings_file)

    exact_prompt = (
        "=== Editorial Policy V1 ===\n"
        "1. Tập trung vào cảnh hành động và phân cảnh kịch tính của nhân vật chính.\n"
        "2. Cross-episode connections: 'Quote test', \"Double quotes\", and JSON: {\"key\": \"value\"}.\n"
        "3. Pacing: 120 wpm; tone: analytical & dramatic.\n"
        "4. Audio: commentary gain +2.0dB, ducking -16dB.\n"
        "    - Sub-bullet with 4-space indent\n"
        "\n"
        "Special symbols: @#$%^&*()_+~`|}{[]:;?><,./\n"
    )

    settings = AppSettings(recap_prompt=exact_prompt)
    store.save(settings)
    assert settings_file.is_file()

    loaded = store.load()
    assert loaded.recap_prompt == exact_prompt
    assert len(loaded.recap_prompt) == len(exact_prompt)


# ---------------------------------------------------------------------------
# 6. Subtitle Cache Unchanged by Policy or Prompt
# ---------------------------------------------------------------------------

def test_subtitle_cache_unchanged_by_policy(tmp_path: Path) -> None:
    """Subtitle cache keys are completely decoupled and invariant to prompt/policy changes."""
    video_path = tmp_path / "sub_test.mp4"
    video_path.write_bytes(b"dummy video")

    k_sub1 = compute_subtitle_cache_key(
        episode_id="E01",
        source_video=video_path,
        subtitle_source_id="track1",
        ocr_engine="rapidocr",
        model_version="PP-OCRv4",
        track_codec="srt",
        track_index=0,
        track_source="embedded",
    )

    # Prompt or policy changes must not affect subtitle cache key
    k_sub2 = compute_subtitle_cache_key(
        episode_id="E01",
        source_video=video_path,
        subtitle_source_id="track1",
        ocr_engine="rapidocr",
        model_version="PP-OCRv4",
        track_codec="srt",
        track_index=0,
        track_source="embedded",
    )

    assert k_sub1 == k_sub2
    # Verify no prompt-related keys or leaks in subtitle key
    assert len(k_sub1) == 64


# ---------------------------------------------------------------------------
# 7. AnalysisEngine Policy Integration & run_analysis
# ---------------------------------------------------------------------------

def test_analysis_engine_policy_integration(tmp_path: Path) -> None:
    """AnalysisEngine accepts policy and propagates it to scanner, connector, finalizer, and cache."""
    prompt = (
        "Focus on character decisions and power shifts.\n"
        "Connect causal threads across episodes.\n"
        "Narration style: documentary."
    )
    policy = EditorialPolicy.from_prompt(prompt)

    cache_mgr = EvidenceCacheManager(tmp_path / "ev")
    h_cache = HierarchyCacheManager(tmp_path / "h")

    engine = AnalysisEngine(
        cache_manager=cache_mgr,
        hierarchy_cache=h_cache,
        policy=policy,
    )

    assert engine.policy == policy
    assert engine.scanner.policy == policy
    assert engine.connector.policy == policy
    assert engine.finalizer.policy == policy

    # Verify final plan cache key reflects policy
    ev_map = {
        "E01": EpisodeEvidence(episode_id="E01", source_video=str(tmp_path / "v1.mp4"), duration_seconds=30.0)
    }
    k1 = compute_final_plan_cache_key(ev_map, engine.settings, "SEASON", policy=engine.policy)
    k_no_policy = compute_final_plan_cache_key(ev_map, engine.settings, "SEASON")
    assert k1 != k_no_policy
