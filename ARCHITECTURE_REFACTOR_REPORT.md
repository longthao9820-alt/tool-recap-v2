# ToolRecap V2 v0.5.0 — Architecture Refactor Engineering Report

## 1. Version and implementation commit

- Version: `0.5.0`
- Implementation commit: `3dd3f1991846715f45318d85683c930b71271b65`
- Branch: `main`

## 2. Files and modules changed

- `toolrecap_v2/analyzer/final_json.py` — new canonical Finalizer transport,
  Final JSON contract, deterministic validator, structured errors, and bounded repair.
- `toolrecap_v2/analyzer/engine.py` — canonical production routing and Final JSON cache.
- `toolrecap_v2/analyzer/evidence.py` — transcript- and Vision-aware Scanner cache identity.
- `toolrecap_v2/analyzer/phases.py`, `toolrecap_v2/ui.py`, `toolrecap_v2/projects.py`
  — real workflow phases, whole-project persistence, dependency signatures, and resume.
- `toolrecap_v2/api_client.py` — process-wide, cancellation-aware Gateway concurrency bound.
- `toolrecap_v2/domain/models.py` — renderer contract metadata for outputs and segments.
- `toolrecap_v2/renderer.py` — verified output-level render resume.
- `toolrecap_v2/updater.py` — verified backup-before-delete, apply verification, rollback verification.
- `toolrecap_v2/settings.py`, `toolrecap_v2/version.py`, `pyproject.toml` — repair setting and v0.5.0.
- `tests/test_canonical_final_json.py`, `tests/conftest.py` — canonical acceptance tests and explicit
  retirement of prohibited legacy-pipeline integration contracts.
- `README.md` — user and architecture documentation rewritten for the canonical path.

## 3. Architecture before the refactor

The enabled-Gateway production path was:

```text
Source Preparation
  -> Scanner / Coverage Ledger / second pass
  -> Season Connection / batching / merge
  -> Compact Episode Summaries
  -> Candidate Discovery
  -> Candidate Consolidation
  -> Candidate / Zero-output Verification
  -> Candidate Finalizer
  -> AnalysisManifest
  -> Renderer
```

Application code therefore made or constrained editorial decisions before the configured Finalizer.
The old parser also accepted missing `outputs` as an empty list in some paths, allowing technical
failures to resemble editorial zero-output decisions.

## 4. Architecture after the refactor

The enabled-Gateway production path is now:

```text
Project discovery (one file or one folder project)
  -> media/subtitle/OCR/STT source preparation
  -> Scanner chunks (grounded observations only)
  -> configured Finalizer + raw Recap Prompt + complete project mapping
  -> one canonical Final JSON
  -> deterministic technical validation
  -> configured Finalizer repair with exact structured errors, if required
  -> valid Final JSON
  -> local voice / source cuts / audio / subtitles / GPU render
  -> exact publication output
```

## 5. Editorial stages removed from the canonical path

- Season Connection
- semantic season batching and season merge
- compact episode summarization as an editorial input layer
- Candidate Discovery
- Candidate Consolidation and ranking/filtering
- Candidate Verification
- Zero-output Verification
- candidate-driven application-side Finalizer grouping

These legacy modules remain import-compatible and retain direct unit coverage for cache migration and
historical data, but `AnalysisEngine.analyze()` does not call them when the Gateway is enabled.

## 6. Bypassed and repurposed stages

- Coverage data remains Scanner/source diagnostic metadata; it no longer gates or selects outputs.
- The hierarchy cache remains readable for compatibility but is not a canonical Final JSON dependency.
- The old `CandidateFinalizer` remains for Gateway-disabled compatibility only.
- Existing media, subtitle, OCR/STT, voice, rendering, output validation, project persistence, retry,
  cancellation, updater, and release modules remain technical infrastructure.

## 7. Exact Scanner responsibility

Scanner receives episode/source identity, technical range, timestamped transcript, optional supported
visual input, metadata, and source-observation instructions. It returns grounded facts, actions,
dialogue, reactions, chronology, timestamps, uncertainty, and other observable evidence. The raw Recap
Prompt is not converted into a Scanner editorial filter. Scanner does not choose outputs, rank stories,
or reject subplots.

Scanner cache identity includes source file identity, normalized transcript hash, Scanner model,
thinking, Vision capability, chunk length, and Scanner prompt/directive version.

## 8. Exact Finalizer responsibility and prompt ownership

`build_finalizer_project_prompt()` places the user's stored prompt verbatim between explicit
`BEGIN/END RAW RECAP PROMPT` markers. The same request includes project identity, scope, ordered source
mapping and durations, all Scanner observations, recap metadata, and the renderer-facing schema.

The configured Finalizer alone chooses output count, stories, characters, subplots, thesis, hook,
titles, narration/dialogue, footage, timestamps, order, and separation. ToolRecap sends the exact
configured `finalizer_model` and `finalizer_thinking` values.

## 9. Generic Gateway roles

Scanner and Finalizer remain settings-driven Gateway roles. Canonical tests use `abc-worker`,
`xyz-editor`, `worker-model-x`, and `editor-model-y` and verify exact transport. No canonical branch
compares a model value to Prime, Sub, Codex, Gemini, Antigravity, or a provider name. Existing saved
settings and UI fields remain compatible.

## 10. Whole-folder projects

Folder selection uses `scan_videos()` to discover supported direct child files and natural-sort them.
`ProjectRecord.from_season_paths()` creates one `SEASON` project with stable `E01..En` source mapping.
The queue probes and prepares every source, runs one project-level Finalizer, validates its one Final
JSON, and renders every returned output. No episode-by-episode manual JSON workflow is involved.

## 11. Final JSON contract

The root must contain `outputs` as an array. An output requires unique `output_id`, `title`, safe unique
`.mp4` `file_name`, `output_type`, `language`, and a non-empty ordered `segments` array. A segment
requires unique `segment_id`, `segment_type`, supported `audio_policy`, publication-compatible
`subtitle_policy`, renderer-compatible visual speed, narration/dialogue fields as appropriate, and
non-empty `source_clips`. A clip requires project-resolvable `episode_id`/`source_file` and exact
`start_ms`/`end_ms`.

Validated objects are converted once into the existing renderer model. `file_name` determines the
safe publication basename; the renderer does not re-edit the plan.

## 12. Technical validation and automatic repair

The validator checks root/schema shape, required fields, unique IDs/names, safe filenames, supported
policies, narration requirements, exact episode/source resolution, on-disk source existence for real
prepared projects, and `0 <= start_ms < end_ms <= duration_ms`. Unsupported renderer requests are
rejected rather than ignored.

Each error includes `path`, stable `code`, message, and exact details such as requested range, resolved
source, and source duration. The invalid complete JSON and error array are sent back to the same
configured Finalizer for up to two repairs by default (bounded 0–3 in settings). No timestamp clamping,
source substitution, output dropping, or narration rewriting occurs in Python.

## 13. Zero-output versus technical errors

- Explicit schema-valid `{"outputs": []}` becomes `VALID_EMPTY_OUTPUT` and does not render.
- Missing `outputs` is `SCHEMA_ERROR`.
- Invalid sources and timestamps remain structured validation errors.
- Malformed/truncated JSON, missing choices/content, empty HTTP success, and transport failures remain
  bounded `APIError` failures in the Gateway client.
- Legacy empty caches without `VALID_EMPTY_OUTPUT` are rejected and cannot masquerade as success.

## 14. Cache dependency and resume rules

- Source/video or selected sidecar identity change: invalidate project analysis and dependent layers.
- Transcript/OCR/STT content change: normalized transcript hash invalidates Scanner evidence.
- Scanner model/thinking/Vision/chunk change: invalidate Scanner and Final JSON.
- Finalizer model/thinking, raw prompt, or recap metadata change: reuse Scanner evidence and regenerate
  Final JSON.
- Voice, encoder, render quality, subtitles, or audio-mix change: preserve Final JSON and rerender only.

Project state stores separate analysis and render signatures. A compatible persisted Final JSON skips
Scanner and Finalizer. On a render retry, previously completed outputs are reused only after the exact
three-file publication folder passes validation; rendering resumes at the first invalid/incomplete
output.

## 15. Cancellation and Gateway reliability

Existing stop propagation remains active through source preparation, OCR/STT, Scanner, cancellation-
aware retry/backoff, Finalizer, repair, voice generation, and FFmpeg. Cancelled work is not marked
complete, while atomic caches and completed output state remain reusable.

The API client retains bounded timeouts, request-size ceilings, safe key redaction, retry for 429/500/
502/503/504 and network errors, and retry for empty/malformed/truncated/missing-content HTTP 200
responses. A process-wide four-slot semaphore now bounds all clients and projects; Scanner parallelism
operates within that global limit.

## 16. Renderer integration

Every technically valid Finalizer output flows directly to `PublicationRenderer`. The preserved V2
renderer resolves source clips, generates local voice, measures narration, rejects insufficient footage
instead of truncating speech, cuts/concatenates source, preserves/mixes original audio, remaps original
dialogue subtitles, creates narration subtitles, GPU-encodes, and validates the publication folder.

Each output folder contains exactly `<title>.mp4`, `<title>.narration.srt`, and
`<title>.original.srt`; work files stay under the external temporary work directory.

## 17. Updater changes

Download/staging still requires the portable ZIP, companion SHA256, safe extraction, and staged
executable. The generated apply script now checks the backup copy command and verifies the backed-up
executable before deleting any installed file. It verifies the new executable after apply, performs a
clean rollback from the verified backup on apply failure, verifies rollback, and preserves the backup
for manual recovery if rollback itself fails. User settings/models/cache remain outside the application
directory.

## 18. Tests executed and exact results

- Full suite: **472 collected; 452 passed; 20 skipped; 0 failed; 0 errors**.
- Canonical architecture suite: **12 passed; 0 skipped; 0 failed**.
- Canonical + updater focused run: **20 passed; 0 failed**.
- `python -m compileall -q toolrecap_v2`: passed.
- `git diff --check`: passed (only Git CRLF conversion notices on Windows).

The 20 skips are enumerated in `tests/conftest.py`. They are old end-to-end contracts that require the
prohibited Season Connection/Candidate Discovery/Consolidation/Verification production path or feed
that path's pre-Final-JSON fixtures into ProjectQueue. The underlying legacy modules still have passing
unit coverage; the new canonical suite replaces their invalid production-topology assertions.

## 19. Portable build result

- Command: `python build_exe.py`
- Result: passed.
- Executable version check: `ToolRecap V2 v0.5.0`.
- Executable self-check: passed (`ffmpeg=True, voice=True, icon=True, gui=True`).
- Bundled encoder detection: NVENC, AMF, and QSV available in bundled FFmpeg; local detected plan used
  NVIDIA NVENC on an RTX 3060.
- Portable ZIP: `release/ToolRecapV2-v0.5.0-windows-portable.zip`
- Size: `245,945,830` bytes.
- SHA256: `bad9f9bdb08f621dd7def8696228a7996bc0170fbd08ae8f7f746a276d90c345`
- Checksum file: `release/ToolRecapV2-v0.5.0-windows-portable.zip.sha256.txt`

## 20. Update test result

Eight automated updater tests and the new backup-order regression passed. They cover release parsing,
asset/checksum selection, SHA256 enforcement, safe extraction, traversal rejection, staging, and apply
script safety. The freshly built ZIP/checksum pair is internally consistent.

## 21. Known limitations

- No GitHub Release was published because publishing credentials/release authorization were not part of
  this request. The portable artifacts are ready for publication.
- A destructive live cross-machine apply/rollback test was not run on another installation. Automated
  updater tests and script-order verification passed, but a release-candidate machine should still run
  the X -> X+1 acceptance scenario before public rollout.
- The current renderer supports deterministic `1.0x` visual timing and the exact two-SRT publication
  policy. Final JSON requesting another speed or subtitle policy is rejected and repaired rather than
  silently ignored.
- The process-wide Gateway concurrency limit is four and is not currently exposed as a separate UI
  setting; per-project Scanner parallelism remains configurable from one to four.
- Legacy editorial modules remain in the repository for compatibility/migration but are bypassed by the
  enabled-Gateway production path. They can be removed in a later cleanup after old cached projects no
  longer require them.
