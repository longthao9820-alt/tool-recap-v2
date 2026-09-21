# ToolRecap V2 v0.6.0 — Voice Runtime Reliability Report

## Result

ToolRecap now owns one authoritative local public voice system. Preview, preflight,
health smoke tests, and publication rendering all call the same `VoiceModelManager`,
managed Python runtime, OmniVoice engine, pinned model revision, adapter, synthesis
method, and WAV validator. External VoiceStudio installations and
`VOICESTUDIO_PYTHON` do not participate in the production path.

## Files changed

- `toolrecap_v2/voice/runtime.py` — authoritative manifest, fingerprint, structured
  health result, dependency/model inspection, integrity checks, and diagnostics.
- `toolrecap_v2/voice/bootstrap.py` — compatible pinned dependencies,
  cancellation-aware hidden installers, structured install failures, staged
  import/synthesis validation, and atomic promotion/rollback.
- `toolrecap_v2/voice/manager.py` — one readiness/preview/render path, real preview
  synthesis, session health proof, repair, model metadata, and diagnostics.
- `toolrecap_v2/voice/omnivoice_adapter.py` — public OmniVoice 0.2.1 API support,
  pinned model revision, NumPy/Tensor output support, and validated PCM16 output.
- `toolrecap_v2/voice/catalog.py`, `toolrecap_v2/voice/__init__.py` — ToolRecap local
  preset terminology and no external-runtime participation.
- `toolrecap_v2/ui.py` — truthful status, background preview/repair, cancellable
  preview, and ToolRecap Local Voice Runtime UI.
- `toolrecap_v2/projects.py` — voice preflight before media/AI work, preserved
  analysis cache, voice runtime/model render dependencies, and safer project-state
  atomic writes.
- `toolrecap_v2/renderer.py` — runtime/model fingerprint in output render signatures.
- `auto_main.py`, `build_exe.py`, `README.md`, `pyproject.toml`,
  `toolrecap_v2/version.py` — self-check, documentation, packaging wording, and v0.6.0.
- `tests/conftest.py`, `tests/test_voice_runtime_reliability.py`,
  `tests/test_voice_system.py`, `tests/test_voice_updater.py`,
  `tests/test_ui_workflow.py`, `tests/test_renderer_queue.py` — regression and
  reliability acceptance coverage.
- `scripts/test_voice_render_e2e.py` — real two-episode/two-output voice-render E2E.

## Old versus new architecture

Old:

```text
Preview may replay preview_<voice>.wav
Render independently probes external VoiceStudio Python/adapter
→ otherwise creates an isolated runtime
→ installs incompatible packages
→ fails on Output 1
```

New:

```text
VoiceRuntimeManifest
→ ToolRecap-managed staged Python runtime
→ exact dependency verification
→ pinned model revision/integrity metadata
→ real synthesis smoke test
→ atomic promotion

VoiceManager.ensure_ready()
├── Project preflight
├── Preview (always real synthesis)
└── PublicationRenderer synthesis
```

## Dependency manifest

- Voice runtime schema: `2`
- Voice runtime version: `2.0.0`
- Dependency manifest version: `2026.09.1`
- Python: `3.11.9` embeddable amd64
- torch: `2.8.0+cpu`
- torchaudio: `2.8.0+cpu`
- transformers: `5.15.1`
- accelerate: `1.14.0`
- soundfile: `0.14.0`
- numpy: `2.2.6`
- pydub: `0.25.1`
- webdataset: `1.0.2`
- sentencepiece: `0.2.2`
- protobuf: `7.36.0`
- safetensors: `0.8.0`
- huggingface-hub: `1.28.0`
- tokenizers: `0.22.2`
- public OmniVoice: `0.2.1`, installed with `--no-deps` after its exact inference
  dependencies are installed from the manifest
- model: `k2-fsa/OmniVoice`
- revision: `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`

The installed runtime fingerprint is:
`023cb1b21d45733441845f0e7d43034d3fe4c596998f72de6b23b6143b2542a0`.

## ResolutionImpossible root cause

The old bootstrap requested both `omnivoice==0.2.1` and
`transformers==4.44.2`. OmniVoice 0.2.1 declares `transformers>=5.3.0`, so no
dependency solution can satisfy both constraints. Additionally, the adapter assumed
the external VoiceStudio project's OmniVoice API returned `torch.Tensor`; public
OmniVoice 0.2.1 returns `numpy.ndarray`. Both the dependency set and adapter API had
to be fixed; changing one version alone would not have produced reliable synthesis.

## Preview/render consistency

`VoiceModelManager.preview()` first proves runtime health, then deliberately
regenerates speech through `_synthesize_production()`. `synthesize()` uses that same
method. A cached WAV is never used to establish READY. READY exists only after a
real synthesis and `validate_wav_audio()` pass for the selected voice/style/runtime/
model fingerprint.

## External VoiceStudio removal

The production manager only resolves:

`%LOCALAPPDATA%\ToolRecapV2\voice_runtime\current\python.exe`

It never calls external VoiceStudio discovery, adapter executables, or
`VOICESTUDIO_PYTHON`. Legacy functions remain import-compatible for migration tests,
but no canonical UI, preview, preflight, or render path uses them. An automated test
sets `VOICESTUDIO_PYTHON` to an unrelated executable and verifies it is ignored.

## Health, model integrity, installation, and repair

Structured `VoiceHealthResult` reports runtime/version/dependencies/imports/model/
synthesis/audio/device state and error code. Model health requires the pinned
snapshot, config/tokenizer files, substantial weights, no `.part`/`.incomplete`
files, and matching ToolRecap model metadata.

Installation occurs entirely in staging. Package imports, exact versions, model
load, real synthesis, and WAV validation run before atomic promotion. The previous
runtime is renamed to backup and restored if promotion fails. Cancellation kills
installer/synthesis process trees and staging is cleaned.

## Cache and resume behavior

Voice ID/style/runtime fingerprint/model revision affect only project/output render
signatures. Analysis signatures are unchanged. A voice failure therefore preserves
subtitle/OCR/STT artifacts, Scanner evidence, Finalizer result, and Final JSON.
Existing queue/resume tests verify cached Final JSON skips `AnalysisEngine.analyze()`
and rendering resumes from the first failed/incomplete output.

## CPU/GPU result

The verified managed runtime selected CPU:

- CUDA available: `False`
- selected voice device: `cpu`
- video NVENC remains independently available and is not treated as voice CUDA.

## Automated tests

- Full suite: **557 collected; 537 passed; 20 retired legacy-pipeline tests skipped;
  0 failures; 0 errors**.
- Voice/runtime/UI focused suite: **37 passed; 0 failures**.
- `python -m compileall`: passed.
- `git diff --check`: passed, with only expected Windows LF/CRLF notices.

Coverage includes fresh/missing runtime behavior, cached preview with broken runtime,
structured dependency conflict, interrupted staging, preview/render consistency,
voice preflight before media/AI, render-only invalidation, external VoiceStudio
absence/presence isolation, second launch, cancellation, runtime/model integrity,
resume, and production prohibition of fake audio.

## Real verification

1. A clean ToolRecap-managed runtime was installed under LocalAppData.
2. Exact package imports/versions passed.
3. The pinned model loaded and actual speech synthesis passed WAV validation.
4. Preview and immediate narration rendering both succeeded through the same path.
5. `scripts/test_voice_render_e2e.py` created two real synthetic episode videos,
   rendered two outputs with actual OmniVoice narration, mixed audio, produced both
   SRT files, encoded MP4 files, and passed strict publication validation:
   `VOICE_RENDER_E2E_PASSED`.

## Portable build

- Version: `0.6.0`
- Executable `--version`: passed.
- Executable `--self-check`: passed (`ffmpeg=True, voice=True, icon=True, gui=True`).
- Voice backend reported: `toolrecap-managed`.
- Robust concat executable self-check: passed.
- ZIP: `release/ToolRecapV2-v0.6.0-windows-portable.zip`
- SHA256: `d230ba2bcc3b4746f427506386f71aa0a99576c6fe011aab054b082e4d080b13`

## Known limitation

The current standalone adapter loads the OmniVoice model per synthesis subprocess.
This is reliable and isolates failures, but CPU narration generation has noticeable
startup cost per segment. A future optimization may introduce a persistent managed
worker while preserving the same runtime/model/health contract.
