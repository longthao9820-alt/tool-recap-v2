# ToolRecap V2

ToolRecap V2 automates the proven workflow:

```text
SELECT VIDEO / SELECT FOLDER
  -> Source Preparation
  -> Scanner
  -> Finalizer + raw Recap Prompt
  -> Final JSON
  -> Technical Validation / bounded AI repair
  -> Voice + deterministic render
  -> Finished videos
```

The configured AI Finalizer is the editor. ToolRecap prepares sources, transports
Scanner observations, validates the Final JSON technically, and executes that plan.
It does not rank candidates, choose stories, impose output quotas, or rewrite the
Finalizer's editorial decisions.

## Project selection

- **Select File** creates one `SINGLE_EPISODE` project.
- **Select Folder** discovers supported videos directly in that folder, naturally
  sorts them, and creates one `SEASON` project containing every discovered source.
- One Start action runs the complete workflow. Manual JSON import is not part of
  the normal production path.

Supported source extensions are `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`,
and `.ts`.

## AI Gateway settings

The Gateway is OpenAI-compatible and provider-neutral. ToolRecap sends the exact
configured role names; no code path depends on a particular provider or model name.

- Enable AI Gateway analysis
- API endpoint and API key
- Scanner model and thinking level
- Scanner Vision support
- Finalizer model and thinking level
- Scanner parallelism
- Scanner chunk length
- Test Scanner / Test Finalizer

The Scanner extracts grounded, timestamped source observations. It does not decide
which stories become outputs. The Finalizer receives the complete project source
mapping, Scanner observations, renderer contract, and the user's raw Recap Prompt.
It returns one canonical Final JSON with zero, one, or any number of outputs.

## Final JSON boundary

Every output contains an ID, title, file/output metadata, language, and ordered
segments. Segments contain their type, purpose, narration/dialogue text, audio and
subtitle policies, visual-speed recommendation, and source clips. Clips identify an
episode/source plus exact millisecond timestamps.

Technical validation checks:

- required root, output, segment, and clip fields;
- unique output and segment IDs;
- supported audio policies;
- exact episode/source resolution;
- `0 <= start_ms < end_ms <= source duration`;
- source existence for prepared projects;
- narration requirements and renderer executability.

Validation never scores story quality. If validation fails, the exact structured
errors are sent back to the configured Finalizer for a bounded repair. ToolRecap
does not clamp timestamps or silently substitute footage.

`{"outputs": []}` is an explicit, valid editorial result. Missing `outputs`, malformed
JSON, transport errors, source errors, and timestamp errors remain technical errors;
they are never converted into “no candidates.”

## Cache and resume

Cache dependencies are layered:

- source extraction depends on source identity and extraction settings;
- Scanner evidence depends on source identity plus Scanner model/thinking/Vision/
  chunk settings;
- Final JSON depends on Scanner evidence, Finalizer model/thinking, raw Recap Prompt,
  and recap metadata;
- render output depends on Final JSON plus voice/audio/encoder/render settings.

A Finalizer retry reuses Scanner evidence. Voice or render changes reuse Final JSON.
Prompt or Finalizer changes regenerate Final JSON while retaining compatible source
and Scanner artifacts. A failed multi-output render validates and skips previously
completed outputs when its render dependency signature is unchanged.

## Renderer and publication output

The existing V2 renderer remains responsible for local voice generation, source
cutting, original-audio preservation/mixing, narration and original-dialogue SRT,
GPU encoding, cancellation, and final output validation. It executes every valid
Finalizer output without candidate filtering.

Each publication folder contains exactly:

```text
<title>.mp4
<title>.narration.srt
<title>.original.srt
```

## Updates

Releases contain a Windows portable ZIP and SHA256 companion file. The updater:

1. fetches release metadata;
2. downloads and verifies SHA256;
3. safely extracts and validates staging;
4. creates and verifies a backup before touching the installation;
5. applies and verifies the new executable;
6. rolls back from the verified backup on failure;
7. restarts ToolRecap.

Settings, caches, downloaded voice models, and project state live outside the app
directory under `%LOCALAPPDATA%\ToolRecapV2` and survive routine updates.

## Development

```powershell
python -m pip install -r requirements.txt
pytest -q
python build_exe.py
```

The build creates a portable directory, ZIP, SHA256 file, and performs executable
`--version` and `--self-check` checks before reporting success.

See `ARCHITECTURE_REFACTOR_REPORT.md` for the implementation audit, code paths,
test evidence, and known limitations.
