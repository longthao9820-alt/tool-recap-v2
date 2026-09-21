"""ToolRecap-owned voice runtime manifest, inspection, and health diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

from ..paths import default_data_directory


VOICE_RUNTIME_SCHEMA = "2"
VOICE_RUNTIME_VERSION = "2.0.0"
DEPENDENCY_MANIFEST_VERSION = "2026.09.1"
VOICE_ENGINE = "omnivoice"
VOICE_ENGINE_VERSION = "0.2.1"
VOICE_MODEL_ID = "k2-fsa/OmniVoice"
VOICE_MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
VOICE_ADAPTER_VERSION = "2"

# This exact core set was resolver-checked and the public OmniVoice 0.2.1 package
# was synthesis-tested against it. OmniVoice is installed with --no-deps because
# its published dependency list includes UI/training packages ToolRecap does not use.
CORE_DEPENDENCIES: dict[str, str] = {
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "transformers": "5.15.1",
    "accelerate": "1.14.0",
    "soundfile": "0.14.0",
    "numpy": "2.2.6",
    "pydub": "0.25.1",
    "webdataset": "1.0.2",
    "sentencepiece": "0.2.2",
    "protobuf": "7.36.0",
    "safetensors": "0.8.0",
    "huggingface-hub": "1.28.0",
    "tokenizers": "0.22.2",
    "omnivoice": VOICE_ENGINE_VERSION,
}


def _canonical_version(value: str) -> str:
    """Treat local wheel labels such as 2.8.0+cpu as the pinned 2.8.0 build."""
    return str(value).split("+", 1)[0]


def runtime_fingerprint() -> str:
    payload = {
        "schema": VOICE_RUNTIME_SCHEMA,
        "runtime": VOICE_RUNTIME_VERSION,
        "manifest": DEPENDENCY_MANIFEST_VERSION,
        "engine": VOICE_ENGINE,
        "engine_version": VOICE_ENGINE_VERSION,
        "adapter": VOICE_ADAPTER_VERSION,
        "model": VOICE_MODEL_ID,
        "model_revision": VOICE_MODEL_REVISION,
        "dependencies": CORE_DEPENDENCIES,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class VoiceHealthResult:
    runtime_present: bool = False
    runtime_version_ok: bool = False
    dependencies_ok: bool = False
    dependency_versions: dict[str, str] = field(default_factory=dict)
    imports_ok: bool = False
    engine_initialized: bool = False
    model_present: bool = False
    model_valid: bool = False
    synthesis_test_ok: bool = False
    audio_validation_ok: bool = False
    ready: bool = False
    state: str = "NOT_INSTALLED"
    failure_stage: str = ""
    error_code: str = ""
    human_message: str = "Local voice runtime is not installed."
    runtime_path: str = ""
    runtime_fingerprint: str = ""
    python_version: str = ""
    engine: str = VOICE_ENGINE
    engine_version: str = ""
    selected_device: str = ""
    cuda_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VoiceRuntimeInspector:
    """Inspect only ToolRecap's managed runtime; never searches external applications."""

    def __init__(self, runtime_dir: Path | None = None, model_cache_dir: Path | None = None) -> None:
        self.runtime_dir = runtime_dir or (default_data_directory() / "voice_runtime" / "current")
        self.model_cache_dir = model_cache_dir or (default_data_directory() / "models" / "voices" / "omnivoice")

    @property
    def python_executable(self) -> Path:
        return self.runtime_dir / "python.exe"

    @property
    def metadata_path(self) -> Path:
        return self.runtime_dir / "voice_runtime.json"

    def model_snapshot_path(self) -> Path:
        return (
            self.model_cache_dir
            / "hub"
            / "models--k2-fsa--OmniVoice"
            / "snapshots"
            / VOICE_MODEL_REVISION
        )

    @property
    def model_metadata_path(self) -> Path:
        return self.model_cache_dir / "toolrecap_model.json"

    def write_model_metadata(self) -> None:
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        temp = self.model_metadata_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {
                    "engine": VOICE_ENGINE,
                    "engine_version": VOICE_ENGINE_VERSION,
                    "model_id": VOICE_MODEL_ID,
                    "model_revision": VOICE_MODEL_REVISION,
                    "runtime_fingerprint": runtime_fingerprint(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temp, self.model_metadata_path)

    def model_is_present(self) -> bool:
        snapshot = self.model_snapshot_path()
        if not snapshot.is_dir():
            return False
        if any(self.model_cache_dir.rglob("*.part")) or any(self.model_cache_dir.rglob("*.incomplete")):
            return False
        required = ("config.json", "tokenizer_config.json", "tokenizer.json")
        if not all((snapshot / name).is_file() for name in required):
            return False
        weights = list(snapshot.rglob("*.safetensors"))
        if not any(p.is_file() and p.stat().st_size > 100 * 1024 * 1024 for p in weights):
            return False
        try:
            metadata = json.loads(self.model_metadata_path.read_text(encoding="utf-8"))
            return (
                metadata.get("model_id") == VOICE_MODEL_ID
                and metadata.get("model_revision") == VOICE_MODEL_REVISION
                and metadata.get("runtime_fingerprint") == runtime_fingerprint()
            )
        except Exception:
            return False

    def _base_result(self) -> VoiceHealthResult:
        return VoiceHealthResult(
            runtime_path=str(self.runtime_dir),
            runtime_fingerprint=runtime_fingerprint(),
            model_present=self.model_is_present(),
            model_valid=self.model_is_present(),
        )

    def inspect(self, *, run_imports: bool = True, timeout: float = 60.0) -> VoiceHealthResult:
        result = self._base_result()
        py_exe = self.python_executable
        if not py_exe.is_file():
            return result
        result.runtime_present = True
        result.state = "REPAIR_REQUIRED"

        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            result.failure_stage = "metadata"
            result.error_code = "RUNTIME_METADATA_MISSING"
            result.human_message = f"Voice runtime metadata is missing or invalid: {exc}"
            return result

        result.runtime_version_ok = (
            str(metadata.get("voice_runtime_schema")) == VOICE_RUNTIME_SCHEMA
            and str(metadata.get("voice_runtime_version")) == VOICE_RUNTIME_VERSION
            and str(metadata.get("dependency_manifest_version")) == DEPENDENCY_MANIFEST_VERSION
            and str(metadata.get("runtime_fingerprint")) == runtime_fingerprint()
        )
        if not result.runtime_version_ok:
            result.failure_stage = "metadata"
            result.error_code = "RUNTIME_VERSION_MISMATCH"
            result.human_message = "Voice runtime manifest does not match this ToolRecap release."
            return result

        if not run_imports:
            result.state = "VERIFYING"
            result.human_message = "Voice runtime metadata is valid; dependency verification is pending."
            return result

        probe = (
            "import json,platform,torch,torchaudio,transformers,accelerate,soundfile,omnivoice;"
            "from importlib.metadata import version;"
            f"names={list(CORE_DEPENDENCIES)!r};"
            "print(json.dumps({'python':platform.python_version(),'versions':{n:version(n) for n in names},"
            "'cuda':bool(torch.cuda.is_available()),'device':('cuda' if torch.cuda.is_available() else 'cpu')}))"
        )
        try:
            completed = subprocess.run(
                [str(py_exe), "-c", probe],
                capture_output=True,
                text=True,
                shell=False,
                timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=self.subprocess_environment(),
            )
        except subprocess.TimeoutExpired:
            result.failure_stage = "imports"
            result.error_code = "IMPORT_TIMEOUT"
            result.human_message = "Voice dependency verification timed out."
            return result
        except OSError as exc:
            result.failure_stage = "runtime"
            result.error_code = "RUNTIME_EXEC_ERROR"
            result.human_message = f"Voice runtime could not start: {exc}"
            return result

        if completed.returncode != 0:
            result.failure_stage = "imports"
            result.error_code = "IMPORT_ERROR"
            result.human_message = f"Voice dependency import failed: {completed.stderr.strip()[:600]}"
            return result
        try:
            info = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception as exc:
            result.failure_stage = "imports"
            result.error_code = "IMPORT_REPORT_INVALID"
            result.human_message = f"Voice dependency report is invalid: {exc}"
            return result

        result.python_version = str(info.get("python", ""))
        result.dependency_versions = {str(k): str(v) for k, v in dict(info.get("versions", {})).items()}
        result.engine_version = result.dependency_versions.get("omnivoice", "")
        result.cuda_available = bool(info.get("cuda", False))
        result.selected_device = str(info.get("device", "cpu"))
        mismatches = {
            name: {"expected": expected, "actual": result.dependency_versions.get(name, "missing")}
            for name, expected in CORE_DEPENDENCIES.items()
            if _canonical_version(result.dependency_versions.get(name, "")) != _canonical_version(expected)
        }
        if mismatches:
            result.failure_stage = "dependencies"
            result.error_code = "DEPENDENCY_VERSION_MISMATCH"
            result.human_message = f"Voice dependency versions do not match the manifest: {mismatches}"
            return result

        result.dependencies_ok = True
        result.imports_ok = True
        result.engine_initialized = True
        if not result.model_present:
            result.state = "DOWNLOADING_MODEL"
            result.failure_stage = "model"
            result.error_code = "MODEL_NOT_DOWNLOADED"
            result.human_message = "The local OmniVoice model is not downloaded yet."
            return result

        result.state = "VERIFYING"
        result.human_message = "Runtime, dependencies, and model are valid; synthesis verification is pending."
        return result

    def subprocess_environment(self) -> dict[str, str]:
        env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
            "PATH": os.environ.get("PATH", ""),
            "TEMP": os.environ.get("TEMP", ""),
            "TMP": os.environ.get("TMP", ""),
            "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
            "APPDATA": os.environ.get("APPDATA", ""),
            "USERPROFILE": os.environ.get("USERPROFILE", ""),
            "HF_HOME": str(self.model_cache_dir),
            "HUGGINGFACE_HUB_CACHE": str(self.model_cache_dir / "hub"),
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        # Prevent PyInstaller/external-environment state from leaking into the managed runtime.
        for key in list(os.environ):
            if key.startswith("_PYI_"):
                continue
            if key in {"CUDA_VISIBLE_DEVICES", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}:
                env[key] = os.environ[key]
        return env

    def diagnostics_text(self, health: VoiceHealthResult) -> str:
        lines = [
            "Voice Runtime Diagnostics",
            "-------------------------",
            f"Runtime version: {VOICE_RUNTIME_VERSION}",
            f"Runtime path: {health.runtime_path}",
            f"Python version: {health.python_version or 'unknown'}",
            f"Engine: {VOICE_ENGINE}",
            f"Engine version: {health.engine_version or 'unknown'}",
        ]
        for name in ("torch", "torchaudio", "transformers", "accelerate", "soundfile"):
            lines.append(f"{name}: {health.dependency_versions.get(name, 'unknown')}")
        lines.extend(
            [
                f"Model: {VOICE_MODEL_ID}@{VOICE_MODEL_REVISION}",
                f"CUDA available: {health.cuda_available}",
                f"Selected device: {health.selected_device or 'unknown'}",
                f"Health result: {'READY' if health.ready else health.state}",
                f"Failure stage: {health.failure_stage or 'none'}",
                f"Error code: {health.error_code or 'none'}",
            ]
        )
        return "\n".join(lines)


__all__ = [
    "CORE_DEPENDENCIES",
    "DEPENDENCY_MANIFEST_VERSION",
    "VOICE_ADAPTER_VERSION",
    "VOICE_ENGINE",
    "VOICE_ENGINE_VERSION",
    "VOICE_MODEL_ID",
    "VOICE_MODEL_REVISION",
    "VOICE_RUNTIME_SCHEMA",
    "VOICE_RUNTIME_VERSION",
    "VoiceHealthResult",
    "VoiceRuntimeInspector",
    "runtime_fingerprint",
]
