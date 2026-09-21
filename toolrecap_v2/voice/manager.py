"""Voice model management, lazy downloading, caching, and TTS synthesis without fake fallback."""
from __future__ import annotations

import array
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Callable

from ..paths import application_root, default_data_directory
from .catalog import (
    DEFAULT_VOICE_ID,
    SUPPORTED_VOICE_STYLES,
    VoiceSpec,
    get_voice_spec,
)
from .bootstrap import VoiceRuntimeBootstrap, BootstrapCancelled, BootstrapError
from .runtime import (
    CORE_DEPENDENCIES,
    VOICE_ADAPTER_VERSION,
    VOICE_ENGINE_VERSION,
    VOICE_MODEL_REVISION,
    VoiceHealthResult,
    VoiceRuntimeInspector,
    runtime_fingerprint,
)


ProgressCallback = Callable[[int, int, float], None]


class VoiceError(RuntimeError):
    pass


class DownloadCancelled(VoiceError):
    pass


def _kill_process_tree(pid: int) -> None:
    """Kill process and all its children safely."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=10)
        else:
            import signal
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def validate_wav_audio(audio_path: Path | str) -> None:
    """Validate that the audio file exists, is valid PCM16 WAV, and is not silent."""
    path = Path(audio_path).resolve()
    if not path.is_file():
        raise VoiceError(f"Không tìm thấy tệp âm thanh: {path}")

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()

            if wav_file.getcomptype() != "NONE" or sample_width != 2:
                raise VoiceError("Định dạng âm thanh phải là PCM16 không nén.")
            if channels <= 0 or frame_rate <= 0 or frame_count <= 0:
                raise VoiceError("Thông số âm thanh WAV không hợp lệ.")

            # Check RMS level
            raw = wav_file.readframes(frame_count)
            samples = array.array("h")
            samples.frombytes(raw)
            if sys.byteorder != "little":
                samples.byteswap()

            if not samples:
                raise VoiceError("Tệp âm thanh rỗng.")

            sum_sq = sum(s * s for s in samples)
            rms = (sum_sq / len(samples)) ** 0.5 / 32768.0
            if rms < 0.0005:
                raise VoiceError("Âm thanh quá nhỏ hoặc bị im lặng hoàn toàn.")
    except (wave.Error, OSError, ValueError) as exc:
        raise VoiceError(f"Tệp WAV không hợp lệ: {exc}") from exc


def download_file_with_progress(
    url: str,
    destination: Path,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    chunk_size: int = 65536,
) -> Path:
    """Download a file with progress updates and cancellation checking."""
    dest = Path(destination).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.parent / f"{dest.name}.part"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ToolRecapV2/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response, open(temp_dest, "wb") as out:
            total_size = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                if cancel_event and cancel_event.is_set():
                    raise DownloadCancelled("Tải xuống đã bị hủy.")
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    percent = min(100.0, (downloaded / total_size) * 100.0)
                    progress_callback(downloaded, total_size, percent)

        if cancel_event and cancel_event.is_set():
            raise DownloadCancelled("Tải xuống đã bị hủy.")

        os.replace(temp_dest, dest)
        return dest
    except Exception:
        if temp_dest.exists():
            try:
                temp_dest.unlink()
            except OSError:
                pass
        raise


class VoiceModelManager:
    """Manages voice models: local cache lookup, lazy downloading, and real TTS synthesis."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        subsystem_dir: Path | None = None,
        *,
        detect_official: bool | None = None,
        auto_bootstrap: bool | None = None,
    ) -> None:
        self.cache_dir = cache_dir or (default_data_directory() / "models" / "voices")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.subsystem_dir = subsystem_dir or (default_data_directory() / "voice_runtime")
        self._custom_subsystem = subsystem_dir is not None
        # Parameter retained for API compatibility; external runtimes are never used.
        self.detect_official = False
        self.auto_bootstrap = auto_bootstrap if auto_bootstrap is not None else (subsystem_dir is None)
        runtime_dir = (self.subsystem_dir / "runtime") if self._custom_subsystem else (self.subsystem_dir / "current")
        self.runtime = VoiceRuntimeInspector(
            runtime_dir=runtime_dir,
            model_cache_dir=self.cache_dir / "omnivoice",
        )
        self._health_cache: dict[str, VoiceHealthResult] = {}
        self._health_lock = threading.RLock()

    def get_voice_dir(self, spec: VoiceSpec) -> Path:
        """Return the directory where this voice model is located."""
        # 1. Check runtime bundled directory first
        bundled_cand = application_root() / "runtime" / "voices" / spec.voice_id
        if self._spec_is_complete_in_dir(spec, bundled_cand):
            return bundled_cand

        # 2. Check user cache directory
        return self.cache_dir / spec.voice_id

    def _spec_is_complete_in_dir(self, spec: VoiceSpec, directory: Path) -> bool:
        if not directory.is_dir():
            return False
        for req_file in spec.required_files:
            if not (directory / req_file).is_file():
                return False
        return True

    def _is_omnivoice_model_cached(self) -> bool:
        """Check if shared OmniVoice model weights exist in local HF/model cache."""
        ov_cache = self.cache_dir / "omnivoice"
        if (ov_cache / "hub" / "models--k2-fsa--OmniVoice").is_dir():
            return True
        if (ov_cache / "models--k2-fsa--OmniVoice").is_dir():
            return True
        if ov_cache.is_dir() and any(p.name != "hub" and p.is_file() and p.stat().st_size > 0 for p in ov_cache.iterdir()):
            return True

        hf_home_env = os.environ.get("HF_HOME")
        if hf_home_env:
            hf_path = Path(hf_home_env)
            if (hf_path / "hub" / "models--k2-fsa--OmniVoice").is_dir() or (hf_path / "models--k2-fsa--OmniVoice").is_dir():
                return True

        default_hf_hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--k2-fsa--OmniVoice"
        if default_hf_hub.is_dir():
            return True

        return False

    def is_voice_installed(self, voice_id: str) -> bool:
        """Check if runtime is ready and required model files exist locally."""
        spec = get_voice_spec(voice_id)
        if spec.engine == "omnivoice":
            cached = self.get_cached_health(voice_id, spec.style)
            return bool(cached and cached.ready)
        v_dir = self.get_voice_dir(spec)
        return self._spec_is_complete_in_dir(spec, v_dir)

    def get_backend_python_executable(self) -> Path | None:
        """Return only ToolRecap's managed runtime Python executable."""
        py_exe = self.runtime.python_executable
        return py_exe if py_exe.is_file() else None

    def ensure_backend_runtime(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        """Ensure a backend runtime is ready, lazily bootstrapping isolated runtime if missing."""
        health = self.runtime.inspect(run_imports=True)
        if health.dependencies_ok:
            if progress_callback:
                progress_callback(100, 100, 100.0)
            return self.runtime.python_executable

        bootstrap = VoiceRuntimeBootstrap(
            target_dir=self.runtime.runtime_dir,
            staging_dir=default_data_directory() / "staging" / "voice_runtime",
        )
        try:
            return bootstrap.bootstrap(
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                force=True,
                validation_callback=lambda py: self._validate_staged_runtime(py, cancel_event=cancel_event),
            )
        except BootstrapCancelled as exc:
            raise VoiceError("Quá trình cài đặt Voice runtime đã bị hủy.") from exc
        except BootstrapError as exc:
            raise VoiceError(f"Cài đặt Voice runtime thất bại: {exc}") from exc

    def get_adapter_executable(self) -> Path | None:
        """External adapter executables are intentionally not part of production."""
        return None

    def _health_key(self, voice_id: str, style: str) -> str:
        spec = get_voice_spec(voice_id)
        raw = json.dumps(
            {
                "voice_id": voice_id,
                "style": style,
                "instruct": spec.instruct,
                "runtime": runtime_fingerprint(),
                "engine": VOICE_ENGINE_VERSION,
                "adapter": VOICE_ADAPTER_VERSION,
                "model_revision": VOICE_MODEL_REVISION,
            },
            sort_keys=True,
        )
        import hashlib
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_cached_health(self, voice_id: str, style: str) -> VoiceHealthResult | None:
        with self._health_lock:
            return self._health_cache.get(self._health_key(voice_id, style))

    def _cache_health(self, voice_id: str, style: str, health: VoiceHealthResult) -> None:
        with self._health_lock:
            self._health_cache[self._health_key(voice_id, style)] = health

    def _validate_staged_runtime(
        self,
        python_executable: Path,
        *,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Validate imports/versions and a real synthesis before runtime promotion."""
        probe = (
            "import torch,torchaudio,transformers,accelerate,soundfile,omnivoice,json;"
            "from importlib.metadata import version;"
            f"expected={CORE_DEPENDENCIES!r};"
            "actual={n:version(n) for n in expected};"
            "bad={n:(expected[n],actual[n]) for n in expected if actual[n].split('+',1)[0]!=expected[n].split('+',1)[0]};"
            "assert not bad,bad;print(json.dumps(actual))"
        )
        completed = subprocess.run(
            [str(python_executable), "-c", probe],
            capture_output=True,
            text=True,
            timeout=90,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=self.runtime.subprocess_environment(),
        )
        if completed.returncode != 0:
            raise BootstrapError(
                f"Voice runtime import validation failed: {completed.stderr.strip()}",
                code="IMPORT_ERROR",
            )

        smoke_path = default_data_directory() / "staging" / "voice_runtime_smoke.wav"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._invoke_omnivoice(
                python_executable=python_executable,
                voice_id=DEFAULT_VOICE_ID,
                text="ToolRecap voice is ready.",
                output_path=smoke_path,
                style="documentary",
                progress_callback=None,
                cancel_event=cancel_event,
                timeout=300.0,
            )
            validate_wav_audio(smoke_path)
        except Exception as exc:
            raise BootstrapError(
                f"Voice runtime synthesis validation failed: {exc}",
                code="SYNTHESIS_ERROR",
            ) from exc
        finally:
            smoke_path.unlink(missing_ok=True)

    def health_check(
        self,
        voice_id: str,
        style: str,
        *,
        synthesis_test: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> VoiceHealthResult:
        health = self.runtime.inspect(run_imports=True)
        if not health.dependencies_ok or not health.model_valid:
            return health
        if synthesis_test:
            smoke_path = default_data_directory() / "cache" / "voice_health" / f"{self._health_key(voice_id, style)}.wav"
            smoke_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._invoke_omnivoice(
                    python_executable=self.runtime.python_executable,
                    voice_id=voice_id,
                    text="ToolRecap voice is ready.",
                    output_path=smoke_path,
                    style=style,
                    progress_callback=None,
                    cancel_event=cancel_event,
                    timeout=300.0,
                )
                validate_wav_audio(smoke_path)
                health.synthesis_test_ok = True
                health.audio_validation_ok = True
            except Exception as exc:
                health.state = "REPAIR_REQUIRED"
                health.failure_stage = "synthesis"
                health.error_code = "SYNTHESIS_ERROR"
                health.human_message = f"Voice synthesis health test failed: {exc}"
                return health
        else:
            health.synthesis_test_ok = True
            health.audio_validation_ok = True
        health.ready = True
        health.state = "READY"
        health.human_message = "ToolRecap local voice is ready."
        self._cache_health(voice_id, style, health)
        return health

    def ensure_ready(
        self,
        voice_id: str,
        style: str,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        smoke_test: bool = True,
        force_repair: bool = False,
    ) -> VoiceHealthResult:
        if cancel_event and cancel_event.is_set():
            raise VoiceError("Voice preparation was cancelled.")
        cached = None if force_repair else self.get_cached_health(voice_id, style)
        if cached and cached.ready:
            return cached

        health = self.runtime.inspect(run_imports=True)
        if force_repair or not health.dependencies_ok:
            if not self.auto_bootstrap:
                raise VoiceError(
                    f"[{health.error_code or 'RUNTIME_NOT_INSTALLED'}] "
                    "ToolRecap local voice runtime is unavailable or unhealthy."
                )
            if progress_callback:
                progress_callback(0, 100, 0.0)
            self.ensure_backend_runtime(progress_callback=progress_callback, cancel_event=cancel_event)

        # Model download occurs through the exact production adapter/from_pretrained
        # path, using the ToolRecap-owned cache and pinned revision.
        health = self.runtime.inspect(run_imports=True)
        if smoke_test:
            smoke_path = default_data_directory() / "cache" / "voice_health" / f"{self._health_key(voice_id, style)}.wav"
            smoke_path.parent.mkdir(parents=True, exist_ok=True)
            self._invoke_omnivoice(
                python_executable=self.runtime.python_executable,
                voice_id=voice_id,
                text="ToolRecap voice is ready.",
                output_path=smoke_path,
                style=style,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                timeout=300.0,
            )
            validate_wav_audio(smoke_path)
            health = self.runtime.inspect(run_imports=True)
            health.synthesis_test_ok = True
            health.audio_validation_ok = True
        elif health.dependencies_ok and health.model_valid:
            health.synthesis_test_ok = True
            health.audio_validation_ok = True
        else:
            health.state = "DOWNLOADING_MODEL" if health.dependencies_ok else health.state
        if not health.ready:
            if health.dependencies_ok and health.model_valid and health.synthesis_test_ok:
                health.ready = True
                health.state = "READY"
                health.human_message = "ToolRecap local voice is ready."
            else:
                raise VoiceError(f"[{health.error_code or 'VOICE_RUNTIME_ERROR'}] {health.human_message}")
        self._cache_health(voice_id, style, health)
        if progress_callback:
            progress_callback(100, 100, 100.0)
        return health

    def preview(
        self,
        voice_id: str,
        style: str,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        """Generate a real preview through the same production synthesis path."""
        self.ensure_ready(
            voice_id,
            style,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            smoke_test=True,
        )
        spec = get_voice_spec(voice_id)
        preview_dir = default_data_directory() / "cache" / "voice_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{self._health_key(voice_id, style)}.wav"
        # Always regenerate when the user explicitly requests Preview. This proves
        # current production synthesis rather than replaying historical audio.
        preview_path.unlink(missing_ok=True)
        self._synthesize_production(
            voice_id,
            spec.preview_text,
            preview_path,
            style=style,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        health = self.health_check(voice_id, style, synthesis_test=False, cancel_event=cancel_event)
        health.synthesis_test_ok = True
        health.audio_validation_ok = True
        health.ready = True
        health.state = "READY"
        self._cache_health(voice_id, style, health)
        return preview_path

    def ensure_voice_model(
        self,
        voice_id: str,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        """Ensure model is downloaded and ready. Downloads lazily if missing with visible progress."""
        if cancel_event and cancel_event.is_set():
            raise DownloadCancelled("Quá trình tải model đã bị hủy.")

        spec = get_voice_spec(voice_id)
        if spec.engine == "omnivoice":
            # All 12 designed voices share single OmniVoice model cache
            target_dir = self.cache_dir / "omnivoice"
            target_dir.mkdir(parents=True, exist_ok=True)
            if self.runtime.model_is_present():
                if progress_callback:
                    progress_callback(100, 100, 100.0)
                return target_dir
            if progress_callback:
                progress_callback(0, 100, 0.0)
            # The pinned adapter downloads the pinned HuggingFace revision into this
            # cache during smoke-test/production synthesis.
            return target_dir

        v_dir = self.get_voice_dir(spec)
        if self._spec_is_complete_in_dir(spec, v_dir):
            if progress_callback:
                progress_callback(100, 100, 100.0)
            return v_dir

        target_dir = self.cache_dir / spec.voice_id
        target_dir.mkdir(parents=True, exist_ok=True)

        if not spec.base_url:
            raise VoiceError(f"Không có địa chỉ tải trực tiếp (base_url) cho model {voice_id}.")

        for filename in spec.files:
            file_path = target_dir / filename
            if file_path.is_file() and file_path.stat().st_size > 0:
                continue
            url = spec.base_url.rstrip("/") + "/" + filename
            download_file_with_progress(
                url,
                file_path,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )

        if not self._spec_is_complete_in_dir(spec, target_dir):
            raise VoiceError(f"Tải model giọng {voice_id} không thành công hoặc thiếu file.")

        return target_dir

    def _invoke_omnivoice(
        self,
        *,
        python_executable: Path,
        voice_id: str,
        text: str,
        output_path: Path,
        style: str,
        progress_callback: ProgressCallback | None,
        cancel_event: threading.Event | None,
        timeout: float = 300.0,
    ) -> Path:
        spec = get_voice_spec(voice_id)
        adapter_script = Path(__file__).parent / "omnivoice_adapter.py"
        request_file: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as handle:
                json.dump(
                    {
                        "voice_id": voice_id,
                        "text": text,
                        "style": style,
                        "output": str(output_path),
                        "instruct": spec.instruct,
                        "model": f"k2-fsa/OmniVoice@{VOICE_MODEL_REVISION}",
                        "cache_dir": str(self.cache_dir / "omnivoice"),
                    },
                    handle,
                    ensure_ascii=False,
                )
                request_file = Path(handle.name)

            command = [str(python_executable), str(adapter_script), "--request", str(request_file)]
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                env=self.runtime.subprocess_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            started = time.monotonic()
            stderr_lines: list[str] = []
            while proc.poll() is None:
                if cancel_event and cancel_event.is_set():
                    _kill_process_tree(proc.pid)
                    raise VoiceError("Voice synthesis was cancelled.")
                if time.monotonic() - started > timeout:
                    _kill_process_tree(proc.pid)
                    raise VoiceError("Voice synthesis timed out.")
                line = proc.stderr.readline() if proc.stderr else ""
                if line:
                    stderr_lines.append(line)
                    try:
                        progress = json.loads(line.strip())
                        if isinstance(progress, dict) and "progress" in progress and progress_callback:
                            value = int(progress["progress"])
                            progress_callback(value, 100, float(value))
                    except Exception:
                        pass
                else:
                    if cancel_event:
                        cancel_event.wait(0.05)
                    else:
                        time.sleep(0.05)
            if proc.stderr:
                stderr_lines.extend(proc.stderr.readlines())
            if proc.returncode != 0:
                clean = []
                for line in stderr_lines:
                    stripped = "".join(ch for ch in line.strip() if ch.isprintable() or ch in "\t\n\r")
                    if not stripped:
                        continue
                    if stripped.startswith("{") and stripped.endswith("}"):
                        try:
                            if "progress" in json.loads(stripped):
                                continue
                        except Exception:
                            pass
                    clean.append(stripped)
                detail = "\n".join(clean)
                if len(detail) > 2000:
                    detail = detail[:2000] + "... [truncated]"
                raise VoiceError(f"Local OmniVoice synthesis failed (code {proc.returncode}): {detail}")
        finally:
            if request_file is not None:
                request_file.unlink(missing_ok=True)
        validate_wav_audio(output_path)
        self.runtime.write_model_metadata()
        return output_path

    def _synthesize_production(
        self,
        voice_id: str,
        text: str,
        output_path: Path,
        *,
        style: str,
        progress_callback: ProgressCallback | None,
        cancel_event: threading.Event | None,
    ) -> Path:
        runtime_python = self.get_backend_python_executable()
        if runtime_python is None:
            raise VoiceError("[RUNTIME_NOT_INSTALLED] ToolRecap local voice runtime is unavailable.")
        self.ensure_voice_model(voice_id, progress_callback=progress_callback, cancel_event=cancel_event)
        return self._invoke_omnivoice(
            python_executable=runtime_python,
            voice_id=voice_id,
            text=text,
            output_path=output_path,
            style=style,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def synthesize(
        self,
        voice_id: str,
        text: str,
        output_path: Path,
        *,
        style: str = "film_recap",
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        allow_mock_synth: bool = False,
    ) -> Path:
        """Synthesize text to a PCM16 WAV file using the chosen voice.

        Strictly enforces real speech synthesis: no silent swallow, no synthetic tone fallback in production.
        """
        if cancel_event and cancel_event.is_set():
            raise VoiceError("Quá trình đọc giọng đã bị hủy.")

        # Invariant 4: Supported voice styles must be validated; unsupported rejected.
        if style not in SUPPORTED_VOICE_STYLES:
            raise VoiceError(
                f"Phong cách giọng '{style}' không được hỗ trợ. "
                f"Các phong cách hợp lệ: {', '.join(SUPPORTED_VOICE_STYLES)}"
            )

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spec = get_voice_spec(voice_id)

        if spec.engine == "omnivoice":
            # Test harness exemption only when explicitly requested
            if allow_mock_synth:
                self._generate_test_audio(text, output_path)
                validate_wav_audio(output_path)
                return output_path
            self.ensure_ready(
                voice_id,
                style,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                smoke_test=True,
            )
            result = self._synthesize_production(
                voice_id,
                text,
                output_path,
                style=style,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            health = self.health_check(voice_id, style, synthesis_test=False, cancel_event=cancel_event)
            health.synthesis_test_ok = True
            health.audio_validation_ok = True
            health.ready = True
            health.state = "READY"
            self._cache_health(voice_id, style, health)
            return result

        elif spec.engine == "piper":
            # Internal compatibility fallback for legacy Piper models
            model_dir = self.ensure_voice_model(
                voice_id,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            onnx_file = next(model_dir.glob("*.onnx"), None)
            json_file = next(model_dir.glob("*.onnx.json"), None)

            if not onnx_file or not json_file:
                raise VoiceError(f"Thiếu tệp cấu hình ONNX cho giọng đọc {voice_id} tại {model_dir}")

            try:
                import piper
                from piper import PiperVoice

                voice = PiperVoice.load(str(onnx_file), config_path=str(json_file))
                with wave.open(str(output_path), "wb") as wav_out:
                    voice.synthesize_wav(text, wav_out)
                validate_wav_audio(output_path)
                return output_path
            except Exception as exc:
                if allow_mock_synth:
                    self._generate_test_audio(text, output_path)
                    validate_wav_audio(output_path)
                    return output_path
                raise VoiceError(f"Tổng hợp giọng đọc {voice_id} bằng Piper thất bại: {exc}") from exc

        elif allow_mock_synth:
            self._generate_test_audio(text, output_path)
            validate_wav_audio(output_path)
            return output_path
        else:
            raise VoiceError(f"Engine giọng đọc '{spec.engine}' chưa được triển khai hoặc không tương thích.")

    def _generate_test_audio(self, text: str, output_path: Path) -> None:
        """Generate test audio ONLY when explicitly requested in test fixtures."""
        import math
        sample_rate = 22050
        words = len(text.split())
        duration = max(1.0, words * 0.42)
        total_frames = int(sample_rate * duration)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(sample_rate)

            samples = array.array("h")
            for i in range(total_frames):
                t = i / sample_rate
                freq = 180.0 + 30.0 * math.sin(2.0 * math.pi * 3.0 * t)
                val = int(8000.0 * math.sin(2.0 * math.pi * freq * t) * (0.8 + 0.2 * math.cos(2.0 * math.pi * 0.5 * t)))
                samples.append(val)

            wav_out.writeframes(samples.tobytes())


_singleton_manager: VoiceModelManager | None = None


def get_voice_manager() -> VoiceModelManager:
    global _singleton_manager
    if _singleton_manager is None:
        _singleton_manager = VoiceModelManager()
    return _singleton_manager
