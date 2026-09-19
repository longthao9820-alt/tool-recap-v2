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
    detect_official_voicestudio_runtime,
    get_isolated_runtime_python,
    get_voice_spec,
    is_executable_adapter_available,
    is_voicestudio_ready,
)
from .bootstrap import VoiceRuntimeBootstrap, BootstrapCancelled, BootstrapError


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
        self.subsystem_dir = subsystem_dir or (default_data_directory() / "voice_subsystem")
        self._custom_subsystem = subsystem_dir is not None
        self.detect_official = detect_official if detect_official is not None else (subsystem_dir is None)
        self.auto_bootstrap = auto_bootstrap if auto_bootstrap is not None else (subsystem_dir is None)

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

        if self.detect_official:
            official_py = detect_official_voicestudio_runtime()
            if official_py:
                vs_root = official_py.parent.parent.parent
                if (vs_root / "omnivoice_data" / "models" / "k2-fsa" / "OmniVoice").is_dir():
                    return True
                if (vs_root / "omnivoice_data" / "models").is_dir() and any((vs_root / "omnivoice_data" / "models").glob("*OmniVoice*")):
                    return True

        return False

    def is_voice_installed(self, voice_id: str) -> bool:
        """Check if runtime is ready and required model files exist locally."""
        spec = get_voice_spec(voice_id)
        if spec.engine == "voicestudio":
            return self._is_omnivoice_model_cached()
        v_dir = self.get_voice_dir(spec)
        return self._spec_is_complete_in_dir(spec, v_dir)

    def get_backend_python_executable(self) -> Path | None:
        """Locate backend Python runtime: official VoiceStudio first (if allowed), then V2 isolated runtime."""
        # 1. Prefer detected official VoiceStudio runtime only when detect_official is True
        if self.detect_official:
            official = detect_official_voicestudio_runtime()
            if official is not None and official.is_file():
                return official

        # 2. Check isolated V2 runtime
        isolated = get_isolated_runtime_python(self.subsystem_dir)
        if isolated is not None and isolated.is_file():
            return isolated

        return None

    def ensure_backend_runtime(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        """Ensure a backend runtime is ready, lazily bootstrapping isolated runtime if missing."""
        backend_python = self.get_backend_python_executable()
        if backend_python:
            if progress_callback:
                progress_callback(100, 100, 100.0)
            return backend_python

        bootstrap = VoiceRuntimeBootstrap(
            target_dir=self.subsystem_dir / "runtime",
            staging_dir=default_data_directory() / "staging" / "voice_runtime",
        )
        try:
            return bootstrap.bootstrap(
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        except BootstrapCancelled as exc:
            raise VoiceError("Quá trình cài đặt Voice runtime đã bị hủy.") from exc
        except BootstrapError as exc:
            raise VoiceError(f"Cài đặt Voice runtime thất bại: {exc}") from exc

    def get_adapter_executable(self) -> Path | None:
        """Locate installed VoiceStudio adapter executable in subsystem or runtime directory."""
        candidates = [
            self.subsystem_dir / "VoiceStudio.exe",
            self.subsystem_dir / "voicestudio.exe",
            self.subsystem_dir / "adapter.exe",
            self.subsystem_dir / "voicestudio_adapter.exe",
            self.subsystem_dir / "adapter.py",
            self.subsystem_dir / "run.cmd",
            self.subsystem_dir / "bin" / "VoiceStudio.exe",
            self.subsystem_dir / "bin" / "voicestudio.exe",
            application_root() / "runtime" / "voices" / "VoiceStudio.exe",
            application_root() / "runtime" / "voices" / "adapter.exe",
            application_root() / "runtime" / "voices" / "adapter.py",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
        return None

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
        if spec.engine == "voicestudio":
            # All 12 designed voices share single OmniVoice model cache
            target_dir = self.cache_dir / "omnivoice"
            target_dir.mkdir(parents=True, exist_ok=True)
            if self._is_omnivoice_model_cached():
                if progress_callback:
                    progress_callback(100, 100, 100.0)
                return target_dir
            if progress_callback:
                progress_callback(0, 100, 0.0)
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

        if spec.engine == "voicestudio":
            # Test harness exemption only when explicitly requested
            if allow_mock_synth:
                self._generate_test_audio(text, output_path)
                validate_wav_audio(output_path)
                return output_path

            # 1. Locate backend Python or executable adapter
            backend_python = self.get_backend_python_executable()
            adapter_exe = self.get_adapter_executable()

            if not backend_python and not adapter_exe:
                if self.auto_bootstrap:
                    # Lazy bootstrap isolated runtime on first use
                    try:
                        backend_python = self.ensure_backend_runtime(
                            progress_callback=progress_callback,
                            cancel_event=cancel_event,
                        )
                    except Exception as exc:
                        if isinstance(exc, VoiceError):
                            raise
                        raise VoiceError(
                            f"VoiceStudio adapter chưa được cài đặt hoặc không khả dụng: {exc}. "
                            f"Không thể tổng hợp giọng đọc cho '{voice_id}'. "
                            f"Vui lòng cài đặt adapter VoiceStudio tương thích qua voice_updater."
                        ) from exc
                else:
                    raise VoiceError(
                        f"VoiceStudio adapter chưa được cài đặt hoặc không khả dụng. "
                        f"Không thể tổng hợp giọng đọc cho '{voice_id}'. "
                        f"Vui lòng cài đặt adapter VoiceStudio tương thích qua voice_updater."
                    )

            # Ensure model cache exists (never swallow ensure model errors)
            self.ensure_voice_model(
                voice_id,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )

            # 2. If backend Python is available, invoke omnivoice_adapter.py via JSON request
            if backend_python:
                adapter_script = Path(__file__).parent / "omnivoice_adapter.py"
                req_file = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
                        json.dump(
                            {
                                "voice_id": voice_id,
                                "text": text,
                                "style": style,
                                "output": str(output_path),
                                "instruct": spec.instruct,
                                "cache_dir": str(self.cache_dir / "omnivoice"),
                            },
                            tf,
                            ensure_ascii=False,
                        )
                        req_file = Path(tf.name)

                    cmd = [str(backend_python), str(adapter_script), "--request", str(req_file)]
                    env = {
                        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
                        "PATH": os.environ.get("PATH", ""),
                        "TEMP": os.environ.get("TEMP", tempfile.gettempdir()),
                        "TMP": os.environ.get("TMP", tempfile.gettempdir()),
                        "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
                        "APPDATA": os.environ.get("APPDATA", ""),
                        "USERPROFILE": os.environ.get("USERPROFILE", ""),
                        "HOMEPATH": os.environ.get("HOMEPATH", ""),
                        "HOMEDRIVE": os.environ.get("HOMEDRIVE", ""),
                        "HF_HOME": str(self.cache_dir / "omnivoice"),
                        "HUGGINGFACE_HUB_CACHE": str(self.cache_dir / "omnivoice" / "hub"),
                    }
                    pythonpath_parts: list[str] = []
                    vs_proj = backend_python.parent.parent.parent
                    if (vs_proj / "omnivoice").is_dir():
                        pythonpath_parts.append(str(vs_proj))
                    if pythonpath_parts:
                        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        shell=False,
                        env=env,
                    )

                    start_time = time.time()
                    timeout = 180.0
                    stderr_lines: list[str] = []

                    while proc.poll() is None:
                        if cancel_event and cancel_event.is_set():
                            _kill_process_tree(proc.pid)
                            try:
                                proc.wait(timeout=5)
                            except Exception:
                                pass
                            raise VoiceError("Quá trình đọc giọng đã bị hủy.")

                        if time.time() - start_time > timeout:
                            _kill_process_tree(proc.pid)
                            try:
                                proc.wait(timeout=5)
                            except Exception:
                                pass
                            raise VoiceError("Quá thời gian tổng hợp giọng đọc từ adapter VoiceStudio.")

                        line = proc.stderr.readline() if proc.stderr else ""
                        if line:
                            stderr_lines.append(line)
                            try:
                                data = json.loads(line.strip())
                                if isinstance(data, dict) and "progress" in data and progress_callback:
                                    prog_val = int(data["progress"])
                                    progress_callback(prog_val, 100, float(prog_val))
                            except Exception:
                                pass
                        else:
                            time.sleep(0.05)

                    if proc.stderr:
                        for rem_line in proc.stderr.readlines():
                            stderr_lines.append(rem_line)

                    if proc.returncode != 0:
                        clean_lines: list[str] = []
                        for line in stderr_lines:
                            line_s = line.strip()
                            if not line_s:
                                continue
                            if line_s.startswith("{") and line_s.endswith("}"):
                                try:
                                    d = json.loads(line_s)
                                    if "stage" in d or "progress" in d:
                                        continue
                                except Exception:
                                    pass
                            clean_line = "".join(ch for ch in line_s if ch.isprintable() or ch in "\t\n\r")
                            clean_lines.append(clean_line)
                        err_msg = "\n".join(clean_lines).strip()
                        if len(err_msg) > 1000:
                            err_msg = err_msg[:1000] + "... [truncated]"
                        raise VoiceError(
                            f"VoiceStudio adapter tổng hợp thất bại (mã {proc.returncode}): {err_msg}"
                        )

                finally:
                    if req_file and req_file.exists():
                        try:
                            req_file.unlink()
                        except OSError:
                            pass

                validate_wav_audio(output_path)
                return output_path

            # 3. Fallback for non-python executable adapter
            elif adapter_exe:
                cmd = [
                    str(adapter_exe),
                    "--voice", voice_id,
                    "--text", text,
                    "--style", style,
                    "--output", str(output_path),
                ]
                if adapter_exe.suffix.lower() == ".py":
                    cmd = [sys.executable, str(adapter_exe), "--voice", voice_id, "--text", text, "--style", style, "--output", str(output_path)]

                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    if res.returncode != 0:
                        raise VoiceError(
                            f"VoiceStudio adapter tổng hợp thất bại (mã {res.returncode}): {res.stderr.strip()}"
                        )
                except subprocess.TimeoutExpired as exc:
                    raise VoiceError("Quá thời gian tổng hợp giọng đọc từ adapter VoiceStudio.") from exc
                except OSError as exc:
                    raise VoiceError(f"Không thể khởi chạy adapter VoiceStudio: {exc}") from exc

                validate_wav_audio(output_path)
                return output_path

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
