"""Voice model management, lazy downloading, caching, and TTS synthesis without fake fallback."""
from __future__ import annotations

import array
import os
import shutil
import sys
import threading
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Callable

from ..paths import application_root, default_data_directory
from .catalog import DEFAULT_VOICE_ID, VoiceSpec, get_voice_spec


ProgressCallback = Callable[[int, int, float], None]


class VoiceError(RuntimeError):
    pass


class DownloadCancelled(VoiceError):
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
    """Manages voice models: local cache lookup, lazy downloading, and real Piper TTS synthesis."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (default_data_directory() / "models" / "voices")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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

    def is_voice_installed(self, voice_id: str) -> bool:
        """Check if all required model files for voice_id exist locally."""
        spec = get_voice_spec(voice_id)
        v_dir = self.get_voice_dir(spec)
        return self._spec_is_complete_in_dir(spec, v_dir)

    def ensure_voice_model(
        self,
        voice_id: str,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        """Ensure model is downloaded and ready. Downloads lazily if missing with visible progress."""
        spec = get_voice_spec(voice_id)
        v_dir = self.get_voice_dir(spec)

        if self._spec_is_complete_in_dir(spec, v_dir):
            if progress_callback:
                progress_callback(100, 100, 100.0)
            return v_dir

        target_dir = self.cache_dir / spec.voice_id
        target_dir.mkdir(parents=True, exist_ok=True)

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
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        allow_mock_synth: bool = False,
    ) -> Path:
        """Synthesize text to a PCM16 WAV file using the chosen voice.

        Strictly enforces real speech synthesis: no silent swallow, no synthetic tone fallback in production.
        """
        if cancel_event and cancel_event.is_set():
            raise VoiceError("Quá trình đọc giọng đã bị hủy.")

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spec = get_voice_spec(voice_id)

        if spec.engine == "piper":
            # Ensure model files are present (with progress report)
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
                    # Only permissible in explicit test harnesses
                    self._generate_test_audio(text, output_path)
                    validate_wav_audio(output_path)
                    return output_path
                # Production MUST fail clearly without fake fallback
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
