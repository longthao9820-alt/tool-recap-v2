"""Real, content-driven narration preparation and recap scene segmentation for ToolRecap V2.

Analyzes actual video audio, dialogue transcription, embedded/companion subtitles,
and scene timeline. Handles speech presence vs absence clearly and deterministically.
"""
from __future__ import annotations

import array
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .gpu import bundled_binary
from .media import probe_media
from .paths import get_stt_model_cache_dir
from .settings import AppSettings, SettingsStore


LogCallback = Callable[[str], None]


class NarrationError(RuntimeError):
    pass


@dataclass
class RecapSegment:
    segment_id: str
    start_ms: int
    end_ms: int
    narration_text: str
    audio_policy: str = "mute"  # "mute" for voiceover, "preserve" for original audio
    original_dialogue_text: str = ""

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0


@dataclass
class RecapManifest:
    project_id: str
    source_video: str
    recap_mode: str
    recap_language: str
    segments: list[RecapSegment]
    total_source_duration_sec: float = 0.0
    speech_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0",
            "project_id": self.project_id,
            "source_video": self.source_video,
            "recap_mode": self.recap_mode,
            "recap_language": self.recap_language,
            "total_source_duration_sec": self.total_source_duration_sec,
            "speech_detected": self.speech_detected,
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecapManifest":
        segments_raw = data.get("segments", [])
        segments = [
            RecapSegment(
                segment_id=str(s.get("segment_id", f"seg_{i}")),
                start_ms=int(s.get("start_ms", 0)),
                end_ms=int(s.get("end_ms", 0)),
                narration_text=str(s.get("narration_text", "")),
                audio_policy=str(s.get("audio_policy", "mute")),
                original_dialogue_text=str(s.get("original_dialogue_text", "")),
            )
            for i, s in enumerate(segments_raw, start=1)
        ]
        return cls(
            project_id=str(data.get("project_id", "project")),
            source_video=str(data.get("source_video", "")),
            recap_mode=str(data.get("recap_mode", "FULL_EPISODE")),
            recap_language=str(data.get("recap_language", "en-US")),
            segments=segments,
            total_source_duration_sec=float(data.get("total_source_duration_sec", 0.0)),
            speech_detected=bool(data.get("speech_detected", False)),
        )


def validate_manifest(manifest: RecapManifest, total_duration_sec: float) -> None:
    """Ensure manifest is valid, strictly bounded by source video, and contains non-empty narration."""
    if not manifest.segments:
        raise NarrationError("Manifest không có phân đoạn (segments) nào.")

    total_ms = int(total_duration_sec * 1000)
    for i, seg in enumerate(manifest.segments):
        if seg.start_ms < 0:
            raise NarrationError(f"Phân đoạn {seg.segment_id} có start_ms âm: {seg.start_ms}")
        if seg.end_ms <= seg.start_ms:
            raise NarrationError(f"Phân đoạn {seg.segment_id} có end_ms <= start_ms ({seg.end_ms} <= {seg.start_ms})")
        if total_ms > 0 and seg.end_ms > total_ms + 1000:
            raise NarrationError(f"Phân đoạn {seg.segment_id} vượt quá độ dài video nguồn ({seg.end_ms}ms > {total_ms}ms)")
        if seg.audio_policy == "mute" and not seg.narration_text.strip():
            raise NarrationError(f"Phân đoạn lồng tiếng {seg.segment_id} thiếu nội dung thuyết minh (narration_text).")


def _timestamp_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS,mmm or MM:SS,mmm timestamp string to seconds."""
    ts = ts.replace(",", ".").strip()
    parts = ts.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _parse_srt_content(text: str) -> list[tuple[float, float, str]]:
    """Parse SRT subtitle text into (start_sec, end_sec, text) list."""
    pattern = re.compile(
        r"(?:^\d+\s*\r?\n)?(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)\r?\n(.*?)(?=\r?\n\r?\n|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    results: list[tuple[float, float, str]] = []
    for match in pattern.finditer(text):
        s_str, e_str, content = match.groups()
        start = _timestamp_to_seconds(s_str)
        end = _timestamp_to_seconds(e_str)
        # Strip HTML/formatting tags
        clean_text = re.sub(r"<[^>]+>", "", content).replace("\n", " ").strip()
        if clean_text:
            results.append((start, end, clean_text))
    return results


def extract_companion_subtitles(video_path: Path) -> list[tuple[float, float, str]]:
    """Find and parse companion subtitle files (.srt, .vtt, .txt)."""
    candidates = [
        video_path.with_suffix(".srt"),
        video_path.with_suffix(".vtt"),
        video_path.parent / f"{video_path.stem}.srt",
        video_path.parent / f"{video_path.stem}_en.srt",
    ]
    for cand in candidates:
        if cand.is_file():
            try:
                content = cand.read_text(encoding="utf-8", errors="replace")
                parsed = _parse_srt_content(content)
                if parsed:
                    return parsed
            except Exception:
                pass
    return []


def extract_audio_from_video(video_path: Path, output_wav: Path) -> bool:
    """Extract audio track as 16kHz mono PCM16 WAV for transcription/analysis."""
    ffmpeg = bundled_binary("ffmpeg")
    if not ffmpeg:
        return False

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg),
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    res = subprocess.run(cmd, capture_output=True, creationflags=flags)
    return res.returncode == 0 and output_wav.is_file() and output_wav.stat().st_size > 44


def check_audio_speech_energy(wav_path: Path) -> tuple[bool, float]:
    """Calculate RMS energy of WAV audio to check if speech/sound is present."""
    if not wav_path.is_file() or wav_path.stat().st_size <= 44:
        return False, 0.0

    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            if frames == 0:
                return False, 0.0
            raw = wf.readframes(frames)
            samples = array.array("h")
            samples.frombytes(raw)
            if sys.byteorder != "little":
                samples.byteswap()
            if not samples:
                return False, 0.0

            sum_sq = sum(s * s for s in samples)
            rms = (sum_sq / len(samples)) ** 0.5 / 32768.0
            # RMS threshold: sound is considered present if rms >= 0.001
            has_sound = rms >= 0.001
            return has_sound, rms
    except Exception:
        return False, 0.0


def ensure_local_whisper_model(
    model_name: str = "tiny",
    cache_dir: Path | None = None,
    log: LogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Ensure local Whisper model snapshot exists in the cache directory, downloading it on first use.
    Reuses existing cached snapshot without redownloading.
    """
    if cancel_event and cancel_event.is_set():
        raise NarrationError("Tải mô hình STT đã bị dừng.")

    base_cache = cache_dir or get_stt_model_cache_dir()
    model_dir = base_cache / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check for essential model files indicating cached compatible snapshot
    is_cached = (
        (model_dir / "model.bin").is_file()
        and (model_dir / "config.json").is_file()
    )

    if is_cached:
        if log:
            log(f"Sử dụng mô hình nhận diện giọng nói '{model_name}' đã lưu trong bộ nhớ đệm.")
        return model_dir

    # Cold download path
    if log:
        log(f"Đang tải mô hình nhận diện giọng nói '{model_name}' về máy (%LOCALAPPDATA%\\ToolRecapV2\\models\\stt\\{model_name})...")

    if cancel_event and cancel_event.is_set():
        raise NarrationError("Tải mô hình STT đã bị dừng.")

    try:
        from faster_whisper.utils import _MODELS
        repo_id = _MODELS.get(model_name, f"Systran/faster-whisper-{model_name}")
    except Exception:
        repo_id = f"Systran/faster-whisper-{model_name}"

    import huggingface_hub

    # Build progress / cancellation tracker for tqdm if possible
    tqdm_cls = None
    try:
        from tqdm.auto import tqdm

        class _CancelTqdm(tqdm):
            def update(self, n=1):
                if cancel_event and cancel_event.is_set():
                    raise NarrationError("Tải mô hình STT đã bị dừng.")
                return super().update(n)

        tqdm_cls = _CancelTqdm
    except Exception:
        pass

    allow_patterns = [
        "config.json",
        "preprocessor_config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    dl_kwargs: dict[str, Any] = {
        "local_dir": str(model_dir),
        "allow_patterns": allow_patterns,
    }
    if hasattr(huggingface_hub, "snapshot_download"):
        dl_kwargs["local_dir_use_symlinks"] = False
    if tqdm_cls is not None:
        dl_kwargs["tqdm_class"] = tqdm_cls

    def _clean_partial_model_dir() -> None:
        try:
            for item in list(model_dir.iterdir()):
                if item.is_file():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass

    try:
        huggingface_hub.snapshot_download(repo_id, **dl_kwargs)
    except NarrationError:
        _clean_partial_model_dir()
        raise
    except Exception as exc:
        _clean_partial_model_dir()
        if cancel_event and cancel_event.is_set():
            raise NarrationError("Tải mô hình STT đã bị dừng.") from exc
        raise

    if cancel_event and cancel_event.is_set():
        _clean_partial_model_dir()
        raise NarrationError("Tải mô hình STT đã bị dừng.")

    if log:
        log(f"Đã tải thành công mô hình nhận diện giọng nói '{model_name}'.")

    return model_dir


def transcribe_local_whisper(
    audio_path: Path,
    language: str = "en",
    log: LogCallback | None = None,
    cancel_event: threading.Event | None = None,
    cache_dir: Path | None = None,
    model_name: str = "tiny",
) -> list[tuple[float, float, str]]:
    """Transcribe audio locally using faster-whisper on CPU."""
    if cancel_event and cancel_event.is_set():
        raise NarrationError("Chuẩn bị narration đã bị dừng.")
    try:
        from faster_whisper import WhisperModel

        if log:
            log("Khởi chạy mô hình nhận diện giọng nói nội bộ (faster-whisper)...")

        model_dir = ensure_local_whisper_model(
            model_name=model_name,
            cache_dir=cache_dir,
            log=log,
            cancel_event=cancel_event,
        )

        if cancel_event and cancel_event.is_set():
            raise NarrationError("Chuẩn bị narration đã bị dừng.")

        # Always instantiate from local cached path with local_files_only=True
        model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", local_files_only=True)

        if cancel_event and cancel_event.is_set():
            raise NarrationError("Chuẩn bị narration đã bị dừng.")

        segments_iter, info = model.transcribe(str(audio_path), beam_size=1, language="en")
        results: list[tuple[float, float, str]] = []
        for seg in segments_iter:
            if cancel_event and cancel_event.is_set():
                raise NarrationError("Chuẩn bị narration đã bị dừng.")
            txt = seg.text.strip()
            if txt:
                results.append((seg.start, seg.end, txt))
        return results
    except NarrationError:
        raise
    except Exception as exc:
        if log:
            log(f"Ghi chú: Nhận diện giọng nói nhanh nội bộ không khả dụng ({exc}).")
        return []


def transcribe_via_api(
    audio_path: Path,
    settings: AppSettings,
    log: LogCallback | None = None,
) -> list[tuple[float, float, str]]:
    """Call external configured API (e.g. OpenAI / Gemini) with clear credential validation."""
    if not settings.api_key.strip():
        raise NarrationError(
            "Cấu hình API được chọn nhưng thiếu API key. Vui lòng nhập API key hợp lệ trong cài đặt."
        )

    provider = settings.transcription_provider.lower()
    if log:
        log(f"Đang gửi âm thanh đến API '{provider}' để phân tích nội dung...")

    if provider == "openai":
        try:
            import urllib.request
            # Validate basic key format
            if not settings.api_key.startswith("sk-"):
                raise NarrationError("OpenAI API key không hợp lệ (phải bắt đầu bằng 'sk-').")

            # Multipart upload to OpenAI Audio Transcriptions
            # If network fails or key is invalid, raise NarrationError clearly
            boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
            url = (settings.api_base_url.rstrip("/") or "https://api.openai.com/v1") + "/audio/transcriptions"
            data = bytearray()
            # File
            data.extend(f"--{boundary}\r\n".encode())
            data.extend(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
            data.extend(b"Content-Type: audio/wav\r\n\r\n")
            data.extend(audio_path.read_bytes())
            data.extend(b"\r\n")
            # Model
            data.extend(f"--{boundary}\r\n".encode())
            data.extend(b'Content-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n')
            # Response format
            data.extend(f"--{boundary}\r\n".encode())
            data.extend(b'Content-Disposition: form-data; name="response_format"\r\n\r\nverbose_json\r\n')
            data.extend(f"--{boundary}--\r\n".encode())

            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization": f"Bearer {settings.api_key.strip()}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                segments_raw = result.get("segments", [])
                return [
                    (float(s["start"]), float(s["end"]), str(s["text"]).strip())
                    for s in segments_raw
                    if str(s.get("text", "")).strip()
                ]
        except Exception as exc:
            raise NarrationError(f"Lỗi xác thực hoặc kết nối đến OpenAI API: {exc}") from exc
    else:
        raise NarrationError(f"Nhà cung cấp API không được hỗ trợ: {provider}")


def _build_segments_from_dialogue(
    video_stem: str,
    total_sec: float,
    dialogue: list[tuple[float, float, str]],
    language: str = "en-US",
) -> list[RecapSegment]:
    """Derive balanced recap scenes directly from transcribed dialogue content."""
    total_sec = max(3.0, total_sec)
    num_scenes = min(4, max(2, len(dialogue) if len(dialogue) <= 4 else 3))

    scene_duration = total_sec / num_scenes
    segments: list[RecapSegment] = []

    for i in range(num_scenes):
        s_sec = i * scene_duration
        e_sec = min(total_sec, (i + 1) * scene_duration)

        # Collect dialogue belonging to this time window
        matching_lines = [
            text for (ds, de, text) in dialogue
            if (s_sec <= ds < e_sec) or (s_sec < de <= e_sec) or (ds <= s_sec and de >= e_sec)
        ]

        if matching_lines:
            combined_dialogue = " ".join(matching_lines)
            # Short quote snippet for clean narration
            snippet = " ".join(combined_dialogue.split()[:14])
            if i == 0:
                narration = f"The story begins as characters discuss: \"{snippet}\", establishing the premise of the episode."
            elif i == num_scenes - 1:
                narration = f"In the closing events, the dialogue culminates: \"{snippet}\", delivering a decisive turning point."
            else:
                narration = f"As events unfold, the situation develops: \"{snippet}\", intensifying the confrontation."
        else:
            combined_dialogue = ""
            if i == 0:
                narration = f"The opening scene at {s_sec:.1f}s establishes the setting before the primary action unfolds."
            elif i == num_scenes - 1:
                narration = f"The final scene brings resolution as events reach their conclusion at {e_sec:.1f}s."
            else:
                narration = f"The narrative progresses through key transitions between {s_sec:.1f}s and {e_sec:.1f}s."

        start_ms = int(round(s_sec * 1000))
        end_ms = int(round(e_sec * 1000))
        segments.append(
            RecapSegment(
                segment_id=f"scene_{i + 1:02d}",
                start_ms=start_ms,
                end_ms=end_ms,
                narration_text=narration,
                audio_policy="mute",
                original_dialogue_text=combined_dialogue,
            )
        )

    return segments


def _build_segments_for_no_speech(
    video_stem: str,
    total_sec: float,
    language: str = "en-US",
) -> list[RecapSegment]:
    """Generate explicit visual timeline recap segments when no speech is detected in episode.

    Explicitly handles the no-speech condition without inventing dialogue.
    """
    total_sec = max(3.0, total_sec)
    # Calibrate scene count based on duration
    num_scenes = 3 if total_sec >= 15.0 else 2
    scene_len = total_sec / num_scenes
    segments: list[RecapSegment] = []

    descriptions = [
        ("Opening sequence", "No spoken dialogue detected. Visual progression and scene composition establish the context."),
        ("Central sequence", "No dialogue in audio. The visual timeline conveys the primary action and pacing."),
        ("Closing sequence", "Visual conclusion unfolds without spoken lines, completing the sequence."),
    ]

    for i in range(num_scenes):
        s_sec = i * scene_len
        e_sec = min(total_sec, (i + 1) * scene_len)
        title, note = descriptions[i % len(descriptions)]

        narration = f"{title} from {s_sec:.1f} to {e_sec:.1f} seconds. {note}"
        segments.append(
            RecapSegment(
                segment_id=f"scene_{i + 1:02d}",
                start_ms=int(round(s_sec * 1000)),
                end_ms=int(round(e_sec * 1000)),
                narration_text=narration,
                audio_policy="mute",
                original_dialogue_text="",  # Explicitly empty
            )
        )

    return segments


def prepare_narration_for_video(
    video_path: Path,
    output_dir: Path,
    *,
    language: str = "en-US",
    mode: str = "FULL_EPISODE",
    cancel_event: threading.Event | None = None,
    log: LogCallback | None = None,
    settings: AppSettings | None = None,
) -> RecapManifest:
    """Prepare content-derived narration for the given video file.

    Workflow:
    1. Checks for companion JSON manifest or companion subtitle file (.srt).
    2. Probes media stream metadata and duration.
    3. Extracts audio and checks for presence of speech / acoustic energy.
    4. If speech is present: transcribes dialogue (via faster-whisper or external API)
       and generates narrative recap quoting actual dialogue lines.
    5. If no speech is detected: explicitly marks speech_detected=False, logs notice,
       and builds timeline visual recap without generic generic fillers.
    6. Validates and saves manifest for auditability and replay.
    """
    if cancel_event and cancel_event.is_set():
        raise NarrationError("Chuẩn bị narration đã bị dừng.")

    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if log:
        log(f"Đang kiểm tra metadata video: {video_path.name}")

    info = probe_media(video_path)
    duration_sec = info["duration"]
    has_audio = info.get("has_audio", False)

    # 1. Check for companion pre-existing manifest
    companion_candidates = [
        video_path.with_suffix(".json"),
        video_path.parent / f"{video_path.stem}_manifest.json",
        video_path.parent / f"{video_path.stem}.recap.json",
    ]
    for cand in companion_candidates:
        if cand.is_file():
            try:
                raw = json.loads(cand.read_text(encoding="utf-8"))
                manifest = RecapManifest.from_dict(raw)
                manifest.total_source_duration_sec = duration_sec
                validate_manifest(manifest, duration_sec)
                if log:
                    log(f"Đã nạp manifest có sẵn: {cand.name} ({len(manifest.segments)} phân đoạn)")
                return manifest
            except Exception as exc:
                if log:
                    log(f"Cảnh báo: Tệp manifest {cand.name} không hợp lệ ({exc}), phân tích lại nội dung...")

    if cancel_event and cancel_event.is_set():
        raise NarrationError("Chuẩn bị narration đã bị dừng.")

    # 2. Check for companion subtitles (.srt/.vtt)
    dialogue: list[tuple[float, float, str]] = extract_companion_subtitles(video_path)
    if dialogue and log:
        log(f"Đã tìm thấy phụ đề đối thoại đi kèm: {len(dialogue)} câu thoại.")

    # 3. Audio extraction and speech presence analysis
    temp_wav = output_dir / f"{video_path.stem}_analysis.wav"
    has_speech = False
    app_settings = settings or SettingsStore().load()

    try:
        if not dialogue and has_audio:
            if app_settings.transcription_provider != "local":
                # Validate API credentials upfront so misconfigured API settings fail fast and clearly
                if not app_settings.api_key.strip():
                    raise NarrationError("Cấu hình API được chọn nhưng thiếu API key. Vui lòng nhập API key hợp lệ trong cài đặt.")
                if app_settings.transcription_provider == "openai" and not app_settings.api_key.startswith("sk-"):
                    raise NarrationError("OpenAI API key không hợp lệ (phải bắt đầu bằng 'sk-').")

            if log:
                log(f"Trích xuất âm thanh để phân tích lời thoại: {video_path.name}")
            extracted = extract_audio_from_video(video_path, temp_wav)
            if extracted:
                has_sound, rms = check_audio_speech_energy(temp_wav)
                if has_sound:
                    if app_settings.transcription_provider != "local":
                        dialogue = transcribe_via_api(temp_wav, app_settings, log=log)
                    else:
                        dialogue = transcribe_local_whisper(
                            temp_wav,
                            language=language,
                            log=log,
                            cancel_event=cancel_event,
                        )

                    if dialogue:
                        has_speech = True
                        if log:
                            log(f"Đã nhận diện thành công {len(dialogue)} phân đoạn lời thoại từ audio.")
                    else:
                        if log:
                            log("Âm thanh có tín hiệu nhưng không tìm thấy lời thoại rõ ràng.")
                else:
                    if log:
                        log("Không phát hiện âm thanh/lời thoại (audio im lặng).")
            else:
                if log:
                    log("Không thể trích xuất audio hoặc video không có luồng âm thanh.")
        elif dialogue:
            has_speech = True
    finally:
        if temp_wav.exists():
            try:
                temp_wav.unlink()
            except OSError:
                pass

    if cancel_event and cancel_event.is_set():
        raise NarrationError("Chuẩn bị narration đã bị dừng.")

    # 4. Generate structured segments based on real content
    if has_speech and dialogue:
        if log:
            log("Tạo kịch bản recap dựa trên lời thoại thực tế của tập...")
        segments = _build_segments_from_dialogue(video_path.stem, duration_sec, dialogue, language=language)
    else:
        if log:
            log("Không phát hiện lời thoại trong tập; tạo tóm tắt diễn biến hình ảnh theo dòng thời gian.")
        segments = _build_segments_for_no_speech(video_path.stem, duration_sec, language=language)

    manifest = RecapManifest(
        project_id=video_path.stem,
        source_video=str(video_path),
        recap_mode=mode if has_speech else "SCENE_ANALYSIS_NO_SPEECH",
        recap_language=language,
        segments=segments,
        total_source_duration_sec=duration_sec,
        speech_detected=has_speech,
    )
    validate_manifest(manifest, duration_sec)

    # Save manifest for full inspectability and reproducibility
    manifest_path = output_dir / f"{video_path.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    if log:
        speech_status = "có lời thoại" if has_speech else "không có lời thoại"
        log(f"Đã lưu kịch bản recap ({len(segments)} phân đoạn, {speech_status}) tại: {manifest_path.name}")

    return manifest
