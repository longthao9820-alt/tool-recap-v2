"""Catalog of supported English voices and model specifications for ToolRecap V2.

Contains verified, real Piper voice models and dynamic loader for voices
installed via compatible VoiceStudio subsystems.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..paths import application_root, default_data_directory


@dataclass(frozen=True)
class VoiceSpec:
    voice_id: str
    display_name: str
    engine: str
    language: str
    gender: str
    description: str
    repo_id: str
    base_url: str
    files: tuple[str, ...]
    required_files: tuple[str, ...]
    preview_text: str = "Welcome to Toolrecap V2, your automated recap video generator."


# Only real, verified Piper voices in builtin catalog.
# OmniVoice is excluded until a compatible real subsystem is installed.
BUILTIN_VOICES: dict[str, VoiceSpec] = {
    "piper.en_US-lessac-medium": VoiceSpec(
        voice_id="piper.en_US-lessac-medium",
        display_name="Lessac (Nữ - Mỹ, Rõ ràng)",
        engine="piper",
        language="en-US",
        gender="Female",
        description="Giọng đọc nữ Mỹ tự nhiên, rõ chữ, thích hợp kể chuyện và recap phim.",
        repo_id="rhasspy/piper-voices",
        base_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/",
        files=("en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"),
        required_files=("en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"),
    ),
    "piper.en_US-ryan-medium": VoiceSpec(
        voice_id="piper.en_US-ryan-medium",
        display_name="Ryan (Nam - Mỹ, Truyền cảm)",
        engine="piper",
        language="en-US",
        gender="Male",
        description="Giọng nam Mỹ ấm áp, phong cách phóng sự hoặc phim tài liệu hành động.",
        repo_id="rhasspy/piper-voices",
        base_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/",
        files=("en_US-ryan-medium.onnx", "en_US-ryan-medium.onnx.json"),
        required_files=("en_US-ryan-medium.onnx", "en_US-ryan-medium.onnx.json"),
    ),
    "piper.en_GB-alba-medium": VoiceSpec(
        voice_id="piper.en_GB-alba-medium",
        display_name="Alba (Nữ - Anh, Điềm tĩnh)",
        engine="piper",
        language="en-GB",
        gender="Female",
        description="Giọng nữ Anh chuẩn (British), truyền tải câu chuyện điềm đạm, cuốn hút.",
        repo_id="rhasspy/piper-voices",
        base_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/",
        files=("en_GB-alba-medium.onnx", "en_GB-alba-medium.onnx.json"),
        required_files=("en_GB-alba-medium.onnx", "en_GB-alba-medium.onnx.json"),
    ),
    "piper.en_GB-alan-medium": VoiceSpec(
        voice_id="piper.en_GB-alan-medium",
        display_name="Alan (Nam - Anh, Trầm ấm)",
        engine="piper",
        language="en-GB",
        gender="Male",
        description="Giọng nam Anh cổ điển, sắc nét, lý tưởng cho recap kịch tính.",
        repo_id="rhasspy/piper-voices",
        base_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/",
        files=("en_GB-alan-medium.onnx", "en_GB-alan-medium.onnx.json"),
        required_files=("en_GB-alan-medium.onnx", "en_GB-alan-medium.onnx.json"),
    ),
}

DEFAULT_VOICE_ID = "piper.en_US-lessac-medium"


def is_executable_adapter_available(base_dir: Path) -> bool:
    """Check if an executable adapter (e.g. VoiceStudio.exe or adapter runner) exists in directory."""
    candidates = [
        base_dir / "VoiceStudio.exe",
        base_dir / "voicestudio.exe",
        base_dir / "adapter.py",
        base_dir / "run.cmd",
        base_dir / "bin" / "VoiceStudio.exe",
        base_dir / "bin" / "voicestudio.exe",
    ]
    return any(c.is_file() for c in candidates)


def get_available_voices(manifest_path: Path | None = None) -> dict[str, VoiceSpec]:
    """Retrieve all available selectable voices: builtin voices plus verified voices from installed subsystem manifest.
    Dynamic catalog is only loaded if an executable adapter exists.
    """
    voices = dict(BUILTIN_VOICES)

    candidate_paths: list[Path] = []
    if manifest_path:
        candidate_paths.append(Path(manifest_path))
    candidate_paths.append(default_data_directory() / "voice_subsystem" / "voice_manifest.json")
    candidate_paths.append(application_root() / "runtime" / "voices" / "voice_manifest.json")

    for cand in candidate_paths:
        if cand.is_file() and is_executable_adapter_available(cand.parent):
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                extra_voices = data.get("installed_voices", [])
                for v in extra_voices:
                    if isinstance(v, dict) and "voice_id" in v:
                        vid = str(v["voice_id"])
                        # Reject unverified omnivoice fake entries
                        if "omnivoice" in vid.lower() and not (cand.parent / "omnivoice.exe").is_file():
                            continue
                        voices[vid] = VoiceSpec(
                            voice_id=vid,
                            display_name=str(v.get("display_name", vid)),
                            engine=str(v.get("engine", "piper")),
                            language=str(v.get("language", "en-US")),
                            gender=str(v.get("gender", "Neutral")),
                            description=str(v.get("description", "")),
                            repo_id=str(v.get("repo_id", "")),
                            base_url=str(v.get("base_url", "")),
                            files=tuple(str(f) for f in v.get("files", ())),
                            required_files=tuple(str(f) for f in v.get("required_files", ())),
                            preview_text=str(v.get("preview_text", "Voice test.")),
                        )
            except Exception:
                pass

    return voices


def get_voice_spec(voice_id: str, manifest_path: Path | None = None) -> VoiceSpec:
    """Retrieve voice spec by ID from available voices or fall back to default voice."""
    available = get_available_voices(manifest_path=manifest_path)
    return available.get(voice_id, BUILTIN_VOICES[DEFAULT_VOICE_ID])
