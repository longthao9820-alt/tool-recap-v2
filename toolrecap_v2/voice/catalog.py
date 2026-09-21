"""Catalog of supported English voices and model specifications for ToolRecap V2.

Contains 12 ToolRecap local OmniVoice presets as the primary production catalog,
with internal compatibility fallbacks for legacy Piper models.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..paths import application_root, default_data_directory
from .runtime import VoiceRuntimeInspector


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
    style: str = "film_recap"
    fallback_piper_voice_id: str = ""
    instruct: str = ""


# V1 Supported Styles
SUPPORTED_VOICE_STYLES: tuple[str, ...] = (
    "film_recap",
    "storytelling",
    "documentary",
    "crime_thriller",
    "drama",
    "soap_emotional",
    "energetic",
    "neutral",
)

STYLE_NAMES: dict[str, str] = {
    "film_recap": "Recap phim",
    "storytelling": "Kể chuyện",
    "documentary": "Documentary",
    "crime_thriller": "Crime / Thriller",
    "drama": "Drama",
    "soap_emotional": "Soap / Emotional",
    "energetic": "Năng động",
    "neutral": "Trung tính",
}

STYLE_INSTRUCTIONS: dict[str, str] = {
    "film_recap": "Narrate a film recap clearly and naturally, engaging but not theatrical, with crisp articulation and a moderate pace.",
    "storytelling": "Speak naturally and clearly, like an engaging storyteller, at a moderate pace.",
    "documentary": "Use a calm, authoritative documentary narration with measured pacing and clear articulation.",
    "crime_thriller": "Use a controlled serious tone with subtle tension, clear words, and no exaggerated acting.",
    "drama": "Use an emotionally aware dramatic narration, restrained and natural rather than theatrical.",
    "soap_emotional": "Use a warm natural emotional tone suitable for a television soap recap, without melodrama.",
    "energetic": "Use an energetic and engaging delivery while keeping every word clear.",
    "neutral": "Use a neutral, clear, steady delivery.",
}


# Exact 12 V1 VoiceStudio English voices as primary production catalog.
# Official archetypes from VoiceStudio v0.5.3 (debpalash/VoiceStudio, backend/core/archetypes.py).
# Upstream engine: voicestudio, model: k2-fsa/OmniVoice@c5fdb5c.
BUILTIN_VOICES: dict[str, VoiceSpec] = {
    # en-US (6 voices)
    "voicestudio.en.neighbor": VoiceSpec(
        voice_id="voicestudio.en.neighbor",
        display_name="Neighbor — Nữ — ToolRecap Local",
        engine="omnivoice",
        language="en-US",
        gender="Female",
        description="Film recap / Storytelling",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="film_recap",
        fallback_piper_voice_id="piper.en_US-lessac-medium",
        instruct="female, young adult, moderate pitch, american accent",
    ),
    "voicestudio.en.companion": VoiceSpec(
        voice_id="voicestudio.en.companion",
        display_name="Companion — Nữ — ToolRecap Local",
        engine="omnivoice",
        language="en-US",
        gender="Female",
        description="Film recap / Storytelling",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="film_recap",
        fallback_piper_voice_id="piper.en_US-lessac-medium",
        instruct="female, middle-aged, moderate pitch, canadian accent",
    ),
    "voicestudio.en.teacher": VoiceSpec(
        voice_id="voicestudio.en.teacher",
        display_name="Teacher — Nữ — ToolRecap Local",
        engine="omnivoice",
        language="en-US",
        gender="Female",
        description="Film recap / Storytelling",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="film_recap",
        fallback_piper_voice_id="piper.en_US-lessac-medium",
        instruct="female, middle-aged, moderate pitch, american accent",
    ),
    "voicestudio.en.anchor": VoiceSpec(
        voice_id="voicestudio.en.anchor",
        display_name="Anchor — Nam — ToolRecap Local",
        engine="omnivoice",
        language="en-US",
        gender="Male",
        description="Documentary / News",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="documentary",
        fallback_piper_voice_id="piper.en_US-ryan-medium",
        instruct="male, middle-aged, moderate pitch, american accent",
    ),
    "voicestudio.en.documentarian": VoiceSpec(
        voice_id="voicestudio.en.documentarian",
        display_name="Documentarian — Nam — ToolRecap Local",
        engine="omnivoice",
        language="en-US",
        gender="Male",
        description="Documentary / Recap (Default US)",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="documentary",
        fallback_piper_voice_id="piper.en_US-ryan-medium",
        instruct="male, middle-aged, low pitch, american accent",
    ),
    "voicestudio.en.promo": VoiceSpec(
        voice_id="voicestudio.en.promo",
        display_name="Promo — Nam — ToolRecap Local",
        engine="omnivoice",
        language="en-US",
        gender="Male",
        description="Energetic / Trailer",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="energetic",
        fallback_piper_voice_id="piper.en_US-ryan-medium",
        instruct="male, middle-aged, low pitch",
    ),
    # en-GB (6 voices)
    "voicestudio.en.librarian": VoiceSpec(
        voice_id="voicestudio.en.librarian",
        display_name="Librarian — Nữ — ToolRecap Local",
        engine="omnivoice",
        language="en-GB",
        gender="Female",
        description="Storytelling / Drama",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="storytelling",
        fallback_piper_voice_id="piper.en_GB-alba-medium",
        instruct="female, middle-aged, low pitch, british accent",
    ),
    "voicestudio.en.podcaster": VoiceSpec(
        voice_id="voicestudio.en.podcaster",
        display_name="Podcaster — Nữ — ToolRecap Local",
        engine="omnivoice",
        language="en-GB",
        gender="Female",
        description="Narrative / Conversation",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="storytelling",
        fallback_piper_voice_id="piper.en_GB-alba-medium",
        instruct="female, young adult, high pitch, australian accent",
    ),
    "voicestudio.en.luxe": VoiceSpec(
        voice_id="voicestudio.en.luxe",
        display_name="Luxe — Nữ — ToolRecap Local",
        engine="omnivoice",
        language="en-GB",
        gender="Female",
        description="Emotional / Drama",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="drama",
        fallback_piper_voice_id="piper.en_GB-alba-medium",
        instruct="female, middle-aged, moderate pitch, british accent",
    ),
    "voicestudio.en.storyteller": VoiceSpec(
        voice_id="voicestudio.en.storyteller",
        display_name="Storyteller — Nam — ToolRecap Local",
        engine="omnivoice",
        language="en-GB",
        gender="Male",
        description="Storytelling / Drama",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="storytelling",
        fallback_piper_voice_id="piper.en_GB-alan-medium",
        instruct="male, elderly, low pitch, british accent",
    ),
    "voicestudio.en.commentator": VoiceSpec(
        voice_id="voicestudio.en.commentator",
        display_name="Commentator — Nam — ToolRecap Local",
        engine="omnivoice",
        language="en-GB",
        gender="Male",
        description="Commentary / Recap (Default UK)",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="storytelling",
        fallback_piper_voice_id="piper.en_GB-alan-medium",
        instruct="male, middle-aged, high pitch, british accent",
    ),
    "voicestudio.en.explainer": VoiceSpec(
        voice_id="voicestudio.en.explainer",
        display_name="Explainer — Nam — ToolRecap Local",
        engine="omnivoice",
        language="en-GB",
        gender="Male",
        description="Documentary / Explainer",
        repo_id="k2-fsa/OmniVoice@c5fdb5c",
        base_url="",
        files=(),
        required_files=(),
        preview_text="Welcome to Toolrecap V2, your automated recap video generator.",
        style="documentary",
        fallback_piper_voice_id="piper.en_GB-alan-medium",
        instruct="male, young adult, moderate pitch, british accent",
    ),
}

# Defaults
DEFAULT_VOICE_ID_US = "voicestudio.en.documentarian"
DEFAULT_VOICE_ID_GB = "voicestudio.en.commentator"
DEFAULT_VOICE_ID = "voicestudio.en.documentarian"
DEFAULT_VOICE_BY_LANGUAGE: dict[str, str] = {
    "en-US": DEFAULT_VOICE_ID_US,
    "en-GB": DEFAULT_VOICE_ID_GB,
}


# Four Piper voices remain internal compatibility fallback only, NOT in selectable production list.
PIPER_COMPATIBILITY_VOICES: dict[str, VoiceSpec] = {
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


OFFICIAL_VOICESTUDIO_PYTHON = Path(
    os.path.expandvars(r"%LOCALAPPDATA%\com.debpalash.omnivoice-studio\project\.venv\Scripts\python.exe")
)
_cached_official_runtime: Path | None | str = "UNCHECKED"


def clear_official_runtime_cache() -> None:
    """Clear memory cache of detected official VoiceStudio runtime."""
    global _cached_official_runtime
    _cached_official_runtime = "UNCHECKED"


def detect_official_voicestudio_runtime(
    override_path: Path | None = None,
    *,
    force_recheck: bool = False,
    timeout: float = 10.0,
) -> Path | None:
    """Detect and verify installed official VoiceStudio Python environment.
    Verifies that required imports (torch, torchaudio, omnivoice) and version are functional.
    Caches verified path in memory to avoid repeated subprocess overhead.
    """
    global _cached_official_runtime
    if override_path is None and not force_recheck and _cached_official_runtime != "UNCHECKED":
        return _cached_official_runtime if isinstance(_cached_official_runtime, Path) else None

    cand = override_path or Path(os.environ.get("VOICESTUDIO_PYTHON") or OFFICIAL_VOICESTUDIO_PYTHON)
    if not cand.is_file():
        if override_path is None:
            _cached_official_runtime = None
        return None

    try:
        import subprocess

        res = subprocess.run(
            [
                str(cand),
                "-c",
                "import torch, torchaudio, omnivoice; from importlib.metadata import version; print(version('omnivoice'))",
            ],
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout,
        )
        if res.returncode == 0 and res.stdout.strip():
            if override_path is None:
                _cached_official_runtime = cand
            return cand
    except Exception:
        pass

    if override_path is None:
        _cached_official_runtime = None
    return None


def get_isolated_runtime_python(subsystem_dir: Path | None = None) -> Path | None:
    """Locate isolated V2 Python runtime executable in subsystem or runtime directory."""
    target = subsystem_dir or (default_data_directory() / "voice_subsystem")
    candidates = [
        target / "runtime" / "Scripts" / "python.exe",
        target / "runtime" / "python.exe",
        target / "venv" / "Scripts" / "python.exe",
        target / "python.exe",
        application_root() / "runtime" / "voices" / "python" / "python.exe",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def is_executable_adapter_available(base_dir: Path) -> bool:
    """Check if an executable adapter exists in directory."""
    candidates = [
        base_dir / "VoiceStudio.exe",
        base_dir / "voicestudio.exe",
        base_dir / "adapter.exe",
        base_dir / "voicestudio_adapter.exe",
        base_dir / "adapter.py",
        base_dir / "run.cmd",
        base_dir / "bin" / "VoiceStudio.exe",
        base_dir / "bin" / "voicestudio.exe",
    ]
    return any(c.is_file() for c in candidates)


def is_voicestudio_ready(
    subsystem_dir: Path | None = None,
    *,
    detect_official: bool | None = None,
) -> bool:
    """Legacy API name: verify managed dependencies and model, never external apps."""
    runtime_dir = (subsystem_dir / "runtime") if subsystem_dir is not None else None
    health = VoiceRuntimeInspector(runtime_dir=runtime_dir).inspect(run_imports=True)
    return health.dependencies_ok and health.model_valid


def is_voice_selectable(
    voice_id: str,
    subsystem_dir: Path | None = None,
    *,
    detect_official: bool | None = None,
) -> bool:
    """Check whether a voice is selectable based on actual backend readiness."""
    spec = get_voice_spec(voice_id)
    if spec.engine == "omnivoice":
        return is_voicestudio_ready(subsystem_dir=subsystem_dir, detect_official=detect_official)
    elif spec.engine == "piper":
        try:
            import piper  # noqa: F401
            return True
        except ImportError:
            return False
    return False


def get_voice_status(
    voice_id: str,
    subsystem_dir: Path | None = None,
    *,
    detect_official: bool | None = None,
) -> dict[str, Any]:
    """Stateless diagnostic status; READY requires VoiceManager's synthesis proof."""
    spec = get_voice_spec(voice_id)
    if spec.engine == "omnivoice":
        runtime_dir = (subsystem_dir / "runtime") if subsystem_dir is not None else None
        health = VoiceRuntimeInspector(runtime_dir=runtime_dir).inspect(run_imports=True)
        prepared = health.dependencies_ok and health.model_valid
        return {
            "voice_id": voice_id,
            "engine": "omnivoice",
            # A stateless catalog probe cannot prove that synthesis just succeeded.
            # VoiceManager health cache is the sole READY authority.
            "ready": False,
            "status": "VERIFYING" if prepared else health.state,
            "install_on_first_use": not health.runtime_present,
            "status_label": (
                "Voice runtime and model are installed — run Preview to verify production synthesis."
                if prepared
                else health.human_message
            ),
            "health": health.to_dict(),
        }
    elif spec.engine == "piper":
        try:
            import piper  # noqa: F401
            ready = True
            msg = "Piper engine sẵn sàng (tương thích nội bộ)."
        except ImportError:
            ready = False
            msg = "Piper engine chưa được cài đặt."
        return {
            "voice_id": voice_id,
            "engine": "piper",
            "ready": ready,
            "status": "READY" if ready else "NOT_INSTALLED",
            "status_label": msg,
        }
    return {
        "voice_id": voice_id,
        "engine": spec.engine,
        "ready": False,
        "status": "UNKNOWN_ENGINE",
        "status_label": f"Engine '{spec.engine}' không được hỗ trợ.",
    }


def migrate_voice_setting(
    current_voice_id: str,
    target_language: str = "en-US",
    *,
    backend_ready: bool | None = None,
) -> tuple[str, str | None]:
    """Migrate legacy Piper selection to a ToolRecap local voice when ready.

    If backend is ready, migrates to corresponding VoiceStudio voice (or language default).
    If backend is not ready, preserves current Piper setting and returns a warning.
    """
    if backend_ready is None:
        backend_ready = is_voicestudio_ready()

    piper_to_voicestudio = {
        "piper.en_US-lessac-medium": "voicestudio.en.neighbor",
        "piper.en_US-ryan-medium": "voicestudio.en.documentarian",
        "piper.en_GB-alba-medium": "voicestudio.en.librarian",
        "piper.en_GB-alan-medium": "voicestudio.en.commentator",
    }

    if current_voice_id in piper_to_voicestudio or current_voice_id.startswith("piper."):
        if backend_ready:
            preferred = piper_to_voicestudio.get(
                current_voice_id,
                DEFAULT_VOICE_ID_US if target_language == "en-US" else DEFAULT_VOICE_ID_GB,
            )
            return preferred, None
        else:
            warning = (
                f"ToolRecap local voice runtime chưa sẵn sàng; giữ cấu hình Piper hiện tại ({current_voice_id})."
            )
            return current_voice_id, warning

    return current_voice_id, None


def get_available_voices(manifest_path: Path | None = None) -> dict[str, VoiceSpec]:
    """Return ToolRecap's fixed, verified local voice preset catalog."""
    return dict(BUILTIN_VOICES)


def get_voice_spec(voice_id: str, manifest_path: Path | None = None) -> VoiceSpec:
    """Retrieve voice spec by ID from available voices, compatibility fallbacks, or default voice."""
    available = get_available_voices(manifest_path=manifest_path)
    if voice_id in available:
        return available[voice_id]
    if voice_id in PIPER_COMPATIBILITY_VOICES:
        return PIPER_COMPATIBILITY_VOICES[voice_id]
    return BUILTIN_VOICES[DEFAULT_VOICE_ID]
