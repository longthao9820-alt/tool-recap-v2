"""Voice subsystem for ToolRecap V2."""
from .catalog import BUILTIN_VOICES, VoiceSpec, get_voice_spec
from .manager import VoiceModelManager, get_voice_manager

__all__ = [
    "BUILTIN_VOICES",
    "VoiceSpec",
    "get_voice_spec",
    "VoiceModelManager",
    "get_voice_manager",
]
