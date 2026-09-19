"""Typed data models for media streams, unified subtitle cues, tracks, and discovery."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def normalize_language_code(raw_code: str | None) -> str:
    """Normalize language codes/names to 3-letter ISO-639-2 (e.g. 'eng', 'vie', 'und')."""
    if not raw_code:
        return "und"
    s = raw_code.strip().lower()
    # Direct mappings for common ISO-639-1 / names to ISO-639-2
    mapping = {
        "en": "eng",
        "eng": "eng",
        "english": "eng",
        "en-us": "eng",
        "en-gb": "eng",
        "vi": "vie",
        "vie": "vie",
        "vietnamese": "vie",
        "ja": "jpn",
        "jpn": "jpn",
        "japanese": "jpn",
        "es": "spa",
        "spa": "spa",
        "spanish": "spa",
        "fr": "fra",
        "fra": "fra",
        "fre": "fra",
        "french": "fra",
        "de": "deu",
        "deu": "deu",
        "ger": "deu",
        "german": "deu",
        "zh": "zho",
        "zho": "zho",
        "chi": "zho",
        "chinese": "zho",
        "ko": "kor",
        "kor": "kor",
        "korean": "kor",
        "it": "ita",
        "ita": "ita",
        "italian": "ita",
        "pt": "por",
        "por": "por",
        "portuguese": "por",
        "ru": "rus",
        "rus": "rus",
        "russian": "rus",
        "und": "und",
        "undefined": "und",
    }
    if s in mapping:
        return mapping[s]
    # Check if prefix matches (e.g. en_US -> eng)
    token = re.split(r"[-_]", s)[0]
    if token in mapping:
        return mapping[token]
    # Return 3-letter code if possible, else original
    return s[:3] if len(s) >= 3 else s


@dataclass
class VideoStreamInfo:
    index: int
    video_index: int
    codec: str
    width: int
    height: int
    fps: float = 0.0
    duration: float = 0.0
    bitrate: int = 0
    title: str = ""
    default: bool = False
    forced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioStreamInfo:
    index: int
    audio_index: int
    codec: str
    language: str
    title: str = ""
    channels: int = 2
    channel_layout: str = "stereo"
    bitrate: int = 0
    default: bool = False
    forced: bool = False
    is_commentary: bool = False
    is_descriptive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubtitleStreamInfo:
    index: int
    subtitle_index: int
    codec: str
    language: str
    title: str = ""
    default: bool = False
    forced: bool = False
    is_bitmap: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioSelectionResult:
    selected_stream: AudioStreamInfo | None
    has_warning: bool = False
    warning: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_stream": self.selected_stream.to_dict() if self.selected_stream else None,
            "has_warning": self.has_warning,
            "warning": self.warning,
            "reason": self.reason,
        }


@dataclass
class MediaProbeResult:
    path: str
    duration: float
    width: int
    height: int
    has_video: bool
    has_audio: bool
    video_codec: str | None
    audio_codec: str | None
    video_streams: list[VideoStreamInfo] = field(default_factory=list)
    audio_streams: list[AudioStreamInfo] = field(default_factory=list)
    subtitle_streams: list[SubtitleStreamInfo] = field(default_factory=list)
    selected_audio: AudioSelectionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "video_streams": [v.to_dict() for v in self.video_streams],
            "audio_streams": [a.to_dict() for a in self.audio_streams],
            "subtitle_streams": [s.to_dict() for s in self.subtitle_streams],
            "selected_audio": self.selected_audio.to_dict() if self.selected_audio else None,
        }


@dataclass
class SubtitleCue:
    """Unified subtitle cue format matching exact contract invariants."""
    start_ms: int
    end_ms: int
    text: str
    source_type: str  # 'sidecar', 'embedded', 'stt', 'ocr'
    source_format: str  # 'srt', 'ass', 'ssa', 'vtt', 'pgs', 'vobsub', 'whisper'
    stream_index: int | None = None
    source_file: str | None = None
    language: str = "eng"
    confidence: float = 1.0
    episode_id: str = ""
    source_video: str = ""

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitleCue:
        return cls(
            start_ms=int(data.get("start_ms", 0)),
            end_ms=int(data.get("end_ms", 0)),
            text=str(data.get("text", "")),
            source_type=str(data.get("source_type", "sidecar")),
            source_format=str(data.get("source_format", "srt")),
            stream_index=int(data["stream_index"]) if data.get("stream_index") is not None else None,
            source_file=str(data["source_file"]) if data.get("source_file") is not None else None,
            language=str(data.get("language", "eng")),
            confidence=float(data.get("confidence", 1.0)),
            episode_id=str(data.get("episode_id", "")),
            source_video=str(data.get("source_video", "")),
        )


@dataclass
class SubtitleTrack:
    """Provenance and metadata for discovered subtitle tracks."""
    track_id: str
    source_type: str  # 'sidecar' or 'embedded'
    source_format: str  # 'srt', 'ass', 'ssa', 'vtt', 'pgs', 'vobsub'
    language: str  # normalized code
    title: str = ""
    stream_index: int | None = None  # None if sidecar
    source_file: str | None = None  # None if embedded
    is_forced: bool = False
    is_full: bool = True
    is_bitmap: bool = False
    score: float = 0.0
    cues: list[SubtitleCue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cues"] = [c.to_dict() for c in self.cues]
        return d


@dataclass
class SubtitleDiscoveryResult:
    video_path: str
    episode_id: str
    all_tracks: list[SubtitleTrack] = field(default_factory=list)
    best_english_full: SubtitleTrack | None = None
    forced_tracks: list[SubtitleTrack] = field(default_factory=list)
    stt_required: bool = False
    selection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "episode_id": self.episode_id,
            "all_tracks": [t.to_dict() for t in self.all_tracks],
            "best_english_full": self.best_english_full.to_dict() if self.best_english_full else None,
            "forced_tracks": [t.to_dict() for t in self.forced_tracks],
            "stt_required": self.stt_required,
            "selection_reason": self.selection_reason,
        }


@dataclass
class PgsSubtitleEvent:
    start_ms: int
    end_ms: int
    image: Any  # PIL.Image.Image | None
    x: int
    y: int
    width: int
    height: int
    composition_number: int = 0
    is_forced: bool = False


@dataclass
class VobSubEvent:
    start_ms: int
    end_ms: int
    image: Any  # PIL.Image.Image | None
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    filepos: int = 0
