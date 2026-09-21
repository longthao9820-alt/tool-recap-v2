"""Standalone OmniVoice TTS adapter invoked by external/isolated Python runtime.

Zero ToolRecap V2 or AGPL imports.
Communicates via JSON request file or CLI args, reports progress to stderr,
and writes standard PCM16 WAV audio.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Official 12 designed voice archetypes from VoiceStudio v0.5.3 (backend/core/archetypes.py)
OFFICIAL_VOICE_INSTRUCTS: dict[str, str] = {
    "voicestudio.en.neighbor": "female, young adult, moderate pitch, american accent",
    "voicestudio.en.companion": "female, middle-aged, moderate pitch, canadian accent",
    "voicestudio.en.teacher": "female, middle-aged, moderate pitch, american accent",
    "voicestudio.en.anchor": "male, middle-aged, moderate pitch, american accent",
    "voicestudio.en.documentarian": "male, middle-aged, low pitch, american accent",
    "voicestudio.en.promo": "male, middle-aged, low pitch",
    "voicestudio.en.librarian": "female, middle-aged, low pitch, british accent",
    "voicestudio.en.podcaster": "female, young adult, high pitch, australian accent",
    "voicestudio.en.luxe": "female, middle-aged, moderate pitch, british accent",
    "voicestudio.en.storyteller": "male, elderly, low pitch, british accent",
    "voicestudio.en.commentator": "male, middle-aged, high pitch, british accent",
    "voicestudio.en.explainer": "male, young adult, moderate pitch, british accent",
}

SUPPORTED_VOICE_STYLES: frozenset[str] = frozenset({
    "film_recap",
    "storytelling",
    "documentary",
    "crime_thriller",
    "drama",
    "soap_emotional",
    "energetic",
    "neutral",
})


def report_progress(stage: str, progress: int, message: str = "") -> None:
    """Report JSON progress stage to stderr."""
    payload = {"stage": stage, "progress": progress}
    if message:
        payload["message"] = message
    try:
        sys.stderr.write(json.dumps(payload) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def validate_request(
    voice_id: str,
    text: str,
    style: str,
    output_path: Path,
) -> None:
    """Validate request against allowlists and path safety rules."""
    if voice_id not in OFFICIAL_VOICE_INSTRUCTS:
        raise ValueError(
            f"Voice ID '{voice_id}' không nằm trong danh sách 12 giọng ToolRecap Local được hỗ trợ. "
            f"Hợp lệ: {', '.join(sorted(OFFICIAL_VOICE_INSTRUCTS.keys()))}"
        )
    if style not in SUPPORTED_VOICE_STYLES:
        raise ValueError(
            f"Phong cách '{style}' không được hỗ trợ. "
            f"Hợp lệ: {', '.join(sorted(SUPPORTED_VOICE_STYLES))}"
        )
    if not text or not text.strip():
        raise ValueError("Văn bản đọc (text) không được để trống.")
    if output_path.suffix.lower() != ".wav":
        raise ValueError(f"Định dạng đầu ra phải là tệp .wav: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ToolRecap V2 OmniVoice standalone adapter")
    parser.add_argument("--request", type=str, default=None, help="Path to JSON request file")
    parser.add_argument("--voice", type=str, default=None, help="Voice ID")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--style", type=str, default="film_recap", help="Voice style")
    parser.add_argument("--output", type=str, default=None, help="Output WAV file path")
    parser.add_argument("--instruct", type=str, default=None, help="Optional override instruct string")
    parser.add_argument("--model", type=str, default=None, help="Model repo or local path")
    parser.add_argument("--cache-dir", type=str, default=None, help="HuggingFace cache directory")

    args = parser.parse_args()

    # 1. Parse input either from request JSON file or CLI args
    if args.request:
        req_path = Path(args.request).resolve()
        if not req_path.is_file():
            sys.stderr.write(f"Tệp yêu cầu không tồn tại: {req_path}\n")
            return 1
        try:
            req_data = json.loads(req_path.read_text(encoding="utf-8"))
        except Exception as exc:
            sys.stderr.write(f"Không thể đọc tệp yêu cầu JSON: {exc}\n")
            return 1
        voice_id = str(req_data.get("voice_id", "")).strip()
        text = str(req_data.get("text", "")).strip()
        style = str(req_data.get("style", "film_recap")).strip()
        output_str = str(req_data.get("output", "")).strip()
        instruct = str(req_data.get("instruct", "")).strip() or None
        model_name = req_data.get("model") or args.model
        cache_dir_str = req_data.get("cache_dir") or args.cache_dir
    else:
        voice_id = str(args.voice or "").strip()
        text = str(args.text or "").strip()
        style = str(args.style or "film_recap").strip()
        output_str = str(args.output or "").strip()
        instruct = str(args.instruct or "").strip() or None
        model_name = args.model
        cache_dir_str = args.cache_dir

    if not output_str:
        sys.stderr.write("Thiếu tham số đường dẫn đầu ra (output).\n")
        return 1

    output_path = Path(output_str).resolve()
    try:
        validate_request(voice_id, text, style, output_path)
    except ValueError as val_err:
        sys.stderr.write(f"Lỗi tham số: {val_err}\n")
        return 1

    # Ensure parent output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine instruct
    final_instruct = instruct or OFFICIAL_VOICE_INSTRUCTS[voice_id]

    # Setup cache directory
    if cache_dir_str:
        hf_home = Path(cache_dir_str).resolve()
    else:
        local_app_data = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        hf_home = Path(local_app_data) / "ToolRecapV2" / "models" / "voices" / "omnivoice"

    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")

    report_progress("init", 10, "Khởi tạo môi trường OmniVoice...")

    # 2. Import torch, torchaudio, and omnivoice
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from omnivoice import OmniVoice
    except ImportError as imp_err:
        sys.stderr.write(f"Lỗi import backend: {imp_err}. Vui lòng cài đặt đầy đủ torch, torchaudio, omnivoice.\n")
        return 2

    # 3. Select device and safe dtype (float16 only on CUDA; float32 on CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_source = model_name or "k2-fsa/OmniVoice@c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
    model_revision = None
    if "@" in model_source:
        model_source, model_revision = model_source.rsplit("@", 1)
    report_progress("loading_model", 25, f"Tải mô hình {model_source} trên {device} ({dtype})...")

    try:
        load_kwargs = {"device_map": device, "dtype": dtype}
        if model_revision:
            load_kwargs["revision"] = model_revision
        model = OmniVoice.from_pretrained(model_source, **load_kwargs)
    except Exception as exc:
        sys.stderr.write(f"Lỗi tải mô hình OmniVoice từ '{model_source}': {exc}\n")
        return 3

    # 4. Generate speech
    report_progress("generating", 55, "Đang tổng hợp giọng đọc...")
    try:
        audios = model.generate(
            text=text,
            language="English",
            instruct=final_instruct,
        )
    except Exception as exc:
        sys.stderr.write(f"Lỗi quá trình generate của OmniVoice: {exc}\n")
        return 4

    if not audios or len(audios) == 0:
        sys.stderr.write("OmniVoice không tạo ra dữ liệu âm thanh nào.\n")
        return 5

    # 5. Save PCM16 WAV
    report_progress("saving", 85, "Lưu tệp âm thanh PCM16 WAV...")
    try:
        audio_value = audios[0]
        if hasattr(audio_value, "detach"):
            audio_value = audio_value.detach().cpu().numpy()
        audio_array = np.asarray(audio_value, dtype=np.float32).squeeze()
        if audio_array.size == 0:
            raise ValueError("OmniVoice returned an empty audio array.")
        sample_rate = getattr(model, "sampling_rate", 24000)
        sf.write(str(output_path), audio_array, sample_rate, subtype="PCM_16")
    except Exception as exc:
        sys.stderr.write(f"Lỗi lưu tệp âm thanh WAV: {exc}\n")
        return 6

    report_progress("completed", 100, "Hoàn tất tổng hợp âm thanh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
