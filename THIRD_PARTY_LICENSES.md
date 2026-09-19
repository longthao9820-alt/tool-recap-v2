# Third-Party Software and Licenses

## Commercial Rights & User Responsibility Disclaimer
ToolRecap V2 is an open-source automation utility distributed under the MIT License. ToolRecap V2 makes no claim of ownership, proprietary rights, or commercial rights over any third-party software, models, weights, datasets, or user-provided source media.

All third-party components and models remain subject to their respective upstream licenses and source terms. Users are solely responsible for ensuring that their use of third-party models, runtimes, and generated media complies with all applicable licenses, terms of service, and copyright laws.

---

## Third-Party Components & Licenses

### 1. VoiceStudio (External Subsystem Reference Only — Not Bundled)
- **Source**: https://github.com/debpalash/VoiceStudio
- **Version Reference**: v0.5.3
- **License**: GNU Affero General Public License v3.0 (AGPL-3.0)
- **Terms & Notes**: Upstream VoiceStudio is referenced as an optional external voice subsystem. VoiceStudio is **not bundled** into the ToolRecap V2 application package. ToolRecap V2 communicates with external processes only across well-defined process, manifest, or network boundaries without static linking or AGPL license contamination.

### 2. OmniVoice (Minimal Adapter & Model)
- **Source**: https://github.com/k2-fsa/OmniVoice
- **Model Reference**: `k2-fsa/OmniVoice@c5fdb5c`
- **License**: Apache License 2.0
- **Terms & Notes**: ToolRecap V2 interfaces with OmniVoice via a minimal, decoupled standalone adapter script (`omnivoice_adapter.py`) under Apache-2.0. Model weights are downloaded directly by the user from public repositories under Apache-2.0 terms.

### 3. PyTorch (Torch & Torchaudio)
- **Source**: https://pytorch.org / https://github.com/pytorch/pytorch
- **Version**: 2.4.0 (isolated voice runtime)
- **License**: Modified BSD License (BSD-3-Clause)
- **Terms & Notes**: Used in isolated voice subsystem runtime for neural speech synthesis.

### 4. RapidOCR (rapidocr-onnxruntime)
- **Source**: https://github.com/RapidAI/RapidOCR
- **Version**: 1.4.4
- **License**: Apache License 2.0
- **Terms & Notes**: Used for local optical character recognition of bitmap subtitles (PGS / VobSub). Pinned PP-OCRv4 ONNX model weights are retrieved from ModelScope under Apache-2.0.

### 5. ONNX Runtime (onnxruntime)
- **Source**: https://github.com/microsoft/onnxruntime
- **Version**: 1.28.0
- **License**: MIT License
- **Terms & Notes**: Cross-platform inference engine used for running OCR and TTS ONNX models.

### 6. OpenCV (opencv-python)
- **Source**: https://github.com/opencv/opencv-python
- **Version**: 5.0.0.93 / 4.8.0+
- **License**: Apache License 2.0
- **Terms & Notes**: Used for image preprocessing in subtitle bitmap OCR pipelines.

### 7. Shapely
- **Source**: https://github.com/shapely/shapely
- **Version**: 2.1.2 / 2.0.0+
- **License**: BSD 3-Clause License
- **Terms & Notes**: Geometric manipulation library utilized by RapidOCR bounding box calculations.

### 8. faster-whisper & CTranslate2
- **Source**: https://github.com/SYSTRAN/faster-whisper
- **Version**: 1.2.1 / 1.0.0+
- **License**: MIT License (faster-whisper) / Apache License 2.0 (CTranslate2)
- **Terms & Notes**: Used for local CPU speech-to-text transcription when subtitles are unavailable or OCR fails.

### 9. FFmpeg & FFprobe
- **Source**: https://ffmpeg.org / https://www.gyan.dev/ffmpeg/builds/
- **License**: GNU Lesser General Public License (LGPL) v2.1+ / GNU General Public License (GPL) v3 (depending on build features)
- **Terms & Notes**: Standalone command-line binaries bundled under `runtime/ffmpeg/bin/`. Invoked strictly via standard CLI subprocesses; no dynamic or static linking into application binaries.

### 10. Piper TTS
- **Source**: https://github.com/rhasspy/piper
- **Version**: 1.8.0 / 1.2.0+
- **License**: MIT License / Apache License 2.0
- **Voice Models**: Public domain / MIT / Open Data Commons from `rhasspy/piper-voices`.

### 11. Pillow
- **Source**: https://github.com/python-pillow/Pillow
- **Version**: 12.3.0
- **License**: Historical Permission Notice and Disclaimer (HPND) / MIT-equivalent

### 12. pyclipper
- **Source**: https://github.com/fonttools/pyclipper
- **Version**: 1.4.0
- **License**: Boost Software License 1.0

### 13. soundfile
- **Source**: https://github.com/bastibe/python-soundfile
- **Version**: 0.14.0
- **License**: BSD 3-Clause License

---

## Public Model Source Terms

Model weights utilized by ToolRecap V2 (including Piper TTS voice models, faster-whisper models, RapidOCR PP-OCRv4 models, and OmniVoice models) are hosted on public repositories (Hugging Face, ModelScope, GitHub).

- **Piper Voices**: Open licenses (MIT / Open Data Commons / Creative Commons).
- **Whisper Models (OpenAI)**: MIT License.
- **RapidOCR PP-OCRv4 (Baidu/RapidAI)**: Apache License 2.0.
- **OmniVoice (k2-fsa)**: Apache License 2.0.

Downloading and utilizing these model weights is performed under public source terms. Users must evaluate whether their intended usage meets upstream terms and applicable jurisdiction requirements.
