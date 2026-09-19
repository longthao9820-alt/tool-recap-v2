# Third-Party Software and Licenses

ToolRecap V2 incorporates or interfaces with several open-source libraries and components:

1. **FFmpeg / FFprobe**
   - Source: https://ffmpeg.org / https://www.gyan.dev/ffmpeg/builds/
   - License: GNU Lesser General Public License (LGPL) v2.1+ / GNU General Public License (GPL) v3
   - Note: Bundled as standalone command-line executables.

2. **Piper TTS**
   - Source: https://github.com/rhasspy/piper
   - License: MIT License / Apache License 2.0
   - Note: Voice models from Hugging Face (`rhasspy/piper-voices`) under open licenses (MIT / Open Data Commons).

3. **faster-whisper & CTranslate2**
   - Source: https://github.com/SYSTRAN/faster-whisper
   - License: MIT License (faster-whisper) / Apache License 2.0 (CTranslate2)
   - Note: Used for fast, local CPU speech transcription.

4. **VoiceStudio (External Subsystem)**
   - Source: https://github.com/debpalash/VoiceStudio
   - License: GNU Affero General Public License (AGPL) v3.0
   - Note: Upstream VoiceStudio is an optional external voice subsystem tracked via tagged releases (v0.5.3). ToolRecap V2 communicates with it via external process/manifest boundaries without static linking or license contamination.
