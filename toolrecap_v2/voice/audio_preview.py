"""Audio preview player for voice auditions without blocking the UI."""
from __future__ import annotations

import sys
import threading
import wave
from pathlib import Path
from typing import Callable


class AudioPreviewPlayer:
    """Plays audio preview files asynchronously with stop and cancel capability."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_playing = False

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def play(self, audio_path: Path | str, on_finished: Callable[[], None] | None = None) -> None:
        """Start playing audio in a daemon thread."""
        self.stop()
        self._stop_event.clear()

        def _worker() -> None:
            self._is_playing = True
            try:
                path = Path(audio_path).resolve()
                if not path.is_file():
                    return

                # Try winsound on Windows first for zero-dependency native playback
                if sys.platform == "win32":
                    try:
                        import winsound
                        # winsound.SND_SYNC plays until completion or until interrupted
                        winsound.PlaySound(str(path), winsound.SND_FILENAME)
                        return
                    except Exception:
                        pass

                # Fallback: sounddevice / simpleaudio if available
                try:
                    import soundfile as sf
                    import sounddevice as sd
                    data, fs = sf.read(str(path))
                    sd.play(data, fs)
                    sd.wait()
                except Exception:
                    pass
            finally:
                self._is_playing = False
                if on_finished:
                    try:
                        on_finished()
                    except Exception:
                        pass

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop any ongoing playback."""
        self._stop_event.set()
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._is_playing = False
