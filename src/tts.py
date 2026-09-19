"""Offline Text-To-Speech (TTS) synthesizer with OS voice detection and graceful degradation.

Discovers system-installed voices via pyttsx3, matches target languages, and avoids
wrong-language phonetic pronunciation by degrading to silent text display when an
authentic voice is unavailable.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyttsx3


class TTSEngine:
    """Offline speech synthesizer managing installed OS voices."""

    def __init__(self, log_path: str | Path = "voices_available.json") -> None:
        self.log_path = Path(log_path)
        self._engine: Optional[pyttsx3.Engine] = None
        self.voices_by_lang: Dict[str, str] = {}
        self.all_voices_meta: List[Dict[str, Any]] = []

        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize pyttsx3 driver and inspect system voices."""
        try:
            self._engine = pyttsx3.init()
            # Set comfortable default speech rate
            self._engine.setProperty("rate", 160)
            voices = self._engine.getProperty("voices")

            for v in voices:
                v_id = str(v.id)
                v_name = str(v.name).lower()
                lang_code = "en"  # default assumption

                # Inspect voice languages attribute or name heuristics
                if hasattr(v, "languages") and v.languages:
                    lang_bytes = v.languages[0]
                    if isinstance(lang_bytes, bytes):
                        lang_code = lang_bytes.decode("utf-8", errors="ignore")[:2].lower()
                    elif isinstance(lang_bytes, str):
                        lang_code = lang_bytes[:2].lower()

                # Common Windows/macOS/Linux voice name detection heuristics
                if "hindi" in v_name or "kalpana" in v_name or "hemant" in v_name:
                    lang_code = "hi"
                elif "tamil" in v_name or "valluvar" in v_name:
                    lang_code = "ta"
                elif "spanish" in v_name or "helena" in v_name or "sabina" in v_name:
                    lang_code = "es"
                elif "french" in v_name or "hortense" in v_name or "julie" in v_name:
                    lang_code = "fr"
                elif "english" in v_name or "david" in v_name or "zira" in v_name or "george" in v_name:
                    lang_code = "en"

                self.all_voices_meta.append({
                    "id": v_id,
                    "name": v.name,
                    "detected_lang": lang_code,
                })

                if lang_code not in self.voices_by_lang:
                    self.voices_by_lang[lang_code] = v_id

            # Save discovered voice inventory
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.all_voices_meta, f, indent=2)

        except Exception as e:
            print(f"Warning: Failed to initialize pyttsx3 audio driver: {e}")
            self._engine = None

    def has_voice_for(self, lang: str) -> bool:
        """Check whether an authentic voice exists for language."""
        return lang.lower() in self.voices_by_lang

    def speak(self, text: str, lang: str = "en") -> bool:
        """Speak the given text offline if a matching voice exists.

        Args:
            text: Text to speak.
            lang: ISO language code ('en', 'hi', 'es', etc.).

        Returns:
            bool: True if speech was performed, False if suppressed due to missing voice.
        """
        if not text or not text.strip():
            return False

        if not self._engine:
            return False

        lang = lang.lower()
        if not self.has_voice_for(lang):
            print(f"[TTS Notice] No native offline voice installed for '{lang}'. Skipping speech to prevent mispronunciation.")
            return False

        voice_id = self.voices_by_lang[lang]
        try:
            self._engine.setProperty("voice", voice_id)
            self._engine.say(text)
            self._engine.runAndWait()
            return True
        except Exception as e:
            print(f"Error during audio speech playback: {e}")
            return False

    def speak_async(self, text: str, lang: str = "en") -> None:
        """Non-blocking asynchronous speech invocation."""
        t = threading.Thread(target=self.speak, args=(text, lang), daemon=True)
        t.start()


# Global engine singleton
_GLOBAL_TTS: Optional[TTSEngine] = None


def get_tts_engine() -> TTSEngine:
    """Return shared TTSEngine instance."""
    global _GLOBAL_TTS
    if _GLOBAL_TTS is None:
        _GLOBAL_TTS = TTSEngine()
    return _GLOBAL_TTS


def speak(text: str, lang: str = "en") -> bool:
    """Convenience function for offline speech synthesis."""
    return get_tts_engine().speak(text, lang)
