"""Multi-tier offline Text-To-Speech (TTS) synthesizer with native multilingual playback.

Architecture:
1. Native Pre-Cached Offline Audio Tier: Instant zero-latency native audio playback for
   all supported Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati,
   Kannada, Malayalam, Punjabi, Urdu, and English pivot) across all 20 canonical
   concepts and phrases via native Windows MCI (ctypes winmm).
2. Dynamic Online Caching Tier: On-demand synthesis & caching for uncached phrases via TTS endpoint.
3. Native OS Voice Tier: Direct Windows SAPI / pyttsx3 playback for system-installed voices.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import queue
import sys
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TTSEngine:
    """Multi-tier multilingual speech synthesizer."""

    def __init__(self, log_path: str | Path = "voices_available.json") -> None:
        self.log_path = Path(log_path)
        self.project_root = Path(__file__).resolve().parent.parent
        self.cache_dir = self.project_root / "assets" / "audio_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.audio_map: Dict[Tuple[str, str], Path] = {}
        self.voices_by_lang: Dict[str, str] = {}
        self.all_voices_meta: List[Dict[str, Any]] = []
        self._queue: queue.Queue = queue.Queue(maxsize=10)
        self._is_windows = sys.platform == "win32"

        self._load_audio_cache_map()
        self._init_voices()
        self._start_worker()

    def _load_audio_cache_map(self) -> None:
        """Load pre-cached audio map for all 20 concepts and curated phrases across all languages."""
        trans_path = self.project_root / "translations.json"
        if not trans_path.exists():
            return

        try:
            with open(trans_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Map concepts
            for concept, lang_dict in data.get("concepts", {}).items():
                concept_clean = concept.strip().lower()
                for lang, entry in lang_dict.items():
                    val = entry.get("text", "") if isinstance(entry, dict) else str(entry)
                    val_clean = val.strip().lower()
                    audio_file = self.cache_dir / f"{lang}_{concept}.mp3"
                    if audio_file.exists():
                        self.audio_map[(lang.lower(), concept_clean)] = audio_file
                        if val_clean:
                            self.audio_map[(lang.lower(), val_clean)] = audio_file

            # Map phrases
            for pkey, lang_dict in data.get("phrases", {}).items():
                safe_pkey = pkey.replace("+", "_").strip().lower()
                for lang, entry in lang_dict.items():
                    val = entry.get("text", "") if isinstance(entry, dict) else str(entry)
                    val_clean = val.strip().lower()
                    audio_file = self.cache_dir / f"{lang}_phrase_{safe_pkey}.mp3"
                    if audio_file.exists():
                        self.audio_map[(lang.lower(), safe_pkey)] = audio_file
                        if val_clean:
                            self.audio_map[(lang.lower(), val_clean)] = audio_file

        except Exception as e:
            print(f"[TTS Cache Map Notice]: {e}")

    def _detect_lang_from_name(self, name: str) -> str:
        nl = name.lower()
        if "hindi" in nl:
            return "hi"
        elif "tamil" in nl:
            return "ta"
        elif "telugu" in nl:
            return "te"
        elif "bengali" in nl or "bangla" in nl:
            return "bn"
        elif "marathi" in nl:
            return "mr"
        elif "gujarati" in nl:
            return "gu"
        elif "kannada" in nl:
            return "kn"
        elif "malayalam" in nl:
            return "ml"
        elif "punjabi" in nl:
            return "pa"
        elif "urdu" in nl:
            return "ur"
        return "en"

    def _init_voices(self) -> None:
        """Inspect and catalog installed system OS voices."""
        if self._is_windows:
            try:
                import pythoncom
                import win32com.client

                pythoncom.CoInitialize()
                spk = win32com.client.Dispatch("SAPI.SpVoice")
                voices = spk.GetVoices()
                for i in range(len(voices)):
                    v = voices.Item(i)
                    desc = v.GetDescription()
                    lang = self._detect_lang_from_name(desc)

                    self.all_voices_meta.append({
                        "id": v.Id,
                        "name": desc,
                        "detected_lang": lang,
                    })
                    if lang not in self.voices_by_lang:
                        self.voices_by_lang[lang] = v.Id

                with open(self.log_path, "w", encoding="utf-8") as f:
                    json.dump(self.all_voices_meta, f, indent=2)
                return
            except Exception as e:
                print(f"[TTS Init Notice] SAPI voice discovery fallback: {e}")

        try:
            import pyttsx3

            eng = pyttsx3.init()
            voices = eng.getProperty("voices")
            for v in voices:
                desc = str(v.name)
                v_id = str(v.id)
                lang = self._detect_lang_from_name(desc)

                self.all_voices_meta.append({
                    "id": v_id,
                    "name": desc,
                    "detected_lang": lang,
                })
                if lang not in self.voices_by_lang:
                    self.voices_by_lang[lang] = v_id

            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.all_voices_meta, f, indent=2)
        except Exception:
            pass

    def _play_audio_file(self, file_path: Path | str) -> bool:
        """Play audio file cleanly via native Windows MCI winmm."""
        p = Path(file_path)
        if not p.exists() or p.stat().st_size < 100:
            return False

        if self._is_windows:
            try:
                mci = ctypes.windll.winmm.mciSendStringW
                alias = f"mp3_tts_{threading.get_ident()}"
                mci(f"close {alias}", None, 0, 0)
                ret = mci(f'open "{p.resolve()}" type mpegvideo alias {alias}', None, 0, 0)
                if ret == 0:
                    mci(f"play {alias} wait", None, 0, 0)
                    mci(f"close {alias}", None, 0, 0)
                    return True
            except Exception as e:
                print(f"[TTS Playback Error]: {e}")
        return False

    def _fetch_and_cache(self, text: str, lang: str) -> Optional[Path]:
        """Fetch and cache audio on-demand via TTS endpoint."""
        try:
            hash_key = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()[:10]
            target_path = self.cache_dir / f"{lang}_dyn_{hash_key}.mp3"
            if target_path.exists() and target_path.stat().st_size > 500:
                return target_path

            encoded_text = urllib.parse.quote(text)
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl={lang}&client=tw-ob&q={encoded_text}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                audio_bytes = resp.read()

            with open(target_path, "wb") as f:
                f.write(audio_bytes)

            self.audio_map[(lang.lower(), text.strip().lower())] = target_path
            return target_path
        except Exception:
            return None

    def _start_worker(self) -> None:
        """Start dedicated background daemon thread for serialized audio playback."""
        worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        worker_thread.start()

    def _worker_loop(self) -> None:
        """Process TTS jobs sequentially on a dedicated thread."""
        sapi_speaker = None
        if self._is_windows:
            try:
                import pythoncom
                import win32com.client

                pythoncom.CoInitialize()
                sapi_speaker = win32com.client.Dispatch("SAPI.SpVoice")
            except Exception as e:
                print(f"[TTS Worker Notice] win32com init failed: {e}")

        pyttsx_eng = None
        if sapi_speaker is None:
            try:
                import pyttsx3

                pyttsx_eng = pyttsx3.init()
            except Exception:
                pass

        while True:
            try:
                item = self._queue.get()
                if item is None:
                    break
                text, lang = item
                text_clean = str(text).strip()
                lang_clean = str(lang).lower().strip()

                # 1. Check Pre-Cached Audio Map First (Instant Native Playback)
                cached_file = self.audio_map.get((lang_clean, text_clean.lower()))
                if cached_file is not None and cached_file.exists():
                    if self._play_audio_file(cached_file):
                        self._queue.task_done()
                        continue

                # 2. Check if a dynamic download is available
                dyn_file = self._fetch_and_cache(text_clean, lang_clean)
                if dyn_file is not None and self._play_audio_file(dyn_file):
                    self._queue.task_done()
                    continue

                # 3. Fallback: OS SAPI / pyttsx3 Synthesis
                if sapi_speaker is not None:
                    # If specific language voice is available, set it
                    v_id = self.voices_by_lang.get(lang_clean)
                    if v_id and hasattr(sapi_speaker, "GetVoices"):
                        try:
                            for idx in range(len(sapi_speaker.GetVoices())):
                                if sapi_speaker.GetVoices().Item(idx).Id == v_id:
                                    sapi_speaker.Voice = sapi_speaker.GetVoices().Item(idx)
                                    break
                        except Exception:
                            pass
                    sapi_speaker.Speak(str(text_clean), 0)
                elif pyttsx_eng is not None:
                    pyttsx_eng.say(str(text_clean))
                    pyttsx_eng.runAndWait()

                self._queue.task_done()
            except Exception as e:
                print(f"[TTS Worker Error]: {e}")

    def has_voice_for(self, lang: str) -> bool:
        """Check whether voice or audio cache exists for language."""
        l = lang.lower()
        if l in self.voices_by_lang:
            return True
        # If we have cached audio for this language
        return any(k[0] == l for k in self.audio_map)

    def speak(self, text: str, lang: str = "en") -> bool:
        """Speak the given text offline using thread-safe queue."""
        self.speak_async(text, lang)
        return True

    def speak_async(self, text: str, lang: str = "en") -> None:
        """Non-blocking asynchronous speech invocation using queue."""
        if not text or not str(text).strip():
            return
        try:
            while self._queue.qsize() > 2:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait((str(text), lang))
        except Exception:
            pass


# Global engine singleton
_GLOBAL_TTS: Optional[TTSEngine] = None


def get_tts_engine() -> TTSEngine:
    """Return shared TTSEngine instance."""
    global _GLOBAL_TTS
    if _GLOBAL_TTS is None:
        _GLOBAL_TTS = TTSEngine()
    return _GLOBAL_TTS


def speak(text: str, lang: str = "en") -> bool:
    """Convenience function for multilingual offline speech synthesis."""
    return get_tts_engine().speak(text, lang)
