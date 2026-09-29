"""Pre-cache high-quality offline audio files for all 20 canonical concepts across all supported languages."""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "assets" / "audio_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "translations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

concepts = data.get("concepts", {})
phrases = data.get("phrases", {})
languages = ["en", "hi", "ta", "es", "fr"]

print(f"Pre-caching audio for {len(concepts)} concepts and {len(phrases)} phrases across {len(languages)} languages...")

headers = {"User-Agent": "Mozilla/5.0"}
success_count = 0

# Cache concepts
for concept, lang_dict in concepts.items():
    for lang in languages:
        target_file = CACHE_DIR / f"{lang}_{concept}.mp3"
        if target_file.exists() and target_file.stat().st_size > 500:
            success_count += 1
            continue

        text_entry = lang_dict.get(lang, "")
        text = text_entry.get("text", "") if isinstance(text_entry, dict) else str(text_entry)
        if not text:
            continue

        try:
            encoded_text = urllib.parse.quote(text)
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl={lang}&client=tw-ob&q={encoded_text}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                audio_bytes = resp.read()
            with open(target_file, "wb") as f:
                f.write(audio_bytes)
            success_count += 1
            print(f"[OK] Cached {lang}_{concept}.mp3 ({len(audio_bytes)} bytes)")
            time.sleep(0.08)
        except Exception as e:
            print(f"[FAIL] {lang}_{concept}: {e}")

# Cache phrases
for pkey, lang_dict in phrases.items():
    safe_pkey = pkey.replace("+", "_")
    for lang in languages:
        target_file = CACHE_DIR / f"{lang}_phrase_{safe_pkey}.mp3"
        if target_file.exists() and target_file.stat().st_size > 500:
            success_count += 1
            continue

        text_entry = lang_dict.get(lang, "")
        text = text_entry.get("text", "") if isinstance(text_entry, dict) else str(text_entry)
        if not text:
            continue

        try:
            encoded_text = urllib.parse.quote(text)
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl={lang}&client=tw-ob&q={encoded_text}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                audio_bytes = resp.read()
            with open(target_file, "wb") as f:
                f.write(audio_bytes)
            success_count += 1
            print(f"[OK] Cached {lang}_phrase_{safe_pkey}.mp3 ({len(audio_bytes)} bytes)")
            time.sleep(0.08)
        except Exception as e:
            print(f"[FAIL] {lang}_phrase_{safe_pkey}: {e}")

print(f"\nCompleted: {success_count} / {total_count} audio files cached in {CACHE_DIR}")
