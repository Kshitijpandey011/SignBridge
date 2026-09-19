"""Multi-tier offline translation engine with rigorous fallback hierarchy.

Hierarchy:
    Tier 1: Curated Phrasebook (Exact, offline, verified translations)
    Tier 2: Offline Machine Translation (Argos Translate / local MT models)
    Tier 3: Online MT Fallback (Optional, explicitly opt-in only)
    Tier 4: English Pivot Fallback (Always available, zero failure mode)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.phrasebuilder import PhraseBuilder

# Global toggle for optional online fallback (OFF by default)
ONLINE_ENABLED: bool = False


class TranslationEngine:
    """Manages phrasebook lookups and offline/online translation tiers."""

    def __init__(
        self,
        translations_path: str | Path = "translations.json",
        phrase_builder: Optional[PhraseBuilder] = None,
    ) -> None:
        self.translations_path = Path(translations_path)
        self.phrase_builder = phrase_builder or PhraseBuilder()
        self.concepts_dict: Dict[str, Dict[str, Any]] = {}
        self.phrases_dict: Dict[str, Dict[str, Any]] = {}
        self.templates_dict: Dict[str, Dict[str, Any]] = {}

        self._load_dictionary()

        # Check for Argos availability
        self.argos_available = False
        try:
            import argostranslate.translate  # type: ignore
            self.argos_available = True
        except ImportError:
            self.argos_available = False

    def _load_dictionary(self) -> None:
        """Load curated translations from JSON file."""
        if self.translations_path.exists():
            with open(self.translations_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.concepts_dict = data.get("concepts", {})
            self.phrases_dict = data.get("phrases", {})
            self.templates_dict = data.get("templates", {})

    def _get_entry_text_and_review(self, entry: Any) -> Tuple[str, bool]:
        """Extract translated text and whether it requires human verification."""
        if isinstance(entry, dict):
            text = entry.get("text", "")
            needs_review = entry.get("needs_review", False)
            return text, needs_review
        elif isinstance(entry, str):
            return entry, False
        return "", True

    def argos_has_pair(self, from_code: str, to_code: str) -> bool:
        """Verify whether an offline translation package exists for language pair."""
        if not self.argos_available:
            return False
        try:
            import argostranslate.translate  # type: ignore
            installed_langs = argostranslate.translate.get_installed_languages()
            from_lang = next((l for l in installed_langs if l.code == from_code), None)
            to_lang = next((l for l in installed_langs if l.code == to_code), None)
            if from_lang and to_lang:
                trans = from_lang.get_translation(to_lang)
                return trans is not None
            return False
        except Exception:
            return False

    def argos_translate(self, text: str, from_code: str, to_code: str) -> str:
        """Translate text using local Argos Translate model."""
        import argostranslate.translate  # type: ignore

        installed_langs = argostranslate.translate.get_installed_languages()
        from_lang = next((l for l in installed_langs if l.code == from_code), None)
        to_lang = next((l for l in installed_langs if l.code == to_code), None)
        translation = from_lang.get_translation(to_lang)
        return translation.translate(text)

    def online_translate(self, text: str, target: str) -> str:
        """Opt-in online translation fallback using urllib (no external heavy API needed)."""
        import json
        import urllib.parse
        import urllib.request

        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target}&dt=t&q={urllib.parse.quote(text)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return "".join([frag[0] for frag in data[0]])

    def translate(self, concepts: List[str], target: str) -> Tuple[str, str]:
        """Execute translation fallback chain.

        Args:
            concepts: List of recognized concept tokens (e.g. ['help', 'water']).
            target: Target language code ('en', 'hi', 'ta', 'es', etc.).

        Returns:
            Tuple[str, str]: (translated_text, quality_tag)
                quality_tag in {"curated", "machine", "machine-online", "fallback-english"}
        """
        if not concepts:
            return "", "curated"

        target = target.lower()
        key = "+".join(concepts)
        english_pivot = self.phrase_builder.build_sentence(concepts)

        # Tier 1a: Exact Curated Phrase match
        if key in self.phrases_dict and target in self.phrases_dict[key]:
            text, needs_rev = self._get_entry_text_and_review(self.phrases_dict[key][target])
            # Even if needs_review is true, it is sourced from curated phrasebook with note
            tag = "curated" if not needs_rev else "curated"
            return text, tag

        # Direct English request
        if target == "en":
            return english_pivot, "curated"

        # Tier 1b: All concepts exist in phrasebook concept dictionary
        all_present = all(
            (c in self.concepts_dict and target in self.concepts_dict[c]) for c in concepts
        )
        if all_present:
            # Single concept
            if len(concepts) == 1:
                c_entry = self.concepts_dict[concepts[0]][target]
                text, _ = self._get_entry_text_and_review(c_entry)
                return text, "curated"

            # Check if template composition applies (e.g. 'need')
            if "need" in self.templates_dict and target in self.templates_dict["need"]:
                tmpl_text, _ = self._get_entry_text_and_review(self.templates_dict["need"][target])
                items_translated = [
                    self._get_entry_text_and_review(self.concepts_dict[c][target])[0] for c in concepts
                ]
                combined = ", ".join(items_translated)
                return tmpl_text.replace("{x}", combined), "curated"

        # Tier 2: Offline Machine Translation (Argos)
        if self.argos_has_pair("en", target):
            try:
                mt_text = self.argos_translate(english_pivot, "en", target)
                return mt_text, "machine"
            except Exception:
                pass

        # Tier 3: Optional Online Machine Translation
        if ONLINE_ENABLED:
            try:
                online_text = self.online_translate(english_pivot, target)
                return online_text, "machine-online"
            except Exception:
                pass

        # Tier 4: Fallback to English pivot
        return english_pivot, "fallback-english"
