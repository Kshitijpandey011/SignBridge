"""Phrase builder mapping sequence of recognized sign concepts to an English pivot sentence.

DISCLAIMER: This is NOT grammatical ISL/ASL natural language translation. It maps
discrete recognized concept tokens into clean, simple English sentences suitable as
an intermediate pivot for spoken-language translation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.label_map import CONCEPT_DISPLAY_TEXT

DEFAULT_PHRASE_TIMEOUT: float = 4.5  # Seconds before concept buffer auto-clears


class PhraseBuilder:
    """Buffers consecutive sign concepts and composes cohesive English pivot sentences."""

    def __init__(
        self,
        phrases_path: str | Path = "phrases.json",
        timeout_seconds: float = DEFAULT_PHRASE_TIMEOUT,
        max_concepts: int = 4,
    ) -> None:
        self.phrases_path = Path(phrases_path)
        self.timeout_seconds = timeout_seconds
        self.max_concepts = max_concepts

        self.phrase_dict: Dict[str, str] = {}
        self.templates: Dict[str, str] = {}
        self._load_phrases()

        self._concept_buffer: List[str] = []
        self._last_added_time: float = 0.0

    def _load_phrases(self) -> None:
        """Load phrase table from JSON."""
        if self.phrases_path.exists():
            with open(self.phrases_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.phrase_dict = data.get("phrases", {})
            self.templates = data.get("templates", {})
        else:
            # Fallback hardcoded defaults
            self.phrase_dict = {
                "help+water": "I need help. I need water.",
                "pain+doctor": "I am in pain. I need a doctor.",
                "medicine+please": "Medicine, please.",
                "thank_you": "Thank you.",
            }

    def add_concept(self, concept: str, current_time: Optional[float] = None) -> str:
        """Add a recognized concept and return the updated composed English sentence.

        Args:
            concept: Concept identifier (e.g. 'water', 'doctor').
            current_time: Optional timestamp override.

        Returns:
            str: Composed English pivot sentence.
        """
        now = current_time if current_time is not None else time.time()

        # Check for buffer expiration
        if (now - self._last_added_time) > self.timeout_seconds and len(self._concept_buffer) > 0:
            self._concept_buffer.clear()

        # Deduplicate consecutive identical concept within short window
        if not self._concept_buffer or self._concept_buffer[-1] != concept:
            self._concept_buffer.append(concept)
            if len(self._concept_buffer) > self.max_concepts:
                self._concept_buffer.pop(0)

        self._last_added_time = now
        return self.build_sentence(self._concept_buffer)

    def build_sentence(self, concepts: List[str]) -> str:
        """Compose an English pivot sentence from a given list of concepts."""
        if not concepts:
            return ""

        key = "+".join(concepts)
        # 1. Exact phrasebook lookup
        if key in self.phrase_dict:
            return self.phrase_dict[key]

        # 2. Check suffix / sub-combinations if buffer has >2 concepts
        if len(concepts) >= 2:
            sub_key = "+".join(concepts[-2:])
            if sub_key in self.phrase_dict:
                return self.phrase_dict[sub_key]

        # 3. Single concept lookup
        if len(concepts) == 1:
            single = concepts[0]
            if single in self.phrase_dict:
                return self.phrase_dict[single]
            return CONCEPT_DISPLAY_TEXT.get(single, single.capitalize())

        # 4. Fallback compositional template
        readable = [CONCEPT_DISPLAY_TEXT.get(c, c) for c in concepts]
        return ". ".join(readable) + "."

    def get_current_phrase(self) -> str:
        """Return sentence from currently buffered concepts."""
        return self.build_sentence(self._concept_buffer)

    def get_buffered_concepts(self) -> List[str]:
        """Return list of currently buffered concepts."""
        return list(self._concept_buffer)

    def reset(self) -> None:
        """Clear concept buffer."""
        self._concept_buffer.clear()
        self._last_added_time = 0.0
