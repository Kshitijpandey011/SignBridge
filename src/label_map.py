"""Label mapping and taxonomy for dual-language (ISL + ASL) sign classification.

Encodes the canonical 20 concepts across 2 languages (40 classes total), maintaining
strict index stability and supporting subset filtering (e.g. Priority-10).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Canonical 20 concepts in fixed order
CANONICAL_CONCEPTS: List[str] = [
    "hello",
    "thank_you",
    "please",
    "sorry",
    "pain",
    "no",
    "yes",
    "help",
    "water",
    "food",
    "medicine",
    "doctor",
    "family",
    "work",
    "school",
    "home",
    "money",
    "phone",
    "happy",
    "sad",
]

# Priority-10 subset for rapid clinic / emergency prototyping
PRIORITY_10_CONCEPTS: List[str] = [
    "help",
    "water",
    "pain",
    "doctor",
    "medicine",
    "yes",
    "no",
    "thank_you",
    "hello",
    "food",
]

CONCEPT_DISPLAY_TEXT: Dict[str, str] = {
    "hello": "Hello",
    "thank_you": "Thank you",
    "please": "Please",
    "sorry": "Sorry",
    "pain": "Pain",
    "no": "No",
    "yes": "Yes",
    "help": "Help",
    "water": "Water",
    "food": "Food",
    "medicine": "Medicine",
    "doctor": "Doctor",
    "family": "Family",
    "work": "Work",
    "school": "School",
    "home": "Home",
    "money": "Money",
    "phone": "Phone",
    "happy": "Happy",
    "sad": "Sad",
}


def build_label_map(concepts: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, str]]:
    """Build the canonical label map mapping class index string to its metadata.

    Classes are grouped into:
    - 0 to N-1: ISL classes (isl_<concept>)
    - N to 2N-1: ASL classes (asl_<concept>)
    where N is the number of concepts (default 20, or subset like 10).

    Args:
        concepts: Optional sequence of concept names to include. Defaults to CANONICAL_CONCEPTS.

    Returns:
        Dict[str, Dict[str, str]]: JSON serializable dictionary mapping str(idx) to info.
    """
    selected_concepts = list(concepts) if concepts is not None else list(CANONICAL_CONCEPTS)
    for c in selected_concepts:
        if c not in CONCEPT_DISPLAY_TEXT:
            raise ValueError(f"Unknown concept '{c}'. Must be one of {CANONICAL_CONCEPTS}")

    n = len(selected_concepts)
    label_map: Dict[str, Dict[str, str]] = {}

    # ISL classes [0 .. n-1]
    for idx, concept in enumerate(selected_concepts):
        label_map[str(idx)] = {
            "lang": "isl",
            "concept": concept,
            "text": CONCEPT_DISPLAY_TEXT[concept],
            "class_name": f"isl_{concept}",
        }

    # ASL classes [n .. 2n-1]
    for idx, concept in enumerate(selected_concepts):
        class_idx = idx + n
        label_map[str(class_idx)] = {
            "lang": "asl",
            "concept": concept,
            "text": CONCEPT_DISPLAY_TEXT[concept],
            "class_name": f"asl_{concept}",
        }

    return label_map


class LabelRegistry:
    """Provides high-performance lookup arrays and masking indices for inference."""

    def __init__(self, label_map: Dict[str, Dict[str, str]]) -> None:
        self.raw_map = label_map
        self.num_classes = len(label_map)

        self.isl_idx: List[int] = []
        self.asl_idx: List[int] = []
        self.concept_of: Dict[int, str] = {}
        self.text_of: Dict[str, str] = dict(CONCEPT_DISPLAY_TEXT)
        self.lang_of: Dict[int, str] = {}
        self.class_name_of: Dict[int, str] = {}
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        for k, v in label_map.items():
            idx = int(k)
            lang = v["lang"]
            concept = v["concept"]
            text = v.get("text", CONCEPT_DISPLAY_TEXT.get(concept, concept))
            class_name = v.get("class_name", f"{lang}_{concept}")

            if lang == "isl":
                self.isl_idx.append(idx)
            elif lang == "asl":
                self.asl_idx.append(idx)

            self.concept_of[idx] = concept
            self.lang_of[idx] = lang
            self.class_name_of[idx] = class_name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            self.text_of[concept] = text

        self.isl_idx.sort()
        self.asl_idx.sort()

    @classmethod
    def from_file(cls, filepath: str | Path) -> "LabelRegistry":
        """Load LabelRegistry from a label_map.json file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Label map file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def default(cls) -> "LabelRegistry":
        """Initialize registry with full 40 canonical classes."""
        return cls(build_label_map(CANONICAL_CONCEPTS))

    def save(self, filepath: str | Path) -> None:
        """Save label map to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.raw_map, f, indent=2)


# Default singleton instance
DEFAULT_REGISTRY = LabelRegistry.default()
isl_idx = DEFAULT_REGISTRY.isl_idx
asl_idx = DEFAULT_REGISTRY.asl_idx
concept_of = DEFAULT_REGISTRY.concept_of
text_of = DEFAULT_REGISTRY.text_of


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate label_map.json")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="label_map.json",
        help="Target output path for label_map.json",
    )
    parser.add_argument(
        "--concepts",
        nargs="*",
        default=None,
        help="Subset of concepts to include (e.g. priority-10). Defaults to all 20 canonical concepts.",
    )
    parser.add_argument(
        "--priority-10",
        action="store_true",
        help="Convenience flag to generate label map using Priority-10 concepts.",
    )
    args = parser.parse_args()

    concepts = None
    if args.priority_10:
        concepts = PRIORITY_10_CONCEPTS
    elif args.concepts:
        concepts = args.concepts

    mapping = build_label_map(concepts)
    registry = LabelRegistry(mapping)
    registry.save(args.output)
    print(f"Generated {args.output} with {registry.num_classes} classes ({len(registry.isl_idx)} ISL, {len(registry.asl_idx)} ASL).")


if __name__ == "__main__":
    main()
