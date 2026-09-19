"""Dataset loading, validation, and signer-held-out splitting.

Loads .npy sequences structured as:
    data/<lang>/<concept>/<signer_id>/<seq_id>.npy
Strictly isolates signers between training and validation partitions to prevent
identity leakage and guarantee authentic generalization metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.features.normalize import FEATURE_DIM
from src.label_map import (
    CANONICAL_CONCEPTS,
    DEFAULT_REGISTRY,
    LabelRegistry,
    build_label_map,
)

SEQUENCE_LENGTH: int = 30


class SignDataset:
    """Dataset container with signer partition awareness."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        label_registry: Optional[LabelRegistry] = None,
        val_signers: Optional[Sequence[str]] = None,
        val_ratio: float = 0.25,
        random_seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.registry = label_registry or DEFAULT_REGISTRY
        self.val_signers_explicit = set(val_signers) if val_signers else None
        self.val_ratio = val_ratio
        self.random_seed = random_seed

    def load_samples(
        self,
        allowed_concepts: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Scan data directory and discover all valid .npy sequence paths with metadata."""
        samples: List[Dict[str, Any]] = []
        if not self.data_dir.exists():
            return samples

        concepts_filter = set(allowed_concepts) if allowed_concepts else set(CANONICAL_CONCEPTS)

        for lang in ("isl", "asl"):
            lang_dir = self.data_dir / lang
            if not lang_dir.exists():
                continue

            for concept_dir in lang_dir.iterdir():
                if not concept_dir.is_dir() or concept_dir.name not in concepts_filter:
                    continue
                concept = concept_dir.name

                for signer_dir in concept_dir.iterdir():
                    if not signer_dir.is_dir():
                        continue
                    signer_id = signer_dir.name

                    for npy_file in signer_dir.glob("*.npy"):
                        class_name = f"{lang}_{concept}"
                        if class_name not in self.registry.class_to_idx:
                            continue

                        samples.append({
                            "path": npy_file,
                            "lang": lang,
                            "concept": concept,
                            "class_name": class_name,
                            "class_idx": self.registry.class_to_idx[class_name],
                            "lang_idx": 0 if lang == "isl" else 1,
                            "signer_id": signer_id,
                        })

        return samples

    def get_splits(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        allowed_concepts: Optional[Sequence[str]] = None,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]],
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]],
    ]:
        """Split samples by signer ID and return (X, y_class, y_lang, meta) for train and val."""
        if samples is None:
            samples = self.load_samples(allowed_concepts)

        if not samples:
            raise ValueError(f"No samples found in {self.data_dir}. Record data or pass synthetic samples.")

        # Identify all unique signers
        all_signers = sorted(list({s["signer_id"] for s in samples}))
        print(f"Total discovered signers ({len(all_signers)}): {all_signers}")

        if self.val_signers_explicit:
            val_signers = self.val_signers_explicit
        else:
            if len(all_signers) == 1:
                print("WARNING: Only 1 unique signer detected! Signer-held-out validation requires >= 2 signers.")
                print("Falling back to sequence-level split for pipeline validation. DO NOT report this as held-out accuracy.")
                val_signers = set()
            else:
                rng = np.random.RandomState(self.random_seed)
                shuffled_signers = list(all_signers)
                rng.shuffle(shuffled_signers)
                num_val = max(1, int(round(len(all_signers) * self.val_ratio)))
                val_signers = set(shuffled_signers[:num_val])

        train_samples: List[Dict[str, Any]] = []
        val_samples: List[Dict[str, Any]] = []

        if len(all_signers) == 1:
            # Emergency fallback: split sequences
            for idx, s in enumerate(samples):
                if idx % 4 == 0:
                    val_samples.append(s)
                else:
                    train_samples.append(s)
        else:
            for s in samples:
                if s["signer_id"] in val_signers:
                    val_samples.append(s)
                else:
                    train_samples.append(s)

        print(f"Dataset split: {len(train_samples)} train sequences, {len(val_samples)} validation sequences.")
        print(f"Train Signers: {sorted(list({s['signer_id'] for s in train_samples}))}")
        print(f"Validation Signers: {sorted(list({s['signer_id'] for s in val_samples}))}")

        num_classes = self.registry.num_classes

        def _materialize(sample_list: List[Dict[str, Any]]):
            n = len(sample_list)
            X = np.zeros((n, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
            y_class = np.zeros((n, num_classes), dtype=np.float32)
            y_lang = np.zeros((n, 2), dtype=np.float32)

            for i, s in enumerate(sample_list):
                data = np.load(s["path"])
                if data.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
                    # Pad or slice to canonical shape
                    fixed = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
                    t = min(data.shape[0], SEQUENCE_LENGTH)
                    d = min(data.shape[1], FEATURE_DIM)
                    fixed[:t, :d] = data[:t, :d]
                    data = fixed

                X[i] = data
                y_class[i, s["class_idx"]] = 1.0
                y_lang[i, s["lang_idx"]] = 1.0

            return X, y_class, y_lang, sample_list

        train_split = _materialize(train_samples)
        val_split = _materialize(val_samples)
        return train_split, val_split


def create_synthetic_dataset(
    target_dir: str | Path = "data",
    concepts: Optional[Sequence[str]] = None,
    samples_per_class: int = 6,
) -> int:
    """Generate synthetic baseline sequences across signers to enable testing pipelines without webcam."""
    target_path = Path(target_dir)
    selected_concepts = list(concepts) if concepts else CANONICAL_CONCEPTS
    signers = ["signer_alpha", "signer_beta", "signer_heldout"]

    total_created = 0
    rng = np.random.RandomState(42)

    for lang in ("isl", "asl"):
        for concept_idx, concept in enumerate(selected_concepts):
            # Create distinct motion signature for this concept and language
            base_signature = rng.normal(loc=0.0, scale=0.2, size=(SEQUENCE_LENGTH, FEATURE_DIM)).astype(np.float32)
            # Add language distinction: ISL has non-zero both hands; ASL has right hand active, left hand zero
            if lang == "asl":
                base_signature[:, 132:195] = 0.0  # left hand zeros
            else:
                base_signature[:, 132:195] += 0.5  # left hand active

            # Concept distinction in right hand
            base_signature[:, 195 + (concept_idx % 20)] += 1.0

            for sample_idx in range(samples_per_class):
                signer = signers[sample_idx % len(signers)]
                signer_dir = target_path / lang / concept / signer
                signer_dir.mkdir(parents=True, exist_ok=True)

                seq_id = f"synth_{lang}_{concept}_{sample_idx:02d}"
                npy_file = signer_dir / f"{seq_id}.npy"
                json_file = signer_dir / f"{seq_id}.json"

                # Perturb sequence
                sample_data = base_signature + rng.normal(0, 0.05, base_signature.shape).astype(np.float32)
                np.save(npy_file, sample_data)

                meta = {
                    "lang": lang,
                    "concept": concept,
                    "signer_id": signer,
                    "source": "synthetic_generator",
                    "frames": SEQUENCE_LENGTH,
                    "features": FEATURE_DIM,
                }
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                total_created += 1

    print(f"Generated {total_created} synthetic sign sequences across 3 signers in {target_dir}.")
    return total_created
