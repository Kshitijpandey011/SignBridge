"""Training pipeline for dual-language sign recognition model.

Loads signer-partitioned sequences, performs augmentations, compiles the multi-task
LSTM model, fits weights with early stopping, and validates on held-out signers.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.augment import augment_sequence
from src.dataset import SignDataset, create_synthetic_dataset
from src.label_map import (
    CANONICAL_CONCEPTS,
    LabelRegistry,
    PRIORITY_10_CONCEPTS,
    build_label_map,
)
from src.model import build_sign_model


def augment_training_split(
    X_train: np.ndarray,
    y_class_train: np.ndarray,
    y_lang_train: np.ndarray,
    multiplier: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand the training split through randomized spatial/temporal augmentations."""
    augmented_X = [X_train]
    augmented_yc = [y_class_train]
    augmented_yl = [y_lang_train]

    for _ in range(multiplier):
        aug_batch = np.zeros_like(X_train)
        for i in range(len(X_train)):
            aug_batch[i] = augment_sequence(X_train[i])
        augmented_X.append(aug_batch)
        augmented_yc.append(y_class_train)
        augmented_yl.append(y_lang_train)

    X_out = np.concatenate(augmented_X, axis=0)
    yc_out = np.concatenate(augmented_yc, axis=0)
    yl_out = np.concatenate(augmented_yl, axis=0)

    # Shuffle augmented dataset
    indices = np.arange(len(X_out))
    np.random.shuffle(indices)

    return X_out[indices], yc_out[indices], yl_out[indices]


def run_training(
    data_dir: str = "data",
    output_model_path: str = "models/model.keras",
    label_map_path: str = "label_map.json",
    epochs: int = 25,
    batch_size: int = 16,
    concepts: Optional[List[str]] = None,
    allow_synthetic: bool = True,
    val_signer: Optional[str] = None,
) -> tf.keras.Model:
    """Execute end-to-end model training run."""
    print("=== Starting Dual-Language Sign Model Training ===")

    # Initialize or load label map
    label_map_file = Path(label_map_path)
    if not label_map_file.exists():
        mapping = build_label_map(concepts)
        registry = LabelRegistry(mapping)
        registry.save(label_map_file)
    else:
        registry = LabelRegistry.from_file(label_map_file)

    num_classes = registry.num_classes
    print(f"Active classes: {num_classes} ({len(registry.isl_idx)} ISL, {len(registry.asl_idx)} ASL)")

    dataset_loader = SignDataset(
        data_dir=data_dir,
        label_registry=registry,
        val_signers=[val_signer] if val_signer else None,
    )
    samples = dataset_loader.load_samples(allowed_concepts=concepts)

    if not samples and allow_synthetic:
        print("No recordings found in data directory. Generating synthetic baseline sequences...")
        create_synthetic_dataset(target_dir=data_dir, concepts=concepts or CANONICAL_CONCEPTS)
        samples = dataset_loader.load_samples(allowed_concepts=concepts)

    if not samples:
        raise RuntimeError("No training samples available! Record clips via tools/record_sign.py")

    (X_train, y_class_train, y_lang_train, train_meta), (X_val, y_class_val, y_lang_val, val_meta) = (
        dataset_loader.get_splits(samples, allowed_concepts=concepts)
    )

    print(f"Raw training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Apply data augmentations
    X_train_aug, y_class_train_aug, y_lang_train_aug = augment_training_split(
        X_train, y_class_train, y_lang_train, multiplier=2
    )
    print(f"Expanded augmented training samples: {len(X_train_aug)}")

    # Build model
    model = build_sign_model(num_classes=num_classes)
    model.summary()

    # Output setup
    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_class_output_accuracy", patience=12, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_class_output_loss", factor=0.5, patience=5, min_lr=1e-5, mode="min"),
        ModelCheckpoint(filepath=str(out_path), monitor="val_class_output_accuracy", save_best_only=True, mode="max"),
    ]

    history = model.fit(
        X_train_aug,
        {"class_output": y_class_train_aug, "language_output": y_lang_train_aug},
        validation_data=(
            X_val,
            {"class_output": y_class_val, "language_output": y_lang_val},
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model bundle
    model.save(str(out_path))
    print(f"\nSaved trained model to: {out_path}")

    # Evaluate same-signer vs held-out signer accuracy explicitly via predictions
    train_class_preds, _ = model.predict(X_train, batch_size=batch_size, verbose=0)
    val_class_preds, _ = model.predict(X_val, batch_size=batch_size, verbose=0)

    same_signer_acc = float(np.mean(np.argmax(train_class_preds, axis=1) == np.argmax(y_class_train, axis=1))) * 100.0
    held_out_acc = float(np.mean(np.argmax(val_class_preds, axis=1) == np.argmax(y_class_val, axis=1))) * 100.0

    print("\n=======================================================")
    print(f"SAME-SIGNER TRAINING ACCURACY : {same_signer_acc:.2f}%")
    print(f"HELD-OUT SIGNER VAL ACCURACY  : {held_out_acc:.2f}%")
    print("=======================================================")
    print("Note: Always report held-out signer validation accuracy for fair generalization.")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-language sign recognition model")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory holding sign dataset")
    parser.add_argument("--output-model", type=str, default="models/model.keras", help="Path to save trained model")
    parser.add_argument("--label-map", type=str, default="label_map.json", help="Path to label_map.json")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--priority-10", action="store_true", help="Train on Priority-10 concepts only")
    parser.add_argument("--concepts", nargs="*", default=None, help="Custom concept list")
    parser.add_argument("--val-signer", type=str, default=None, help="Explicitly specify held-out validation signer ID")
    parser.add_argument("--no-synthetic", action="store_true", help="Disable synthetic dataset generation fallback")
    args = parser.parse_args()

    concepts = None
    if args.priority_10:
        concepts = PRIORITY_10_CONCEPTS
    elif args.concepts:
        concepts = args.concepts

    run_training(
        data_dir=args.data_dir,
        output_model_path=args.output_model,
        label_map_path=args.label_map,
        epochs=args.epochs,
        batch_size=args.batch_size,
        concepts=concepts,
        allow_synthetic=not args.no_synthetic,
        val_signer=args.val_signer,
    )


if __name__ == "__main__":
    main()
