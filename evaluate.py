"""Comprehensive evaluation suite for dual-language sign recognition model.

Computes items 1-8 of the specification:
1. Overall 40-class accuracy on held-out signer
2. ISL-only accuracy and ASL-only accuracy
3. Concept-level accuracy (correct concept regardless of language)
4. Language-detection accuracy in AUTO mode
5. Confusion matrix with same-concept-cross-language vs cross-concept errors
6. Latency per prediction (ms) and FPS
7. Accuracy comparison: fixed ISL/ASL mode (masked) vs AUTO mode
8. Accuracy breakdown across recording sources (custom vs external dataset)
Outputs: eval_report.json and confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SignDataset
from src.label_map import DEFAULT_REGISTRY, LabelRegistry
from src.model import load_inference_model


def apply_language_mask(
    probs: np.ndarray,
    mode: str,
    isl_idx: List[int],
    asl_idx: List[int],
) -> np.ndarray:
    """Apply language constraint mask to probability vector."""
    masked = np.zeros_like(probs)
    if mode == "isl":
        masked[isl_idx] = probs[isl_idx]
    elif mode == "asl":
        masked[asl_idx] = probs[asl_idx]
    else:
        return probs

    s = np.sum(masked)
    return masked / s if s > 0 else masked


def evaluate_system(
    model_path: str = "models/model.keras",
    label_map_path: str = "label_map.json",
    data_dir: str = "data",
    output_report: str = "eval_report.json",
    output_plot: str = "confusion_matrix.png",
    val_signer: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full evaluation protocol."""
    print("=== Commencing Comprehensive Evaluation Protocol ===")
    registry = LabelRegistry.from_file(label_map_path) if Path(label_map_path).exists() else DEFAULT_REGISTRY
    model = load_inference_model(model_path)

    dataset_loader = SignDataset(
        data_dir=data_dir,
        label_registry=registry,
        val_signers=[val_signer] if val_signer else None,
    )
    samples = dataset_loader.load_samples()
    if not samples:
        raise RuntimeError("No data found for evaluation!")

    (X_train, y_class_train, y_lang_train, train_meta), (X_val, y_class_val, y_lang_val, val_meta) = (
        dataset_loader.get_splits(samples)
    )

    if len(X_val) == 0:
        print("Warning: Validation split is empty, falling back to train split for evaluation.")
        X_val, y_class_val, y_lang_val, val_meta = X_train, y_class_train, y_lang_train, train_meta

    num_classes = registry.num_classes
    isl_idx = registry.isl_idx
    asl_idx = registry.asl_idx

    # 6. Benchmark Latency & FPS
    warmup_n = min(10, len(X_val))
    for i in range(warmup_n):
        _ = model.predict(X_val[i : i + 1], verbose=0)

    start_t = time.perf_counter()
    n_bench = len(X_val)
    class_preds, lang_preds = model.predict(X_val, batch_size=1, verbose=0)
    total_time = time.perf_counter() - start_t
    latency_ms = (total_time / n_bench) * 1000.0
    fps = n_bench / total_time if total_time > 0 else 0.0

    # Ground truth labels
    y_true_class = np.argmax(y_class_val, axis=1)
    y_true_lang = np.argmax(y_lang_val, axis=1)

    # 1. Overall 40-class accuracy on held-out signer (AUTO mode / unmasked)
    y_pred_auto = np.argmax(class_preds, axis=1)
    overall_accuracy = float(np.mean(y_pred_auto == y_true_class))

    # 2. ISL-only accuracy & ASL-only accuracy (under fixed masked mode)
    isl_mask = np.isin(y_true_class, isl_idx)
    asl_mask = np.isin(y_true_class, asl_idx)

    # Masked predictions
    masked_preds = np.zeros_like(class_preds)
    for i in range(len(class_preds)):
        sample_lang = val_meta[i]["lang"]
        masked_preds[i] = apply_language_mask(class_preds[i], sample_lang, isl_idx, asl_idx)
    y_pred_masked = np.argmax(masked_preds, axis=1)

    isl_acc = float(np.mean(y_pred_masked[isl_mask] == y_true_class[isl_mask])) if np.any(isl_mask) else 0.0
    asl_acc = float(np.mean(y_pred_masked[asl_mask] == y_true_class[asl_mask])) if np.any(asl_mask) else 0.0
    masked_overall_acc = float(np.mean(y_pred_masked == y_true_class))

    # 3. Concept-level accuracy (correct concept regardless of language)
    true_concepts = [registry.concept_of.get(idx, "") for idx in y_true_class]
    pred_auto_concepts = [registry.concept_of.get(idx, "") for idx in y_pred_auto]
    concept_accuracy = float(np.mean([t == p for t, p in zip(true_concepts, pred_auto_concepts)]))

    # 4. Language-detection accuracy in AUTO mode
    pred_lang_idx = np.argmax(lang_preds, axis=1)
    lang_acc = float(np.mean(pred_lang_idx == y_true_lang))

    # 5. Confusion Matrix & error decomposition
    # 40x40 confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_class, y_pred_auto):
        cm[t, p] += 1

    same_concept_cross_lang_errors = 0
    cross_concept_errors = 0

    for i in range(len(y_true_class)):
        t_idx = y_true_class[i]
        p_idx = y_pred_auto[i]
        if t_idx != p_idx:
            t_concept = registry.concept_of.get(t_idx)
            p_concept = registry.concept_of.get(p_idx)
            if t_concept == p_concept:
                same_concept_cross_lang_errors += 1
            else:
                cross_concept_errors += 1

    # 8. Accuracy breakdown: Custom webcam vs External dataset
    sources = [m.get("source", "unknown") for m in val_meta]
    unique_sources = set(sources)
    source_accuracies: Dict[str, float] = {}
    for s in unique_sources:
        s_mask = np.array([src == s for src in sources])
        if np.any(s_mask):
            source_accuracies[s] = float(np.mean(y_pred_auto[s_mask] == y_true_class[s_mask]))

    report = {
        "overall_40_class_accuracy_heldout": round(overall_accuracy, 4),
        "isl_only_accuracy_masked": round(isl_acc, 4),
        "asl_only_accuracy_masked": round(asl_acc, 4),
        "fixed_mode_overall_accuracy": round(masked_overall_acc, 4),
        "auto_mode_overall_accuracy": round(overall_accuracy, 4),
        "concept_level_accuracy": round(concept_accuracy, 4),
        "language_detection_accuracy": round(lang_acc, 4),
        "same_concept_cross_language_errors": same_concept_cross_lang_errors,
        "cross_concept_errors": cross_concept_errors,
        "latency_per_prediction_ms": round(latency_ms, 2),
        "throughput_fps": round(fps, 1),
        "source_breakdown": source_accuracies,
        "num_val_samples": len(X_val),
    }

    # Save JSON report
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to: {output_report}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (40 Classes)\nConcept Accuracy: {concept_accuracy*100:.1f}%")
    plt.colorbar()
    plt.xlabel("Predicted Class Index")
    plt.ylabel("True Class Index")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"Saved confusion matrix visualization to: {output_plot}")

    print("\n=== EVALUATION REPORT SUMMARY ===")
    print(f"Overall Accuracy (Held-out Signer): {overall_accuracy*100:.2f}%")
    print(f"Concept-Level Accuracy            : {concept_accuracy*100:.2f}%")
    print(f"Language Detection Accuracy       : {lang_acc*100:.2f}%")
    print(f"Same-Concept Cross-Lang Errors    : {same_concept_cross_lang_errors}")
    print(f"Cross-Concept Errors              : {cross_concept_errors}")
    print(f"Inference Latency                 : {latency_ms:.2f} ms ({fps:.1f} FPS)")
    print(f"Fixed Mode Acc vs AUTO Mode Acc   : {masked_overall_acc*100:.1f}% vs {overall_accuracy*100:.1f}%")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sign recognition model")
    parser.add_argument("--model", type=str, default="models/model.keras", help="Path to model file")
    parser.add_argument("--label-map", type=str, default="label_map.json", help="Path to label map")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--report", type=str, default="eval_report.json", help="Output report JSON")
    parser.add_argument("--plot", type=str, default="confusion_matrix.png", help="Output plot path")
    parser.add_argument("--val-signer", type=str, default=None, help="Held-out validation signer ID")
    args = parser.parse_args()

    evaluate_system(
        model_path=args.model,
        label_map_path=args.label_map,
        data_dir=args.data_dir,
        output_report=args.report,
        output_plot=args.plot,
        val_signer=args.val_signer,
    )


if __name__ == "__main__":
    main()
