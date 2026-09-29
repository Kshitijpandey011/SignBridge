"""Single-command automated pipeline for sign language model training and evaluation.

Orchestrates:
1. tools/fetch_public_clips.py: Auto-fetches matching public clips (skipped if cached, unless --refresh).
2. tools/convert_public_dataset.py: Converts raw videos into standardized (30, 258) .npy arrays.
3. train.py: Fits the multi-task LSTM model on all available sequences.
4. evaluate.py: Evaluates model on held-out signers and produces eval_report.json.
5. export_tflite.py: Exports model to optimized TFLite for edge deployment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.label_map import CANONICAL_CONCEPTS, PRIORITY_10_CONCEPTS


def run_command_logged(cmd: List[str], step_name: str) -> bool:
    """Run a subprocess command and stream output."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {step_name}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {res.returncode}")
        return False
    return True


def inspect_dataset_distribution(data_dir: Path, target_concepts: List[str]) -> Dict[str, Any]:
    """Inspect data/ directory to categorize data sources per concept."""
    distribution: Dict[str, Dict[str, Any]] = {"isl": {}, "asl": {}}

    for lang in ("isl", "asl"):
        for concept in target_concepts:
            concept_path = data_dir / lang / concept
            public_count = 0
            human_count = 0

            if concept_path.exists():
                for npy_file in concept_path.rglob("*.npy"):
                    json_file = npy_file.with_suffix(".json")
                    is_public = False
                    if json_file.exists():
                        try:
                            with open(json_file, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                                if meta.get("source") in ("raw_clip", "wlasl", "asl_citizen", "include", "public_clip"):
                                    is_public = True
                        except Exception:
                            pass
                    if is_public or "conv_" in npy_file.name or "public" in str(npy_file):
                        public_count += 1
                    else:
                        human_count += 1

            distribution[lang][concept] = {
                "public_samples": public_count,
                "human_samples": human_count,
                "total": public_count + human_count,
            }

    return distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end automated sign model training pipeline")
    parser.add_argument(
        "--concepts",
        nargs="*",
        default=None,
        help="Concepts subset to include (defaults to all 20 canonical concepts).",
    )
    parser.add_argument("--priority-10", action="store_true", help="Restrict pipeline to Priority-10 concepts")
    parser.add_argument("--refresh", action="store_true", help="Force refetch of public clips even if cached")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for train.py")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip public clip fetching step")
    parser.add_argument("--val-signer", type=str, default=None, help="Explicit held-out validation signer ID")
    args = parser.parse_args()

    target_concepts = CANONICAL_CONCEPTS
    if args.priority_10:
        target_concepts = PRIORITY_10_CONCEPTS
    elif args.concepts:
        target_concepts = args.concepts

    py_exe = sys.executable
    fetch_report_path = PROJECT_ROOT / "fetch_report.json"
    raw_videos_dir = PROJECT_ROOT / "raw_videos"
    data_dir = PROJECT_ROOT / "data"

    print("\n" + "=" * 60)
    print("🚀 LAUNCHING AUTOMATED SIGN RECOGNITION PIPELINE")
    print(f"Target Concepts ({len(target_concepts)}): {target_concepts}")
    print("=" * 60)

    # 1. Fetch public clips
    if not args.skip_fetch:
        if fetch_report_path.exists() and not args.refresh:
            print("\n[INFO] Cached fetch_report.json found. Skipping fetch step (pass --refresh to refetch).")
        else:
            fetch_cmd = [
                py_exe,
                str(PROJECT_ROOT / "tools" / "fetch_public_clips.py"),
                "--concepts",
                *target_concepts,
            ]
            if args.refresh:
                fetch_cmd.append("--refresh")
            if not run_command_logged(fetch_cmd, "Fetch Public Clips"):
                print("[WARNING] Fetch step had warnings or errors. Continuing with available clips.")

    # 2. Convert public dataset / raw videos
    if raw_videos_dir.exists():
        convert_cmd = [
            py_exe,
            str(PROJECT_ROOT / "tools" / "convert_public_dataset.py"),
            "--source-dir",
            str(raw_videos_dir),
            "--source-format",
            "auto",
            "--concepts",
            *target_concepts,
        ]
        run_command_logged(convert_cmd, "Convert Video Datasets to Landmarks")

    # 3. Train model
    train_cmd = [
        py_exe,
        str(PROJECT_ROOT / "train.py"),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--concepts",
        *target_concepts,
    ]
    if args.val_signer:
        train_cmd.extend(["--val-signer", args.val_signer])

    if not run_command_logged(train_cmd, "Model Training"):
        print("[ERROR] Training failed. Halting pipeline.")
        sys.exit(1)

    # 4. Evaluate model
    eval_cmd = [
        py_exe,
        str(PROJECT_ROOT / "evaluate.py"),
    ]
    if args.val_signer:
        eval_cmd.extend(["--val-signer", args.val_signer])
    run_command_logged(eval_cmd, "Model Evaluation")

    # 5. Export to TFLite
    export_cmd = [
        py_exe,
        str(PROJECT_ROOT / "export_tflite.py"),
    ]
    run_command_logged(export_cmd, "Export TFLite Model")

    # 6. Final Summary Report
    distribution = inspect_dataset_distribution(data_dir, target_concepts)

    fetch_report = {}
    if fetch_report_path.exists():
        try:
            with open(fetch_report_path, "r", encoding="utf-8") as f:
                fetch_report = json.load(f)
        except Exception:
            pass

    eval_report = {}
    eval_report_path = PROJECT_ROOT / "eval_report.json"
    if eval_report_path.exists():
        try:
            with open(eval_report_path, "r", encoding="utf-8") as f:
                eval_report = json.load(f)
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print("\n[Data Sourcing Breakdown per Concept]:")
    for lang in ("isl", "asl"):
        print(f"\n  --- {lang.upper()} ---")
        for concept in target_concepts:
            stat = distribution[lang].get(concept, {"public_samples": 0, "human_samples": 0, "total": 0})
            pub = stat["public_samples"]
            hum = stat["human_samples"]
            tot = stat["total"]
            status = "✅ Trained" if tot > 0 else "⚠️ No Data (Use tools/record_sign.py to add clips)"
            print(f"  • {concept:<12}: Total={tot:<3} (Public: {pub}, Human: {hum}) -> {status}")

    unmatched_isl = fetch_report.get("unmatched", {}).get("isl", [])
    unmatched_asl = fetch_report.get("unmatched", {}).get("asl", [])
    if unmatched_isl or unmatched_asl:
        print("\n[Concepts without public dataset clips]:")
        if unmatched_isl:
            print(f"  • ISL Unmatched: {unmatched_isl}")
        if unmatched_asl:
            print(f"  • ASL Unmatched: {unmatched_asl}")
        print("  Tip: Add recordings via webcam: python tools/record_sign.py --lang isl --concept <name>")

    if eval_report:
        print("\n[Validation Metrics]:")
        print(f"  • Overall Held-Out Signer Accuracy: {eval_report.get('overall_accuracy', 0)*100:.2f}%")
        print(f"  • Concept-Level Accuracy          : {eval_report.get('concept_level_accuracy', 0)*100:.2f}%")
        print(f"  • Language Identification Accuracy: {eval_report.get('language_detection_accuracy', 0)*100:.2f}%")
        print(f"  • Prediction Latency               : {eval_report.get('latency_ms', 0):.2f} ms")

    print("\n[Artifacts Produced]:")
    print(f"  • Keras Model   : {PROJECT_ROOT / 'models' / 'model.keras'}")
    print(f"  • TFLite Model  : {PROJECT_ROOT / 'models' / 'model.tflite'}")
    print(f"  • Eval Report   : {eval_report_path}")
    print(f"  • Confusion Mtx : {PROJECT_ROOT / 'confusion_matrix.png'}")
    print("=" * 60)
    print("✅ Automated pipeline finished successfully.\n")


if __name__ == "__main__":
    main()
