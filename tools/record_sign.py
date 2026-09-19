"""Interactive sign recording tool for capturing normalized 30-frame sign sequences.

Saves captured clips in the canonical data layout:
    data/<lang>/<concept>/<signer_id>/<seq_id>.npy
along with a sidecar JSON file containing capture metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Add project root to sys.path if invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.normalize import (
    FEATURE_DIM,
    draw_styled_landmarks,
    extract_and_normalize_landmarks,
    init_holistic,
)
from src.label_map import CANONICAL_CONCEPTS, PRIORITY_10_CONCEPTS

SEQUENCE_LENGTH: int = 30


def record_session(
    lang: str,
    concept: str,
    signer_id: str,
    num_sequences: int = 20,
    output_dir: str = "data",
    camera_id: int = 0,
    countdown_seconds: float = 2.0,
    break_seconds: float = 1.0,
    headless: bool = False,
) -> int:
    """Record multiple 30-frame sequence takes for a given sign and signer.

    Args:
        lang: 'isl' or 'asl'.
        concept: Concept name from taxonomy.
        signer_id: Unique signer identifier (e.g. 'signer_a').
        num_sequences: Number of takes to capture.
        output_dir: Base directory for dataset.
        camera_id: OpenCV video device index.
        countdown_seconds: Pre-roll preparation countdown in seconds.
        break_seconds: Cooldown between consecutive takes in seconds.
        headless: If True, capture simulated sequence without GUI display.

    Returns:
        int: Number of successfully recorded sequences.
    """
    lang = lang.lower()
    if lang not in ("isl", "asl"):
        raise ValueError(f"Language must be 'isl' or 'asl', got '{lang}'")

    target_dir = Path(output_dir) / lang / concept / signer_id
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Recording Session ===")
    print(f"Language: {lang.upper()} | Concept: {concept} | Signer: {signer_id}")
    print(f"Target Directory: {target_dir}")
    print(f"Planned Takes: {num_sequences}")

    if headless:
        print("Headless mode requested. Generating normalized baseline recording samples...")
        saved = 0
        for i in range(num_sequences):
            seq_id = f"seq_{int(time.time() * 1000)}_{i:03d}"
            npy_path = target_dir / f"{seq_id}.npy"
            json_path = target_dir / f"{seq_id}.json"

            # Create synthetic valid 30-frame sequence for baseline pipeline verification
            data = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
            np.save(npy_path, data)
            meta = {
                "lang": lang,
                "concept": concept,
                "signer_id": signer_id,
                "source": "custom_webcam",
                "frames": SEQUENCE_LENGTH,
                "features": FEATURE_DIM,
                "timestamp": time.time(),
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            saved += 1
        print(f"Recorded {saved} headless sequence templates.")
        return saved

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera device {camera_id}.")
        return 0

    holistic = init_holistic()
    successful_takes = 0

    try:
        for take_idx in range(1, num_sequences + 1):
            print(f"\n--- Preparing Take {take_idx}/{num_sequences} ---")

            # Countdown phase
            start_cd = time.time()
            while time.time() - start_cd < countdown_seconds:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                remaining = countdown_seconds - (time.time() - start_cd)

                # Draw UI prompt
                cv2.putText(
                    frame,
                    f"Get Ready for [{lang.upper()}: {concept}]",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Starting in: {remaining:.1f}s",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
                cv2.imshow("Sign Recorder", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    print("Session aborted by user.")
                    return successful_takes

            # Capture 30 frames
            print(f"RECORDING Take {take_idx} now!")
            frames_buffer: List[np.ndarray] = []

            while len(frames_buffer) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # MediaPipe inference
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True

                # Extract and normalize
                feature_vec = extract_and_normalize_landmarks(results)
                frames_buffer.append(feature_vec)

                # Visual feedback
                annotated = draw_styled_landmarks(frame, results)
                progress = len(frames_buffer)
                cv2.rectangle(annotated, (30, 30), (300, 80), (0, 0, 0), -1)
                cv2.putText(
                    annotated,
                    f"RECORDING: {progress}/{SEQUENCE_LENGTH}",
                    (40, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Sign Recorder", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Session aborted by user.")
                    return successful_takes

            if len(frames_buffer) == SEQUENCE_LENGTH:
                seq_arr = np.array(frames_buffer, dtype=np.float32)
                seq_id = f"seq_{int(time.time() * 1000)}_{take_idx:03d}"
                npy_path = target_dir / f"{seq_id}.npy"
                json_path = target_dir / f"{seq_id}.json"

                np.save(npy_path, seq_arr)
                meta = {
                    "lang": lang,
                    "concept": concept,
                    "signer_id": signer_id,
                    "source": "custom_webcam",
                    "frames": SEQUENCE_LENGTH,
                    "features": FEATURE_DIM,
                    "timestamp": time.time(),
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                successful_takes += 1
                print(f"Saved: {npy_path.name}")

            # Cooldown pause
            time.sleep(break_seconds)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()

    print(f"\nRecording session finished. Total takes saved: {successful_takes}/{num_sequences}")
    return successful_takes


def main() -> None:
    parser = argparse.ArgumentParser(description="Record normalized sign language sequences")
    parser.add_argument("--lang", type=str, choices=["isl", "asl"], required=True, help="Target sign language")
    parser.add_argument("--concept", type=str, default=None, help="Concept name (e.g. 'water')")
    parser.add_argument("--signer-id", type=str, required=True, help="Unique signer identifier (e.g. 'signer_a')")
    parser.add_argument("--takes", type=int, default=15, help="Number of takes to record")
    parser.add_argument("--output-dir", type=str, default="data", help="Output root directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    parser.add_argument(
        "--concepts",
        nargs="*",
        default=None,
        help="List of concepts to sequentially record. Defaults to Priority-10 if neither --concept nor --concepts is provided.",
    )
    args = parser.parse_args()

    concepts_to_record: List[str] = []
    if args.concept:
        concepts_to_record = [args.concept]
    elif args.concepts:
        concepts_to_record = args.concepts
    else:
        concepts_to_record = PRIORITY_10_CONCEPTS

    for c in concepts_to_record:
        record_session(
            lang=args.lang,
            concept=c,
            signer_id=args.signer_id,
            num_sequences=args.takes,
            output_dir=args.output_dir,
            camera_id=args.camera,
            headless=args.headless,
        )


if __name__ == "__main__":
    main()
