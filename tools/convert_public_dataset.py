"""Dataset conversion utility for public sign language corpuses (INCLUDE, WLASL, ASL Citizen).

Converts raw video files into standardized (30, 258) normalized numpy sequences and sidecar JSON metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.normalize import (
    FEATURE_DIM,
    extract_and_normalize_landmarks,
    init_holistic,
)
from src.label_map import CANONICAL_CONCEPTS

SEQUENCE_LENGTH: int = 30


def sample_video_frames(video_path: Path, target_frame_count: int = SEQUENCE_LENGTH) -> List[np.ndarray]:
    """Read a video file and uniformly resample it to target_frame_count frames.

    Args:
        video_path: Path to video file.
        target_frame_count: Target number of frames (30).

    Returns:
        List of BGR frames of length target_frame_count.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError(f"Video contains 0 frames: {video_path}")

    # Uniform temporal resampling
    indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)
    resampled = [frames[idx] for idx in indices]
    return resampled


def process_video_to_sequence(
    video_path: Path, holistic: Any
) -> np.ndarray:
    """Process a single video into a normalized (30, 258) array."""
    frames = sample_video_frames(video_path, target_frame_count=SEQUENCE_LENGTH)
    features_list: List[np.ndarray] = []

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        vec = extract_and_normalize_landmarks(results)
        features_list.append(vec)

    seq_arr = np.array(features_list, dtype=np.float32)
    return seq_arr


def parse_include_layout(source_dir: Path) -> List[Tuple[Path, str, str, str]]:
    """Parse INCLUDE dataset folder convention:

    Expected: source_dir/<concept>/<signer_id>_<clip_id>.mp4 or source_dir/<concept>/<clip>.mp4
    Returns: list of (video_path, concept, signer_id, 'include')
    """
    items = []
    for video_path in source_dir.rglob("*.mp4"):
        concept = video_path.parent.name.lower().replace(" ", "_")
        # Extract signer ID from filename if formatted as signer01_clip02.mp4
        parts = video_path.stem.split("_")
        signer_id = parts[0] if len(parts) > 1 and "signer" in parts[0].lower() else "include_signer"
        items.append((video_path, concept, signer_id, "include"))
    return items


def parse_wlasl_layout(source_dir: Path, json_index_path: Optional[Path] = None) -> List[Tuple[Path, str, str, str]]:
    """Parse WLASL dataset folder convention:

    Expected: source_dir/<concept>/<video_id>.mp4 or video folder indexed by WLASL_v0.3.json.
    """
    items = []
    if json_index_path and json_index_path.exists():
        with open(json_index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            gloss = entry.get("gloss", "").lower().replace(" ", "_")
            for instance in entry.get("instances", []):
                vid_id = instance.get("video_id")
                signer_id = f"wlasl_signer_{instance.get('signer_id', 'unknown')}"
                vid_path = source_dir / f"{vid_id}.mp4"
                if vid_path.exists():
                    items.append((vid_path, gloss, signer_id, "wlasl"))
    else:
        # Direct folder structure: source_dir/<concept>/<clip>.mp4
        for video_path in source_dir.rglob("*.mp4"):
            concept = video_path.parent.name.lower().replace(" ", "_")
            signer_id = f"wlasl_{video_path.stem}"
            items.append((video_path, concept, signer_id, "wlasl"))

    return items


def parse_asl_citizen_layout(source_dir: Path) -> List[Tuple[Path, str, str, str]]:
    """Parse ASL Citizen convention:

    Expected: source_dir/<concept>/<user_id>_<instance>.mp4
    """
    items = []
    for video_path in source_dir.rglob("*.mp4"):
        concept = video_path.parent.name.lower().replace(" ", "_")
        signer_id = video_path.stem.split("_")[0] if "_" in video_path.stem else "asl_citizen_signer"
        items.append((video_path, concept, signer_id, "asl_citizen"))
    return items


def convert_dataset(
    source_dir: Path,
    source_format: str,
    target_lang: str,
    output_dir: Path,
    allowed_concepts: Optional[List[str]] = None,
    index_json: Optional[Path] = None,
) -> int:
    """Batch convert external dataset into canonical data/ directory structure."""
    source_format = source_format.lower()
    target_lang = target_lang.lower()

    if source_format == "include":
        entries = parse_include_layout(source_dir)
    elif source_format == "wlasl":
        entries = parse_wlasl_layout(source_dir, index_json)
    elif source_format in ("asl_citizen", "aslcitizen"):
        entries = parse_asl_citizen_layout(source_dir)
    else:
        raise ValueError(f"Unsupported source format: {source_format}")

    # Filter concepts
    filtered = []
    for v_path, concept, signer_id, src in entries:
        if allowed_concepts and concept not in allowed_concepts:
            continue
        filtered.append((v_path, concept, signer_id, src))

    print(f"Found {len(filtered)} matching videos for conversion.")
    if not filtered:
        return 0

    holistic = init_holistic(static_image_mode=False)
    converted_count = 0

    try:
        for v_path, concept, signer_id, src in tqdm(filtered, desc="Converting videos"):
            dest_dir = output_dir / target_lang / concept / signer_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            seq_id = f"conv_{src}_{v_path.stem}"
            npy_path = dest_dir / f"{seq_id}.npy"
            json_path = dest_dir / f"{seq_id}.json"

            if npy_path.exists():
                continue

            try:
                seq_data = process_video_to_sequence(v_path, holistic)
                np.save(npy_path, seq_data)

                meta = {
                    "lang": target_lang,
                    "concept": concept,
                    "signer_id": signer_id,
                    "source": src,
                    "original_video": str(v_path),
                    "frames": SEQUENCE_LENGTH,
                    "features": FEATURE_DIM,
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                converted_count += 1
            except Exception as e:
                print(f"Warning: Failed to convert {v_path}: {e}")
    finally:
        holistic.close()

    print(f"Successfully converted {converted_count} video sequences.")
    return converted_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert external video datasets to canonical 30-frame npy sequences")
    parser.add_argument("--source-dir", type=str, required=True, help="Input directory containing raw dataset videos")
    parser.add_argument(
        "--source-format",
        type=str,
        choices=["include", "wlasl", "asl_citizen"],
        required=True,
        help="Format layout of source dataset",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["isl", "asl"],
        required=True,
        help="Target sign language tag (isl for INCLUDE, asl for WLASL/ASL Citizen)",
    )
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory root (default: data)")
    parser.add_argument("--index-json", type=str, default=None, help="Optional index JSON file (e.g. WLASL_v0.3.json)")
    parser.add_argument(
        "--concepts",
        nargs="*",
        default=None,
        help="Concept filter list. Defaults to all 20 canonical concepts.",
    )
    args = parser.parse_args()

    allowed_concepts = args.concepts if args.concepts else CANONICAL_CONCEPTS
    convert_dataset(
        source_dir=Path(args.source_dir),
        source_format=args.source_format,
        target_lang=args.lang,
        output_dir=Path(args.output_dir),
        allowed_concepts=allowed_concepts,
        index_json=Path(args.index_json) if args.index_json else None,
    )


if __name__ == "__main__":
    main()
