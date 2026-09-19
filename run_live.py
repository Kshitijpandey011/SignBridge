"""Live real-time dual-language sign recognition, translation, and speech system.

Executes the unified runtime pipeline:
Webcam -> MediaPipe Holistic -> Feature Normalization -> Rolling Sequence Buffer
-> Multi-Task LSTM -> Language Mode Masking & Soft-Lock Auto-Detection
-> Prediction Stabilization & Motion Energy Gating -> Concept Phrase Builder
-> Multi-Tier Spoken Translation Fallback -> Offline TTS Speech Playback & HUD Overlay
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.buffer import SequenceBuffer
from src.features.normalize import (
    draw_styled_landmarks,
    extract_and_normalize_landmarks,
    init_holistic,
)
from src.inference.mode import InferenceModeManager
from src.inference.stabilize import PredictionStabilizer
from src.label_map import DEFAULT_REGISTRY, LabelRegistry
from src.model import load_inference_model
from src.phrasebuilder import PhraseBuilder
from src.translate import TranslationEngine
from src.tts import get_tts_engine


def run_live_inference(
    model_path: str = "models/model.keras",
    label_map_path: str = "label_map.json",
    mode: str = "auto",
    target_lang: str = "en",
    camera_id: int = 0,
    enable_tts: bool = True,
    headless_frames: int = 0,
) -> None:
    """Run real-time inference loop."""
    print("=== Launching Live Dual-Language Sign Language Translator ===")

    registry = LabelRegistry.from_file(label_map_path) if Path(label_map_path).exists() else DEFAULT_REGISTRY
    model = load_inference_model(model_path)

    buffer = SequenceBuffer(sequence_length=30, feature_dim=258)
    mode_mgr = InferenceModeManager(isl_idx=registry.isl_idx, asl_idx=registry.asl_idx)
    mode_mgr.set_mode(mode)

    stabilizer = PredictionStabilizer(
        voting_window=10,
        fixed_mode_threshold=0.90,
        auto_mode_threshold=0.80,
    )
    phrase_builder = PhraseBuilder()
    translator = TranslationEngine(phrase_builder=phrase_builder)
    tts = get_tts_engine() if enable_tts else None

    history_log: Deque[str] = deque(maxlen=6)
    last_hud_prediction = "Listening..."
    last_hud_translation = ""
    last_hud_tag = ""

    if headless_frames > 0:
        print(f"Running in headless verification mode for {headless_frames} frames...")
        dummy_feat = np.zeros(258, dtype=np.float32)
        dummy_feat[132:195] = 0.5  # Simulate hands
        for f in range(headless_frames):
            buffer.append(dummy_feat)
            if buffer.is_ready():
                seq = np.expand_dims(buffer.get_sequence(), axis=0)
                class_preds, lang_preds = model.predict(seq, verbose=0)
                masked_probs, detected_lang, is_locked = mode_mgr.update(class_preds[0], lang_preds[0])
                stb_result = stabilizer.process(dummy_feat, masked_probs, mode=mode_mgr.current_mode)
                if stb_result:
                    c_idx, conf = stb_result
                    concept = registry.concept_of[c_idx]
                    phrase_builder.add_concept(concept)
                    trans, tag = translator.translate(phrase_builder.get_buffered_concepts(), target_lang)
                    history_log.append(f"[{detected_lang.upper()}] {concept.capitalize()} ({conf:.2f}) -> {trans}")
        print("Headless verification completed successfully.")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Unable to open camera device index {camera_id}.")
        return

    holistic = init_holistic()
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
            prev_time = curr_time

            # MediaPipe extraction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            annotated = draw_styled_landmarks(frame, results)
            feature_vector = extract_and_normalize_landmarks(results)
            buffer.append(feature_vector)

            detected_lang = "auto"
            is_locked = False

            if buffer.is_ready():
                seq = np.expand_dims(buffer.get_sequence(), axis=0)
                class_preds, lang_preds = model.predict(seq, verbose=0)
                class_probs = class_preds[0]
                lang_probs = lang_preds[0] if lang_preds is not None else None

                masked_probs, detected_lang, is_locked = mode_mgr.update(class_probs, lang_probs)
                stabilized = stabilizer.process(feature_vector, masked_probs, mode=mode_mgr.current_mode)

                if stabilized is not None:
                    class_idx, confidence = stabilized
                    concept = registry.concept_of.get(class_idx, "unknown")
                    lang_tag = registry.lang_of.get(class_idx, detected_lang).upper()

                    last_hud_prediction = f"[{lang_tag}] {concept.capitalize()} ({confidence:.2f})"

                    # Phrase building & translation
                    phrase_builder.add_concept(concept)
                    current_phrase = phrase_builder.get_current_phrase()
                    translated_text, quality_tag = translator.translate(
                        phrase_builder.get_buffered_concepts(), target_lang
                    )
                    last_hud_translation = f"{translated_text}"
                    last_hud_tag = quality_tag

                    log_entry = f"{last_hud_prediction} -> {translated_text} ({quality_tag})"
                    history_log.append(log_entry)

                    # Speech synthesis
                    if tts:
                        tts.speak_async(translated_text, target_lang)

            # Draw HUD Overlays
            # Top Banner
            cv2.rectangle(annotated, (0, 0), (w, 60), (30, 30, 30), -1)
            lock_label = " (Locked)" if is_locked else ""
            mode_display = f"Mode: {mode_mgr.current_mode.upper()}{lock_label} | Sign Lang: {detected_lang.upper()}"
            cv2.putText(annotated, mode_display, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(
                annotated,
                f"Target: {target_lang.upper()} | FPS: {fps:.1f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            # Bottom Recognition Box
            cv2.rectangle(annotated, (15, h - 130), (w - 15, h - 15), (20, 20, 20), -1)
            cv2.putText(
                annotated,
                f"Detected: {last_hud_prediction}",
                (30, h - 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (100, 255, 100),
                2,
            )

            if last_hud_translation:
                tag_color = (0, 255, 0) if "curated" in last_hud_tag else (0, 165, 255)
                cv2.putText(
                    annotated,
                    f"Translated [{last_hud_tag}]: {last_hud_translation}",
                    (30, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    tag_color,
                    2,
                )

            # Text Log in Top Right
            cv2.rectangle(annotated, (w - 360, 70), (w - 10, 200), (0, 0, 0), -1)
            cv2.putText(annotated, "Recent Signs:", (w - 350, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            for i, item in enumerate(list(history_log)[-4:]):
                cv2.putText(
                    annotated, item, (w - 350, 115 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
                )

            cv2.imshow("Dual-Language Sign Translator (ISL + ASL)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("1"):
                mode_mgr.set_mode("isl")
            elif key == ord("2"):
                mode_mgr.set_mode("asl")
            elif key == ord("3"):
                mode_mgr.set_mode("auto")
            elif key == ord("c"):
                phrase_builder.reset()
                last_hud_translation = ""

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live real-time dual-language sign translator")
    parser.add_argument("--model", type=str, default="models/model.keras", help="Trained model path")
    parser.add_argument("--label-map", type=str, default="label_map.json", help="Label map path")
    parser.add_argument("--mode", type=str, choices=["isl", "asl", "auto"], default="auto", help="Inference mode")
    parser.add_argument("--target-lang", type=str, default="en", help="Target spoken language (en, hi, ta, es, fr)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--no-tts", action="store_true", help="Disable audio speech synthesis")
    parser.add_argument("--headless-frames", type=int, default=0, help="Simulate frames without GUI display")
    args = parser.parse_args()

    run_live_inference(
        model_path=args.model,
        label_map_path=args.label_map,
        mode=args.mode,
        target_lang=args.target_lang,
        camera_id=args.camera,
        enable_tts=not args.no_tts,
        headless_frames=args.headless_frames,
    )


if __name__ == "__main__":
    main()
