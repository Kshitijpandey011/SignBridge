import json
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_checks():
    print("--- 1. Testing Feature & Normalization Parity ---")
    from src.features.normalize import FEATURE_DIM, POSE_DIM, HAND_DIM
    assert FEATURE_DIM == 258, f"Expected 258 features, got {FEATURE_DIM}"
    assert POSE_DIM == 132, f"Expected 132 pose features, got {POSE_DIM}"
    assert HAND_DIM == 63, f"Expected 63 hand features, got {HAND_DIM}"

    print("--- 2. Testing Buffer & Temporal Shapes ---")
    from src.features.buffer import SequenceBuffer
    buf = SequenceBuffer(30, 258)
    for _ in range(30):
        buf.append(np.zeros(258, dtype=np.float32))
    seq = buf.get_sequence()
    assert seq.shape == (30, 258), f"Expected (30, 258), got {seq.shape}"

    print("--- 3. Testing Label Taxonomy ---")
    from src.label_map import DEFAULT_REGISTRY
    assert DEFAULT_REGISTRY.num_classes == 40, f"Expected 40 classes, got {DEFAULT_REGISTRY.num_classes}"
    assert len(DEFAULT_REGISTRY.isl_idx) == 20
    assert len(DEFAULT_REGISTRY.asl_idx) == 20

    print("--- 4. Testing Augmentations ---")
    from src.augment import augment_sequence
    aug = augment_sequence(seq)
    assert aug.shape == (30, 258)

    print("--- 5. Testing Model Weights & Topologies ---")
    from src.model import load_inference_model
    model = load_inference_model("models/model.keras")
    assert model.input_shape == (None, 30, 258)
    assert len(model.outputs) == 2
    assert model.outputs[0].shape[-1] == 40
    assert model.outputs[1].shape[-1] == 2

    print("--- 6. Testing Inference Mode & Stabilization ---")
    from src.inference.mode import InferenceModeManager, apply_language_mask
    from src.inference.stabilize import PredictionStabilizer
    mgr = InferenceModeManager(DEFAULT_REGISTRY.isl_idx, DEFAULT_REGISTRY.asl_idx)
    probs = np.ones(40) / 40.0
    masked_isl = apply_language_mask(probs, "isl", DEFAULT_REGISTRY.isl_idx, DEFAULT_REGISTRY.asl_idx)
    assert np.all(masked_isl[DEFAULT_REGISTRY.asl_idx] == 0.0)

    stab = PredictionStabilizer(voting_window=5, fixed_mode_threshold=0.5)
    f_dummy = np.zeros(258, dtype=np.float32)
    f_dummy[140] = 1.0  # hand active
    p_dummy = np.zeros(40, dtype=np.float32)
    p_dummy[0] = 0.99
    # Feed frames with simulated motion
    out = None
    for i in range(10):
        f_in = f_dummy.copy()
        f_in[140] += i * 0.1  # motion energy
        res = stab.process(f_in, p_dummy, mode="isl", current_time=100.0 + i * 0.5)
        if res is not None:
            out = res
    assert out is not None and out[0] == 0

    print("--- 7. Testing Phrase Building & Translation ---")
    from src.phrasebuilder import PhraseBuilder
    from src.translate import TranslationEngine
    pb = PhraseBuilder()
    phrase = pb.build_sentence(["help", "water"])
    assert "help" in phrase.lower() and "water" in phrase.lower()

    te = TranslationEngine(phrase_builder=pb)
    trans_hi, tag_hi = te.translate(["help", "water"], "hi")
    assert trans_hi != ""
    assert tag_hi in ("curated", "machine", "fallback-english")

    print("--- 8. Testing TFLite & Evaluation Assets ---")
    assert Path("models/model.tflite").exists()
    assert Path("eval_report.json").exists()
    assert Path("confusion_matrix.png").exists()

    print("\nALL SYSTEM INTEGRATION CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_checks()
