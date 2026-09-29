import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.mock_sequence import create_mock_sequence
from src.features.kinematics import KinematicGestureClassifier
from src.inference.mode import InferenceModeManager
from src.label_map import DEFAULT_REGISTRY

def test_auto_mode_tracking():
    kin_clf = KinematicGestureClassifier()
    mode_mgr = InferenceModeManager(DEFAULT_REGISTRY.isl_idx, DEFAULT_REGISTRY.asl_idx)
    mode_mgr.set_mode("auto")

    test_cases = [
        ("isl_please", "please", "isl", dict(gesture="namaste", shape="open", zone="chest", motion="still", two_hands=True)),
        ("isl_hello", "hello", "isl", dict(shape="open", zone="chest", motion="waving")),
        ("isl_home", "home", "isl", dict(gesture="roof", shape="open", zone="chest", motion="still", two_hands=True)),
        ("isl_sorry", "sorry", "isl", dict(shape="bunched", zone="head", motion="still", two_hands=True)),
        ("isl_no", "no", "isl", dict(shape="index", zone="chin", motion="waving")),
        ("isl_pain", "pain", "isl", dict(shape="index", zone="chest", motion="still")),
        ("isl_doctor", "doctor", "isl", dict(gesture="doctor", shape="index", zone="chest", motion="still", two_hands=True)),
        ("isl_school", "school", "isl", dict(gesture="school", shape="open", zone="chest", motion="waving", two_hands=True)),
        ("asl_hello", "hello", "asl", dict(shape="open", zone="head", motion="waving")),
        ("asl_please", "please", "asl", dict(shape="open", zone="chest", motion="circular")),
        ("asl_sorry", "sorry", "asl", dict(shape="fist", zone="chest", motion="circular")),
        ("asl_water", "water", "asl", dict(shape="w", zone="chin", motion="still")),
        ("asl_home", "home", "asl", dict(shape="bunched", zone="head", motion="still")),
        ("asl_yes", "yes", "asl", dict(shape="fist", zone="chest", motion="nodding")),
        ("asl_help", "help", "asl", dict(shape="thumbs_up", zone="chest", motion="still")),
        ("asl_pain", "pain", "asl", dict(shape="index", zone="chest", motion="still", two_hands=True)),
    ]

    passed = 0
    for tag, expected_concept, expected_lang, params in test_cases:
        seq = create_mock_sequence(**params)
        is_valid, concept, conf, lang_hint = kin_clf.evaluate_absolute_sign(seq)
        kin_scores = kin_clf.score_sequence(seq)
        
        masked_probs, detected_lang, is_locked = mode_mgr.update(kin_scores, None)
        top_idx = int(np.argmax(masked_probs))
        top_concept = DEFAULT_REGISTRY.concept_of.get(top_idx, "")

        is_lang_correct = (detected_lang == expected_lang)
        is_concept_correct = (concept == expected_concept)
        assert is_lang_correct and is_concept_correct, f"Failed on {tag}: got ({detected_lang}, {concept}), expected ({expected_lang}, {expected_concept})"
        passed += 1

    assert passed == len(test_cases)

if __name__ == "__main__":
    test_auto_mode_tracking()
    print("Auto-mode tracking tests passed successfully!")
