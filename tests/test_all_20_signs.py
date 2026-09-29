import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.mock_sequence import create_mock_sequence
from src.features.kinematics import KinematicGestureClassifier

def test_all_20_signs_canonical():
    clf = KinematicGestureClassifier()

    test_catalog = [
        ("hello", dict(shape="open", zone="head", motion="waving")),
        ("thank_you", dict(shape="open", zone="chin", motion="still")),
        ("please", dict(shape="open", zone="chest", motion="circular")),
        ("sorry", dict(shape="fist", zone="chest", motion="circular")),
        ("pain", dict(shape="index", zone="chest", motion="still")),
        ("no", dict(shape="index", zone="chin", motion="waving")),
        ("yes", dict(shape="fist", zone="chest", motion="nodding")),
        ("help", dict(shape="thumbs_up", zone="chest", motion="still")),
        ("water", dict(shape="w", zone="chin", motion="still")),
        ("food", dict(shape="fist", zone="chin", motion="still")),
        ("doctor", dict(gesture="doctor", shape="index", zone="chest", motion="still", two_hands=True)),
        ("work", dict(shape="fist", zone="chest", motion="still", two_hands=True)),
        ("school", dict(gesture="school", shape="open", zone="chest", motion="waving", two_hands=True)),
        ("medicine", dict(gesture="medicine", shape="index", zone="chest", motion="still", two_hands=True)),
        ("family", dict(shape="open", zone="chest", motion="circular", two_hands=True)),
        ("home", dict(shape="bunched", zone="head", motion="still")),
        ("money", dict(shape="bunched", zone="chest", motion="still")),
        ("phone", dict(shape="y", zone="head", motion="still")),
        ("happy", dict(shape="open", zone="chest", motion="upward")),
        ("sad", dict(shape="open", zone="chin", motion="downward")),
    ]

    passed = 0
    for expected, params in test_catalog:
        seq = create_mock_sequence(**params)
        is_valid, concept, conf, lang = clf.evaluate_absolute_sign(seq)
        is_ok = is_valid and (concept == expected)
        if is_ok:
            passed += 1
        assert is_ok, f"Failed on sign {expected}: detected={concept}, valid={is_valid}"

    assert passed == len(test_catalog)

if __name__ == "__main__":
    test_all_20_signs_canonical()
    print("All 20 signs passed successfully!")
