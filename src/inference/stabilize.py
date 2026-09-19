"""Prediction stabilization, temporal voting, idle rejection, and emission cooldown.

Prevents rapid flicker, false positives during idle stillness, and repeated word
triggers in live webcam video streams.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Deque, Optional, Tuple

import numpy as np

from src.features.normalize import (
    FEATURE_DIM,
    HAND_DIM,
    POSE_DIM,
    compute_motion_energy,
)


class PredictionStabilizer:
    """Multi-stage temporal filter for sign recognition outputs."""

    def __init__(
        self,
        voting_window: int = 10,
        fixed_mode_threshold: float = 0.90,
        auto_mode_threshold: float = 0.80,
        motion_energy_threshold: float = 0.08,
        cooldown_seconds: float = 1.0,
    ) -> None:
        self.voting_window = voting_window
        self.fixed_mode_threshold = fixed_mode_threshold
        self.auto_mode_threshold = auto_mode_threshold
        self.motion_energy_threshold = motion_energy_threshold
        self.cooldown_seconds = cooldown_seconds

        self._prediction_history: Deque[Tuple[int, float]] = deque(maxlen=voting_window)
        self._prev_frame_features: Optional[np.ndarray] = None
        self._last_emission_time: float = 0.0
        self._last_emitted_class: Optional[int] = None

    def are_hands_present(self, frame_features: np.ndarray) -> bool:
        """Check whether at least one hand landmark coordinate is non-zero."""
        if frame_features is None or len(frame_features) < FEATURE_DIM:
            return False
        hand_features = frame_features[POSE_DIM:FEATURE_DIM]
        return bool(np.any(hand_features != 0.0))

    def process(
        self,
        frame_features: np.ndarray,
        masked_probs: np.ndarray,
        mode: str = "auto",
        current_time: Optional[float] = None,
    ) -> Optional[Tuple[int, float]]:
        """Process latest prediction and return (class_idx, confidence) if stabilized and triggered.

        Args:
            frame_features: Latest normalized 258-dim feature vector.
            masked_probs: Re-normalized class probability vector.
            mode: 'isl', 'asl', or 'auto' (determines confidence threshold).
            current_time: Optional timestamp override for testing.

        Returns:
            Optional[Tuple[int, float]]: Stabilized (class_idx, confidence), or None if suppressed.
        """
        now = current_time if current_time is not None else time.time()

        # 1. Idle & Hand Rejection: Emit nothing if hands are absent
        if not self.are_hands_present(frame_features):
            self._prediction_history.clear()
            self._prev_frame_features = frame_features
            return None

        # 2. Motion Energy Rejection: Emit nothing if movement is below stillness floor
        if self._prev_frame_features is not None:
            energy = compute_motion_energy(self._prev_frame_features, frame_features)
            if energy < self.motion_energy_threshold:
                # Signer is pausing / motionless
                self._prev_frame_features = frame_features
                return None
        self._prev_frame_features = frame_features.copy()

        # 3. Top class and confidence check for this single frame
        top_idx = int(np.argmax(masked_probs))
        top_conf = float(masked_probs[top_idx])
        threshold = self.auto_mode_threshold if mode == "auto" else self.fixed_mode_threshold

        self._prediction_history.append((top_idx, top_conf))

        # Check voting window
        if len(self._prediction_history) < self.voting_window:
            return None

        # 4. Multi-frame majority voting
        classes = [c for c, _ in self._prediction_history]
        counts = Counter(classes)
        majority_class, count = counts.most_common(1)[0]

        # Require majority agreement (e.g. >= 7 out of 10)
        if count < int(self.voting_window * 0.7):
            return None

        # Calculate average confidence for majority class
        relevant_confs = [conf for c, conf in self._prediction_history if c == majority_class]
        avg_conf = sum(relevant_confs) / len(relevant_confs)

        if avg_conf < threshold:
            return None

        # 5. Cooldown gate: Prevent duplicate trigger within ~1 second
        if (now - self._last_emission_time) < self.cooldown_seconds and majority_class == self._last_emitted_class:
            return None

        # Emit stabilized classification
        self._last_emission_time = now
        self._last_emitted_class = majority_class
        return (majority_class, avg_conf)

    def reset(self) -> None:
        """Clear voting buffer and emission states."""
        self._prediction_history.clear()
        self._prev_frame_features = None
        self._last_emission_time = 0.0
        self._last_emitted_class = None
