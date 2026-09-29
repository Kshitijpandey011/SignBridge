"""Language inference mode masking and dynamic auto-detection with soft-locking.

Controls classification constraints under three run modes:
- ISL mode: Masks out all non-ISL classes.
- ASL mode: Masks out all non-ASL classes.
- AUTO mode: Dynamic real-time tracking of ISL vs ASL signs without lockouts.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple
import numpy as np


def apply_language_mask(
    probs: np.ndarray,
    mode: str,
    isl_idx: List[int],
    asl_idx: List[int],
) -> np.ndarray:
    """Apply language constraint mask to class probability vector.

    Args:
        probs: 1D probability distribution across classes (40,).
        mode: 'isl', 'asl', or 'auto'.
        isl_idx: List of integer indices corresponding to ISL classes.
        asl_idx: List of integer indices corresponding to ASL classes.

    Returns:
        np.ndarray: Re-normalized probability vector.
    """
    mode = mode.lower()
    if mode == "isl":
        masked = np.zeros_like(probs)
        masked[isl_idx] = probs[isl_idx]
    elif mode == "asl":
        masked = np.zeros_like(probs)
        masked[asl_idx] = probs[asl_idx]
    else:
        return probs.copy()

    total = float(np.sum(masked))
    return (masked / total) if total > 0 else masked


class InferenceModeManager:
    """Manages active run mode, temporal language voting, and dynamic AUTO tracking."""

    def __init__(
        self,
        isl_idx: List[int],
        asl_idx: List[int],
        history_window: int = 6,
        lock_threshold: int = 4,
        unlock_threshold: int = 5,
        mean_prob_threshold: float = 0.60,
    ) -> None:
        self.isl_idx = isl_idx
        self.asl_idx = asl_idx
        self.history_window = history_window
        self.lock_threshold = lock_threshold
        self.unlock_threshold = unlock_threshold
        self.mean_prob_threshold = mean_prob_threshold

        # Rolling history of (winner_lang, p_isl, p_asl)
        self._vote_history: deque[Tuple[str, float, float]] = deque(maxlen=history_window)

        # State
        self.current_mode: str = "auto"  # 'isl', 'asl', 'auto'
        self.locked_language: Optional[str] = None  # None, 'isl', 'asl'
        self.manual_override: Optional[str] = None  # None, 'isl', 'asl'

    def set_mode(self, mode: str) -> None:
        """Set user operational mode: 'isl', 'asl', or 'auto'."""
        mode = mode.lower()
        if mode not in ("isl", "asl", "auto"):
            raise ValueError(f"Invalid mode '{mode}'. Choose 'isl', 'asl', or 'auto'.")
        self.current_mode = mode
        if mode in ("isl", "asl"):
            self.locked_language = mode
        else:
            self.locked_language = None

    def force_override(self, lang: Optional[str]) -> None:
        """Manually lock or unlock language in AUTO mode."""
        if lang is not None and lang.lower() not in ("isl", "asl"):
            raise ValueError("Override must be 'isl', 'asl', or None")
        self.manual_override = lang.lower() if lang else None

    def update(
        self, class_probs: np.ndarray, lang_probs: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, str, bool]:
        """Update voting state with new frame probabilities and return effective output.

        Args:
            class_probs: Raw 40-way softmax output from model / kinematic fusion.
            lang_probs: Optional 2-way secondary head output.

        Returns:
            Tuple of:
                - masked_probs: Probabilities adjusted by current language constraints
                - detected_language: 'isl' or 'asl'
                - is_locked: Whether currently strongly locked/consensus
        """
        # Calculate language probabilities directly from class probabilities (which contain kinematic ground-truth)
        p_isl = float(np.sum(class_probs[self.isl_idx]))
        p_asl = float(np.sum(class_probs[self.asl_idx]))

        if lang_probs is not None and len(lang_probs) == 2 and np.sum(lang_probs) > 0:
            p_isl = 0.90 * p_isl + 0.10 * float(lang_probs[0])
            p_asl = 0.90 * p_asl + 0.10 * float(lang_probs[1])

        frame_winner = "isl" if p_isl >= p_asl else "asl"
        self._vote_history.append((frame_winner, p_isl, p_asl))

        # Handle manual override or fixed modes
        effective_mode = self.current_mode
        if self.manual_override:
            effective_mode = self.manual_override
            masked_probs = apply_language_mask(class_probs, effective_mode, self.isl_idx, self.asl_idx)
            detected_lang = effective_mode
            is_locked = True
            return masked_probs, detected_lang, is_locked

        if self.current_mode in ("isl", "asl"):
            effective_mode = self.current_mode
            masked_probs = apply_language_mask(class_probs, effective_mode, self.isl_idx, self.asl_idx)
            detected_lang = effective_mode
            is_locked = True
            return masked_probs, detected_lang, is_locked

        # AUTO MODE: Dynamic rolling tracking without lockouts
        isl_votes = sum(1 for v, _, _ in self._vote_history if v == "isl")
        asl_votes = len(self._vote_history) - isl_votes
        mean_p_isl = sum(p for _, p, _ in self._vote_history) / max(len(self._vote_history), 1)
        mean_p_asl = sum(p for _, _, p in self._vote_history) / max(len(self._vote_history), 1)

        # Immediate responsive frame tracking: if current frame is distinct, it takes immediate priority
        if abs(p_isl - p_asl) > 0.40:
            detected_lang = frame_winner
        elif isl_votes > asl_votes:
            detected_lang = "isl"
        elif asl_votes > isl_votes:
            detected_lang = "asl"
        else:
            detected_lang = frame_winner

        # In AUTO mode, allow both ISL and ASL classes to compete!
        # Apply a mild prior (1.10x) to the active language without zeroing out the other
        masked_probs = class_probs.copy()
        if detected_lang == "isl":
            masked_probs[self.isl_idx] *= 1.10
        else:
            masked_probs[self.asl_idx] *= 1.10

        total = float(np.sum(masked_probs))
        if total > 0:
            masked_probs = masked_probs / total

        is_locked = bool(len(self._vote_history) >= 4 and abs(isl_votes - asl_votes) >= 3)
        return masked_probs, detected_lang, is_locked

    def reset(self) -> None:
        """Reset vote history and lock states."""
        self._vote_history.clear()
        if self.current_mode == "auto":
            self.locked_language = None
