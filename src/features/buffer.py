"""Rolling sequence buffer for temporal landmark representations.

Maintains a FIFO queue of normalized landmark vectors with shape (SEQUENCE_LENGTH, FEATURE_DIM).
Used identically in dataset recording and real-time model inference.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from src.features.normalize import FEATURE_DIM

DEFAULT_SEQUENCE_LENGTH: int = 30


class SequenceBuffer:
    """Fixed-capacity FIFO buffer holding normalized landmark frames."""

    def __init__(self, sequence_length: int = DEFAULT_SEQUENCE_LENGTH, feature_dim: int = FEATURE_DIM) -> None:
        """Initialize rolling sequence buffer.

        Args:
            sequence_length: Number of consecutive frames in a window (default: 30).
            feature_dim: Dimension of each frame's normalized feature vector (default: 258).
        """
        if sequence_length <= 0:
            raise ValueError(f"Sequence length must be positive, got {sequence_length}")
        if feature_dim <= 0:
            raise ValueError(f"Feature dimension must be positive, got {feature_dim}")

        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self._buffer: deque[np.ndarray] = deque(maxlen=sequence_length)

    def append(self, frame_features: np.ndarray) -> None:
        """Append a frame feature vector to the buffer.

        Args:
            frame_features: 1D numpy array of shape (feature_dim,) or (258,).
        """
        if not isinstance(frame_features, np.ndarray):
            frame_features = np.asarray(frame_features, dtype=np.float32)

        if frame_features.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected frame feature shape ({self.feature_dim},), got {frame_features.shape}"
            )

        self._buffer.append(frame_features.astype(np.float32, copy=False))

    def is_full(self) -> bool:
        """Check if the buffer has accumulated the full window of frames."""
        return len(self._buffer) == self.sequence_length

    def is_ready(self) -> bool:
        """Alias for is_full() to verify window readiness."""
        return self.is_full()

    def get_sequence(self) -> np.ndarray:
        """Return the current sequence window as a numpy array of shape (sequence_length, feature_dim).

        If the buffer is not yet full, zero-pads the beginning of the sequence.
        """
        if len(self._buffer) == 0:
            return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)

        arr = np.array(self._buffer, dtype=np.float32)
        if len(self._buffer) < self.sequence_length:
            padding = np.zeros(
                (self.sequence_length - len(self._buffer), self.feature_dim), dtype=np.float32
            )
            return np.vstack([padding, arr])
        return arr

    def clear(self) -> None:
        """Reset the buffer."""
        self._buffer.clear()

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()

    def __len__(self) -> int:
        """Return number of frames currently in buffer."""
        return len(self._buffer)
