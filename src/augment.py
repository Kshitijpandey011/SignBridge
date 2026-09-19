"""Data augmentation pipeline for 30-frame temporal landmark sequences.

Implements the five canonical augmentations specified for sign recognition:
1. Temporal speed jitter / time warping (±15%)
2. Additive Gaussian landmark noise
3. Spatial scale jitter (0.9 - 1.1x)
4. Planar rotation (±10 degrees)
5. Mirror augmentation (x-axis reflection + Left/Right hand & pose landmark swap)
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import numpy as np

from src.features.normalize import FEATURE_DIM, HAND_DIM, POSE_DIM

# Pose left/right symmetrical landmark pairs in MediaPipe (33 landmarks total):
# (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
# (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
POSE_SYMMETRIC_PAIRS = [
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]


def time_warp_jitter(sequence: np.ndarray, max_jitter: float = 0.15) -> np.ndarray:
    """Warp temporal speed by ±15% and resample back to 30 frames.

    Args:
        sequence: Array of shape (T, 258) where T is sequence length (30).
        max_jitter: Maximum speed jitter ratio (default: 0.15).

    Returns:
        np.ndarray: Temporal augmented array of shape (30, 258).
    """
    orig_len, feat_dim = sequence.shape
    factor = 1.0 + random.uniform(-max_jitter, max_jitter)
    warped_len = max(5, int(round(orig_len * factor)))

    # Resample to warped_len then back to orig_len via linear interpolation
    orig_indices = np.linspace(0, orig_len - 1, num=warped_len)
    warped = np.zeros((warped_len, feat_dim), dtype=np.float32)

    for d in range(feat_dim):
        warped[:, d] = np.interp(orig_indices, np.arange(orig_len), sequence[:, d])

    target_indices = np.linspace(0, warped_len - 1, num=orig_len)
    result = np.zeros((orig_len, feat_dim), dtype=np.float32)
    for d in range(feat_dim):
        result[:, d] = np.interp(target_indices, np.arange(warped_len), warped[:, d])

    return result.astype(np.float32)


def add_gaussian_noise(sequence: np.ndarray, sigma: float = 0.015) -> np.ndarray:
    """Add zero-mean Gaussian noise to landmark coordinate values."""
    noisy = sequence.copy()
    noise = np.random.normal(0, sigma, sequence.shape).astype(np.float32)

    # Do not add noise to visibility channel (every 4th element in pose [0:132])
    mask = np.ones(sequence.shape, dtype=bool)
    for p in range(0, POSE_DIM, 4):
        mask[:, p + 3] = False  # preserve visibility

    # Do not add noise to zero-padded absent hands
    # Hand slots are [132:195] (LH) and [195:258] (RH)
    lh_zeros = (noisy[:, POSE_DIM : POSE_DIM + HAND_DIM] == 0.0)
    noisy[:, POSE_DIM : POSE_DIM + HAND_DIM] += (noise[:, POSE_DIM : POSE_DIM + HAND_DIM] * (~lh_zeros))

    rh_zeros = (noisy[:, POSE_DIM + HAND_DIM : FEATURE_DIM] == 0.0)
    noisy[:, POSE_DIM + HAND_DIM : FEATURE_DIM] += (noise[:, POSE_DIM + HAND_DIM : FEATURE_DIM] * (~rh_zeros))

    # Add noise to pose coordinates
    pose_noisy = noisy[:, :POSE_DIM] + noise[:, :POSE_DIM] * mask[:, :POSE_DIM]
    noisy[:, :POSE_DIM] = pose_noisy

    return noisy


def scale_jitter(sequence: np.ndarray, min_scale: float = 0.9, max_scale: float = 1.1) -> np.ndarray:
    """Scale landmark coordinates uniformly by a factor in [min_scale, max_scale]."""
    factor = random.uniform(min_scale, max_scale)
    scaled = sequence.copy()

    # Scale pose x, y, z (skip visibility)
    for p in range(0, POSE_DIM, 4):
        scaled[:, p : p + 3] *= factor

    # Scale hands
    scaled[:, POSE_DIM:FEATURE_DIM] *= factor
    return scaled.astype(np.float32)


def rotate_sequence_2d(sequence: np.ndarray, max_degrees: float = 10.0) -> np.ndarray:
    """Apply 2D planar rotation by theta in [-max_degrees, +max_degrees] around origin."""
    angle_rad = math.radians(random.uniform(-max_degrees, max_degrees))
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated = sequence.copy()

    # Rotate pose landmarks
    for p in range(0, POSE_DIM, 4):
        x = sequence[:, p]
        y = sequence[:, p + 1]
        rotated[:, p] = x * cos_a - y * sin_a
        rotated[:, p + 1] = x * sin_a + y * cos_a

    # Rotate hand landmarks
    for h in range(POSE_DIM, FEATURE_DIM, 3):
        x = sequence[:, h]
        y = sequence[:, h + 1]
        # Only rotate non-zero coordinates
        mask = (x != 0.0) | (y != 0.0)
        rotated[:, h] = np.where(mask, x * cos_a - y * sin_a, 0.0)
        rotated[:, h + 1] = np.where(mask, x * sin_a + y * cos_a, 0.0)

    return rotated.astype(np.float32)


def mirror_and_swap(sequence: np.ndarray) -> np.ndarray:
    """Reflect along horizontal x-axis and swap left/right symmetrical landmark slots.

    This covers left-handed signers seamlessly:
    1. Invert x coordinates: x -> -x.
    2. Swap pose symmetrical landmark pairs (e.g. left shoulder <-> right shoulder).
    3. Swap Left Hand feature slots [132:195] with Right Hand feature slots [195:258].
    """
    mirrored = sequence.copy()

    # Invert x coordinates across pose
    for p in range(0, POSE_DIM, 4):
        mirrored[:, p] = -mirrored[:, p]

    # Invert x coordinates across hands
    for h in range(POSE_DIM, FEATURE_DIM, 3):
        non_zero = mirrored[:, h] != 0.0
        mirrored[:, h] = np.where(non_zero, -mirrored[:, h], 0.0)

    # Swap pose symmetrical pairs
    for left_idx, right_idx in POSE_SYMMETRIC_PAIRS:
        l_base = left_idx * 4
        r_base = right_idx * 4
        temp = mirrored[:, l_base : l_base + 4].copy()
        mirrored[:, l_base : l_base + 4] = mirrored[:, r_base : r_base + 4]
        mirrored[:, r_base : r_base + 4] = temp

    # Swap left and right hand blocks
    lh = mirrored[:, POSE_DIM : POSE_DIM + HAND_DIM].copy()
    rh = mirrored[:, POSE_DIM + HAND_DIM : FEATURE_DIM].copy()
    mirrored[:, POSE_DIM : POSE_DIM + HAND_DIM] = rh
    mirrored[:, POSE_DIM + HAND_DIM : FEATURE_DIM] = lh

    return mirrored.astype(np.float32)


def augment_sequence(
    sequence: np.ndarray,
    enable_timewarp: bool = True,
    enable_noise: bool = True,
    enable_scale: bool = True,
    enable_rotate: bool = True,
    enable_mirror: bool = True,
    mirror_prob: float = 0.5,
) -> np.ndarray:
    """Apply stochastic augmentation combination to an individual sequence."""
    seq = sequence.copy()

    if enable_timewarp and random.random() < 0.7:
        seq = time_warp_jitter(seq, max_jitter=0.15)

    if enable_scale and random.random() < 0.7:
        seq = scale_jitter(seq, min_scale=0.9, max_scale=1.1)

    if enable_rotate and random.random() < 0.7:
        seq = rotate_sequence_2d(seq, max_degrees=10.0)

    if enable_noise and random.random() < 0.7:
        seq = add_gaussian_noise(seq, sigma=0.012)

    if enable_mirror and random.random() < mirror_prob:
        seq = mirror_and_swap(seq)

    return seq.astype(np.float32)
