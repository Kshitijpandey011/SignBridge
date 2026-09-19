"""Feature extraction and geometric landmark normalization module.

This is the canonical module for transforming raw MediaPipe Holistic results into
standardized, scale-invariant 258-dimensional feature vectors.
Every data collection script, dataset converter, and live inference worker MUST import
from here to ensure numerical consistency across the pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Total feature vector dimension:
# Pose: 33 landmarks * 4 (x, y, z, visibility) = 132
# Left Hand: 21 landmarks * 3 (x, y, z) = 63
# Right Hand: 21 landmarks * 3 (x, y, z) = 63
# Total = 132 + 63 + 63 = 258
FEATURE_DIM: int = 258
POSE_DIM: int = 132
HAND_DIM: int = 63

# MediaPipe Pose Landmark Indices for Shoulders:
# 11: LEFT_SHOULDER, 12: RIGHT_SHOULDER
LEFT_SHOULDER_IDX: int = 11
RIGHT_SHOULDER_IDX: int = 12


def init_holistic(
    static_image_mode: bool = False,
    model_complexity: int = 1,
    smooth_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp.solutions.holistic.Holistic:
    """Initialize and return a MediaPipe Holistic model instance."""
    return mp.solutions.holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_and_normalize_landmarks(results: Any) -> np.ndarray:
    """Extract and normalize landmarks from MediaPipe Holistic output into a 258-dim vector.

    Normalization algorithm:
    1. Origin translation: Calculate the midpoint of the left and right shoulders (indices 11 and 12).
       Translate all coordinates (pose, left hand, right hand) by subtracting this midpoint.
    2. Scale normalization: Compute Euclidean distance between left and right shoulders.
       Divide all coordinates by this distance (with minimum epsilon clamp 1e-4) to make
       features invariant to distance from camera and body size.
    3. Missing hands: If a hand is not detected, fill its 63 slots with zeros.
    4. Slot order: [0:132] Pose, [132:195] Left Hand, [195:258] Right Hand.

    Args:
        results: MediaPipe Holistic detection results object.

    Returns:
        np.ndarray: float32 vector of shape (258,).
    """
    feature_vector = np.zeros(FEATURE_DIM, dtype=np.float32)

    pose_landmarks = results.pose_landmarks
    left_hand_landmarks = results.left_hand_landmarks
    right_hand_landmarks = results.right_hand_landmarks

    # Determine reference origin and scale from shoulders if pose is available
    if pose_landmarks is not None and len(pose_landmarks.landmark) > max(LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX):
        ls = pose_landmarks.landmark[LEFT_SHOULDER_IDX]
        rs = pose_landmarks.landmark[RIGHT_SHOULDER_IDX]

        mid_x = (ls.x + rs.x) / 2.0
        mid_y = (ls.y + rs.y) / 2.0
        mid_z = (ls.z + rs.z) / 2.0

        dx = ls.x - rs.x
        dy = ls.y - rs.y
        dz = ls.z - rs.z
        shoulder_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        scale = max(shoulder_dist, 1e-4)

        # Normalize 33 pose landmarks (33 * 4 = 132 features)
        pose_arr = np.zeros(POSE_DIM, dtype=np.float32)
        for i, lm in enumerate(pose_landmarks.landmark):
            base_idx = i * 4
            pose_arr[base_idx] = (lm.x - mid_x) / scale
            pose_arr[base_idx + 1] = (lm.y - mid_y) / scale
            pose_arr[base_idx + 2] = (lm.z - mid_z) / scale
            pose_arr[base_idx + 3] = float(lm.visibility)
        feature_vector[0:POSE_DIM] = pose_arr
    else:
        # Fallback if no pose detected: origin at (0.5, 0.5, 0.0), scale = 1.0
        mid_x, mid_y, mid_z = 0.5, 0.5, 0.0
        scale = 1.0

    # Normalize Left Hand if detected (21 * 3 = 63 features)
    if left_hand_landmarks is not None and len(left_hand_landmarks.landmark) == 21:
        lh_arr = np.zeros(HAND_DIM, dtype=np.float32)
        for i, lm in enumerate(left_hand_landmarks.landmark):
            base_idx = i * 3
            lh_arr[base_idx] = (lm.x - mid_x) / scale
            lh_arr[base_idx + 1] = (lm.y - mid_y) / scale
            lh_arr[base_idx + 2] = (lm.z - mid_z) / scale
        feature_vector[POSE_DIM : POSE_DIM + HAND_DIM] = lh_arr
    # If not detected, remains zeros (zero-fill rule)

    # Normalize Right Hand if detected (21 * 3 = 63 features)
    if right_hand_landmarks is not None and len(right_hand_landmarks.landmark) == 21:
        rh_arr = np.zeros(HAND_DIM, dtype=np.float32)
        for i, lm in enumerate(right_hand_landmarks.landmark):
            base_idx = i * 3
            rh_arr[base_idx] = (lm.x - mid_x) / scale
            rh_arr[base_idx + 1] = (lm.y - mid_y) / scale
            rh_arr[base_idx + 2] = (lm.z - mid_z) / scale
        feature_vector[POSE_DIM + HAND_DIM : FEATURE_DIM] = rh_arr
    # If not detected, remains zeros (zero-fill rule)

    return feature_vector


def compute_motion_energy(feat_prev: np.ndarray, feat_curr: np.ndarray) -> float:
    """Compute motion energy between two consecutive normalized feature frames.

    Evaluates coordinate deltas specifically across hands and upper arms to detect
    meaningful signing motion vs idle stillness.

    Args:
        feat_prev: Previous frame normalized feature vector (258,).
        feat_curr: Current frame normalized feature vector (258,).

    Returns:
        float: L2 norm of positional delta in active signing segments.
    """
    if feat_prev is None or feat_curr is None:
        return 0.0

    # Focus primarily on hand features (indices 132 to 258)
    hands_prev = feat_prev[POSE_DIM:FEATURE_DIM]
    hands_curr = feat_curr[POSE_DIM:FEATURE_DIM]

    # Delta of hand coordinates
    diff = hands_curr - hands_prev
    energy = float(np.linalg.norm(diff))
    return energy


def draw_styled_landmarks(image: np.ndarray, results: Any) -> np.ndarray:
    """Render skeleton landmarks on a copy of the input image for debugging/UI."""
    annotated = image.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )

    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1),
        )

    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1),
        )

    return annotated
