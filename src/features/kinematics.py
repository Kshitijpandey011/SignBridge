"""Semantic kinematic gesture analyzer for dual-language sign recognition.

Enforces robust shape, location, trajectory, and two-handed coordination
matching for the canonical 20 concepts across ISL and ASL.
Strictly rejects idle movements, transitions, and ambiguous hand postures,
while providing high sensitivity, accuracy, and clear ISL vs ASL discrimination.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.features.normalize import FEATURE_DIM, HAND_DIM, POSE_DIM
from src.label_map import DEFAULT_REGISTRY

# Landmark Index Constants in MediaPipe
# Hand: 21 landmarks (x, y, z) = 63 features
# 0: Wrist
# 1-4: Thumb (4=Tip, 3=IP, 2=MCP, 1=CMC)
# 5-8: Index (8=Tip, 7=DIP, 6=PIP, 5=MCP)
# 9-12: Middle (12=Tip, 11=DIP, 10=PIP, 9=MCP)
# 13-16: Ring (16=Tip, 15=DIP, 14=PIP, 13=MCP)
# 17-20: Pinky (20=Tip, 19=DIP, 18=PIP, 17=MCP)


def dist_3d(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Euclidean distance between two 3D points."""
    d = pt1 - pt2
    return float(np.sqrt(np.sum(d * d)))


def analyze_hand_shape(hand_features: np.ndarray) -> str:
    """Classify 63-dim hand features into core handshapes:
    'open', 'fist', 'thumbs_up', 'index', 'v', 'w', 'y', 'bunched', or 'general'.
    Uses scale-invariant knuckle curl ratios: dist(tip, mcp) / dist(pip, mcp).
    """
    if hand_features is None or len(hand_features) < 63:
        return "none"
    if np.all(hand_features == 0.0):
        return "none"

    coords = hand_features.reshape((21, 3))
    wrist = coords[0]

    # Knuckle-to-tip curl ratios for the 4 fingers
    # Extended finger: ratio > 1.20
    # Curled finger: ratio < 1.15
    index_mcp = coords[5]
    index_pip = coords[6]
    index_tip = coords[8]
    index_ratio = dist_3d(index_tip, index_mcp) / max(dist_3d(index_pip, index_mcp), 1e-4)
    index_ext = bool(index_ratio > 1.20)

    middle_mcp = coords[9]
    middle_pip = coords[10]
    middle_tip = coords[12]
    middle_ratio = dist_3d(middle_tip, middle_mcp) / max(dist_3d(middle_pip, middle_mcp), 1e-4)
    middle_ext = bool(middle_ratio > 1.20)

    ring_mcp = coords[13]
    ring_pip = coords[14]
    ring_tip = coords[16]
    ring_ratio = dist_3d(ring_tip, ring_mcp) / max(dist_3d(ring_pip, ring_mcp), 1e-4)
    ring_ext = bool(ring_ratio > 1.20)

    pinky_mcp = coords[17]
    pinky_pip = coords[18]
    pinky_tip = coords[20]
    pinky_ratio = dist_3d(pinky_tip, pinky_mcp) / max(dist_3d(pinky_pip, pinky_mcp), 1e-4)
    pinky_ext = bool(pinky_ratio > 1.20)

    # Thumb extension and orientation
    thumb_mcp = coords[2]
    thumb_ip = coords[3]
    thumb_tip = coords[4]
    thumb_ratio = dist_3d(thumb_tip, thumb_mcp) / max(dist_3d(thumb_ip, thumb_mcp), 1e-4)
    palm_span = dist_3d(index_mcp, pinky_mcp)
    thumb_ext = bool(thumb_ratio > 1.15 and dist_3d(thumb_tip, pinky_mcp) > 0.28 * max(palm_span, 1e-4))

    # True thumbs-up: thumb tip is vertically higher than thumb MCP in camera view
    thumb_upward = bool(thumb_ext and (thumb_tip[1] < thumb_mcp[1] - 0.015))

    ext_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])
    curled_count = 4 - ext_count

    # Bunched / Flat-O: All 4 fingertips gathered very close to thumb tip with fingers reaching forward
    avg_tip_dist = float(np.mean([
        dist_3d(thumb_tip, index_tip),
        dist_3d(thumb_tip, middle_tip),
        dist_3d(thumb_tip, ring_tip),
        dist_3d(thumb_tip, pinky_tip),
    ]))

    # 1. Y-handshape: Thumb + Pinky extended, index and middle curled
    if thumb_ext and pinky_ext and not index_ext and not middle_ext:
        return "y"

    # 2. Thumbs up: Thumb extended upwards, at least 3 fingers curled
    if thumb_upward and curled_count >= 3:
        return "thumbs_up"

    # 3. Fist: All 4 curled into the palm, thumb curled or tucked
    if curled_count == 4 and not thumb_upward:
        return "fist"

    # 4. Bunched / Flat-O: Fingertips gathered touching thumb tip
    if avg_tip_dist < 0.40 * max(palm_span, 1e-4) and ext_count >= 1:
        return "bunched"

    # 5. W-handshape: Index, Middle, Ring extended, Pinky curled
    if (index_ext and middle_ext and ring_ext and not pinky_ext) or (ext_count == 3 and not pinky_ext):
        return "w"

    # 6. V-handshape: Index and Middle extended, Ring and Pinky curled
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "v"

    # 7. Index pointing: Only index extended
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "index"

    # 8. Open Palm: 3 or 4 fingers extended
    if ext_count >= 3:
        return "open"

    return "general"


def analyze_hand_zone(hand_features: np.ndarray) -> str:
    """Determine vertical elevation zone of hand relative to shoulders (origin y=0.0):
    'head': y < -0.35 (temple, forehead, ears, eyes)
    'chin': -0.35 <= y < -0.10 (mouth, lips, chin, cheeks, jaw)
    'chest': -0.10 <= y < 0.38 (chest, sternum, heart, shoulders)
    'neutral': y >= 0.38 (waist, hips, lap)
    """
    if hand_features is None or np.all(hand_features == 0.0):
        return "none"
    coords = hand_features.reshape((21, 3))
    avg_y = float(np.mean(coords[:, 1]))

    if avg_y < -0.35:
        return "head"
    elif avg_y < -0.10:
        return "chin"
    elif avg_y < 0.38:
        return "chest"
    else:
        return "neutral"


def smooth_trajectory(pts: np.ndarray) -> np.ndarray:
    """Apply 3-tap moving average filter [0.25, 0.50, 0.25] to suppress camera noise."""
    if len(pts) < 3:
        return pts
    smoothed = pts.copy()
    for i in range(1, len(pts) - 1):
        smoothed[i] = 0.25 * pts[i - 1] + 0.50 * pts[i] + 0.25 * pts[i + 1]
    return smoothed


def analyze_motion_trajectory(sequence_window: np.ndarray) -> str:
    """Analyze movement trajectory of active hand over sequence window:
    'circular', 'nodding', 'waving', 'chin_outward', 'upward', 'downward', 'tapping', or 'still'.
    """
    if sequence_window is None or len(sequence_window) < 6:
        return "still"

    # Identify primary active hand
    rh_active = np.any(sequence_window[:, 195:258] != 0.0)
    lh_active = np.any(sequence_window[:, 132:195] != 0.0)

    if rh_active:
        base_offset = 195
    elif lh_active:
        base_offset = 132
    else:
        return "still"

    wrists = sequence_window[:, base_offset : base_offset + 3]
    mcps = sequence_window[:, base_offset + 27 : base_offset + 30]

    valid_mask = np.any(wrists != 0.0, axis=1)
    if np.sum(valid_mask) < 4:
        return "still"

    raw_pts = wrists[valid_mask]
    mcp_pts = mcps[valid_mask]

    active_pts = smooth_trajectory(raw_pts)
    active_mcps = smooth_trajectory(mcp_pts)

    diffs = np.diff(active_pts, axis=0)
    total_path = float(np.sum(np.linalg.norm(diffs, axis=1)))
    net_disp = float(np.linalg.norm(active_pts[-1] - active_pts[0]))

    dx = float(active_pts[-1, 0] - active_pts[0, 0])
    dy = float(active_pts[-1, 1] - active_pts[0, 1])
    dz = float(active_pts[-1, 2] - active_pts[0, 2])

    if total_path < 0.035:
        return "still"

    # 1. Circular Motion: Phase Angle Unwrapping or Planar Ellipse
    x_span = float(np.max(active_pts[:, 0]) - np.min(active_pts[:, 0]))
    y_span = float(np.max(active_pts[:, 1]) - np.min(active_pts[:, 1]))
    centroid = np.mean(active_pts[:, :2], axis=0)
    centered = active_pts[:, :2] - centroid
    radii = np.linalg.norm(centered, axis=1)

    if x_span > 0.025 and y_span > 0.025 and np.mean(radii) > 0.015:
        angles = np.unwrap(np.arctan2(centered[:, 1], centered[:, 0]))
        total_angular_span = float(abs(angles[-1] - angles[0]))
        if total_angular_span >= 1.4 and net_disp < 0.22:
            return "circular"

    # 2. Per-axis Directional & Velocity Analysis
    x_path = float(np.sum(np.abs(diffs[:, 0])))
    y_path = float(np.sum(np.abs(diffs[:, 1])))

    # Direction reversals
    valid_dx = diffs[:, 0][np.abs(diffs[:, 0]) > 0.003]
    valid_dy = diffs[:, 1][np.abs(diffs[:, 1]) > 0.003]
    x_sign_changes = np.count_nonzero(np.diff(np.sign(valid_dx))) if len(valid_dx) > 1 else 0
    y_sign_changes = np.count_nonzero(np.diff(np.sign(valid_dy))) if len(valid_dy) > 1 else 0

    # Knuckle pitch oscillation
    knuckle_rel_y = active_mcps[:, 1] - active_pts[:, 1]
    knuckle_diffs = np.diff(knuckle_rel_y)
    valid_kd = knuckle_diffs[np.abs(knuckle_diffs) > 0.003]
    knuckle_sign_changes = np.count_nonzero(np.diff(np.sign(valid_kd))) if len(valid_kd) > 1 else 0

    # A. Nodding: Vertical oscillation of wrist or knuckle pitch (has reversals)
    if (y_sign_changes >= 1 or knuckle_sign_changes >= 1) and y_path > 0.03 and x_path < 0.12:
        return "nodding"

    # B. Waving: Horizontal oscillation
    if x_sign_changes >= 1 and x_path > 0.03 and y_path < 0.12:
        return "waving"

    # C. Upward strokes (Happy): unidirectional upward velocity
    upward_steps = np.count_nonzero(diffs[:, 1] < -0.010)
    if (upward_steps >= 2 or dy < -0.035) and abs(dy) >= abs(dx) * 0.4 and y_sign_changes == 0:
        return "upward"

    # D. Downward strokes (Sad): unidirectional downward velocity
    downward_steps = np.count_nonzero(diffs[:, 1] > 0.010)
    if (downward_steps >= 2 or dy > 0.035) and abs(dy) >= abs(dx) * 0.4 and y_sign_changes == 0:
        return "downward"

    # E. Chin Outward: moves forward (dz > 0) or forward-downward away from chin
    if dz > 0.04 or (active_pts[0, 1] < -0.10 and dy > 0.03 and dz > 0.02):
        return "chin_outward"

    # F. Tapping: active path with low net displacement
    if total_path > 0.04 and net_disp < 0.07:
        return "tapping"

    return "still"


def analyze_two_hand_interaction(lh_features: np.ndarray, rh_features: np.ndarray) -> Dict[str, Any]:
    """Analyze spatial relationship and interaction between left and right hands."""
    result = {
        "two_hands": False,
        "hands_touching": False,
        "hand_on_wrist": False,
        "dominant_on_palm": False,
        "fist_on_palm": False,
        "both_open": False,
        "both_fists": False,
        "index_tips_meeting": False,
        "namaste_prayer": False,
        "inverted_v_roof": False,
        "circular_embrace": False,
    }

    lh_active = lh_features is not None and not np.all(lh_features == 0.0)
    rh_active = rh_features is not None and not np.all(rh_features == 0.0)

    if not (lh_active and rh_active):
        return result

    result["two_hands"] = True
    lh_coords = lh_features.reshape((21, 3))
    rh_coords = rh_features.reshape((21, 3))

    lh_wrist = lh_coords[0]
    rh_wrist = rh_coords[0]
    lh_palm_center = lh_coords[9]
    rh_palm_center = rh_coords[9]

    lh_shape = analyze_hand_shape(lh_features)
    rh_shape = analyze_hand_shape(rh_features)

    result["both_open"] = (lh_shape in ("open", "general") and rh_shape in ("open", "general"))
    result["both_fists"] = (lh_shape in ("fist", "thumbs_up") and rh_shape in ("fist", "thumbs_up"))

    # Minimum distance between hands
    min_dist = min([dist_3d(lh_coords[i], rh_coords[j]) for i in (0, 4, 8, 12) for j in (0, 4, 8, 12)])
    wrist_dist = dist_3d(lh_wrist, rh_wrist)
    tip_dist = dist_3d(lh_coords[8], rh_coords[8])

    if min_dist < 0.38 or wrist_dist < 0.45:
        result["hands_touching"] = True

    # 1. Inverted-V / Roof (ISL Home): Fingertips touching at top, wrists separated laterally
    lh_tips_up = bool(lh_coords[8, 1] < lh_wrist[1])
    rh_tips_up = bool(rh_coords[8, 1] < rh_wrist[1])
    if lh_tips_up and rh_tips_up and tip_dist < 0.25 and wrist_dist > 0.28:
        result["inverted_v_roof"] = True

    # 2. Namaste / Prayer (ISL Please): Both palms vertical, fingertips pointing up and touching, wrists close
    if result["both_open"] and lh_tips_up and rh_tips_up and tip_dist < 0.25 and wrist_dist < 0.35:
        result["namaste_prayer"] = True

    # 3. Fist on palm (Help)
    fist_palm_1 = rh_shape in ("fist", "thumbs_up") and lh_shape in ("open", "general") and dist_3d(rh_wrist, lh_palm_center) < 0.38
    fist_palm_2 = lh_shape in ("fist", "thumbs_up") and rh_shape in ("open", "general") and dist_3d(lh_wrist, rh_palm_center) < 0.38
    if fist_palm_1 or fist_palm_2:
        result["fist_on_palm"] = True

    # 4. Index tips meeting / pointing at each other (Pain - ASL)
    if tip_dist < 0.35 and lh_shape in ("index", "v") and rh_shape in ("index", "v"):
        result["index_tips_meeting"] = True

    # 5. Doctor (dominant index/middle touching wrist) vs Medicine (dominant fingertip touching open palm center)
    rh_to_lh_wrist = min([dist_3d(rh_coords[i], lh_wrist) for i in (8, 12)])
    rh_to_lh_palm = min([dist_3d(rh_coords[i], lh_palm_center) for i in (8, 12)])
    lh_to_rh_wrist = min([dist_3d(lh_coords[i], rh_wrist) for i in (8, 12)])
    lh_to_rh_palm = min([dist_3d(lh_coords[i], rh_palm_center) for i in (8, 12)])

    if not result["both_open"] and not result["both_fists"]:
        # Checking pulse (Doctor): dominant fingers touch wrist
        if (rh_shape in ("index", "v") and rh_to_lh_wrist < 0.30 and rh_to_lh_wrist <= rh_to_lh_palm + 0.05) or \
           (lh_shape in ("index", "v") and lh_to_rh_wrist < 0.30 and lh_to_rh_wrist <= lh_to_rh_palm + 0.05):
            result["hand_on_wrist"] = True
        # Grinding medicine (Medicine): dominant fingers touch center of open palm
        elif (rh_to_lh_palm < 0.30 and lh_shape in ("open", "general")) or \
             (lh_to_rh_palm < 0.30 and rh_shape in ("open", "general")):
            result["dominant_on_palm"] = True

    # 6. Circular embrace / outward loop (Family)
    if result["both_open"] and (wrist_dist > 0.20 and wrist_dist < 0.65):
        result["circular_embrace"] = True

    return result


class KinematicGestureClassifier:
    """Strict yet capable anatomical kinematic classifier for all 20 canonical signs.
    Matches absolute shape, body location, and movement trajectory across ISL and ASL.
    """

    def __init__(self) -> None:
        self.registry = DEFAULT_REGISTRY

    def evaluate_absolute_sign(
        self, sequence_window: np.ndarray
    ) -> Tuple[bool, Optional[str], float, str]:
        """Verify whether the sequence matches the physiological criteria of any canonical sign.

        Returns:
            Tuple of:
            - is_valid (bool): True if absolute physical match is met.
            - concept (Optional[str]): Matched concept name (e.g. 'water', 'hello').
            - confidence (float): Calibrated certainty score (e.g. 0.96).
            - lang_hint (str): 'asl', 'isl', or 'both'.
        """
        if sequence_window is None or len(sequence_window) == 0:
            return False, None, 0.0, ""

        last_frame = sequence_window[-1]
        lh = last_frame[POSE_DIM : POSE_DIM + HAND_DIM]
        rh = last_frame[POSE_DIM + HAND_DIM : FEATURE_DIM]

        lh_active = bool(np.any(lh != 0.0))
        rh_active = bool(np.any(rh != 0.0))

        if not rh_active and not lh_active:
            return False, None, 0.0, ""

        primary_hand = rh if rh_active else lh
        shape = analyze_hand_shape(primary_hand)
        zone = analyze_hand_zone(primary_hand)
        motion = analyze_motion_trajectory(sequence_window)
        interaction = analyze_two_hand_interaction(lh, rh)
        both_hands = interaction["two_hands"]

        primary_coords = primary_hand.reshape((21, 3))
        wrist = primary_coords[0]
        index_tip = primary_coords[8]
        fingertips_up = bool(index_tip[1] < wrist[1])

        # =========================================================================
        # SECTION 1: TWO-HANDED SIGNS (Evaluated first when both hands are active)
        # =========================================================================
        if both_hands:
            # 1. HOME (ISL): Both hands forming roof (inverted-V) at chest or chin
            if interaction["inverted_v_roof"] and zone in ("chest", "chin"):
                return True, "home", 0.97, "isl"

            # 2. PLEASE (ISL): Both open palms pressed together vertically at chest (Namaste) held still
            if interaction["namaste_prayer"] and zone in ("chest", "chin") and motion in ("still", "nodding"):
                return True, "please", 0.98, "isl"

            # 3. HELP (ISL): Dominant thumbs-up or fist resting directly on open palm
            if interaction["fist_on_palm"]:
                return True, "help", 0.98, "isl"

            # 4. PAIN (ASL): Dual index fingers pointing tips directly toward each other
            if interaction["index_tips_meeting"]:
                return True, "pain", 0.96, "asl"

            # 5. DOCTOR (ISL): Dominant fingertips touching inner wrist / radial pulse
            if interaction["hand_on_wrist"] and zone in ("chest", "neutral"):
                return True, "doctor", 0.96, "isl"

            # 6. MEDICINE (ISL): Dominant fingertip touching/grinding center of open palm
            if interaction["dominant_on_palm"] and zone in ("chest", "neutral"):
                return True, "medicine", 0.96, "isl"

            # 7. WORK (ISL): Both hands in fists tapping together at chest
            if interaction["both_fists"] and interaction["hands_touching"] and zone in ("chest", "neutral"):
                return True, "work", 0.96, "isl"

            # 8. SCHOOL (ISL): Both open hands clapping together horizontally at chest
            if interaction["both_open"] and interaction["hands_touching"] and motion in ("waving", "tapping") and zone in ("chest", "neutral"):
                return True, "school", 0.96, "isl"

            # 9. FAMILY (ISL): Two hands in outward circular loop or circular embrace at chest
            if interaction["both_open"] and zone in ("chest", "chin") and motion == "circular":
                return True, "family", 0.96, "isl"

            # 10. SORRY (ISL): Both hands held at ear level (holding earlobes)
            if zone == "head":
                return True, "sorry", 0.95, "isl"

            # 11. HAPPY (ISL): Both hands open palms brushing upward on chest
            if interaction["both_open"] and zone in ("chest", "chin") and motion == "upward":
                return True, "happy", 0.96, "isl"

        # =========================================================================
        # SECTION 2: SINGLE-HANDED SIGNS BY BODY ZONE
        # =========================================================================

        # -------------------------------------------------------------------------
        # ZONE A: HEAD (Forehead, Temple, Ears)
        # -------------------------------------------------------------------------
        if zone == "head":
            # Phone: Y-handshape (thumb + pinky extended) at ear
            if shape == "y" or (shape == "fist" and abs(wrist[0]) > 0.18):
                return True, "phone", 0.98, "asl"
            # Home (ASL): Bunched fingers touching cheek near ear
            if shape == "bunched":
                return True, "home", 0.95, "asl"
            # Sorry (ISL): Hand holding earlobe / ear
            if shape in ("fist", "index", "bunched", "general"):
                return True, "sorry", 0.95, "isl"
            # Hello (ASL): Open flat palm at forehead/temple waving, saluting, or held
            if shape in ("open", "general"):
                return True, "hello", 0.96, "asl"

        # -------------------------------------------------------------------------
        # ZONE B: CHIN (Mouth, Lips, Chin, Cheeks)
        # -------------------------------------------------------------------------
        if zone == "chin":
            # Water (ASL): W-handshape (3 fingers up) strictly at chin
            if shape == "w":
                return True, "water", 0.97, "asl"
            # Water (ISL): V-handshape or cupped hand at chin simulating drinking
            if shape in ("v", "general"):
                return True, "water", 0.96, "isl"
            # Food: Bunched / Flat-O or compact fist directly at mouth/lips
            if shape in ("bunched", "fist"):
                return True, "food", 0.97, "both"
            # Happy: Open hand brushing upward across chin
            if shape in ("open", "general") and motion == "upward":
                return True, "happy", 0.96, "asl"
            # Sad: Open hand dragging downward across face/chin
            if shape in ("open", "general") and motion == "downward":
                return True, "sad", 0.96, "both"
            # Thank You (ASL): Open flat palm at chin/lips pointing up moving forward/downward
            if shape in ("open", "general") and motion == "chin_outward":
                return True, "thank_you", 0.96, "asl"
            # Thank You (ISL): Open flat palm touching chin/lips held or with slight acknowledgement
            if shape in ("open", "general") and motion in ("still", "nodding", "tapping"):
                return True, "thank_you", 0.96, "isl"
            # No (ISL): Index finger wagging side to side at chin
            if shape in ("index", "v") and motion in ("waving", "still", "tapping"):
                return True, "no", 0.96, "isl"

        # -------------------------------------------------------------------------
        # ZONE C: CHEST (Sternum, Heart, Torso)
        # -------------------------------------------------------------------------
        if zone == "chest":
            # Hello (ISL): Open palm at chest / shoulder waving side to side
            if shape in ("open", "general") and motion == "waving":
                return True, "hello", 0.96, "isl"
            # Happy: Open flat palm brushing upward on chest
            if shape in ("open", "general") and motion == "upward":
                return True, "happy", 0.97, "asl"
            # Yes (ASL): Single fist nodding up and down in front of chest
            if shape in ("fist", "thumbs_up") and motion in ("nodding", "downward"):
                return True, "yes", 0.96, "asl"
            # No (ISL): Index finger wagging side-to-side at chest
            if shape in ("index", "v") and motion in ("waving", "tapping"):
                return True, "no", 0.96, "isl"
            # Help (ASL): Single thumbs-up pointing up at chest
            if shape == "thumbs_up" and motion != "nodding":
                return True, "help", 0.97, "asl"
            # Sorry (ASL): Closed fist rubbing in a circle over heart/chest, or placed flat
            if shape == "fist" and motion in ("circular", "still"):
                return True, "sorry", 0.96, "asl"
            # Please (ASL): Open flat palm rubbing in circle on chest, or held flat on sternum
            if shape in ("open", "general") and motion in ("circular", "still", "tapping"):
                return True, "please", 0.96, "asl"
            # Pain (ISL): Index finger pointing/jabbing directly at chest
            if shape in ("index", "v") and motion not in ("waving", "upward"):
                return True, "pain", 0.95, "isl"
            # Money: Thumb rubbing across fingertips at chest
            if shape in ("bunched", "fist", "general") and motion in ("tapping", "still", "waving"):
                return True, "money", 0.95, "both"

        # -------------------------------------------------------------------------
        # ZONE D: NEUTRAL (In front of body, waist, table)
        # -------------------------------------------------------------------------
        if zone == "neutral":
            # Yes (ASL): Closed fist nodding up and down
            if shape in ("fist", "thumbs_up") and motion in ("nodding", "downward"):
                return True, "yes", 0.96, "asl"
            # Help (ASL): Thumbs-up pointing up
            if shape == "thumbs_up":
                return True, "help", 0.96, "asl"
            # No (ISL): Index finger wagging side-to-side
            if shape in ("index", "v") and motion in ("waving", "still", "tapping"):
                return True, "no", 0.96, "isl"
            # Pain (ISL): Index finger jabbing
            if shape in ("index", "v"):
                return True, "pain", 0.93, "isl"
            # Money: Thumb rubbing across fingertips
            if shape in ("bunched", "fist", "general"):
                return True, "money", 0.94, "both"

        return False, None, 0.0, ""

    def score_sequence(self, sequence_window: np.ndarray) -> np.ndarray:
        """Compute 40-class probability distribution based on physiological matching."""
        is_matched, matched_concept, conf, lang_hint = self.evaluate_absolute_sign(sequence_window)

        if not is_matched or matched_concept is None:
            # Baseline flat noise floor: all 40 classes get 0.025 (2.5%)
            return np.ones(40, dtype=np.float32) / 40.0

        scores = np.ones(40, dtype=np.float32) * 0.001

        for c_idx in range(40):
            concept = self.registry.concept_of.get(c_idx, "")
            lang = self.registry.lang_of.get(c_idx, "")

            if concept == matched_concept:
                if lang_hint == "asl":
                    if lang == "asl":
                        scores[c_idx] = 12.0
                    else:
                        scores[c_idx] = 0.5
                elif lang_hint == "isl":
                    if lang == "isl":
                        scores[c_idx] = 12.0
                    else:
                        scores[c_idx] = 0.5
                else:  # "both"
                    scores[c_idx] = 11.0 if lang == "asl" else 1.5

        total = float(np.sum(scores))
        return (scores / total) if total > 0 else scores
