import numpy as np
from src.features.normalize import FEATURE_DIM, POSE_DIM, HAND_DIM

def create_mock_sequence(
    gesture=None,
    active_hand="right",
    shape="open",
    zone="head",
    motion="waving",
    two_hands=False,
    length=30
):
    """Generate synthetic 30-frame MediaPipe sequence with specified anatomical properties."""
    seq = np.zeros((length, FEATURE_DIM), dtype=np.float32)

    # Pose: shoulders at (0,0,0)
    for t in range(length):
        seq[t, 11*4:11*4+4] = [-0.5, 0.0, 0.0, 1.0] # Left shoulder
        seq[t, 12*4:12*4+4] = [ 0.5, 0.0, 0.0, 1.0] # Right shoulder

    # Zone y offsets
    zone_y = {
        "head": -0.65,
        "chin": -0.25,
        "chest": 0.10,
        "neutral": 0.45
    }[zone]

    # Motion trajectories
    for t in range(length):
        progress = t / float(length - 1)
        if motion == "waving":
            dx = 0.08 * np.sin(progress * 4 * np.pi)
            dy = 0.0
            dz = 0.0
        elif motion == "nodding":
            dx = 0.0
            dy = 0.06 * np.sin(progress * 4 * np.pi)
            dz = 0.0
        elif motion == "circular":
            dx = 0.06 * np.cos(progress * 2 * np.pi)
            dy = 0.06 * np.sin(progress * 2 * np.pi)
            dz = 0.0
        elif motion == "chin_outward":
            dx = 0.0
            dy = 0.04 * progress
            dz = 0.10 * progress
        elif motion == "upward":
            dx = 0.0
            dy = -0.12 * progress
            dz = 0.0
        elif motion == "downward":
            dx = 0.0
            dy = 0.12 * progress
            dz = 0.0
        else: # still
            dx = dy = dz = 0.0

        # Construct right hand landmarks
        rh_start = POSE_DIM + HAND_DIM # 195
        wrist = np.array([0.25 + dx, zone_y + dy, 0.0 + dz], dtype=np.float32)
        hand_lms = np.zeros((21, 3), dtype=np.float32)
        hand_lms[0] = wrist

        # Helper to set finger
        def set_finger(lms, base_w, mcp_idx, tip_idx, extended):
            mcp = base_w + np.array([(mcp_idx - 9)*0.02, -0.05, 0.0], dtype=np.float32)
            pip = mcp + np.array([0.0, -0.03, 0.0], dtype=np.float32)
            dip = pip + np.array([0.0, -0.02, 0.0], dtype=np.float32)
            if extended:
                tip = dip + np.array([0.0, -0.03, 0.0], dtype=np.float32)
            else:
                tip = mcp + np.array([0.0, 0.01, 0.02], dtype=np.float32)
            lms[mcp_idx] = mcp
            lms[mcp_idx+1] = pip
            lms[mcp_idx+2] = dip
            lms[tip_idx] = tip

        # Thumb
        thumb_ext = shape in ("thumbs_up", "y", "open")
        hand_lms[1] = wrist + np.array([-0.02, -0.02, 0.0])
        hand_lms[2] = wrist + np.array([-0.04, -0.03, 0.0])
        hand_lms[3] = wrist + np.array([-0.06, -0.04, 0.0])
        if thumb_ext:
            hand_lms[4] = wrist + np.array([-0.08, -0.07, 0.0])
        else:
            hand_lms[4] = wrist + np.array([-0.03, -0.03, 0.0])

        # Fingers: index (5..8), middle (9..12), ring (13..16), pinky (17..20)
        ext_map = {
            "open": (True, True, True, True),
            "fist": (False, False, False, False),
            "thumbs_up": (False, False, False, False),
            "index": (True, False, False, False),
            "v": (True, True, False, False),
            "w": (True, True, True, False),
            "y": (False, False, False, True),
        }
        idx_e, mid_e, rng_e, pnk_e = ext_map.get(shape, (True, True, True, True))
        set_finger(hand_lms, wrist, 5, 8, idx_e)
        set_finger(hand_lms, wrist, 9, 12, mid_e)
        set_finger(hand_lms, wrist, 13, 16, rng_e)
        set_finger(hand_lms, wrist, 17, 20, pnk_e)

        if shape == "bunched":
            hand_lms[4] = wrist + np.array([-0.02, -0.04, 0.0])
            for k in (8, 12, 16, 20):
                hand_lms[k] = wrist + np.array([-0.02, -0.04, 0.01])

        if gesture == "doctor":
            lh_wrist = np.array([0.05, zone_y, 0.0], dtype=np.float32)
            hand_lms[8] = lh_wrist + np.array([0.01, 0.01, 0.0])
            hand_lms[12] = lh_wrist + np.array([0.02, 0.01, 0.0])

        elif gesture == "medicine":
            lh_wrist = np.array([0.05, zone_y, 0.0], dtype=np.float32)
            lh_palm = lh_wrist + np.array([0.0, -0.10, 0.0])
            hand_lms[8] = lh_palm + np.array([0.01, 0.01, 0.0])

        elif gesture == "roof":
            rh_wrist = np.array([0.20, zone_y, 0.0], dtype=np.float32)
            hand_lms[0] = rh_wrist
            hand_lms[8] = np.array([0.02, zone_y - 0.12, 0.0], dtype=np.float32)

        seq[t, rh_start:rh_start+HAND_DIM] = hand_lms.flatten()

        if two_hands:
            lh_start = POSE_DIM # 132
            lh_wrist = np.array([-0.20 if gesture == "roof" else 0.05, zone_y, 0.0], dtype=np.float32)
            lh_lms = np.zeros((21, 3), dtype=np.float32)
            lh_lms[0] = lh_wrist

            lh_shape = "open" if gesture in ("school", "medicine", "doctor", "roof") else shape
            l_idx_e, l_mid_e, l_rng_e, l_pnk_e = ext_map.get(lh_shape, (True, True, True, True))
            set_finger(lh_lms, lh_wrist, 5, 8, l_idx_e)
            set_finger(lh_lms, lh_wrist, 9, 12, l_mid_e)
            set_finger(lh_lms, lh_wrist, 13, 16, l_rng_e)
            set_finger(lh_lms, lh_wrist, 17, 20, l_pnk_e)

            if gesture == "doctor":
                lh_lms[9] = lh_wrist + np.array([0.0, -0.12, 0.0])
            elif gesture == "medicine":
                lh_lms[9] = lh_wrist + np.array([0.0, -0.10, 0.0])
            elif gesture == "roof":
                lh_lms[8] = np.array([-0.02, zone_y - 0.12, 0.0], dtype=np.float32)

            seq[t, lh_start:lh_start+HAND_DIM] = lh_lms.flatten()

    return seq
