"""Interactive Streamlit application for dual-language sign language translation.

Features:
- Real-time video feed / webcam loop with MediaPipe landmark skeleton rendering
- Tri-mode operational toggle: ISL | ASL | AUTO (Beta) with detected language badges
- Real-time recognized signs log and interactive phrase composer
- Multi-tier translation panel in large font with verified/machine quality badges
- Offline Text-To-Speech audio player
- "Add New Sign" recording flow and "Retrain Model" launcher
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Setup page config
st.set_page_config(
    page_title="SignBridge",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.buffer import SequenceBuffer
from src.features.kinematics import (
    KinematicGestureClassifier,
    analyze_hand_shape,
    analyze_hand_zone,
    analyze_motion_trajectory,
    analyze_two_hand_interaction,
)
from src.features.normalize import (
    draw_styled_landmarks,
    draw_unicode_text,
    extract_and_normalize_landmarks,
    init_holistic,
)
from src.inference.mode import InferenceModeManager
from src.inference.stabilize import PredictionStabilizer
from src.label_map import (
    DEFAULT_REGISTRY,
    LabelRegistry,
)
from src.model import FastSignPredictor
from src.phrasebuilder import PhraseBuilder
from src.translate import TranslationEngine
from src.tts import get_tts_engine
from src.avatar3d import ALL_20_SIGNS, render_3d_avatar

# Custom CSS for modern glassmorphic look
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .badge-isl {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-asl {
        background-color: #3B82F6;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-auto {
        background-color: #8B5CF6;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .trans-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        margin-top: 10px;
    }
    .trans-text {
        font-size: 2rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .quality-curated {
        background-color: #059669;
        color: white;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
    }
    .quality-machine {
        background-color: #D97706;
        color: white;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #94A3B8;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_system_resources():
    """Load model, label registry, translator, and TTS engine."""
    tflite_path = PROJECT_ROOT / "models" / "model.tflite"
    keras_path = PROJECT_ROOT / "models" / "model.keras"
    label_path = PROJECT_ROOT / "label_map.json"

    registry = LabelRegistry.from_file(label_path) if label_path.exists() else DEFAULT_REGISTRY
    model = FastSignPredictor(tflite_path=tflite_path, keras_path=keras_path)
    phrase_builder = PhraseBuilder()
    translator = TranslationEngine(phrase_builder=phrase_builder)
    tts = get_tts_engine()
    kinematic_clf = KinematicGestureClassifier()

    return registry, model, phrase_builder, translator, tts, kinematic_clf


registry, model, phrase_builder, translator, tts, kinematic_clf = load_system_resources()

# Initialize session states
if "recent_signs" not in st.session_state:
    st.session_state.recent_signs = []
if "current_phrase" not in st.session_state:
    st.session_state.current_phrase = ""
if "current_translation" not in st.session_state:
    st.session_state.current_translation = ""
if "quality_tag" not in st.session_state:
    st.session_state.quality_tag = ""
if "detected_lang" not in st.session_state:
    st.session_state.detected_lang = "AUTO"
if "is_locked" not in st.session_state:
    st.session_state.is_locked = False
if "live_sign" not in st.session_state:
    st.session_state.live_sign = ""
if "last_spoken_output" not in st.session_state:
    st.session_state.last_spoken_output = ""
if "clear_output_at" not in st.session_state:
    st.session_state.clear_output_at = 0.0

# Sidebar Controls
st.sidebar.markdown("### ⚙️ System Controls")

run_mode = "AUTO"
mode_key = "auto"

target_lang_choice = st.sidebar.selectbox(
    "Target Spoken Language (Indian Languages)",
    [
        ("ta", "Tamil (தமிழ்)"),
        ("te", "Telugu (తెలుగు)"),
        ("bn", "Bengali (বাংলা)"),
        ("mr", "Marathi (मराठी)"),
        ("gu", "Gujarati (ગુજરાતી)"),
        ("kn", "Kannada (ಕನ್ನಡ)"),
        ("ml", "Malayalam (മലയാളം)"),
        ("pa", "Punjabi (ਪੰਜਾਬੀ)"),
        ("ur", "Urdu (اردو)"),
        ("en", "English (Pivot)"),
        ("hi", "Hindi (हिंदी)"),
    ],
    format_func=lambda x: x[1],
)
target_lang_code = target_lang_choice[0]

enable_audio = st.sidebar.checkbox("Enable Offline Speech (TTS)", value=True)
show_skeleton = st.sidebar.checkbox("Show Tracing Lines", value=False)
enable_webcam = st.sidebar.checkbox("Enable Live Webcam Tracking", value=True)

# Sign Catalog Download
catalog_path = PROJECT_ROOT / "SIGN_CATALOG.txt"
if catalog_path.exists():
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_content = f.read()
    st.sidebar.download_button(
        label="📄 Download Sign Catalog (TXT)",
        data=catalog_content,
        file_name="SIGN_CATALOG.txt",
        mime="text/plain",
        help="Download complete dictionary of 40 ASL & ISL signs with handshapes, motions, and translations.",
    )

# Main Header
col_header, col_badge = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">SignBridge</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer">Recognized-signs-to-simple-English pivot with multi-tier offline translation. Real-time inference.</div>',
        unsafe_allow_html=True,
    )

with col_badge:
    cur_lang = getattr(st.session_state, "detected_lang", "AUTO")
    badge_cls = "badge-isl" if cur_lang == "ISL" else ("badge-asl" if cur_lang == "ASL" else "badge-auto")
    badge_text = f"AUTO: {cur_lang}"
    lock_status = "🔒 Locked" if st.session_state.is_locked else "⚡ Live Tracking"
    st.markdown(
        f'<div style="text-align:right; margin-top:10px;"><span class="{badge_cls}">{badge_text}</span><br><small>{lock_status}</small></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")


def get_trans_html(translation_text: str, quality_tag: str, target_lang_name: str) -> str:
    """Generate HTML card for real-time translation."""
    q_badge = "quality-curated" if "curated" in quality_tag else "quality-machine"
    badge_label = "Verified Phrasebook" if "curated" in quality_tag else ("Machine Translation" if quality_tag else "")
    tag_html = f'<span class="{q_badge}">{badge_label}</span>' if badge_label else ""
    return (
        f'<div class="trans-box">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
        f'<span style="color:#94A3B8; font-size:0.9rem;">Target Language: {target_lang_name}</span>'
        f'{tag_html}'
        f'</div>'
        f'<div class="trans-text">{translation_text or "Awaiting sign input..."}</div>'
        f'</div>'
    )


# Declare both columns FIRST so real-time placeholders can be populated during video stream
col_video, col_translation = st.columns([1.3, 1.0])

with col_translation:
    st.subheader("🌐 Real-Time Translation & Output")
    trans_box_placeholder = st.empty()
    trans_box_placeholder.markdown(
        get_trans_html(st.session_state.current_translation, st.session_state.quality_tag, target_lang_choice[1]),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 📝 English Pivot Sentence")
    pivot_status_placeholder = st.empty()
    pivot_status_placeholder.info(f"Pivot: {st.session_state.current_phrase or '(awaiting signs...)'}")

    st.markdown("---")
    st.subheader("👤 3D Sign Language Avatar")
    st.caption("Click any sign in the interactive panel below to see the 3D human avatar perform the gesture.")
    render_3d_avatar("hello", height=580)

with col_video:
    frame_placeholder = st.empty()

    if not enable_webcam:
        frame_placeholder.info("📹 Live webcam feed paused. Select any sign on the right to view the 3D human signer, or re-enable the camera in the sidebar.")
    elif model is None:
        st.error("Trained model not found! Please run python run_pipeline.py first.")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam device 0. Please verify your camera is connected.")
            if st.button("🔄 Retry Camera Connection"):
                st.rerun()
        else:
            # Set high-speed 640x480 capture with zero buffer lag
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # High-precision Holistic pipeline with full body, face, and hand tracking
            holistic = init_holistic(model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.55, min_tracking_confidence=0.55)
            buffer = SequenceBuffer(sequence_length=30, feature_dim=258)
            mode_mgr = InferenceModeManager(isl_idx=registry.isl_idx, asl_idx=registry.asl_idx)
            mode_mgr.set_mode(mode_key)
            stabilizer = PredictionStabilizer(
                voting_window=4,
                fixed_mode_threshold=0.30,
                auto_mode_threshold=0.25,
                margin_threshold=0.05,
                motion_energy_threshold=0.0,
                cooldown_seconds=1.2,
            )

            frame_idx = 0
            prev_time = time.time()
            fps_display = 30.0

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    # Compute rolling FPS
                    curr_time = time.time()
                    dt = curr_time - prev_time
                    prev_time = curr_time
                    if dt > 0:
                        fps_display = 0.85 * fps_display + 0.15 * (1.0 / dt)

                    # Run MediaPipe on raw camera frame for anatomically accurate left vs right hand tracking
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = holistic.process(rgb)
                    rgb.flags.writeable = True

                    if show_skeleton:
                        annotated = draw_styled_landmarks(rgb, results)
                    else:
                        annotated = rgb.copy()

                    # Flip annotated frame horizontally for natural selfie mirror preview
                    annotated = cv2.flip(annotated, 1)

                    feature_vector = extract_and_normalize_landmarks(results)
                    buffer.append(feature_vector)

                    # Check if hands are in frame (MediaPipe non-zero coordinates)
                    hands_present = bool(np.any(feature_vector[132:258] != 0.0))

                    # Check if currently in word display hold (clearing cycle)
                    is_clearing = getattr(st.session_state, "clear_output_at", 0.0) > 0.0

                    # Check for expiration of word display window: CLEAR the output column first!
                    if is_clearing and time.time() >= st.session_state.clear_output_at:
                        st.session_state.clear_output_at = 0.0
                        st.session_state.current_translation = ""
                        st.session_state.quality_tag = ""
                        st.session_state.current_phrase = ""
                        st.session_state.last_spoken_output = ""
                        st.session_state.live_sign = ""
                        phrase_builder.reset()
                        stabilizer.reset()
                        trans_box_placeholder.markdown(
                            get_trans_html("", "", target_lang_choice[1]),
                            unsafe_allow_html=True,
                        )
                        pivot_status_placeholder.info("Pivot: (awaiting signs...)")
                        is_clearing = False

                    # STRICT GATING: Only recognize and show a new word if the output column has cleared!
                    if not is_clearing and frame_idx % 2 == 0 and len(buffer) >= 6 and hands_present:
                        seq = np.expand_dims(buffer.get_sequence(), axis=0)

                        # PHYSIOLOGICAL SHAPE MATCHING: Matches exact handshape, zone, motion & two-hand interaction
                        is_matched, matched_concept, match_conf, lang_hint = kinematic_clf.evaluate_absolute_sign(seq[0])

                        lh_feat = feature_vector[132:195]
                        rh_feat = feature_vector[195:258]
                        primary_feat = rh_feat if np.any(rh_feat != 0.0) else lh_feat
                        cur_shape = analyze_hand_shape(primary_feat)
                        cur_zone = analyze_hand_zone(primary_feat)

                        if not is_matched:
                            if cur_shape != "none" and cur_zone != "none":
                                st.session_state.live_sign = f"Tracking: {cur_shape.upper()} in {cur_zone.upper()}"
                            else:
                                st.session_state.live_sign = "(Hold sign clearly)"
                            stabilizer.reset()
                        else:
                            class_preds, lang_preds = model.predict(seq)
                            l_probs = lang_preds[0] if lang_preds is not None else None

                            # Real-time anatomical kinematic feature scoring (absolute shape, zone, motion)
                            kin_scores = kinematic_clf.score_sequence(seq[0])

                            # Multi-modal fusion: 85% Absolute Kinematic Grounding + 15% Temporal Neural Prior
                            fused_scores = 0.85 * kin_scores + 0.15 * class_preds[0]
                            fused_scores = fused_scores / float(np.sum(fused_scores))

                            masked_probs, detected_lang, is_locked = mode_mgr.update(fused_scores, l_probs)

                            st.session_state.detected_lang = detected_lang.upper()
                            st.session_state.is_locked = is_locked

                            # Real-time top candidate for immediate visual feedback
                            top_c_idx = int(np.argmax(masked_probs))
                            top_c_conf = float(masked_probs[top_c_idx])
                            top_concept = registry.concept_of.get(top_c_idx, "")
                            top_lang = registry.lang_of.get(top_c_idx, detected_lang).upper()
                            st.session_state.live_sign = f"[{top_lang}] {top_concept.capitalize()} ({top_c_conf*100:.0f}%)"

                            stabilized = stabilizer.process(feature_vector, masked_probs, mode=mode_key)
                            if stabilized is not None:
                                c_idx, conf = stabilized
                                concept = registry.concept_of.get(c_idx, "")
                                lang_sign = registry.lang_of.get(c_idx, detected_lang).upper()

                                st.session_state.recent_signs.append(f"[{lang_sign}] {concept.capitalize()} ({conf:.2f})")
                                if len(st.session_state.recent_signs) > 8:
                                    st.session_state.recent_signs.pop(0)

                                # Discrete single-word output: reset phrase builder and add single concept
                                phrase_builder.reset()
                                phrase_builder.add_concept(concept)
                                phrase = phrase_builder.get_current_phrase()
                                st.session_state.current_phrase = phrase

                                trans_text, tag = translator.translate([concept], target_lang_code)
                                st.session_state.current_translation = trans_text
                                st.session_state.quality_tag = tag

                                # RENDER IN TRANSLATION PANEL FIRST:
                                trans_box_placeholder.markdown(
                                    get_trans_html(trans_text, tag, target_lang_choice[1]),
                                    unsafe_allow_html=True,
                                )
                                pivot_status_placeholder.info(f"Pivot: {phrase}")

                                # DO NOT SPEAK until valid output has come out in the output column
                                if (
                                    enable_audio
                                    and trans_text
                                    and trans_text.strip()
                                    and trans_text != getattr(st.session_state, "last_spoken_output", "")
                                ):
                                    st.session_state.last_spoken_output = trans_text
                                    tts.speak_async(trans_text, target_lang_code)

                                # HOLD on screen for 1.8s, during which NO new signs will be recognized or shown
                                st.session_state.clear_output_at = time.time() + 1.8
                                stabilizer.reset()

                    # REAL-TIME ON-FRAME HEADS-UP DISPLAY (HUD):
                    h, w, _ = annotated.shape

                    # Top Telemetry Banner for Live Body, Hands, Zones & Movement Recognition
                    lh_feat = feature_vector[132:195]
                    rh_feat = feature_vector[195:258]
                    lh_shape = analyze_hand_shape(lh_feat) if np.any(lh_feat != 0.0) else "none"
                    rh_shape = analyze_hand_shape(rh_feat) if np.any(rh_feat != 0.0) else "none"
                    lh_zone = analyze_hand_zone(lh_feat) if lh_shape != "none" else ""
                    rh_zone = analyze_hand_zone(rh_feat) if rh_shape != "none" else ""
                    motion_curr = analyze_motion_trajectory(buffer.get_sequence()) if len(buffer) >= 6 else "still"

                    # Two-hand interaction diagnosis
                    inter = analyze_two_hand_interaction(lh_feat, rh_feat)
                    inter_str = ""
                    if inter["namaste_prayer"]:
                        inter_str = " | Pose: NAMASTE"
                    elif inter["inverted_v_roof"]:
                        inter_str = " | Pose: ROOF"
                    elif inter["fist_on_palm"]:
                        inter_str = " | Pose: FIST-ON-PALM"
                    elif inter["hand_on_wrist"]:
                        inter_str = " | Pose: PULSE-TOUCH"
                    elif inter["dominant_on_palm"]:
                        inter_str = " | Pose: PALM-TOUCH"
                    elif inter["both_fists"] and inter["hands_touching"]:
                        inter_str = " | Pose: FISTS-TOUCH"
                    elif inter["both_open"] and inter["hands_touching"]:
                        inter_str = " | Pose: CLAP"
                    elif inter["index_tips_meeting"]:
                        inter_str = " | Pose: INDEX-TOUCH"

                    # Draw sleek semi-transparent top banner
                    top_bar = annotated.copy()
                    cv2.rectangle(top_bar, (0, 0), (w, 32), (15, 23, 42), -1)
                    cv2.addWeighted(top_bar, 0.85, annotated, 0.15, 0, annotated)
                    cv2.line(annotated, (0, 32), (w, 32), (56, 189, 248), 1)

                    telemetry_str = f"Body: Active | LH: {lh_shape.upper()}{' (' + lh_zone + ')' if lh_zone else ''} | RH: {rh_shape.upper()}{' (' + rh_zone + ')' if rh_zone else ''} | Motion: {motion_curr.upper()}{inter_str}"
                    cv2.putText(annotated, telemetry_str, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)

                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (0, h - 55), (w, h), (15, 23, 42), -1)
                    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

                    status_str = f"Mode: {run_mode.split()[0]} | Detected: {st.session_state.detected_lang} | {fps_display:.0f} FPS"
                    cv2.putText(annotated, status_str, (12, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (148, 163, 184), 1, cv2.LINE_AA)

                    if is_clearing and (st.session_state.current_phrase or st.session_state.current_translation):
                        recognized_word = (st.session_state.current_phrase or "Recognized").title()
                        cv2.putText(annotated, f"Word: {recognized_word}", (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (16, 185, 129), 2, cv2.LINE_AA)
                    elif hands_present:
                        live_text = st.session_state.get("live_sign", "Hands active - tracking...")
                        cv2.putText(annotated, f"Sign: {live_text}", (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (56, 189, 248), 2, cv2.LINE_AA)
                    else:
                        st.session_state.live_sign = ""
                        cv2.putText(annotated, "Place hands in view to sign", (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (148, 163, 184), 1, cv2.LINE_AA)

                    if st.session_state.current_translation:
                        preview_trans = st.session_state.current_translation[:30]
                        annotated = draw_unicode_text(
                            annotated,
                            f"Trans: {preview_trans}",
                            (w // 2, h - 30),
                            color=(16, 185, 129),
                            font_size=18,
                            is_bgr=False,
                        )

                    frame_placeholder.image(annotated, channels="RGB", use_column_width=True)
            finally:
                cap.release()
                holistic.close()

