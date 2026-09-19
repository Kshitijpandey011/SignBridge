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
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st

# Setup page config
st.set_page_config(
    page_title="ISL + ASL Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.buffer import SequenceBuffer
from src.features.normalize import (
    draw_styled_landmarks,
    extract_and_normalize_landmarks,
    init_holistic,
)
from src.inference.mode import InferenceModeManager
from src.inference.stabilize import PredictionStabilizer
from src.label_map import (
    CANONICAL_CONCEPTS,
    CONCEPT_DISPLAY_TEXT,
    DEFAULT_REGISTRY,
    LabelRegistry,
)
from src.model import load_inference_model
from src.phrasebuilder import PhraseBuilder
from src.translate import TranslationEngine
from src.tts import get_tts_engine

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
    model_path = PROJECT_ROOT / "models" / "model.keras"
    label_path = PROJECT_ROOT / "label_map.json"

    registry = LabelRegistry.from_file(label_path) if label_path.exists() else DEFAULT_REGISTRY
    model = load_inference_model(model_path) if model_path.exists() else None
    phrase_builder = PhraseBuilder()
    translator = TranslationEngine(phrase_builder=phrase_builder)
    tts = get_tts_engine()

    return registry, model, phrase_builder, translator, tts


registry, model, phrase_builder, translator, tts = load_system_resources()

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

# Sidebar Controls
st.sidebar.markdown("### ⚙️ System Controls")

run_mode = st.sidebar.radio(
    "Operational Run Mode",
    ["AUTO (Beta)", "ISL Mode", "ASL Mode"],
    index=0,
    help="AUTO allows competition between ISL and ASL with soft-lock. Fixed modes mask opposite language classes to 0.",
)
mode_key = "auto" if "AUTO" in run_mode else ("isl" if "ISL" in run_mode else "asl")

target_lang_choice = st.sidebar.selectbox(
    "Target Spoken Language",
    [
        ("en", "English (Pivot)"),
        ("hi", "Hindi (हिंदी)"),
        ("ta", "Tamil (தமிழ்)"),
        ("es", "Spanish (Español)"),
        ("fr", "French (Français)"),
    ],
    format_func=lambda x: x[1],
)
target_lang_code = target_lang_choice[0]

enable_audio = st.sidebar.checkbox("Enable Offline Speech (TTS)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Add New Sign / Calibration")
with st.sidebar.expander("Record New Sign Samples"):
    calib_lang = st.selectbox("Calibration Language", ["ISL", "ASL"]).lower()
    calib_concept = st.selectbox("Sign Concept", CANONICAL_CONCEPTS)
    calib_signer = st.text_input("Signer ID", value="user_new")
    calib_takes = st.slider("Takes to Record", 5, 30, 10)

    if st.button("Start Recording Clips"):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "record_sign.py"),
            "--lang",
            calib_lang,
            "--concept",
            calib_concept,
            "--signer-id",
            calib_signer,
            "--takes",
            str(calib_takes),
        ]
        st.info("Launching recording window... Look at your camera.")
        res = subprocess.run(cmd, capture_output=True, text=True)
        st.success(f"Finished recording {calib_takes} takes!")

    if st.button("Retrain Model Now"):
        with st.spinner("Retraining model on all recordings (this will take a minute)..."):
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "train.py"),
                "--epochs",
                "15",
                "--batch-size",
                "16",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            st.success("Model retraining complete! Reloading weights.")
            st.cache_resource.clear()
            st.rerun()

# Main Header
col_header, col_badge = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">Offline ISL + ASL Sign Translator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer">Recognized-signs-to-simple-English pivot with multi-tier offline translation. Not grammatical sign syntax.</div>',
        unsafe_allow_html=True,
    )

with col_badge:
    badge_cls = "badge-auto" if mode_key == "auto" else ("badge-isl" if mode_key == "isl" else "badge-asl")
    lock_status = "🔒 Locked" if st.session_state.is_locked else "🔓 Tracking"
    st.markdown(
        f'<div style="text-align:right; margin-top:10px;"><span class="{badge_cls}">{run_mode}</span><br><small>{lock_status}</small></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# Main Interface Layout
col_video, col_translation = st.columns([1.3, 1.0])

with col_video:
    st.subheader("📹 Live Video Stream")
    camera_active = st.toggle("Activate Webcam", value=False)
    frame_placeholder = st.empty()

    if not camera_active:
        frame_placeholder.info("Webcam is inactive. Click 'Activate Webcam' above to start real-time recognition.")
    else:
        if model is None:
            st.error("Trained model not found! Please run train.py first.")
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access webcam device 0.")
            else:
                holistic = init_holistic()
                buffer = SequenceBuffer(sequence_length=30, feature_dim=258)
                mode_mgr = InferenceModeManager(isl_idx=registry.isl_idx, asl_idx=registry.asl_idx)
                mode_mgr.set_mode(mode_key)
                stabilizer = PredictionStabilizer(voting_window=10)

                stop_stream = st.button("Stop Webcam Stream")
                while cap.isOpened() and not stop_stream:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = holistic.process(rgb)
                    rgb.flags.writeable = True

                    annotated = draw_styled_landmarks(rgb, results)
                    feature_vector = extract_and_normalize_landmarks(results)
                    buffer.append(feature_vector)

                    if buffer.is_ready():
                        seq = np.expand_dims(buffer.get_sequence(), axis=0)
                        class_preds, lang_preds = model.predict(seq, verbose=0)
                        l_probs = lang_preds[0] if lang_preds is not None else None
                        masked_probs, detected_lang, is_locked = mode_mgr.update(class_preds[0], l_probs)

                        st.session_state.detected_lang = detected_lang.upper()
                        st.session_state.is_locked = is_locked

                        stabilized = stabilizer.process(feature_vector, masked_probs, mode=mode_key)
                        if stabilized is not None:
                            c_idx, conf = stabilized
                            concept = registry.concept_of.get(c_idx, "")
                            lang_sign = registry.lang_of.get(c_idx, detected_lang).upper()

                            st.session_state.recent_signs.append(f"[{lang_sign}] {concept.capitalize()} ({conf:.2f})")
                            if len(st.session_state.recent_signs) > 8:
                                st.session_state.recent_signs.pop(0)

                            phrase_builder.add_concept(concept)
                            phrase = phrase_builder.get_current_phrase()
                            st.session_state.current_phrase = phrase

                            trans_text, tag = translator.translate(
                                phrase_builder.get_buffered_concepts(), target_lang_code
                            )
                            st.session_state.current_translation = trans_text
                            st.session_state.quality_tag = tag

                            if enable_audio:
                                tts.speak_async(trans_text, target_lang_code)

                    frame_placeholder.image(annotated, channels="RGB", use_container_width=True)

                cap.release()
                holistic.close()

with col_translation:
    st.subheader("🌐 Translation & Output")

    # Translation Banner
    st.markdown('<div class="trans-box">', unsafe_allow_html=True)
    q_badge = "quality-curated" if "curated" in st.session_state.quality_tag else "quality-machine"
    badge_label = "Verified Phrasebook" if "curated" in st.session_state.quality_tag else "Machine Translation"
    tag_html = (
        f'<span class="{q_badge}">{badge_label}</span>' if st.session_state.quality_tag else ""
    )

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#94A3B8; font-size:0.9rem;">Target Language: {target_lang_choice[1]}</span>
            {tag_html}
        </div>
        <div class="trans-text">{st.session_state.current_translation or "Awaiting sign input..."}</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Audio & Action Buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("🔊 Speak Translation"):
            if st.session_state.current_translation:
                tts.speak(st.session_state.current_translation, target_lang_code)
    with col_btn2:
        if st.button("🧹 Clear Phrase"):
            phrase_builder.reset()
            st.session_state.current_phrase = ""
            st.session_state.current_translation = ""
            st.session_state.quality_tag = ""
            st.rerun()
    with col_btn3:
        if st.button("📋 Copy Text"):
            st.toast("Text copied to clipboard!")

    st.markdown("---")
    st.markdown("#### 📝 English Pivot Sentence")
    user_edited_phrase = st.text_input(
        "Edit or manually enter pivot words:",
        value=st.session_state.current_phrase,
        placeholder="e.g. I need water.",
    )
    if st.button("Translate Manual Phrase"):
        if user_edited_phrase:
            # Tokenize words into concepts
            tokens = [w.strip().lower() for w in user_edited_phrase.replace(".", "").split()]
            trans_res, q_tag = translator.translate(tokens, target_lang_code)
            st.session_state.current_translation = trans_res
            st.session_state.quality_tag = q_tag
            st.rerun()

    st.markdown("#### 📜 Recent Signs Log")
    if st.session_state.recent_signs:
        for s in reversed(st.session_state.recent_signs):
            st.text(s)
    else:
        st.caption("No signs registered yet.")
