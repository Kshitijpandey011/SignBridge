# Project Architectural Decisions & Assumptions Log

This document tracks technical decisions, numeric constants, and conventions adopted across all project phases.

---

## Phase 1: Environment, Feature Pipeline & Data Ingestion

1. **UI Framework Selection**:
   - **Decision**: Streamlit (`app.py`) was selected over Tkinter/PyQt.
   - **Rationale**: Streamlit provides rapid, reactive component rendering for real-time video feeds, badge indicators, dropdowns, and dynamic styling while preserving multi-platform portability without native GUI thread conflicts.

2. **Landmark Normalization & Geometric Origin**:
   - **Decision**: Mid-shoulder origin translation and shoulder-width Euclidean distance scaling.
   - **MediaPipe Pose Landmarks**:
     - Left Shoulder: Index `11`
     - Right Shoulder: Index `12`
     - Origin = `((lm[11].x + lm[12].x) / 2, (lm[11].y + lm[12].y) / 2, (lm[11].z + lm[12].z) / 2)`
     - Scale = `max(sqrt(dx^2 + dy^2 + dz^2), 1e-4)`
   - **Rationale**: Normalizing all landmarks (pose and hands) relative to the mid-shoulder anchor and dividing by shoulder span ensures scale, height, camera distance, and perspective invariance.

3. **Feature Dimensionality Allocation**:
   - **Decision**: Exact 258 features per frame:
     - Pose: Indices `0:132` (33 landmarks * [x, y, z, visibility])
     - Left Hand: Indices `132:195` (21 landmarks * [x, y, z])
     - Right Hand: Indices `195:258` (21 landmarks * [x, y, z])
     - Face landmarks (468) are excluded to ensure real-time >30 FPS CPU inference.
   - **Zero-Filling**: Missing hands are zero-filled (63 float zeros). This preserves an explicit signal distinguishing one-handed ASL signs from two-handed ISL signs.

4. **Temporal Resampling for Public Datasets**:
   - **Decision**: Uniform linear temporal interpolation using `np.linspace(0, total_frames - 1, 30, dtype=int)`.
   - **Rationale**: Deterministic, zero-jitter sampling that accurately maps varying video FPS (24, 29.97, 60 FPS) to the model's fixed 30-frame window.

5. **Data Layout & Sidecar Metadata**:
   - **Decision**: `data/<lang>/<concept>/<signer_id>/<seq_id>.npy` accompanied by `<seq_id>.json` sidecars containing source, signer, and capture timestamps.

---

## Phase 2: Dataset Assembly, Model Training & Evaluation

6. **Signer Partitioning Strategy**:
   - **Decision**: Strict split by `signer_id`. Sequences from validation signers never appear in training.
   - **Rationale**: Prevents biometric leakage and memorization of individual signing styles, body proportions, or background artifacts.

7. **Data Augmentations**:
   - **Decision**: Five stochastic transformations:
     - Speed/time jitter: `±15%` temporal spline interpolation.
     - Gaussian coordinate noise: `N(0, 0.012)` applied only to active spatial coordinates (visibility and zero-filled hand slots protected).
     - Scale jitter: Uniform scaling in `[0.9, 1.1]`.
     - 2D rotation: `±10°` around mid-shoulder origin.
     - Mirror & Hand swap: Horizontal reflection (`x -> -x`), pose symmetric pairs swapped, and Left Hand slot `[132:195]` swapped with Right Hand slot `[195:258]`.

8. **Multi-Task Objective & Loss Weights**:
   - **Decision**: Combined objective: `class_loss + 0.3 * language_loss`.
   - **Rationale**: The secondary language classification head acts as an auxiliary regularizer, encouraging the LSTM recurrent backbone to extract sign-language typology features while preserving primary focus on concept classification.

9. **Model Format & Checkpoints**:
   - **Decision**: Saved in modern Keras `.keras` zip-container format, preserving custom multi-output topology and optimizer state.

---

## Phase 3: Real-Time Inference & Translation Layer

10. **Auto-Detection Hysteresis & Soft-Locking**:
    - **Decision**: 10-frame voting window. Locking onto a language requires >=7/10 frame wins with mean probability > 0.70. Once locked, switching requires >=8/10 votes in the opposite language.
    - **Rationale**: Prevents erratic UI flickering mid-sign during transitional hand movements.

11. **Stabilization & False-Positive Mitigation**:
    - **Decision**: Multi-stage gating:
      1. Hand presence check: If both hand slots are zero-filled, suppress classification immediately.
      2. Motion energy floor (0.08): Suppresses classification during resting/still poses.
      3. 10-frame top-class voting with 0.90 threshold (0.80 in AUTO mode).
      4. 1.0s emission cooldown on repeated class predictions.

12. **Multi-Tier Translation Hierarchy**:
    - **Decision**: Closed-vocabulary concept sequence -> English pivot sentence -> 4-tier fallback:
      - Tier 1: Curated phrasebook (`translations.json`)
      - Tier 2: Offline MT (Argos Translate)
      - Tier 3: Opt-in Online MT
      - Tier 4: English Pivot Fallback (guarantees zero UI crash)

13. **Audio Degradation Safeguard**:
    - **Decision**: Verify OS voice language mapping; if native voice is unavailable for the target language, suppress audio speech and render prominent text banner. Never mispronounce foreign phrases with wrong-language voices.

---

## Phase 4: UI, Optimization & Deployment

14. **TensorFlow Lite Flex Delegate Integration**:
    - **Decision**: Enabled `SELECT_TF_OPS` in `TFLiteConverter` for the recurrent LSTM layer ops while verifying max absolute error against Keras float32 output is below `1e-6`.

15. **Streamlit State Management**:
    - **Decision**: Use `@st.cache_resource` for deep model graph loading and isolate session buffer queues to prevent stream stalling during frame re-rendering.

---

## OPEN HUMAN TASKS

The automated codebase provides the complete software, mathematical, and architectural infrastructure. The following physical and domain-expert tasks require human execution:

1. **Sign Variant Verification**:
   - Have a certified/fluent ISL and ASL signer verify the variant definitions in [`sign_cards/verification_checklist.csv`](sign_cards/verification_checklist.csv) against ISLRTC and WLASL standards.
2. **External Dataset Licensing & Download**:
   - Obtain license clearance and download raw video archives for INCLUDE, WLASL 0.3, and ASL Citizen.
   - Run `python tools/convert_public_dataset.py` to ingest public clips into `data/`.
3. **Physical Signer Recording**:
   - Record genuine webcam footage across multiple physical signers using `python tools/record_sign.py` to bridge the domain gap between studio datasets and local cameras.
4. **Native Speaker Translation Review**:
   - Have fluent speakers of Hindi, Tamil, Spanish, etc., inspect and approve the translations in [`translations.json`](translations.json), toggling `"needs_review": false`.
5. **TTS Voice Installation**:
   - Install OS-level language voice packs (e.g. Windows Language Settings -> Speech -> Add Hindi / Tamil / Spanish speech voices) or configure local Piper TTS voices.



