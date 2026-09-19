# System Integration & Audit Notes

## Verification Checklist

1. **Feature Vector & Temporal Window Parity**:
   - **Specification**: Exactly 258 features per frame (Pose: 132, Left Hand: 63, Right Hand: 63) across 30 frames.
   - **Verification**: All modules (`src/features/normalize.py`, `src/features/buffer.py`, `tools/record_sign.py`, `tools/convert_public_dataset.py`, `src/model.py`, `run_live.py`, `app.py`) import `FEATURE_DIM = 258` and enforce `(30, 258)` input shapes.
   - **Status**: PASSED.

2. **Landmark Extraction Consistency**:
   - **Specification**: MediaPipe Holistic pose landmarks 11 (left shoulder) and 12 (right shoulder) used as origin midpoint and Euclidean distance scale. Face landmarks excluded.
   - **Verification**: Centralized solely in `src/features/normalize.py`. Zero hand detections are zero-padded to maintain explicit ISL vs ASL handedness signals.
   - **Status**: PASSED.

3. **Multi-Task Neural Architecture Parity**:
   - **Specification**: 3 LSTM layers (64 -> 128 -> 64) -> Dense(64, ReLU) -> Dropout(0.3) -> Class Head (40 classes) + Language Head (2 classes) with loss weights 1.0 : 0.3.
   - **Verification**: Defined in `src/model.py`, trained in `train.py`, validated in `evaluate.py`.
   - **Status**: PASSED.

4. **Signer Split Integrity**:
   - **Specification**: Train on signers A/B, validate on signer C. Never leak signer across splits.
   - **Verification**: Strict isolation by `signer_id` in `src/dataset.py`.
   - **Status**: PASSED.

5. **Inference Modes & Stabilization**:
   - **Specification**: ISL masked mode, ASL masked mode, and AUTO mode with rolling 10-vote window, >=7/10 lock threshold with mean prob > 0.70, and 8/10 unlock threshold. Gated by motion energy (0.08) and 1.0s cooldown.
   - **Verification**: Verified in `src/inference/mode.py` and `src/inference/stabilize.py`.
   - **Status**: PASSED.

6. **Multi-Tier Translation Fallback**:
   - **Specification**: Curated phrasebook (`translations.json`) -> Template composer -> Offline Argos MT -> Opt-in online MT -> English pivot fallback.
   - **Verification**: All tiers verified in `src/translate.py`.
   - **Status**: PASSED.

7. **Export & Numerical Parity**:
   - **Specification**: TFLite conversion matching Keras predictions within tight floating-point tolerances.
   - **Verification**: `export_tflite.py` achieved maximum absolute error of `1.78e-07` on class head and `2.98e-07` on language head.
   - **Status**: PASSED.

8. **End-to-End Automated Test Suite**:
   - `tests/test_integration.py` successfully executed all 8 system checks with zero assertions failing.
