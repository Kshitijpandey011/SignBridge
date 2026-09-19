# Dual-Language Offline Sign Language Translator (ISL + ASL)

A complete, real-time, offline sign language recognition and spoken-language translation system supporting both **Indian Sign Language (ISL)** and **American Sign Language (ASL)**.

Features a multi-task recurrent neural network (LSTM) predicting 40 classes (20 concepts × 2 sign languages) alongside auxiliary sign-language discrimination, coupled with an English pivot phrase builder, multi-tier translation fallback, and offline text-to-speech (TTS).

---

## ⚠️ Linguistic & Architectural Disclaimer
> **IMPORTANT:** This system performs **recognized-signs-to-simple-English pivot mapping**, NOT full grammatical ISL or ASL translation. Sign languages possess distinct grammar, facial grammar, topic-comment syntax, and spatial referencing. This application maps discrete recognized sign concepts into concise English pivot sentences, which are subsequently translated into target spoken languages (Hindi, Tamil, Spanish, French, etc.).

---

## 1. Quick Setup & Installation

### Requirements
- Python 3.11+
- Webcam

### Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional)* To enable local offline machine translation via Argos Translate:
```bash
python tools/setup_translation_offline.py --languages hi es fr
```

---

## 2. Running the System

### Option A: Interactive Streamlit Web App (Recommended)
Launch the modern web UI featuring live skeleton landmarks, mode toggles, translation badges, and calibration tools:
```bash
streamlit run app.py
```

### Option B: Fast CLI Real-Time Inference
Run live inference directly in an OpenCV HUD window:
```bash
# AUTO Mode (Automatic ISL / ASL detection with soft-lock)
python run_live.py --mode auto --target-lang hi

# Fixed ISL Mode (Masks ASL to zero)
python run_live.py --mode isl --target-lang hi

# Fixed ASL Mode (Masks ISL to zero)
python run_live.py --mode asl --target-lang es
```

---

## 3. Operational Modes

1. **ISL Mode**:
   - Only `isl_*` classes can be predicted. Non-ISL probabilities are masked to zero.
2. **ASL Mode**:
   - Only `asl_*` classes can be predicted. Non-ASL probabilities are masked to zero.
3. **AUTO Mode (Beta)**:
   - All 40 classes compete dynamically.
   - Evaluates a 10-prediction rolling window. Locks onto a language when $\ge 7/10$ votes agree with mean probability $> 0.70$.
   - Applies hysteresis soft-locking: requiring $\ge 8/10$ opposite votes to switch languages mid-conversation, eliminating UI flicker.

---

## 4. Concept Taxonomy & Verification Status

The model is trained on 20 canonical concepts across both languages (40 classes total). Sign variants are tracked in [`sign_cards/verification_checklist.csv`](sign_cards/verification_checklist.csv):

| # | Concept | ISL Variant (INCLUDE / ISLRTC) | ASL Variant (WLASL / ASL Citizen) |
|---|---|---|---|
| 1 | `hello` | Open palm wave / forehead salute | Open B-hand from temple |
| 2 | `thank_you` | Open hand chin outwards | Flat hand from chin outwards |
| 3 | `please` | Chest circular rub / prayer hands | Flat hand circles on chest |
| 4 | `sorry` | Closed fist rub over chest | 'A' hand circular motion on chest |
| 5 | `pain` | Double index twist at pain location | Index fingers twisted towards each other |
| 6 | `no` | Two fingers shake / head shake | Index + middle tap thumb |
| 7 | `yes` | Nodding fist up and down | 'S' hand nod up and down |
| 8 | `help` | Flat palm supporting fist | Thumbs-up on flat open palm, lifted |
| 9 | `water` | 'W' hand / cupped hand to mouth | 'W' hand tapping lower lip |
| 10 | `food` | Flattened O-hand tapping mouth | Flattened O-hand to mouth |
| 11 | `medicine` | Crushing pills motion in palm | Middle finger rubs center of palm |
| 12 | `doctor` | Pulse check on wrist | 'M'/'D' fingers tap inside wrist |
| 13 | `family` | 'F' hands circle together | 'F' hands circular contact |
| 14 | `work` | Fist tap over wrist | Active fist taps passive wrist |
| 15 | `school` | Flat clap hands twice | Flat hands clapping horizontally |
| 16 | `home` | Flat hands rooftop or cheek-chin | Flat O-hand touches cheek to jaw |
| 17 | `money` | Thumb rubs fingers / coin flick | Flat O-hand taps open palm |
| 18 | `phone` | 'Y' hand to ear | 'Y' hand pressed against ear |
| 19 | `happy` | Open hands brush upward on chest | Open flat hand brushes upward on chest |
| 20 | `sad` | Open hands drag down face | Open 5-hands drawn downward in front of face |

---

## 5. Feature Extraction & Normalization

All inputs adhere to the exact 258-feature standard:
- **Pose Landmarks**: $33 \times (x, y, z, \text{visibility}) = 132$ dimensions.
- **Left Hand**: $21 \times (x, y, z) = 63$ dimensions (zero-filled when not detected).
- **Right Hand**: $21 \times (x, y, z) = 63$ dimensions (zero-filled when not detected).
- **Total**: $132 + 63 + 63 = 258$ values per frame across a 30-frame rolling window $\rightarrow$ Input Shape `(30, 258)`.
- **Normalization**: Mid-shoulder origin translation and shoulder-width Euclidean scaling.

---

## 6. Translation Fallback Hierarchy

Every recognized sign sequence is resolved through the multi-tier translation chain:
1. **Tier 1 (Curated Phrasebook)**: Exact matches in [`translations.json`](translations.json). Marked as **Verified**.
2. **Tier 2 (Offline MT)**: Local neural MT via Argos Translate. Marked as **Machine**.
3. **Tier 3 (Online MT)**: Opt-in web translation fallback.
4. **Tier 4 (English Pivot)**: Direct English text fallback.

---

## 7. Model Training & Evaluation

### Train New Model
```bash
python train.py --epochs 25 --batch-size 16 --output-model models/model.keras
```

### Comprehensive Evaluation Suite
Generates [`eval_report.json`](eval_report.json) and [`confusion_matrix.png`](confusion_matrix.png):
```bash
python evaluate.py --model models/model.keras --data-dir data
```

### Export to TensorFlow Lite
```bash
python export_tflite.py --model models/model.keras --output models/model.tflite
```

---

## 8. Calibrating & Adding a New Sign

To add a new sign or calibrate a specific user's webcam:
1. Run `python tools/record_sign.py --lang isl --concept water --signer-id my_name --takes 15`.
2. Review the captured `.npy` sequences under `data/isl/water/my_name/`.
3. Retrain the model: `python train.py`.
