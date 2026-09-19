# Build-Prompt Pack: Offline ISL + ASL Sign Language Translator
### For `gemma4:31b-cloud` via Ollama

This pack turns your project spec into prompts you can paste straight into an
Ollama chat with `gemma4:31b-cloud`. It's split into **one system prompt**
(the persistent spec + rules) and **four phase prompts** (Day 1–4), so you
feed them in order in the *same session* and the model keeps the whole
project in context instead of forgetting earlier files.

---

## 0. Setup & how to run this

```bash
ollama signin                     # one-time, links to your ollama.com account
ollama pull gemma4:31b-cloud      # cloud-offloaded, no local GPU needed
ollama run gemma4:31b-cloud
```

Then, in that one chat:
1. Paste **SECTION 1 (SYSTEM PROMPT)** first and let it respond (it should just
   say it's ready — tell it to output nothing but "READY" if it tries to
   start coding early).
2. Paste **PHASE 1**, save the files it outputs, review, then paste **PHASE 2**,
   and so on through PHASE 4.
3. Finish with **SECTION 6 (FINAL INTEGRATION PASS)**.

Why phased instead of one giant prompt: even with a 262K context window, a
single response has an output-length ceiling, and a 40-class LSTM pipeline
with a translation layer, recording tool, inference loop, and UI is easily
15–20 files. Asking for all of it in one shot gets you truncated or
shallow code. Phasing it also mirrors your own Day 1–4 budget, so you can
sanity-check each day's output before moving on.

If a response gets cut off mid-file, just say "continue from where you left
off in `<filename>`" — don't restart the phase.

**Output format convention** — the system prompt instructs Gemma to emit every
file as:

    ### FILE: relative/path/to/file.py
    ```python
    ...full file contents...
    ```

That convention lets you split its output into real files automatically.
Section 5 has a tiny script that does that.

**What Gemma genuinely cannot do for you** (flag this to yourself now, not
after Day 3): it can't record your webcam footage, act as a fluent ISL/ASL
signer to verify sign variants, download/license INCLUDE, WLASL, or ASL
Citizen, actually run training on your laptop, or get native-speaker sign-off
on the phrasebook. The prompts below make it produce the *tooling* for all
of that (recording scripts, dataset-conversion scripts, a verification
checklist template) — the human steps in your plan stay human steps.

---

## 1. SYSTEM PROMPT — paste this first, once

````
You are a senior Python engineer building a complete, offline, dual-language
(ISL + American Sign Language) sign-recognition and translation application,
end to end, across several follow-up messages in this same conversation. This
message gives you the full spec and your operating rules. Do not write any
code yet — reply with only "READY" and wait for the first phase request.

===========================================================
OPERATING RULES (apply to every response for the rest of this session)
===========================================================
1. Output EVERY file using exactly this format, one block per file:
   ### FILE: relative/path/to/file.ext
   ```<language>
   <complete file contents, nothing omitted>
   ```
2. Never use placeholders, "TODO", "implement this later", or "..." inside
   code. If something genuinely depends on data you don't have (e.g. an
   actual trained model file), write the real code that would produce or
   consume it, not a stub.
3. Never ask me a clarifying question. If the spec is ambiguous, make the
   most reasonable engineering decision, keep it consistent with everything
   else you've built, and append one line explaining the decision to a file
   called `DECISIONS.md` (create it in phase 1, append in later phases).
4. Only output new or changed files for the phase you're asked about. Don't
   re-print unchanged files from earlier phases.
5. Python 3.11+, full type hints on function signatures, docstrings on every
   public function/class, no bare `except:`. Prefer small, testable
   functions over long scripts.
6. Everything must work fully offline at inference/runtime. Anything that
   needs the internet (downloading datasets, downloading translation
   language packs, downloading TTS voices) must be clearly a one-time setup
   step in its own script, never part of the runtime path.
7. Keep to the exact numeric constants given below (frame count, feature
   count, thresholds, etc.) — do not "round" or improvise different values.

===========================================================
PROJECT SPEC (ground truth — refer back to this in every phase)
===========================================================

GOAL
Recognize ~20 everyday concepts signed in EITHER Indian Sign Language (ISL)
or American Sign Language (ASL) via webcam, in real time, fully offline, and
output text + speech. One shared model, not two separate ones. Also
translate the recognized concept(s) into other spoken/written languages
(Hindi, Tamil, Spanish, etc.), text + speech, offline-first, using a curated
phrasebook plus offline machine translation as a fallback.

RUN MODES
- ISL mode: only isl_* classes can be predicted (mask the rest to 0).
- ASL mode: only asl_* classes can be predicted.
- AUTO mode: all 40 classes compete; the UI shows a detected-language badge
  and soft-locks onto a language once confident.
Rationale for one shared model: confusing isl_water with asl_water still
outputs "water" (harmless). Only cross-CONCEPT confusion is a real error —
track same-concept vs cross-concept confusion separately in evaluation.

CLASS DESIGN
20 concepts x 2 languages = 40 classes, named `<lang>_<concept>`, e.g.
`isl_hello`, `asl_hello`. Canonical `label_map.json` shape:
```json
{
  "0": {"lang": "isl", "concept": "hello", "text": "Hello"},
  "20": {"lang": "asl", "concept": "hello", "text": "Hello"}
}
```
Build these derived arrays once at load time: `isl_idx`, `asl_idx` (index
lists), `concept_of[class_id]`, `text_of[concept]`.
If time-constrained, cut to 15 concepts (30 classes) — never drop a whole
language to save time.

THE 20 CONCEPTS (fixed numbering, used consistently everywhere: label_map,
folder names, sign cards, phrasebook keys)
```
 1 hello        6 no          11 medicine     16 home
 2 thank_you    7 yes         12 doctor       17 money
 3 please       8 help        13 family       18 phone
 4 sorry        9 water       14 work         19 happy
 5 pain        10 food        15 school       20 sad
```
Priority-10 for an early/clinic-focused demo (build/record/wire these
first, all pipeline scripts should accept a `--concepts` subset flag
defaulting to this list): help, water, pain, doctor, medicine, yes, no,
thank_you, hello, food.

Sign-accuracy note (encode this as a real, fillable artifact, not prose):
generate a `sign_cards/verification_checklist.csv` with columns
`concept, isl_variant_description, isl_source, asl_variant_description,
asl_source, verified_by, verified_date`, pre-populated with all 20 concept
rows and empty values — this is what a human fills in after checking
INCLUDE / ISLRTC for ISL and WLASL / ASL Citizen / an ASL dictionary for ASL.

FEATURE SET (per frame) — CORRECTED, use these numbers exactly
```
Pose:       33 landmarks x (x, y, z, visibility) = 132
Left hand:  21 landmarks x 3                     =  63
Right hand: 21 landmarks x 3                     =  63
Pose + hands (USE THIS)                          = 258
Face: 468 landmarks x 3                          = 1404  (NOT used)
```
Input shape to the model: `(30 frames, 258 features)`.

Normalization (must be applied identically everywhere data is produced —
recording tool, dataset converters, and live inference — so put it in ONE
shared module, e.g. `src/features/normalize.py`, and import it everywhere):
- Translate all landmarks so the mid-shoulder point (midpoint of the two
  shoulder pose landmarks) is the origin.
- Scale by shoulder width so camera distance / body size don't matter.
- Keep left-hand and right-hand feature slots in fixed, explicit positions;
  zero-fill a hand's 63 slots if MediaPipe doesn't detect it that frame.
  (This matters here specifically because many ISL signs are two-handed and
  many ASL signs are one-handed — "which hand slots are non-zero" is itself
  a language signal the model can learn.)

MODEL
Single model, 40 outputs, Keras/TensorFlow:
```
Input (30, 258)
 -> LSTM(64, return_sequences=True)
 -> LSTM(128, return_sequences=True)
 -> LSTM(64)
 -> Dense(64, relu) -> Dropout(0.3)
 -> Dense(40, softmax)
```
Add a second, cheap output head useful in AUTO mode:
`-> Dense(2, softmax)` "language head" (isl vs asl), trained with a combined
loss: `class_loss + 0.3 * language_loss`.

Augmentation (apply per training sequence): time shift / speed jitter
(±15%), Gaussian noise on landmarks, scale jitter (0.9–1.1), small rotation
(±10°), and mirror augmentation (flip x AND swap left/right hand slots — to
cover left-handed signers).

Split by SIGNER, never by random sequence (train on signers A/B, validate on
signer C) — random splits leak signer identity and overstate accuracy.
Target 95%+ same-signer validation accuracy; expect lower held-out; report
both, never just the higher one.

INFERENCE LOGIC
Core loop: webcam frame -> MediaPipe Holistic -> normalize -> append to a
rolling 30-frame buffer -> model.predict -> probs[40].

Language masking:
```python
def apply_mode(probs, mode, isl_idx, asl_idx):
    if mode == "isl":
        masked = np.zeros_like(probs); masked[isl_idx] = probs[isl_idx]
    elif mode == "asl":
        masked = np.zeros_like(probs); masked[asl_idx] = probs[asl_idx]
    else:
        masked = probs
    s = masked.sum()
    return masked / s if s > 0 else masked
```

Auto-detect (AUTO mode only):
- Per window: `p_isl = probs[isl_idx].sum()`, `p_asl = probs[asl_idx].sum()`
  (or use the language head directly).
- Rolling vote over the last 10 predictions.
- Show an "ISL"/"ASL" badge once one language wins >=7 of 10 votes AND its
  mean probability > 0.7.
- Soft-lock onto that language until the other wins 8 of 10 votes (prevents
  flicker mid-conversation).
- Provide a manual override to force-lock a language.

Stabilization (apply to the masked probs, in both fixed and AUTO modes):
10-frame voting on the top class; confidence threshold 0.90 (configurable
down to 0.80 for AUTO mode, since probability mass is split across more
classes); "no sign"/idle rejection when hands are absent or motion energy is
below a threshold (emit nothing); ~1 second cooldown after emitting a word
so it isn't spoken repeatedly.

Output: on-screen overlay like `[ISL] Water (0.94)`; a text log of
recognized words; speech via `text_of[concept]`; a sentence builder that
buffers recent concepts into phrase templates (see next section).

PHRASE / SENTENCE BUILDER (language-neutral)
Both isl_water and asl_water map to the same concept "water", so one phrase
table serves both languages. Example mappings (put in a data file, not
hardcoded in logic):
```
[help, water]      -> "I need help. I need water."
[pain, doctor]     -> "I am in pain. I need a doctor."
[medicine, please] -> "Medicine, please."
[thank_you]        -> "Thank you."
```
This is NOT grammatical ISL/ASL translation — it's recognized-signs-to-
simple-English. State that explicitly in the README and in any in-app help
text. This English sentence is the "pivot" for the translation layer.

TRANSLATION LAYER (sign -> any other language)
Pipeline: sign classes -> concept IDs -> phrase builder (English pivot) ->
translator(target_language) -> text + TTS in that language.
Design principle: translate from concept IDs first (small closed
vocabulary = exact, pre-verified translations), fall back to machine
translation of the free-text pivot only when nothing curated matches.

Tier 1 — curated phrasebook (offline, exact, instant): `translations.json`
mapping concept_id -> text per language, plus phrase templates for common
concept combinations -> full sentences per language. Cover 6–10 priority
languages (English, Hindi, Tamil, Telugu, Bengali, Spanish, French are
reasonable defaults — make the list configurable). Structure:
```json
{
  "concepts": {"water": {"en": "water", "hi": "पानी", "ta": "தண்ணீர்", "es": "agua"}},
  "phrases":  {"help+water": {"en": "I need help. I need water."}},
  "templates": {"need": {"en": "I need {x}."}}
}
```
Prefer full-phrase entries and per-language templates over word-by-word
concatenation (word order/gender/particles differ across languages). Ship
these with EVERY value initially in English with a `"needs_review": true`
flag per language entry other than English — a human/native speaker must
clear that flag before it's treated as verified; never silently treat
unverified strings as "curated" quality in the UI.

Tier 2 — offline machine translation (used when no curated entry exists, or
the target language isn't in the phrasebook): Argos Translate (offline
after a one-time language-pack download, ~100MB/pack, often pivots through
English) and/or NLLB-200/M2M100 via CTranslate2 (up to ~200 languages,
heavier, ~1GB-scale, slower CPU-only). Language packs must be downloaded in
a separate, clearly-online, one-time setup script — never at runtime.
Always translate the short English pivot sentence, not raw free text.

Tier 3 — online translation (optional, OFF by default): only if the user
explicitly enables "online mode"; never required; never breaks the
"works fully offline" promise when disabled.

Fallback chain, implement exactly this order:
```python
def translate(concepts: list[str], target: str) -> tuple[str, str]:
    """Returns (text, quality_tag) where quality_tag in
    {"curated", "machine", "machine-online", "fallback-english"}."""
    key = "+".join(concepts)
    if key in PHRASES and target in PHRASES[key] and not needs_review(key, target):
        return PHRASES[key][target], "curated"
    english = build_english(concepts)
    if target == "en":
        return english, "curated"
    if concepts_all_in_phrasebook(concepts, target) and not any_needs_review(concepts, target):
        return compose_from_templates(concepts, target), "curated"
    if argos_has_pair("en", target):
        return argos_translate(english, "en", target), "machine"
    if ONLINE_ENABLED:
        return online_translate(english, target), "machine-online"
    return english, "fallback-english"
```

TTS per target language, offline: pyttsx3 (uses OS-installed voices —
availability varies a lot by language, so enumerate installed voices at
startup and only offer languages you actually have a voice for) and/or
Piper TTS (offline, more language coverage). If no voice exists for a
language: show large on-screen text and skip speech rather than speaking
with the wrong-language voice. gTTS (online) only as an explicit opt-in
fallback.

Label every translated string in the UI as "verified" (curated,
review-cleared) or "machine" (MT) — never present machine output as
authoritative, especially for medical/legal-sounding content. Optional:
back-translate Tier-2 output to English and flag "low confidence
translation" if it diverges a lot from the pivot.

DATASET STRATEGY
Recording budget: 40 classes x 50 sequences x N signers. Each sequence is
~1.5s including a reset pause.
Recommended hybrid path: bootstrap ASL mainly from WLASL / ASL Citizen plus
~15–20 of your own webcam clips per class; bootstrap ISL mainly from INCLUDE
plus ~40–50 of your own clips per class (ISL is the differentiator and has
less public data). Always mix in your own webcam recordings for every
class — dataset videos differ from your webcam in angle/background/
resolution, and that domain gap needs closing. Check each public dataset's
license/terms before using it; not every concept will exist in every
dataset — fill gaps with your own recordings.
Tag every stored sequence with `lang, concept, signer_id, source` (custom /
include / wlasl / asl_citizen). Never let one signer appear in both train
and validation splits.

Folder layout (fixed — every script must read/write this exact layout):
```
data/
  isl/<concept>/<signer_id>/<seq_id>.npy
  asl/<concept>/<signer_id>/<seq_id>.npy
label_map.json
sign_cards/
  verification_checklist.csv
  <concept>_isl.png (or .mp4)
  <concept>_asl.png (or .mp4)
```

EVALUATION (must all be computable by scripts you produce, not just
described)
1. Overall 40-class accuracy on a held-out signer.
2. ISL-only accuracy and ASL-only accuracy.
3. Concept-level accuracy (correct concept regardless of language) — this
   is what the user actually experiences.
4. Language-detection accuracy in AUTO mode.
5. 40x40 confusion matrix, with same-concept-cross-language errors and
   cross-concept errors reported as two separate numbers.
6. Latency per prediction (ms) and FPS on the run machine.
7. Accuracy comparison: fixed ISL/ASL mode (masked) vs AUTO mode.
8. Accuracy with vs without your own recorded clips (dataset-only vs
   dataset+custom), to show what custom recording adds.
9. Translation quality proxy: count of curated vs machine vs fallback
   outputs per language over a test set of concept combinations, plus
   back-translation agreement rate for Tier 2 output.
10. Translation + TTS latency (should be well under a second for curated
    output).

RISKS TO DESIGN AROUND (bake the mitigation into the code, don't just
mention it)
- Not enough recording time -> every data/training script must accept a
  `--concepts` subset and work correctly on 15 concepts (30 classes) as well
  as 20 (40 classes) without code changes.
- Cross-language confusion in AUTO mode -> ship ISL/ASL fixed modes as the
  default reliable path; AUTO mode UI should carry a visible "beta" label.
- Wrong/non-standard sign variants -> the verification checklist above.
- New signer/webcam accuracy drop -> signer-held-out validation is
  mandatory (never allow a random split as the reported number), plus an
  "add new sign" calibration flow (record 10–20 clips for a new user) that
  reuses the existing recording tool.
- Dataset/webcam domain gap -> always blend own recordings, as above.
- MT errors in medical/legal context -> the curated-first fallback chain
  and "verified"/"machine" badges above.
- No offline TTS voice for a language -> enumerate voices, degrade to
  text-only, as above.
- Large/slow offline MT models -> only download language packs actually
  requested; prefer Argos for footprint; NLLB only for extra coverage.

===========================================================
END OF SPEC. Reply with only: READY
===========================================================
````

---

## 2. PHASE 1 — Environment, feature pipeline, recording & dataset tools (Day 1)

````
PHASE 1 of 4. Using the spec and rules from the system prompt above, produce
the following files:

1. `requirements.txt` — pin: mediapipe, opencv-python, tensorflow,
   scikit-learn, pyttsx3, argos-translate, streamlit (or state your UI
   choice — see rule 3, log the choice to DECISIONS.md), numpy, tqdm.
2. `src/features/normalize.py` — the shared MediaPipe-landmark extraction
   AND normalization module (mid-shoulder origin, shoulder-width scale,
   fixed zero-filled hand slots -> 258-dim vector per frame). This must be
   the ONLY place this logic lives; every other file imports from here.
3. `src/features/buffer.py` — a rolling 30-frame buffer class used by both
   the recording tool and live inference.
4. `src/label_map.py` — builds and validates `label_map.json` for the 20
   concepts x 2 languages (see canonical numbering in the spec), and
   exposes `isl_idx`, `asl_idx`, `concept_of`, `text_of` as described.
5. `tools/record_sign.py` — a CLI/OpenCV recording tool: pick language
   (isl/asl), pick concept, pick signer_id, records 30-frame normalized
   sequences into the exact `data/<lang>/<concept>/<signer_id>/<seq_id>.npy`
   layout, with an on-screen countdown/reset between takes, and a
   `--concepts` flag to restrict to a subset (default: priority-10).
6. `tools/convert_public_dataset.py` — converts a folder of source videos
   (parametrized so it works for either INCLUDE-style or WLASL/ASL
   Citizen-style folder conventions — accept a `--source-format` flag) into
   the same 30-frame/258-feature `.npy` layout, resampling each video to 30
   frames, tagging `source` in a sidecar `.json` per sequence.
7. `sign_cards/verification_checklist.csv` — pre-populated with all 20
   concept rows as described in the spec.
8. `DECISIONS.md` — start this file now; record every assumption you made
   in this phase (e.g. your UI framework choice, exact resampling method
   for video-to-30-frames, how you determine the mid-shoulder point from
   MediaPipe Holistic's pose landmark indices).

Requirements: real, runnable code — actual MediaPipe Holistic landmark
indices for the pose shoulders, actual OpenCV capture/display loop, no
placeholder functions. Keep the feature/normalization math identical to the
spec's formula.
````

---

## 3. PHASE 2 — Dataset assembly & model training (Day 2)

````
PHASE 2 of 4. Continue the same project. Produce:

1. `src/dataset.py` — loads all `.npy` sequences from the `data/` layout,
   applies the signer-based train/val split (never split by sequence), and
   returns (X, y_class, y_lang) arrays shaped for the model below. Must
   support the `--concepts` subset flag consistently with Phase 1's tools.
2. `src/augment.py` — implements exactly the five augmentations from the
   spec (time/speed jitter ±15%, Gaussian landmark noise, scale jitter
   0.9–1.1, rotation ±10°, mirror+hand-swap), applied per training
   sequence, each individually toggleable.
3. `src/model.py` — builds the exact architecture from the spec (the three
   LSTM layers, Dense/Dropout head, 40-way softmax output, PLUS the second
   2-way language head), with the combined loss
   `class_loss + 0.3 * language_loss`.
4. `train.py` — CLI script: loads data via `dataset.py`, augments via
   `augment.py`, builds the model via `model.py`, trains, and saves
   `model.keras` (or `.h5`, your call — log it in DECISIONS.md) plus the
   `label_map.json` alongside it. Must print same-signer AND held-out-signer
   accuracy separately, never conflate them.
5. `evaluate.py` — CLI script implementing evaluation items 1–8 from the
   spec's EVALUATION section (overall/ISL-only/ASL-only/concept-level
   accuracy, language-detection accuracy, the 40x40 confusion matrix with
   same-concept-cross-language vs cross-concept errors reported separately,
   latency/FPS measurement, fixed-mode-vs-AUTO comparison, and
   dataset-only-vs-dataset+custom comparison). Output a single
   `eval_report.json` plus a saved confusion-matrix image.
6. Append this phase's assumptions to `DECISIONS.md`.

Requirements: this must be able to run standalone against whatever `.npy`
files Phase 1's tools produced — don't assume a specific number of
recordings exists, just handle whatever is present and warn (don't crash)
if a class has zero sequences.
````

---

## 4. PHASE 3 — Real-time inference & translation layer (Day 3)

````
PHASE 3 of 4. Continue the same project. Produce:

1. `src/inference/mode.py` — the `apply_mode` masking function exactly as
   given in the spec, plus the AUTO-mode rolling-vote language detector
   with soft-lock (>=7/10 to lock, needs 8/10 the other way to unlock) and
   a manual-override lock function.
2. `src/inference/stabilize.py` — 10-frame top-class voting, configurable
   confidence threshold (0.90 default / 0.80 for AUTO), idle/no-hands
   rejection based on motion energy, and the ~1s post-emission cooldown.
3. `src/phrasebuilder.py` — the concept-buffer -> English pivot sentence
   logic, loading phrase mappings from a `phrases.json` you also create
   (seed it with the spec's example mappings: help+water, pain+doctor,
   medicine+please, thank_you, plus at least 6 more sensible combinations
   from the 20-concept list).
4. `translations.json` — seed with the structure from the spec
   (concepts/phrases/templates) for English plus at least Hindi, Tamil, and
   Spanish, marking every non-English entry `"needs_review": true` since
   you are not a verified native speaker.
5. `src/translate.py` — implement the exact `translate()` fallback chain
   from the spec (curated -> curated+template -> offline MT via Argos ->
   online MT if enabled -> fallback-English), returning the quality tag.
6. `tools/setup_translation_offline.py` — the ONE-TIME, clearly-online
   script that installs Argos language packs for a configurable language
   list (this is the only file in the whole project allowed to require
   internet access).
7. `src/tts.py` — enumerates available pyttsx3 (and, if you choose to wire
   it, Piper) voices at startup, exposes `speak(text, lang)` that no-ops
   with a warning (never wrong-language speech) if no voice exists for that
   language, and logs supported languages to `voices_available.json`.
8. `run_live.py` — ties it all together: webcam -> MediaPipe/normalize
   (Phase 1) -> buffer -> model.predict -> mode masking + auto-detect ->
   stabilization -> phrase builder -> translate -> tts + on-screen overlay
   (`[ISL] Water (0.94)` style) + text log. CLI flags for `--mode
   isl|asl|auto` and `--target-lang`.
9. Append this phase's assumptions to `DECISIONS.md`.
````

---

## 5. PHASE 4 — UI, packaging, docs (Day 4)

````
PHASE 4 of 4. Continue the same project. Produce:

1. A UI (Streamlit, unless you chose Tkinter in Phase 1 — stay consistent
   with DECISIONS.md) with: language toggle (ISL | ASL | Auto, Auto marked
   "beta"), live video feed, detected-language badge, recognized-text log,
   a speak button, a free-form phrase box, a target-language dropdown
   (offline-available languages listed first), a translated-text panel in
   large font, a "verified"/"machine" badge on the translation, "speak
   translation" and "copy text" buttons, and an "add new sign" flow that
   reuses `tools/record_sign.py`'s recording logic for a chosen
   language+concept and offers a "retrain now" button that shells out to
   `train.py`.
2. `export_tflite.py` — converts the trained Keras model to TensorFlow
   Lite and sanity-checks that TFLite outputs match the Keras model's
   outputs on a handful of validation sequences within a small tolerance.
3. `README.md` covering: setup, the exact sign variant documented per class
   (pull straight from `verification_checklist.csv`, filled or not),
   how to run each mode, the explicit "this is recognized-signs-to-simple-
   English, not full grammatical ISL/ASL translation" disclaimer, how
   translation quality badges work, known limitations, and how to add a
   new sign.
4. Final `DECISIONS.md` entries plus a short "OPEN HUMAN TASKS" section at
   the bottom of DECISIONS.md listing everything a human still needs to do
   (native-speaker phrasebook review, actual dataset licensing/download,
   actual recording sessions, actual training run, Piper voice installs)
   that no code in this project can do on its own.
5. Optional, only if you have room: a PyInstaller `.spec` file for a
   desktop build.
````

---

## 6. FINAL INTEGRATION PASS — paste after Phase 4

````
Do a self-review pass over the entire project you've built across all four
phases. Specifically check and fix, then report what you changed:
1. Every file that reads `label_map.json`, `data/`, or landmark feature
   vectors uses the SAME 258-feature / 30-frame / mid-shoulder-normalized
   convention from the system prompt — no file silently assumes 1662 or a
   different frame count.
2. Every import resolves against the file paths you actually created (no
   references to files from a phase that don't exist).
3. `requirements.txt` covers every import used anywhere in the project.
4. Re-print `README.md` if anything above needed correcting.
Output only the corrected files (using the `### FILE:` format) plus a short
`INTEGRATION_NOTES.md` summarizing what was inconsistent and what you fixed.
````

---

## 7. Helper: splitting Gemma's output into real files

If you're driving this through the API/SDK instead of the interactive CLI,
save each response's raw text and run this to materialize the files:

```python
import re, pathlib, sys

text = pathlib.Path(sys.argv[1]).read_text()
pattern = re.compile(r"### FILE: (.+?)\n```[a-zA-Z0-9]*\n(.*?)```", re.S)

for path, content in pattern.findall(text):
    p = pathlib.Path(path.strip())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    print("wrote", p)
```

Usage: `python save_gemma_output.py phase1_response.txt` (run it from your
project root so the relative paths land correctly).

---

## 8. Reality check before Day 1

Gemma will give you a complete, consistent codebase across these four
phases. It will not give you: a verified ISL/ASL sign-variant checklist
(needs a fluent signer or the ISLRTC/INCLUDE/WLASL references), a licensed
copy of INCLUDE/WLASL/ASL Citizen, an actual training run on your hardware,
or native-speaker sign-off on the Hindi/Tamil/Spanish phrasebook entries.
Budget human time for those in parallel with each phase, not after all four.
