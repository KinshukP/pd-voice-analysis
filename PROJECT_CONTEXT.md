# Parkinson's Voice Screening — Project Context

Context brief for a fresh Claude Code session. Summarizes decisions already made
in model development so app work can start without re-deriving them.

## What this is

A web app where a user records or uploads a short voice recording (sustained
vowel "ahhh") and receives:

1. **Detection** — likely Parkinson's / no strong signs
2. **Confidence** — a calibrated score
3. **Explanation** — why the model said what it said
4. **Severity** — estimated severity (NOT YET BUILT, deliberately deferred)

Team: Rukshik Nelluri, Kinshuk Panchakarla. Project name: "The Parkinson's Pod" (POD5).

## Model development status: DONE (in Google Colab)

Model work is complete and artifacts are saved. Do not retrain — load the
artifacts.

### The detector

- **Data:** Parkinson's Disease Classification dataset (Sakar et al. 2019, UCI).
  252 speakers (188 PD / 64 HC), 756 recordings (3 per speaker), 754 features.
  Sustained vowel /a/, recorded at 44.1 kHz.
- **Model:** `HistGradientBoostingClassifier` (scikit-learn), class-weighted,
  early stopping.
- **Honest performance:** ~0.88 AUC, ~0.86 accuracy at speaker level, under
  speaker-grouped cross-validation (no speaker appears in both train and test).
- **Sensitivity ~0.92 / specificity ~0.65** at the chosen threshold.

### Why this dataset and not the others

Three other feature datasets exist (Oxford Detection, Oxford Telemonitoring,
Multiple Types). They only share ~13 basic MDVP features (jitter/shimmer/HNR).
Investigation found:

- Merging Oxford Detection + Multiple Types → weak (~0.55 AUC), does not
  transfer across datasets.
- Basic features alone, even single-dataset → ~0.73-0.75 AUC ceiling.
- The Classification dataset's accuracy comes from **exotic** features:
  TQWT wavelet (~50% of importance) and MFCC-family (~45%). Basic
  jitter/shimmer features contribute ~0.5%.

Known limitations accepted for v1: single-dataset model (cannot be externally
validated — no other dataset has these features), and it over-flags healthy
people (the "likely healthy" band is only ~77% accurate, so never phrase a
negative result as "you're clear").

### Saved artifacts (Google Drive: `Code/checkpoints/`)

| File | Contents |
|---|---|
| `detector_v1.joblib` | fitted model, `feature_cols` (753 names, ordered), metadata |
| `detector_v1_data.joblib` | prepared feature matrix, labels, speaker groups |
| `detector_v1_shap.joblib` | SHAP values matrix, base value, global importance |
| `detector_v1_concepts.joblib` | per-recording concept contributions, feature→concept map |
| `confidence_v1.joblib` | isotonic calibrator, threshold (0.65), band edges (0.35/0.65) |

## The explanation architecture (important — read before changing it)

The exotic features (e.g. `tqwt_entropy_log_dec_33`) are **not interpretable**.
A correlation study found only 3 of the top 20 exotic features have any strong
relationship (|r| >= 0.5) to interpretable jitter/shimmer measures; median |r|
is 0.40.

**Therefore: never let an LLM narrate raw feature names.** It would confabulate
meanings that were never validated. This is a hard design rule for a health tool.

Instead, the pipeline is:

1. SHAP computes per-feature contributions for the recording.
2. Features are bucketed into 6 human-readable **concepts**:
   - `temporal_fluctuation` — "moment-to-moment energy and spectral fluctuation"
   - `wavelet_complexity` — "fine-grained irregularity in the sound"
   - `spectral_shape` — "overall tone and timbre of the voice"
   - `vocal_tract` — "vocal-tract resonance (formants)"
   - `voice_regularity` — "steadiness of pitch and loudness"
   - `other_spectral` — "other spectral detail"
3. SHAP values are summed within each concept.
4. Grounded facts (band, calibrated probability, top concepts + direction) are
   assembled.
5. An LLM phrases those facts into a paragraph — **phrasing only**, using a
   prompt that supplies concept labels, never raw feature names, so
   confabulation is structurally prevented.
6. A deterministic template exists as both a mock and a production fallback if
   the LLM call fails.

## THE CRITICAL UNSOLVED PROBLEM — start here

The model consumes 753 pre-extracted features. **Nothing yet computes those
features from a raw recording.** This is the hardest remaining piece and the app
cannot work without it.

Feature blocks and difficulty:

- **MFCC family** — standard, `librosa` handles it. Low risk.
- **Baseline jitter/shimmer/HNR/pulses** — standard, `parselmouth` (Praat
  bindings). Low risk.
- **Formants / bandwidth / intensity** — `parselmouth`. Medium risk.
- **Vocal fold features** — medium-high risk, less standard.
- **TQWT (tunable Q-factor wavelet transform)** — HIGH RISK. This was the novel
  contribution of the source paper, implemented via a MATLAB toolbox. No
  mainstream Python equivalent guaranteed to match. ~50% of model importance
  lives here.

The dataset ships only the finished CSV, not extraction code, so exact
reproduction is uncertain.

### Required validation step (do not skip)

After implementing extraction, compare the distributions of extracted features
against the training CSV's columns (means, ranges, correlations). If they don't
line up, the model is receiving out-of-distribution input and its output is
meaningless — which for a health screening tool is worse than no output.

### Fallback if TQWT can't be reproduced faithfully

Retrain the detector on only the feature subset that CAN be reliably extracted
(MFCC + baseline + formants). Expected performance lands somewhere between the
0.73 basic-only and 0.88 full-feature numbers. This is the honest option: the app
computes exactly what the model was trained on. Do NOT guess at TQWT parameters
and hope.

## Files in this folder

- `PROJECT_CONTEXT.md` — this document.
- `colab_reference.py` — **reference implementation, not app code.** The exported
  Google Colab notebook where all model development happened. It will not run
  outside Colab (Drive mounts, `google.colab` imports). Read it to port logic
  from, don't try to execute it.
- `models/` — the five `.joblib` artifacts listed above, downloaded from Drive.
- `data/` — all four feature datasets, folder names matching the Drive layout:
  - `data/parkinsons-disease-classification-dataset/pd_speech_features.csv` —
    **the training data for the detector.** Needed as the ground-truth reference
    for validating extracted features (exact column names, value ranges).
  - `data/oxford-parkinsons-disease-detection-dataset/parkinsons.data` — basic
    MDVP features, HC+PD. Not used by the detector; kept for limited external
    validation on shared features.
  - `data/oxford-parkinsons-disease-telemonitoring-dataset/parkinsons_updrs.data`
    — PD only, UPDRS labels. For the deferred severity model.
  - `data/parkinsons-speech-with-multiple-types-of-sound-recordings/train_data.txt`
    and `test_data.txt` — HC+PD with UPDRS for PD subjects. Its PD rows feed the
    deferred severity model. (Note: `test_data.txt` is all-PD with no controls,
    so it cannot validate a detector.)

Only the Classification dataset is needed for app v1. The other three are present
for the severity model and external validation later.

### What to port from `colab_reference.py`

Port these — they are already written and validated, and reimplementing them
risks silent drift between what was measured and what the app does:

- The feature→concept mapping (`concept_of`) and the `CONCEPT_LABEL` dictionary.
  This must match exactly, or the app's explanations stop corresponding to the
  saved concept SHAP values.
- `build_facts`, `build_prompt`, `render_template`, `phrase_explanation` — the
  whole explanation-phrasing chain, including the LLM guardrails in the prompt
  and the deterministic template fallback.
- The confidence band logic (threshold and band edges are also stored in
  `confidence_v1.joblib`).

Ignore these — they were investigation, and the app loads saved artifacts
instead of retraining:

- All training, cross-validation, and diagnostic cells.
- Dataset comparison / leave-one-dataset-out / feature importance cells.
- Drive mounting and dataset copying.

**Note:** `colab_reference.py` contains NO feature-extraction code. The entire
notebook operates on pre-extracted CSV features. It helps with build steps 2-4
below, but contributes nothing to step 1 — extraction is not started.

## Suggested build order

1. **Feature extractor in isolation** — `extract_features(wav_path) -> dict`,
   plus the validation-against-CSV step above. Nothing else matters until this
   works.
2. **Inference pipeline module** — port the Colab logic into a normal Python
   module: load artifacts → extract features → predict → calibrate → SHAP →
   concept aggregation → assemble facts → phrase explanation. One function, used
   by both evaluation and the app so what's measured is what users get.
3. **Backend** — Python API endpoint: audio in, structured result out.
4. **Frontend** — record/upload, display result + confidence + explanation.
5. **Severity model** — deferred. Separate model predicting UPDRS, trained on
   Oxford Telemonitoring + Multiple Types PD rows, only runs when detection says
   PD. Flagged as the weakest pillar (interpolated labels, PD-only data); may end
   up as a coarse mild/moderate/severe band rather than a number.

## Product guardrails

- Always present results as **screening, not diagnosis**; recommend consulting a
  clinician.
- Never phrase a negative result as "you don't have Parkinson's" — the model's
  healthy-band accuracy is only ~77%. Use "no strong signs detected."
- Recordings landing in the uncertain band should be flagged for follow-up rather
  than given a confident call.
- The user recording protocol should match training data: sustained "ahhh",
  quiet room, a few seconds.