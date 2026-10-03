# Feature extraction — findings so far

Working notes from build step 1. Records what we established about how the
training features were computed, so it does not have to be re-derived.

## Confirmed: the baseline block is Praat's voice report

The 18 non-nonlinear Baseline columns line up one-for-one with the fields of
Praat's **Voice report**, and three internal identities in the CSV confirm it
rather than merely suggesting it:

| Check | Praat's definition | Measured in CSV |
|---|---|---|
| `ddpJitter / rapJitter` | exactly 3 | mean 3.0024 (spread is rounding at the CSV's precision) |
| `ddaShimmer / apq3Shimmer` | exactly 3 | mean 2.99999, range 2.997–3.002 |
| `numPulses - numPeriodsPulses` | 1, plus 1 per voice break | 1 for 679/756 rows |
| `meanNoiseToHarmHarmonicity` vs `(1-ac)/ac` | monotone | r = 0.995 |

So `extract_baseline_praat` parses the voice report directly instead of issuing
individual queries. This block is on solid ground.

**Units:** Praat prints jitter and shimmer as percentages; the CSV stores
fractions (`locPctJitter` averages 0.0023, not 0.23), so those fields are
divided by 100. `locAbsJitter` (seconds) and `locDbShimmer` (dB) are already in
the CSV's units.

**Pitch range:** implied F0 (`1 / meanPeriodPulses`) spans 77–475 Hz, sitting
comfortably inside Praat's default 75–600 Hz floor/ceiling. No reason to change
those defaults.

## Confirmed: `gender=0` is female, `gender=1` is male

The dataset ships no data dictionary for its `gender` column, but the speaker
counts recover it unambiguously. Sakar et al. describe a cohort of 188 PD (107
men, 81 women) and 64 HC (23 men, 41 women). The CSV's speakers cross-tabulate
as:

| code | HC | PD | total | matches |
|---|---:|---:|---:|---|
| `gender=1` | 23 | 107 | 130 | the paper's **men** |
| `gender=0` | 41 | 81 | 122 | the paper's **women** |

All four cells agree exactly, so this is settled rather than guessed. It matters
beyond the model input itself: the formant ceiling is chosen by sex, so getting
this backwards would mis-measure f1–f4 and b1–b4 for every speaker.

## Derived: the training rows used a fixed ~1.93 s window

`numPeriodsPulses * meanPeriodPulses` recovers the voiced duration each row was
computed over:

```
median 1.9238 s   75th pct 1.9258 s   99th pct 1.9291 s   max 1.9297 s
92% of rows fall in 1.90–1.93 s
```

That hard ceiling is not what a set of naturally-varying recordings looks like —
the authors analysed a fixed-length window. The extractor therefore centre-trims
to 1.93 s by default (`window_s`). Skipping this would inflate `numPulses` and
shift every frame-count-sensitive statistic on a longer recording.

Open: the CSV fixes the window *length* but not its *anchor*. Centre-trimming is
our choice, to avoid onset/offset instability in a sustained vowel.

## Resolved: the MFCC convention

librosa's `feature.mfcc` helper defaults to a **dB-scaled** (10·log₁₀) mel
spectrogram over a **128-band** filterbank. Both defaults are wrong here — they
put 71 of 84 MFCC features outside the training range entirely.

Scanning conventions against the training columns (filterbank size × log base ×
DCT normalisation × liftering × delta width), scored on the 66 MFCC features
that do not depend on recording level:

| Convention | In range | Median z |
|---|---:|---:|
| **26 mels, log₁₀(power), ortho DCT-II, no lifter, delta width 9** | **62/66** | **0.69** |
| 24 mels, same | 62/66 | 0.77 |
| 20 mels, same | 62/66 | 0.87 |
| 26 mels, ln(amplitude) | 60/66 | 0.65 |
| 26 mels, ln(power) | 48/66 | 3.15 |
| 128 mels (librosa default), log₁₀ | 47/66 | 2.84 |

Adopted: 26 filters, log₁₀ of mel power, orthonormal DCT-II, no cepstral
liftering, delta width 9.

Caveats on what is and isn't pinned:
- **Filterbank size is not identifiable** — 20/22/24/26 all score 62/66. 26 is
  chosen for the tightest median z and because it is the classic HTK size.
- **Amplitude vs power is not identifiable** — log₁₀(power) and ln(amplitude)
  differ only by a constant 0.869 factor and score within noise of each other.
  The *base* is pinned firmly (natural log of power is clearly wrong); the exact
  amplitude convention is not.
- Liftering (HTK's lifter=22) and narrower delta windows both made it worse.

## Two features remain unmatched, for a structural reason

`mean_Log_energy` and `mean_MFCC_0th_coef` are still out of range, and this is
expected rather than a bug: they encode **absolute signal level**, which depends
on microphone gain, distance and the sample scaling convention (float vs int16).
They cannot be matched without either the original recordings or a calibrated
level. The same applies to the three Intensity features.

Empirically this costs nothing — see below, dropping all level-dependent
features leaves AUC unchanged.

## Validation result against real recordings

Three sustained-vowel takes, 5 s each, 44.1 kHz mono, adult male speaker,
compared against the training distribution (mean of the three takes):

| Family | In range | Median z | Take-to-take drift |
|---|---|---:|---:|
| Baseline (jitter/shimmer/HNR/pulses) | **18/18** | 0.20 | 0.07 |
| Formant frequencies | **4/4** | 1.12 | 0.03 |
| Bandwidths | **4/4** | 0.36 | 0.01 |
| Intensity (mic-dependent) | 3/3 | 2.09 | 0.13 |
| MFCC | 71/84 | 0.68 | 0.10 |

The Praat-derived block reproduces the training method essentially exactly — a
median z of 0.20 means the values sit almost on the training median. Take drift
well under 0.2 everywhere means the features are stable across recordings of the
same person, so they are measuring the voice rather than the noise.

Jitter sits high (rap/ddp at the ~95th percentile). Most likely explanation is
recording technique — holding a perfectly steady pitch is hard without practice —
and note the training distribution is ~75% PD, so its percentiles are not a
healthy-population reference. This is a validation artefact, not a clinical
signal, and jitter carries ~0.5% of model importance regardless.

## The fallback is not a fallback — it is the better option

Speaker-grouped 5-fold CV on the training CSV, speaker-level AUC, 3 seeds:

| Feature set | n | Speaker AUC |
|---|---:|---:|
| All 753 (current detector) | 753 | 0.888 ± 0.016 |
| **Extractable 114** | 114 | **0.872 ± 0.010** |
| Validated 107 (level-dependent dropped) | 107 | 0.872 ± 0.008 |
| MFCC only | 84 | 0.865 ± 0.006 |
| TQWT only | 432 | 0.802 ± 0.013 |
| Baseline + formants + intensity only | 30 | 0.799 ± 0.018 |
| Wavelet only | 182 | 0.692 ± 0.016 |

The full-feature number reproduces the saved model's 0.877, so the harness is
sound.

**The 114 features we can already extract score 0.872 against the full model's
0.888 — a gap of 0.016, smaller than the ±0.016 seed-to-seed variance.** TQWT's
~50% SHAP importance is real but largely *redundant* with MFCC: on its own TQWT
scores 0.802, below MFCC's 0.865. Gradient boosting spreads importance across
correlated features, so a large SHAP share does not imply unique information.

Implication: reproducing TQWT — the highest-risk, least-documented block, 432
features from a MATLAB toolbox with unknown Q/redundancy/level parameters — buys
at most ~0.016 AUC, and buys *negative* value if reproduced imperfectly, because
approximate values would be out-of-distribution input to a model that trusts them.

## detector_v2: the 107-feature model

Built by `scripts/retrain_v2.py`, using v1's recipe unchanged (same estimator
settings, same speaker-grouped 5-fold CV, same isotonic calibration) so the
comparison isolates the feature set. v1 artifacts are untouched.

### Discrimination — unchanged

Speaker-level AUC across 5 CV seeds, both models through the identical pipeline:

| Model | Features | Speaker AUC |
|---|---:|---|
| v1 | 753 | 0.880 ± 0.012 |
| v2 | 107 | 0.875 ± 0.012 |

Paired difference −0.005 (range −0.029 to +0.014) — well inside seed noise. The
two models discriminate equally well; v2 simply does it with features that can
actually be computed from a recording.

Calibration is sound: ECE drops from 0.039 raw to 0.000 after isotonic fitting.

### Operating point — this DID move

At a fixed threshold the two models are not interchangeable, because the
calibrated probability distribution shifted with the feature set:

| Threshold | v1 sens / spec | v2 sens / spec |
|---|---|---|
| 0.65 | 0.87 / 0.66 | **0.81 / 0.74** |
| 0.70 | 0.80 / 0.76 | 0.78 / 0.79 |

v2 trades ~6 points of sensitivity for ~8 of specificity at the same threshold.
(Note v1's *published* 0.92/0.65 does not reproduce here — through this pipeline
v1 gives 0.87/0.66 at 0.65. The published figure likely came from a different
seed. All comparisons above are like-for-like.)

**Threshold kept at 0.65.** It is within 0.01 Youden-J of v2's optimum (0.70)
while retaining more sensitivity, and it already improves on the over-flagging
weakness v1 was criticised for (specificity 0.74 vs 0.66).

### Band edges — re-tuned

Sensitivity at the threshold is the wrong lens for this product: a recording
scoring 0.50 is not told it is healthy, it lands in `uncertain` and is flagged
for follow-up. The number that matters is **what share of PD lands in the
reassuring band**.

| low / high | healthy band | its accuracy | uncertain | PD wrongly reassured |
|---|---:|---:|---:|---:|
| 0.30 / 0.65 **(chosen)** | 6.3% | **0.83** | 26.7% | **1.4%** |
| 0.35 / 0.65 (v1 edges) | 9.1% | 0.78 | 23.9% | 2.7% |
| 0.40 / 0.60 | 18.0% | 0.71 | 11.4% | 7.1% |

**`band_low` moved 0.35 → 0.30.** It halves the share of PD recordings wrongly
reassured (2.7% → 1.4%) and lifts the healthy band's accuracy from 0.78 to 0.83,
for a slightly wider uncertain band. PROJECT_CONTEXT flags the ~77%-accurate
healthy band as a known weakness; this directly improves it. `band_high` stays
at 0.65.

The guardrail still stands: 0.83 is not high enough to phrase a negative result
as "you're clear."

### Concept mapping — three corrections

Applying v1's `concept_of` to the 107 features exposed problems that were
invisible at 753. See `pdvoice/concepts.py`:

| Issue | At 753 | At 107 | Fix |
|---|---|---|---|
| `wavelet_complexity` held only TQWT | 432 features | **0 features** | concept dropped |
| `b1`–`b4` fell to `other_spectral` (rule tested for the substring "bandwidth", columns are named `b1`) | 4/753, invisible | 4/107 | now `vocal_tract` with `f1`–`f4` |
| `gender` fell to `other_spectral`, i.e. would be described as "other spectral detail" | 1/753 | 1/107 | new silent `demographic` concept, never narrated |

Final coverage — every feature claimed, nothing in a catch-all:

| Concept | Features | Mean contribution toward PD |
|---|---:|---:|
| temporal_fluctuation | 56 | +0.694 |
| spectral_shape | 24 | +0.294 |
| voice_regularity | 18 | +0.138 |
| vocal_tract | 8 | +0.115 |
| demographic *(silent)* | 1 | +0.000 |

`gender` contributing +0.000 retroactively justifies suppressing it from
narration — it costs the explanation nothing.

## Hyperparameter tuning: attempted, rejected

`scripts/tune_hyperparams.py`. Nested cross-validation — outer 5-fold
speaker-grouped for evaluation, inner 3-fold speaker-grouped for the search,
25 candidates per outer fold, repeated over 5 seeds (125 independent searches).
The search never touches the fold it is scored on.

| | Speaker AUC |
|---|---|
| Baseline (v2 fixed params, same harness) | 0.8752 ± 0.0121 |
| Tuned (nested CV) | 0.8754 ± 0.0187 |
| **Difference** | **+0.0002** |

Per-seed paired differences: +0.0111, +0.0053, −0.0141, −0.0002, −0.0008. They
straddle zero; mean +0.0002 against a ~0.015 noise floor.

**Verdict: no improvement. v2 stands; no v3 was created.**

Two details worth keeping:

- **Tuning made the model *less* stable.** Spread widened from ±0.0121 to
  ±0.0187. Choosing hyperparameters per fold adds variance without adding
  performance.
- **The chosen hyperparameters were near-random**, which is the real tell. Across
  25 outer folds, the modal choice was barely above what uniform sampling would
  give: `learning_rate` 7/25 (chance ≈ 4), `max_depth` 7/25 (chance ≈ 5),
  `min_samples_leaf` 8/25 (chance ≈ 5). Only `max_leaf_nodes=7` showed real
  signal at 13/25 (chance ≈ 6) — a mild preference for smaller trees. A flat
  response surface like this means the configuration was never the bottleneck.

The tempting number was seed 42's tuned 0.9028. Seed 1 gave 0.8487 on the same
configuration. Reporting the former would have been exactly the self-deception
the nested design exists to prevent.

**What this implies:** the ceiling here is set by the data — 252 speakers, one
corpus, one recording session each — not by model configuration. More speakers
or a second corpus would move the number; more search will not.

## Step 2: the inference pipeline, and what running it revealed

`pdvoice/pipeline.py` chains extraction → detector → calibration → SHAP →
concepts → facts → phrasing behind `run_inference(wav_path, gender)`.

`gender` is a required argument, not optional: it is one of the 107 model inputs
and it selects the formant ceiling. It cannot be read off the audio, and
guessing it would silently mis-measure f1–f4 and b1–b4.

### The prevalence problem — the most important finding here

The SHAP base value is **1.357 log-odds = 0.795 probability**. That is the
model's answer *before it looks at a single feature*, and it is that high because
the training cohort is 74.6% PD patients.

Consequences, both observed on real recordings:

1. A recording can land in `likely_pd` while **every concept points toward
   healthy**. Take 3 scored 0.679 with a net SHAP of −0.53: the voice looked
   healthier than the average training speaker, but the average training speaker
   is a patient. The explanation template originally asserted "shows acoustic
   signs associated with Parkinson's" directly beside a list of toward-healthy
   factors — a plain self-contradiction. `build_facts` now computes
   `evidence_vs_band`, and the phrasing states when the verdict is coming from
   the baseline rather than from the voice.
2. **The calibrated probabilities do not transfer to real-world screening.** They
   are calibrated to a 75%-PD population. General-population PD prevalence is
   well under 1% (roughly 1–2% over age 60). A "0.97 probability" from this
   pipeline means 0.97 *given a cohort that is three-quarters patients* — not
   0.97 for a walk-up user. Any deployment needs an explicit prevalence
   adjustment, or the number must not be shown as a probability of having PD.

This is a property of the source dataset, not of anything built here, and it
would apply equally to the v1 753-feature model.

### Stability across takes

Three recordings of the same speaker, minutes apart, scored 0.974 / 0.872 /
0.679 — a spread of 0.29. The underlying features were stable (take-to-take
drift under 0.1 of the training range), so the model is amplifying small feature
differences into large probability differences. A single recording is therefore
not a reliable basis for a result; averaging several takes is the obvious
mitigation and matches how the training data was collected (3 per speaker).

### Input-quality gating

`run_inference` refuses rather than guesses. Verified behaviour:

| Input | Result |
|---|---|
| Silence | rejected — 24 features non-finite, no voiced segment |
| White noise | rejected — no voiced segment |
| 20 ms clip | rejected — Praat error wrapped as `InferenceError` |
| Missing file | rejected — `InferenceError` |
| Real recording | scored, 4/107 features out of range (3.7%) |

Recordings with more than 20% of features outside the training range are marked
`reliable: false` and the warning is prepended to `explanation_text`, so a
caller cannot display the score without also carrying the caveat.

### Voicing guard

Silence is not neutral input. Praat's measures skip unvoiced frames, but the
MFCC block — 84 of the 107 features — averages over every frame including
silence, and the energy cliff at a silence boundary wrecks the derivative
statistics. Injecting a one-second pause into a clean recording moved
`std_delta_delta_log_energy` from **0.0104 to 11.7** (~1000x) and swung
confidence by 46 points, without ever tripping the out-of-range check.

`extract.measure_voicing()` counts voiced frames with Praat's own pitch tracker
— the same machinery that decides which frames the jitter/shimmer features use,
so the answer is consistent with what extraction actually does. (An RMS gate
disagrees: it called these recordings 2–4% silent where Praat calls them 0%
unvoiced.)

Threshold set from measurement, not taste:

| Recording | Unvoiced |
|---|---:|
| 7 clean sustained-vowel takes | 0.0% |
| One take where the vowel was not held throughout | 21.9% |
| Clean take + 1 s pause | 52.3% |
| Only 1 s of voice in a 5 s file | 58.6% |
| Noise / silence | 100% |

`MAX_UNVOICED_FRACTION = 0.10` sits in a wide empty gap between the clean takes
and everything else. `run_inference` refuses above it; `run_inference_multi`
checks per take and excludes the offender by name, since one bad take is a
reason to drop that take rather than the whole session. If too few usable takes
remain, the run is refused with the failing filenames listed.

Validated on the real recordings: `ahhh_b_4.wav` — independently suspicious for
having 175 pulses against ~297 for every other take — measures 21.9% unvoiced
and is now excluded. It had previously scored joint-highest at 0.9735.

### Fixed: fabricated frames in the delta computation

Six `mean_*_delta_delta` features sat outside the training range **for both
classes** — the tell that they were an extraction artifact rather than anything
about the speaker.

Cause, measured rather than inferred: librosa's `delta` uses scipy's
`savgol_filter` with `mode='interp'`, which fits a polynomial across the edge
window and emits its derivative there. Those values are fabricated. On a
194-frame recording, the first 5 frames were all exactly −0.1848 and the last 5
all exactly −0.2052 — single repeated constants, ~10x the interior magnitude.
The arithmetic closes exactly:

```
(5 x -0.1848 + 5 x -0.2052) / 194 = -0.01005
observed mean_1st_delta_delta      = -0.009903
interior mean (edges removed)      = +0.000116   (training median +0.00003)
```

Fix: discard `width // 2 + 1` = 5 frames per end before computing the mean/std
statistics, from the static block too so all 84 statistics describe the same
stretch of audio.

Result on 7 voiced takes: **107/107 features in the training range**, on every
individual take (previously 4–16 out of range per take). Median z 0.559.

The MFCC convention was re-scanned afterwards in case the artifact had distorted
the original fit. It had not: the production config (26 mels, log10, ortho DCT,
no lifter, width 9) now scores 66/66, and the structural choices — orthonormal
DCT, no liftering, width 9, log rather than dB — win throughout the top ranks.
Nine configurations tie at 66/66, so filterbank size remains unidentifiable and
26 is retained rather than chased.

### Fixed: outlier detection now compares inputs, not scores

Score-based outlier detection has a blind spot. When the model saturates — every
take landing at 0.7–0.97 — a genuinely bad recording produces an unremarkable
probability and passes. A deliberately-bad take scored 0.9500, squarely inside
the group.

Comparing takes on their *features* separates them cleanly. Each feature is
scaled by its training MAD so the 107 are commensurable, and each take is
summarised by its median scaled deviation from the element-wise median take:

| Take | Deviation | vs group typical |
|---|---:|---:|
| 6 clean takes | 0.208–0.302 | 0.76–1.11x |
| deliberately bad take | 0.667 | **2.44x** |
| take with a silent gap | 0.958 | **3.51x** |

Both bad takes are now excluded and named. Verified not to over-prune: given
only the 6 clean takes, it excludes none. The two checks are capped jointly at
one third of the batch, so they cannot prune down to whatever agrees.

### Fixed: the decision threshold contradicted its own evidence

The calibrated probability carries the training set's prevalence inside it
(74.6% patients), so it answers "how likely is PD *in a cohort that is
three-quarters patients*". Dividing that prior out leaves the **likelihood
ratio** — how much more the voice resembles a patient's than a control's:

```
LR = [p / (1-p)] / [pi / (1-pi)]      LR = 1 exactly at p = pi = 0.7460
```

Measured against the old band edges:

| Edge | p | LR |
|---|---:|---:|
| `band_low` | 0.30 | 0.15 |
| `band_high` (old) | **0.65** | **0.63** |
| evidence-neutral | 0.7460 | 1.00 |

**"Likely Parkinson's" was being declared at LR 0.63** — on evidence roughly
1.6x *more* consistent with a healthy control. Everything between p = 0.65 and
p = 0.746 was a guaranteed contradiction: 28 of 756 training recordings (3.7%),
including **9 of 192 healthy controls**. It was not a display quirk; the
explanation layer's "the patterns point the other way" clause was correctly
reporting that the verdict disagreed with the evidence.

`band_high` and the decision threshold are now **pinned to the evidence-neutral
point** rather than chosen (`evidence_neutral_point()` in `scripts/retrain_v2.py`),
so the contradiction is impossible by construction.

| | Old | New |
|---|---|---|
| band_high | 0.65 (LR 0.63) | **0.7460 (LR 1.00)** |
| Sensitivity | 0.81 | 0.78 |
| Specificity | 0.74 | **0.79** |
| PD wrongly reassured | 1.4% | **1.4%** (unchanged) |

The sensitivity cost is not a safety cost: recordings dropping out of
`likely_pd` land in `uncertain` and are still flagged for follow-up. The rate
that matters — PD patients wrongly *reassured* — is governed by `band_low`,
which did not move.

This changes where the line is drawn, not how well the model discriminates: AUC
is 0.8634 either way, since the likelihood ratio is a monotone rescaling.

`confidence_v2.joblib` now also stores `training_prevalence` and the band edges'
likelihood ratios, so a caller can recover prior-free evidence or re-apply a
stated prevalence.

### The verdict is binary

Product decision: both band edges sit on the evidence-neutral point, so the
middle "inconclusive" band is zero-width and every recording is called one way
or the other.

This also repaired the confidence score. `decision_margin` is zero for anything
*inside* a band, so with a three-band layout every inconclusive result reported
0% confidence whether it sat at 0.31 or 0.74 — it could not distinguish
"barely uncertain" from "dead centre". With a single edge the margin is non-zero
on both sides and grades properly in both directions.

What it costs, on the training cohort: 230 recordings move from "inconclusive,
follow up" to "no strong signs", and the share of PD recordings given a
reassuring answer rises from 1.4% to **22.3%**.

That figure overstates the harm, because it is measured on a cohort that is
74.6% patients. What matters in use is how often "no strong signs" is right:

| Population screened | NPV — "no strong signs" is correct | PPV — "likely PD" is correct |
|---|---:|---:|
| training cohort (75% patients) | 54.7% | 91.6% |
| adults over 60 (~2%) | 99.4% | 7.1% |
| general adult (~1%) | 99.7% | 3.6% |

The negative call is strong in any realistic population; the positive call is
mostly false alarms. That is inherent to screening a rare condition with an
imperfect test, and is why the wording never claims more than a signal and the
"no strong signs" phrasing always carries "this does not rule Parkinson's out".

The bottom confidence level was renamed `inconclusive` → `very low`, since a
call is now always made and "inconclusive" would imply otherwise.

### Still outstanding

**Proper voiced-region windowing** — choosing the analysis window by where the
voice actually is, rather than the geometric centre. The guard rejects bad
input; it does not yet make the most of usable input.

**Prevalence-adjusted probability.** `internal.prevalence_adjusted_probability`
is still `None`. With the likelihood ratio available it is now a short step:
`odds = LR x q/(1-q)` for a stated prevalence `q`. It needs a decision about
which population is assumed, and the assumed value must be displayed beside the
number — at 1% prevalence a recording with LR 0.72 comes out at 0.7%, versus the
67.9% the raw calibrated probability reports.

## Not implemented, deliberately

| Block | Count | Why not |
|---|---:|---|
| PPE / DFA / RPDE | 3 | Little et al. nonlinear measures, not Praat. Tractable, but embedding/scaling parameters unstated. |
| Vocal Fold (GQ/GNE/VFER/IMF) | 22 | No mainstream Python implementation matching the paper's. |
| Wavelet | 182 | Mother wavelet and input signal (F0 contour vs. waveform) unstated. |
| TQWT | 432 | Q / redundancy / level count unknown; MATLAB toolbox original. ~50% of model importance. |

These raise `NotImplementedError` rather than returning approximations. A
guessed value in the TQWT block would yield a confident-looking score computed
from input the model has never seen, which for a screening tool is worse than
refusing to score.

**Current coverage: 114 of 753 features (15%).**

## Environment note

`models/*.joblib` was pickled under **scikit-learn 1.6.1**, and newer versions
cannot load it at all (1.9.0 fails with `ModuleNotFoundError: No module named
'_loss'`). requirements.txt pins 1.6.1; all five artifacts then load with no
version warnings. Do not bump that pin without re-saving the artifacts.
