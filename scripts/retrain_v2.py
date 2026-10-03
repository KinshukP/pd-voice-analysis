"""Retrain the detector on only the features the extractor can reliably produce.

v1 consumed 753 features, 639 of which nothing can compute from a .wav file, so
it cannot run outside Colab. v2 consumes the 107 features that `pdvoice.extract`
produces and that were validated against the training distribution on real
recordings (see docs/extraction_findings.md).

Follows the v1 recipe from colab_reference.py exactly — same estimator settings,
same speaker-grouped CV, same isotonic calibration, same Youden-J threshold
sweep — so the before/after comparison reflects the feature set and nothing else.

    .venv/bin/python scripts/retrain_v2.py

Writes models/*_v2.joblib. Does not touch the v1 artifacts.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdvoice import concepts, feature_spec  # noqa: E402

SEED = 42
N_SPLITS = 5

# v1's operating point, kept only as the comparison baseline.
V1_THRESHOLD = 0.65
V1_BAND_LOW, V1_BAND_HIGH = 0.35, 0.65

# v2's operating point. Re-derived from the sweep this script prints, NOT
# inherited: the calibrated probability distribution shifted with the feature
# set, so the same numeric threshold buys a different sensitivity/specificity
# trade. Rationale is in docs/extraction_findings.md; change these only
# alongside that sweep.
#
# The verdict is BINARY: both band edges sit on the evidence-neutral point, so
# the middle "uncertain" band is zero-width and every recording is called one
# way or the other. Product decision, taken deliberately -- see below.
#
# What it costs, measured on the training cohort: recordings between p=0.30 and
# p=0.746 used to be "inconclusive, follow up" and are now "no strong signs".
# 230 recordings move, and the share of PD recordings given a reassuring answer
# goes from 1.4% to 22.3%.
#
# Why that number overstates the harm: it is computed on a cohort that is 74.6%
# patients. What matters in use is how often "no strong signs" is actually
# right, which depends on who is screened:
#
#     training cohort (75% patients)   NPV 54.7%
#     adults over 60 (~2% prevalence)  NPV 99.4%
#     general adult  (~1% prevalence)  NPV 99.7%
#
# The flip side is the positive call: at 1% prevalence a "likely PD" result is
# right about 3.6% of the time. That is inherent to screening a rare condition
# with an imperfect test, and is why the wording never claims more than a signal.
#
# A single edge also repairs the confidence score. `decision_margin` was zero
# for anything inside the old band, so every inconclusive result reported 0%
# confidence whether it sat at 0.31 or 0.74. With one edge the margin is
# non-zero on both sides and grades properly in both directions.
#
# Both edges are pinned to `evidence_neutral_point` -- not a free choice; see
# that function for why.


def evidence_neutral_point(y: np.ndarray) -> float:
    """The calibrated probability at which the voice stops favouring "healthy".

    The calibrated probability carries the training set's own prevalence baked
    in (74.6% patients), so it answers "how likely is PD *in a cohort that is
    three-quarters patients*". Dividing that prior out leaves the likelihood
    ratio -- how much more the voice resembles a patient's than a control's:

        LR = [p / (1-p)] / [pi / (1-pi)]

    LR = 1 is the point where the recording says nothing either way, and it
    falls exactly at p = pi.

    This matters because the previous threshold of 0.65 sits at LR = 0.63 --
    "likely Parkinson's" was being declared on evidence roughly 1.6x more
    consistent with a healthy control. 28 of 756 training recordings (3.7%) fell
    in that contradiction zone, including 9 of 192 healthy controls, and the
    explanation layer had to narrate the contradiction rather than avoid it.

    Anchoring the decision here makes that outcome impossible by construction:
    below this point the recording favours "healthy", so calling it PD would
    contradict the evidence the explanation is about to show.

    Both band edges now sit here, making the verdict binary -- see the module
    header for what that trades away.

    Note this changes where the line is drawn, not how well the model
    discriminates: AUC is identical either way (0.8634), since the likelihood
    ratio is a monotone rescaling of the calibrated probability.
    """
    return float(np.mean(y))


def likelihood_ratio(p: float, prevalence: float) -> float:
    """Strip the training prior out of a calibrated probability."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return (p / (1 - p)) / (prevalence / (1 - prevalence))

MODELS_DIR = feature_spec.REPO_ROOT / "models"

#: Features excluded from v2: their absolute values track microphone gain and
#: distance rather than the voice, so they cannot be reproduced at inference
#: time. Dropping them costs no AUC (0.872 either way).
LEVEL_DEPENDENT = {
    "minIntensity",
    "maxIntensity",
    "meanIntensity",
    "mean_Log_energy",
    "std_Log_energy",
    "mean_MFCC_0th_coef",
    "std_MFCC_0th_coef",
}


def base_model() -> HistGradientBoostingClassifier:
    """v1's estimator settings, unchanged (colab_reference.py:985)."""
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        learning_rate=0.05,
        max_iter=600,
        max_depth=3,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=SEED,
    )


def ece(prob: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error (colab_reference.py:998)."""
    total = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        hi = lo + 1 / bins
        m = (prob >= lo) & (prob < hi)
        if m.sum():
            total += m.mean() * abs(truth[m].mean() - prob[m].mean())
    return total


def v2_feature_cols() -> list[str]:
    return [c for c in feature_spec.implemented_feature_names() if c not in LEVEL_DEPENDENT]


def out_of_fold(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    oof = np.zeros(len(X))
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in sgkf.split(X, y, groups):
        m = base_model()
        m.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]
    return oof


def speaker_level(prob: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the 3 recordings per speaker to one score, as v1 reported it."""
    frame = pd.DataFrame({"g": groups, "y": y, "p": prob})
    agg = frame.groupby("g").agg({"y": "first", "p": "mean"})
    return agg["p"].to_numpy(), agg["y"].to_numpy()


def sens_spec(prob: np.ndarray, y: np.ndarray, threshold: float) -> tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y, (prob >= threshold).astype(int)).ravel()
    return tp / (tp + fn), tn / (tn + fp)


def band_report(prob: np.ndarray, y: np.ndarray, low: float, high: float) -> list[tuple]:
    band = np.where(prob < low, "likely_healthy", np.where(prob > high, "likely_pd", "uncertain"))
    rows = []
    for name in ("likely_healthy", "uncertain", "likely_pd"):
        m = band == name
        share = float(m.mean() * 100)
        if name == "uncertain" or not m.sum():
            rows.append((name, share, float("nan")))
        else:
            acc = float((((prob[m] >= 0.5).astype(int)) == y[m]).mean())
            rows.append((name, share, acc))
    return rows


def missed_pd_rate(prob: np.ndarray, y: np.ndarray, low: float) -> float:
    """Share of PD recordings that land in the likely_healthy band.

    This is the number that matters for a screening tool, not sensitivity at the
    decision threshold. A PD recording scoring 0.50 is not told it is healthy --
    it lands in the uncertain band and is flagged for clinical follow-up. Only
    recordings below `low` are actively reassured, so only those are true misses.
    """
    pd_mask = y == 1
    return float((prob[pd_mask] < low).mean())


def main() -> int:
    df = pd.read_csv(feature_spec.TRAINING_CSV, header=1)
    y = df["class"].to_numpy()
    groups = df["id"].to_numpy()

    cols_v1 = feature_spec.load_feature_cols()
    cols_v2 = v2_feature_cols()
    print(f"v1 features: {len(cols_v1)}   v2 features: {len(cols_v2)}")

    # ---------------- concept coverage ----------------
    index = concepts.concept_index(cols_v2)
    print("\nConcept coverage on the v2 feature set:")
    for name, positions in index.items():
        flag = "" if name in concepts.NARRATABLE_CONCEPTS else "   (silent, not narrated)"
        print(f"  {name:<22} {len(positions):>4}{flag}")
    claimed = sum(len(v) for v in index.values())
    assert claimed == len(cols_v2), f"concept map covers {claimed} of {len(cols_v2)}"
    empty = [c for c in concepts.NARRATABLE_CONCEPTS if c not in index]
    print(f"  every feature is claimed; narratable concepts with no features: {empty or 'none'}")

    # ---------------- cross-validated comparison ----------------
    print("\n" + "=" * 78)
    print("SPEAKER-GROUPED CROSS-VALIDATION")
    print("=" * 78)
    results = {}
    for label, cols in (("v1 (753 features)", cols_v1), ("v2 (107 features)", cols_v2)):
        X = df[cols].astype(float)
        oof = out_of_fold(X, y, groups)
        sp_p, sp_y = speaker_level(oof, y, groups)
        results[label] = {
            "oof": oof,
            "rec_auc": roc_auc_score(y, oof),
            "spk_auc": roc_auc_score(sp_y, sp_p),
        }
        print(
            f"{label:<22} recording AUC {results[label]['rec_auc']:.3f}   "
            f"speaker AUC {results[label]['spk_auc']:.3f}"
        )

    oof = results["v2 (107 features)"]["oof"]

    # ---------------- calibration ----------------
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    cal = calibrator.transform(oof)
    print("\nCalibration (v2, recording level):")
    print(f"  raw        Brier={brier_score_loss(y, oof):.3f}  ECE={ece(oof, y):.3f}")
    print(f"  calibrated Brier={brier_score_loss(y, cal):.3f}  ECE={ece(cal, y):.3f}")

    # ---------------- threshold sweep ----------------
    print("\nThreshold sweep (calibrated probability, recording level):")
    best_t, best_j = 0.5, -1.0
    for t in np.round(np.arange(0.30, 0.71, 0.05), 2):
        sens, spec = sens_spec(cal, y, t)
        j = sens + spec - 1
        mark = "   <- v1 threshold" if abs(t - V1_THRESHOLD) < 1e-9 else ""
        print(f"  t={t:.2f}  sensitivity={sens:.2f}  specificity={spec:.2f}  youdenJ={j:.2f}{mark}")
        if j > best_j:
            best_t, best_j = float(t), j
    print(f"\n  Youden-optimal threshold = {best_t:.2f}")
    for t, name in ((V1_THRESHOLD, "v1 threshold 0.65"), (best_t, f"v2 optimal {best_t:.2f}")):
        sens, spec = sens_spec(cal, y, t)
        print(f"    {name:<22} sensitivity={sens:.2f}  specificity={spec:.2f}")

    # ---------------- like-for-like operating point vs v1 ----------------
    # v1's published 0.92/0.65 came from a different run; recomputing it through
    # this exact pipeline is the only fair comparison.
    print("\nOperating point, v1 vs v2 through the identical pipeline:")
    oof_v1 = results["v1 (753 features)"]["oof"]
    cal_v1 = IsotonicRegression(out_of_bounds="clip").fit(oof_v1, y).transform(oof_v1)
    print(f"  {'model':<8}{'thr':>7}{'sens':>8}{'spec':>8}")
    for label, probs in (("v1", cal_v1), ("v2", cal)):
        for t in (V1_THRESHOLD, best_t):
            s, sp = sens_spec(probs, y, t)
            print(f"  {label:<8}{t:>7.2f}{s:>8.2f}{sp:>8.2f}")

    # ---------------- confidence bands ----------------
    print(f"\nConfidence bands at v1 edges ({V1_BAND_LOW} / {V1_BAND_HIGH}):")
    for name, share, acc in band_report(cal, y, V1_BAND_LOW, V1_BAND_HIGH):
        acc_txt = "flagged for follow-up" if np.isnan(acc) else f"accuracy={acc:.2f}"
        print(f"  {name:<16} {share:5.1f}% of recordings   {acc_txt}")

    print("\nBand-edge sweep (what share of PD is wrongly reassured?):")
    print(
        f"  {'low':>6}{'high':>6}{'healthy%':>10}{'hACC':>7}"
        f"{'uncertain%':>12}{'pd%':>7}{'pdACC':>7}{'PD missed':>11}"
    )
    for low, high in ((0.30, 0.65), (0.35, 0.65), (0.40, 0.60), (0.35, 0.70), (0.30, 0.70)):
        rows = {r[0]: r for r in band_report(cal, y, low, high)}
        missed = missed_pd_rate(cal, y, low)
        mark = "   <- v1 edges" if (low, high) == (V1_BAND_LOW, V1_BAND_HIGH) else ""
        print(
            f"  {low:>6.2f}{high:>6.2f}{rows['likely_healthy'][1]:>10.1f}"
            f"{rows['likely_healthy'][2]:>7.2f}{rows['uncertain'][1]:>12.1f}"
            f"{rows['likely_pd'][1]:>7.1f}{rows['likely_pd'][2]:>7.2f}{missed:>11.1%}{mark}"
        )

    # ---------------- final fit on all data ----------------
    X_v2 = df[cols_v2].astype(float)
    model = base_model().fit(X_v2, y)

    sp_p, sp_y = speaker_level(oof, y, groups)
    joblib.dump(
        {
            "model": model,
            "feature_cols": cols_v2,
            "label_order": ["HC", "PD"],
            "dataset": "parkinsons-disease-classification",
            "n_speakers": int(len(np.unique(groups))),
            "cv_speaker_auc": float(roc_auc_score(sp_y, sp_p)),
            "cv_recording_auc": float(roc_auc_score(y, oof)),
            "excluded_level_dependent": sorted(LEVEL_DEPENDENT),
            "provenance": (
                "Retrained from pd_speech_features.csv on the subset of features "
                "pdvoice.extract can reproduce and that validated against the "
                "training distribution. See docs/extraction_findings.md."
            ),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
        MODELS_DIR / "detector_v2.joblib",
    )

    # Same schema as detector_v1_data.joblib, so downstream code is uniform.
    joblib.dump(
        {"X": X_v2, "y": y, "groups": groups, "feature_cols": cols_v2},
        MODELS_DIR / "detector_v2_data.joblib",
    )

    neutral = evidence_neutral_point(y)
    chosen_sens, chosen_spec = sens_spec(cal, y, neutral)
    print(f"\nEvidence-neutral point (LR = 1): p = {neutral:.4f}")
    print(f"  previous band_high 0.65 sat at LR = {likelihood_ratio(0.65, neutral):.3f}")
    contradiction = ((cal >= 0.65) & (cal < neutral)).sum()
    print(
        f"  recordings that were flagged 'likely PD' on healthy-leaning evidence: "
        f"{contradiction}/{len(y)} ({contradiction/len(y):.1%}), "
        f"{((cal >= 0.65) & (cal < neutral) & (y == 0)).sum()} of them healthy controls"
    )
    joblib.dump(
        {
            "calibrator": calibrator,
            "threshold": float(neutral),
            "band_low": float(neutral),
            "band_high": float(neutral),
            "oof_auc": float(roc_auc_score(y, oof)),
            "youden_optimal_threshold": float(best_t),
            "sensitivity": float(chosen_sens),
            "specificity": float(chosen_spec),
            "pd_missed_in_healthy_band": missed_pd_rate(cal, y, neutral),
            # Everything needed to turn a calibrated probability back into
            # prior-free evidence, or to re-apply a stated prevalence.
            "training_prevalence": float(np.mean(y)),
            "band_high_likelihood_ratio": 1.0,
            "band_low_likelihood_ratio": 1.0,
            "note": (
                "band_high is pinned to the evidence-neutral point (likelihood "
                "ratio 1.0), not chosen: below it the recording favours 'healthy', "
                "so 'likely PD' there would contradict its own evidence. "
                "See evidence_neutral_point() and docs/extraction_findings.md."
            ),
        },
        MODELS_DIR / "confidence_v2.joblib",
    )

    # ---------------- SHAP ----------------
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_v2)
    if isinstance(shap_values, list):  # older shap returns one array per class
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # (n, features, classes)
        shap_values = shap_values[:, :, 1]
    print(f"\nSHAP values: {shap_values.shape}")

    joblib.dump(
        {
            "shap_values": shap_values,
            "base_value": float(np.ravel(explainer.expected_value)[-1]),
            "feature_cols": cols_v2,
            # key name matches detector_v1_shap.joblib
            "mean_abs": pd.Series(
                np.abs(shap_values).mean(axis=0), index=cols_v2
            ).sort_values(ascending=False),
        },
        MODELS_DIR / "detector_v2_shap.joblib",
    )

    concept_contrib = pd.DataFrame(
        {name: shap_values[:, positions].sum(axis=1) for name, positions in index.items()}
    )
    joblib.dump(
        {
            "concept_contrib": concept_contrib,
            "feature_concept": concepts.build_feature_concept(cols_v2),
            "concept_label": concepts.ALL_CONCEPT_LABEL,
            "concepts": list(index.keys()),
            "narratable": [c for c in concepts.NARRATABLE_CONCEPTS if c in index],
        },
        MODELS_DIR / "detector_v2_concepts.joblib",
    )

    predicted_pd = model.predict(X_v2) == 1
    print("\nMean concept contribution among predicted-PD recordings (+ pushes toward PD):")
    for name, value in concept_contrib[predicted_pd].mean().sort_values(ascending=False).items():
        tag = "" if name in concepts.NARRATABLE_CONCEPTS else "  (silent)"
        print(f"  {value:+.3f}  {concepts.ALL_CONCEPT_LABEL[name]}{tag}")

    print(
        f"\nChosen operating point: threshold={neutral:.4f} (LR=1.0)  "
        f"bands={neutral:.4f}/{neutral:.4f} (binary)  "
        f"sensitivity={chosen_sens:.2f}  specificity={chosen_spec:.2f}  "
        f"PD wrongly reassured={missed_pd_rate(cal, y, neutral):.1%}"
    )
    print(
        "\nWrote: detector_v2, detector_v2_data, confidence_v2, "
        "detector_v2_shap, detector_v2_concepts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
