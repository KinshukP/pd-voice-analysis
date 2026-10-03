"""Nested-CV hyperparameter search for the v2 detector, without fooling ourselves.

The trap this script exists to avoid: tuning hyperparameters against the same
folds used to report the final score. That inflates the result exactly the way a
non-speaker-grouped split does, just one level up -- the reported number would
partly measure how well the search memorised those folds.

So the search is nested:

  outer loop  (5-fold, speaker-grouped)  -> honest evaluation, never seen by the search
    inner loop (3-fold, speaker-grouped, on the outer TRAINING portion only)
                                         -> picks hyperparameters
    refit best params on the outer training portion
    predict the held-out outer fold

Speaker grouping is enforced at BOTH levels, so no speaker appears in both sides
of any split, inner or outer.

The whole nested procedure is then repeated across 5 seeds, because a single run
is not a trustworthy number -- as established when v2's single-seed 0.892
collapsed to 0.875 +/- 0.012 over 5 seeds.

    .venv/bin/python scripts/tune_hyperparams.py

Reports only. Writes no artifacts; promoting a winner is a separate decision.
"""

from __future__ import annotations

import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdvoice import feature_spec  # noqa: E402
from retrain_v2 import v2_feature_cols  # noqa: E402

SEEDS = (42, 0, 1, 2, 3)
OUTER_SPLITS = 5
INNER_SPLITS = 3
N_CANDIDATES = 25

#: The published baseline this must beat to be worth adopting.
BASELINE_MEAN, BASELINE_STD = 0.875, 0.012

#: Noise floor observed throughout this project. An improvement smaller than
#: this is not an improvement.
NOISE_FLOOR = 0.015

#: v2's settings, carried from colab_reference.py:985.
V2_PARAMS = dict(
    learning_rate=0.05,
    max_iter=600,
    max_depth=3,
    l2_regularization=1.0,
    min_samples_leaf=20,
    max_leaf_nodes=31,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
)

SEARCH_SPACE = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
    "max_depth": [2, 3, 4, 5, None],
    "max_iter": [200, 400, 600, 800],
    "l2_regularization": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
    "min_samples_leaf": [5, 10, 20, 30, 50],
    "max_leaf_nodes": [7, 15, 31, None],
    "n_iter_no_change": [15, 30, 50],
}


def make_model(seed: int, **params) -> HistGradientBoostingClassifier:
    """class_weight and early stopping are fixed setup, not tuned."""
    base = dict(class_weight="balanced", early_stopping=True, validation_fraction=0.15)
    base.update(params)
    return HistGradientBoostingClassifier(random_state=seed, **base)


def speaker_auc(prob: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    frame = pd.DataFrame({"g": groups, "y": y, "p": prob}).groupby("g").agg(
        {"y": "first", "p": "mean"}
    )
    return float(roc_auc_score(frame["y"], frame["p"]))


def baseline_run(X: pd.DataFrame, y, groups, seed: int) -> float:
    """v2's fixed hyperparameters through the same outer CV, for a fair compare."""
    oof = np.zeros(len(X))
    outer = StratifiedGroupKFold(OUTER_SPLITS, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y, groups):
        m = make_model(seed, **V2_PARAMS).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return speaker_auc(oof, y, groups)


def nested_run(X: pd.DataFrame, y, groups, seed: int) -> tuple[float, list[dict]]:
    """One full nested-CV pass. Returns the honest AUC and the params chosen."""
    oof = np.zeros(len(X))
    chosen: list[dict] = []
    outer = StratifiedGroupKFold(OUTER_SPLITS, shuffle=True, random_state=seed)

    for tr, te in outer.split(X, y, groups):
        X_tr, y_tr, g_tr = X.iloc[tr], y[tr], groups[tr]

        # The inner splitter also groups by speaker, so hyperparameters are
        # never chosen using a speaker that appears in the inner training half.
        inner = StratifiedGroupKFold(INNER_SPLITS, shuffle=True, random_state=seed)
        search = RandomizedSearchCV(
            estimator=make_model(seed),
            param_distributions=SEARCH_SPACE,
            n_iter=N_CANDIDATES,
            scoring="roc_auc",
            cv=inner,
            random_state=seed,
            n_jobs=-1,
            refit=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_tr, y_tr, groups=g_tr)

        chosen.append(search.best_params_)
        # The outer fold has been touched by nothing above.
        oof[te] = search.best_estimator_.predict_proba(X.iloc[te])[:, 1]

    return speaker_auc(oof, y, groups), chosen


def main() -> int:
    df = pd.read_csv(feature_spec.TRAINING_CSV, header=1)
    cols = v2_feature_cols()
    X = df[cols].astype(float)
    y = df["class"].to_numpy()
    groups = df["id"].to_numpy()

    print(f"Features: {len(cols)}   recordings: {len(X)}   speakers: {len(np.unique(groups))}")
    print(f"Outer {OUTER_SPLITS}-fold / inner {INNER_SPLITS}-fold, both speaker-grouped.")
    print(f"{N_CANDIDATES} candidates per outer fold, {len(SEEDS)} seeds.\n")

    print("=" * 78)
    print("BASELINE (v2 fixed hyperparameters, same outer CV)")
    print("=" * 78)
    base_scores = []
    for seed in SEEDS:
        s = baseline_run(X, y, groups, seed)
        base_scores.append(s)
        print(f"  seed {seed:<3} speaker AUC {s:.4f}")
    base_mean, base_std = float(np.mean(base_scores)), float(np.std(base_scores))
    print(f"  -> {base_mean:.4f} +/- {base_std:.4f}")
    print(f"     (published v2 baseline: {BASELINE_MEAN:.3f} +/- {BASELINE_STD:.3f})")

    print("\n" + "=" * 78)
    print("NESTED CV (hyperparameters searched inside each outer training fold)")
    print("=" * 78)
    tuned_scores = []
    all_chosen: list[dict] = []
    start = time.time()
    for seed in SEEDS:
        score, chosen = nested_run(X, y, groups, seed)
        tuned_scores.append(score)
        all_chosen.extend(chosen)
        print(f"  seed {seed:<3} speaker AUC {score:.4f}   ({time.time() - start:.0f}s elapsed)")
    tuned_mean, tuned_std = float(np.mean(tuned_scores)), float(np.std(tuned_scores))
    print(f"  -> {tuned_mean:.4f} +/- {tuned_std:.4f}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    delta = tuned_mean - base_mean
    paired = np.array(tuned_scores) - np.array(base_scores)
    print(f"  baseline (this harness) : {base_mean:.4f} +/- {base_std:.4f}")
    print(f"  tuned (nested CV)       : {tuned_mean:.4f} +/- {tuned_std:.4f}")
    print(f"  difference              : {delta:+.4f}")
    print(f"  paired per-seed diffs   : {[f'{d:+.4f}' for d in paired]}")
    print(f"  mean paired difference  : {paired.mean():+.4f} (sd {paired.std():.4f})")
    print(f"  noise floor             : ~{NOISE_FLOOR:.3f}")
    if delta > NOISE_FLOOR:
        print("\n  -> Improvement EXCEEDS the noise floor. Worth considering for v3.")
    else:
        print("\n  -> NOT a real improvement. The difference is within seed noise.")
        print("     Keep v2. Adopting the tuned model would be picking a higher")
        print("     number out of a noisy search, not a better model.")

    print("\n" + "=" * 78)
    print("HYPERPARAMETERS CHOSEN ACROSS THE 25 OUTER FOLDS")
    print("=" * 78)
    print("(stability matters: a parameter that jumps around every fold is being")
    print(" fitted to fold quirks, not to a real property of the data)\n")
    for key in SEARCH_SPACE:
        counts = Counter(str(c.get(key)) for c in all_chosen)
        top, n = counts.most_common(1)[0]
        spread = ", ".join(f"{v}x{c}" for v, c in counts.most_common())
        print(f"  {key:<20} most common {top!s:<8} ({n}/{len(all_chosen)})   [{spread}]")
        print(f"  {'':<20} v2 default: {V2_PARAMS.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
