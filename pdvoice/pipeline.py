"""Build step 2: the whole model side of the project behind one function.

    run_inference("recording.wav", gender=1) -> dict

Chains extraction -> detector -> calibration -> SHAP -> concept aggregation ->
grounded facts -> phrasing. Pure function, no API and no UI; the backend and the
evaluation harness are both meant to call this, so what gets measured is what
users get.

On surfacing failures. A screening tool that returns a confident wrong answer is
worse than one that returns nothing, so this module checks its own inputs rather
than trusting them:

  * the extracted feature names and their order must match the model's
    `feature_cols` exactly, or it raises;
  * any non-finite feature value raises;
  * every feature is compared against the training range, and a recording with
    too many out-of-range values is marked unreliable, with the warning carried
    into the returned explanation text rather than hidden in a log line.

That last check is the important one. The model was trained on 252 speakers
recorded in one session on one setup; fed something outside that, it will still
return a confident-looking probability, and nothing about the number would
reveal that it is meaningless.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from pdvoice import concepts, confidence, explain
from pdvoice.extract import (
    FeatureExtractionError,
    InsufficientVoicingError,
    extract_features,
    measure_voicing,
    require_voicing,
)
from pdvoice.feature_spec import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"

#: Fraction of features allowed outside the training range before the result is
#: marked unreliable. Some drift is normal -- a different microphone in a
#: different room moves things -- but a large share means the extractor is
#: measuring something the model never saw.
MAX_OUT_OF_RANGE_FRACTION = 0.20

#: Minimum recordings `run_inference_multi` will accept. One recording is not a
#: reliable basis for a result: three takes from one speaker minutes apart
#: spread 0.29 in calibrated probability.
MIN_TAKES = 5
#: What the app should ask a user for.
RECOMMENDED_TAKES = 7
#: Below this many usable takes after exclusions, the median is not meaningful.
MIN_USABLE_TAKES = 3

#: A take is an outlier if it sits further than this many scaled MADs from the
#: median AND further than OUTLIER_MIN_DEVIATION in absolute terms. Both must
#: hold: with 5-7 takes the MAD is itself noisy, and a small MAD would otherwise
#: exclude takes that merely differ a little.
OUTLIER_MAD_MULTIPLIER = 3.5
OUTLIER_MIN_DEVIATION = 0.25

#: Thresholds for comparing takes on their FEATURES rather than their scores.
#: Score-based detection has a blind spot: when the model saturates -- every take
#: scoring 0.7-0.97 -- a genuinely bad recording produces a perfectly ordinary
#: probability and slips through. Measured on 8 real takes, the six clean ones
#: deviated 0.76-1.11x the group typical while a deliberately-bad take hit 2.44x
#: and a take with a silent gap hit 3.51x, so both criteria below clear a wide gap.
INPUT_OUTLIER_MAD_MULTIPLIER = 3.5
INPUT_OUTLIER_MIN_RATIO = 1.5
#: Never prune more than this fraction. If more takes look aberrant than this,
#: the recordings are inconsistent as a set and that is the finding to report --
#: pruning until they agree would manufacture a confident answer.
MAX_EXCLUDED_FRACTION = 1 / 3


class InferenceError(RuntimeError):
    """Raised when a recording cannot be scored trustworthily."""


class _Artifacts:
    """The four saved v2 artifacts plus the training ranges, loaded once."""

    def __init__(self) -> None:
        missing = [
            name
            for name in (
                "detector_v2.joblib",
                "detector_v2_data.joblib",
                "confidence_v2.joblib",
                "detector_v2_shap.joblib",
                "detector_v2_concepts.joblib",
            )
            if not (MODELS_DIR / name).exists()
        ]
        if missing:
            raise InferenceError(
                f"Missing model artifacts: {missing}. Run scripts/retrain_v2.py first."
            )

        detector = joblib.load(MODELS_DIR / "detector_v2.joblib")
        # Not named `confidence`: that would shadow the pdvoice.confidence module.
        confidence_bundle = joblib.load(MODELS_DIR / "confidence_v2.joblib")
        shap_bundle = joblib.load(MODELS_DIR / "detector_v2_shap.joblib")
        concept_bundle = joblib.load(MODELS_DIR / "detector_v2_concepts.joblib")
        data = joblib.load(MODELS_DIR / "detector_v2_data.joblib")

        self.model = detector["model"]
        self.feature_cols: list[str] = list(detector["feature_cols"])
        self.detector_meta = {
            k: v for k, v in detector.items() if k not in ("model", "feature_cols")
        }

        self.calibrator = confidence_bundle["calibrator"]
        self.threshold = float(confidence_bundle["threshold"])
        self.band_low = float(confidence_bundle["band_low"])
        self.band_high = float(confidence_bundle["band_high"])
        # The decision is made on the likelihood ratio, so the prevalence the
        # calibrator was fitted against is what matters, not the stored edges.
        self.training_prevalence = float(
            confidence_bundle.get("training_prevalence", self.band_high)
        )

        self.concept_label: dict[str, str] = concept_bundle["concept_label"]
        self.narratable: list[str] = list(concept_bundle["narratable"])
        self.concept_index = concepts.concept_index(self.feature_cols)

        # Training ranges, for the out-of-distribution check.
        X = data["X"]
        self.train_min = X.min(axis=0)
        self.train_max = X.max(axis=0)
        # Robust per-feature scale, so the 107 features can be compared on one
        # ruler when looking for a take whose inputs are unlike the others.
        self.train_median = X.median(axis=0)
        self.train_mad = (X - self.train_median).abs().median(axis=0) * 1.4826

        # An explainer over the same model the predictions come from.
        import shap

        self.explainer = shap.TreeExplainer(self.model)
        self.shap_base_value = float(shap_bundle["base_value"])

        if list(data["feature_cols"]) != self.feature_cols:
            raise InferenceError(
                "detector_v2_data.joblib and detector_v2.joblib disagree on feature order."
            )


@lru_cache(maxsize=1)
def _artifacts() -> _Artifacts:
    return _Artifacts()


def _check_input_quality(row: pd.DataFrame, art: _Artifacts) -> dict[str, Any]:
    """Compare every feature against the range the model was trained on."""
    values = row.iloc[0]
    below = values < art.train_min
    above = values > art.train_max
    out = [c for c in art.feature_cols if bool(below[c]) or bool(above[c])]

    fraction = len(out) / len(art.feature_cols)
    reliable = fraction <= MAX_OUT_OF_RANGE_FRACTION
    warning = None
    if not reliable:
        warning = (
            f"{len(out)} of {len(art.feature_cols)} features ({fraction:.0%}) fall outside "
            "the range the model was trained on. This recording is unlike the training "
            "data -- different microphone, room, or vocal task -- so the score below is "
            "not trustworthy and should not be acted on."
        )
    return {
        "reliable": reliable,
        "n_out_of_range": len(out),
        "n_features": len(art.feature_cols),
        "fraction_out_of_range": round(fraction, 4),
        "out_of_range_features": out,
        "warning": warning,
    }


def _concept_contributions(row: pd.DataFrame, art: _Artifacts) -> tuple[dict[str, float], dict[str, float]]:
    """SHAP values for this recording, summed within each concept.

    Returns (all concepts, narratable concepts only). `demographic` is computed
    but withheld from narration -- see pdvoice/concepts.py.
    """
    raw = art.explainer.shap_values(row)
    values = np.asarray(raw[1] if isinstance(raw, list) else raw)
    if values.ndim == 3:  # (n, features, classes)
        values = values[:, :, 1]
    values = values[0]

    if values.shape[0] != len(art.feature_cols):
        raise InferenceError(
            f"SHAP returned {values.shape[0]} values for {len(art.feature_cols)} features."
        )

    everything = {
        name: float(values[positions].sum())
        for name, positions in art.concept_index.items()
    }
    narratable = {k: v for k, v in everything.items() if k in art.narratable}
    return everything, narratable


def _breakdown(contributions: dict[str, float], art: _Artifacts) -> list[dict[str, Any]]:
    return [
        {
            "concept": name,
            "label": art.concept_label.get(name, name),
            "contribution": round(float(value), 4),
            "direction": "toward PD" if value > 0 else "toward healthy",
            "narrated": name in art.narratable,
        }
        for name, value in sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    ]


def _adjusted_block(lr: float) -> dict[str, Any]:
    """The probability that IS safe to show -- with its assumption attached.

    Bundled with `assumed_prevalence` rather than returned bare, so the number
    cannot travel without the input that produced it. At a different assumed
    prevalence it is a different number, and quoting it alone would be as
    misleading as quoting the training-calibrated one.
    """
    q = confidence.TARGET_PREVALENCE
    return {
        "probability": round(confidence.adjust_prevalence(lr, q), 5),
        "assumed_prevalence": q,
        "baseline_probability": q,
        "likelihood_ratio": round(float(lr), 4),
        "note": (
            f"Probability of Parkinson's given this voice, assuming {q:.0%} "
            "prevalence in the population being screened. Compare it against "
            f"that {q:.0%} baseline, not against 0. The assumed prevalence is an "
            "input, not a measurement -- see confidence.TARGET_PREVALENCE."
        ),
    }


def _internal_block(raw: float, calibrated: float) -> dict[str, Any]:
    """Model probabilities, quarantined from display.

    These decide the band and nothing else. They are calibrated to a cohort that
    was 74.6% Parkinson's patients, so presenting either as "your chance of
    having Parkinson's" would overstate it by orders of magnitude for a
    general-population user. `confidence_pct` is the number that is safe to show.
    """
    return {
        "DO_NOT_DISPLAY_AS_DISEASE_PROBABILITY": True,
        "calibrated_probability": round(float(calibrated), 4),
        "raw_probability": round(float(raw), 4),
        "calibrated_to_prevalence": 0.746,
        # Reserved. Producing this needs an assumed population prior and an
        # explicit disclosure of that prior alongside the number; deliberately
        # not implemented yet.
        "prevalence_adjusted_probability": None,
        "note": (
            "Calibrated against a 74.6%-PD training cohort, not a general "
            "population (<1-2% prevalence). Used to select the band only."
        ),
    }


def run_inference(
    wav_path: str | Path,
    gender: int,
    llm_fn: Callable[[str], str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Score one recording of a sustained vowel, end to end.

    Args:
        wav_path: the recording.
        gender: dataset gender code, 0 = female / 1 = male. Required rather than
            optional: it is one of the model's 107 inputs and it selects the
            formant ceiling. It cannot be read off the audio, and guessing it
            would silently mis-measure the formant features.
        llm_fn: takes a prompt, returns phrased text. None uses the
            deterministic template, which is also the production fallback.

    Returns:
        A dict with `result`, `confidence_pct`, `raw_probability`,
        `calibrated_probability`, `explanation_text`, `concept_breakdown`,
        `input_quality`, `facts` and `meta`.

    Raises:
        InferenceError: the recording cannot be scored trustworthily.
    """
    art = _artifacts()

    # ---- voicing pre-check ---------------------------------------------
    # Runs before extraction because silence corrupts the MFCC block silently:
    # the features come out looking numerically ordinary, so nothing downstream
    # would reveal the problem. One cheap pitch pass settles it up front.
    #
    # `force` waives the gate but still measures, so the recording is scored and
    # the voicing figure stays in the output rather than being quietly dropped --
    # an override should be visible in the result, not invisible.
    try:
        voicing = measure_voicing(wav_path) if force else require_voicing(wav_path)
    except FileNotFoundError as error:
        raise InferenceError(f"Recording not found: {error}") from error
    except InsufficientVoicingError as error:
        raise InferenceError(f"Not enough voice in the recording: {error}") from error
    except FeatureExtractionError as error:
        raise InferenceError(f"Could not analyse the recording: {error}") from error

    # ---- extract -------------------------------------------------------
    # Every way extraction can fail becomes one exception type, so a caller
    # cannot accidentally let a missing file or a Praat error escape as
    # something that looks like a crash rather than a refusal to score.
    try:
        features = extract_features(wav_path, gender=gender)
    except FileNotFoundError as error:
        raise InferenceError(f"Recording not found: {error}") from error
    except FeatureExtractionError as error:
        raise InferenceError(f"Could not extract features: {error}") from error

    missing = [c for c in art.feature_cols if c not in features]
    if missing:
        raise InferenceError(
            f"Extractor did not produce {len(missing)} feature(s) the model needs: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    # Built by explicit column order, never by dict insertion order, so a
    # reordering cannot silently feed the model a shuffled vector.
    row = pd.DataFrame([[features[c] for c in art.feature_cols]], columns=art.feature_cols)

    non_finite = [c for c in art.feature_cols if not np.isfinite(row.iloc[0][c])]
    if non_finite:
        raise InferenceError(
            f"Extraction produced non-finite values for {non_finite}. The recording is "
            "probably too short, silent, or has no voiced segment."
        )

    quality = _check_input_quality(row, art)

    # ---- predict + calibrate -------------------------------------------
    raw_probability = float(art.model.predict_proba(row)[0, 1])
    calibrated = float(art.calibrator.transform([raw_probability])[0])

    # ---- explain --------------------------------------------------------
    all_concepts, narratable = _concept_contributions(row, art)
    # Everything downstream is decided on prevalence-free evidence, not on the
    # training-calibrated probability. See pdvoice/confidence.py.
    lr = confidence.likelihood_ratio(calibrated, art.training_prevalence)
    band = confidence.band_for(lr)
    evidence = confidence.evidence_strength(lr)
    # A single recording carries no agreement evidence, so no spread discount.
    score = confidence.confidence_score(lr, spread_std=None)
    facts = explain.build_facts(
        band=band,
        concept_contributions=narratable,
        confidence=score,
        evidence=evidence,
    )
    explanation_text = explain.phrase_explanation(facts, llm_fn=llm_fn)
    if quality["warning"]:
        explanation_text = f"UNRELIABLE INPUT: {quality['warning']}\n\n{explanation_text}"

    return {
        "result": facts["band"],
        "confidence_pct": score["confidence_pct"],
        "confidence_level": score["confidence_level"],
        "confidence_detail": score,
        "evidence": evidence,
        "adjusted_probability": _adjusted_block(lr),
        "explanation_text": explanation_text,
        "concept_breakdown": _breakdown(all_concepts, art),
        # Returned so run_inference_multi can compare takes on their inputs
        # rather than only on their scores -- see _find_input_outliers.
        "features": features,
        "input_quality": {**quality, "voicing": voicing},
        "facts": facts,
        "internal": _internal_block(raw_probability, calibrated),
        "meta": {
            "wav": str(wav_path),
            "gender": gender,
            "model": "detector_v2",
            "n_features": len(art.feature_cols),
            "decision_threshold": art.threshold,
            "band_edges": [art.band_low, art.band_high],
            "cv_speaker_auc": art.detector_meta.get("cv_speaker_auc"),
            "shap_base_value": art.shap_base_value,
        },
    }


# --------------------------------------------------------------------------
# Multi-take aggregation
# --------------------------------------------------------------------------
def _find_outliers(probabilities: list[float]) -> list[int]:
    """Indices of takes too far from the median to combine with the rest.

    Uses median absolute deviation rather than standard deviation, because the
    thing being guarded against -- one cough, crack or noisy take -- would
    inflate a standard deviation and hide itself.
    """
    values = np.asarray(probabilities, dtype=float)
    median = float(np.median(values))
    deviations = np.abs(values - median)
    mad = float(np.median(deviations)) * 1.4826

    threshold = max(OUTLIER_MAD_MULTIPLIER * mad, OUTLIER_MIN_DEVIATION)
    flagged = [i for i, d in enumerate(deviations) if d > threshold]

    # Prune only the worst offenders if too many are flagged; the remainder are
    # reported through the spread warning instead.
    max_excluded = int(len(values) * MAX_EXCLUDED_FRACTION)
    if len(flagged) > max_excluded:
        flagged = sorted(flagged, key=lambda i: deviations[i], reverse=True)[:max_excluded]
    return sorted(flagged)


def _find_input_outliers(
    scored: list[dict[str, Any]], art: _Artifacts
) -> tuple[list[int], list[float]]:
    """Indices of takes whose FEATURES disagree with the rest, plus each take's score.

    Complements `_find_outliers`, which compares calibrated probabilities. That
    check cannot see a bad recording once the model has saturated -- if every
    take lands at 0.7-0.97, a cough produces a probability indistinguishable from
    a clean take. Comparing the inputs instead catches it, because the features
    themselves are visibly unlike the others.

    Each feature is scaled by its *training* median-absolute-deviation, so the
    107 features are commensurable and the scale does not shift with the number
    of takes supplied. A take is summarised by its median scaled deviation from
    the element-wise median take, which ignores a handful of odd features and
    responds to a recording that is broadly different.
    """
    matrix = np.array([[s["features"][c] for c in art.feature_cols] for s in scored], dtype=float)

    scale = art.train_mad.to_numpy()
    scale = np.where(scale > 0, scale, np.nan)
    centre = np.median(matrix, axis=0)
    with np.errstate(invalid="ignore"):
        deviation = np.abs(matrix - centre) / scale
    per_take = np.nanmedian(deviation, axis=1)

    group = float(np.median(per_take))
    mad = float(np.median(np.abs(per_take - group))) * 1.4826

    flagged = []
    for i, value in enumerate(per_take):
        ratio = value / group if group > 0 else 1.0
        z = (value - group) / mad if mad > 0 else 0.0
        # Both must hold: the MAD test alone fires on trivial differences when
        # the takes are near-identical, and the ratio alone is scale-blind.
        if z > INPUT_OUTLIER_MAD_MULTIPLIER and ratio > INPUT_OUTLIER_MIN_RATIO:
            flagged.append(i)

    max_excluded = int(len(scored) * MAX_EXCLUDED_FRACTION)
    if len(flagged) > max_excluded:
        flagged = sorted(flagged, key=lambda i: per_take[i], reverse=True)[:max_excluded]
    return sorted(flagged), [float(v) for v in per_take]


def run_inference_multi(
    wav_paths: list[str | Path],
    gender: int,
    llm_fn: Callable[[str], str] | None = None,
    force_include: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Score several recordings of the same speaker and combine them.

    A thin wrapper over `run_inference`, which is unchanged and still scores one
    file at a time. Combination is by median rather than mean, so a single bad
    take shifts the result far less; takes that disagree sharply with the rest
    are excluded outright and named in the output.

    Args:
        wav_paths: at least MIN_TAKES recordings of the same speaker.
            RECOMMENDED_TAKES is what the app should ask for.
        gender: dataset gender code, 0 female / 1 male.
        llm_fn: phrasing seam, as in `run_inference`.
        force_include: paths or bare filenames to combine regardless of the
            voicing gate or the outlier checks -- a deliberate override for
            inspecting a borderline take. It cannot rescue a recording that
            fails extraction outright (silence, no voiced segment): there are no
            features to score, so nothing to force. Overridden takes are named
            in the result and flagged in the explanation text, because a result
            that ignored its own quality checks must not read like one that
            passed them.

    Returns:
        The same shape as `run_inference` plus `takes` and `spread` blocks.

    Raises:
        InferenceError: too few recordings given, or too few scored successfully.
    """
    paths = [Path(p) for p in wav_paths]
    # Match on the full path or the bare filename, so a caller that only knows
    # the name the user uploaded (the API does) can still address a take.
    forced_keys = {str(f) for f in (force_include or [])}
    is_forced = lambda p: str(p) in forced_keys or p.name in forced_keys  # noqa: E731
    if len(paths) < MIN_TAKES:
        raise InferenceError(
            f"run_inference_multi needs at least {MIN_TAKES} recordings "
            f"({RECOMMENDED_TAKES} recommended); got {len(paths)}. A single take is "
            "not a reliable basis for a result."
        )

    art = _artifacts()

    scored: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    voicing_excluded: list[dict[str, Any]] = []

    forced_used: list[dict[str, Any]] = []
    for path in paths:
        forced = is_forced(path)
        # Checked per take rather than for the batch: one take with a pause is a
        # reason to drop that take, not to refuse the whole recording session.
        if not forced:
            try:
                require_voicing(path)
            except InsufficientVoicingError as error:
                voicing_excluded.append(
                    {
                        "kind": "insufficient_voicing",
                        "wav": str(path),
                        "unvoiced_fraction": error.unvoiced_fraction,
                        "reason": (
                            f"{error.unvoiced_fraction:.0%} of the analysed audio carries no "
                            "voice; the vowel was not held for the whole recording"
                        ),
                    }
                )
                continue
            except (FileNotFoundError, FeatureExtractionError) as error:
                failed.append({"wav": str(path), "reason": str(error)})
                continue

        try:
            result = run_inference(path, gender=gender, llm_fn=None, force=forced)
        except InferenceError as error:
            # A forced take that cannot be extracted at all still fails: the
            # override waives the quality gates, not the laws of arithmetic.
            failed.append({"wav": str(path), "reason": str(error)})
            continue
        scored.append(result)
        if forced:
            forced_used.append({
                "wav": str(path),
                "unvoiced_fraction": result["input_quality"]["voicing"]["unvoiced_fraction"],
            })

    if len(scored) < MIN_USABLE_TAKES:
        detail = []
        if voicing_excluded:
            detail.append(
                f"{len(voicing_excluded)} had too little voice "
                f"({', '.join(Path(v['wav']).name for v in voicing_excluded)})"
            )
        if failed:
            detail.append(f"{len(failed)} could not be analysed: {failed}")
        raise InferenceError(
            f"Only {len(scored)} of {len(paths)} recordings were usable, below the "
            f"minimum of {MIN_USABLE_TAKES}. "
            + ("; ".join(detail) if detail else "")
            + " Re-record, holding a steady 'ahhh' for the whole recording."
        )

    probabilities = [s["internal"]["calibrated_probability"] for s in scored]
    median_all = float(np.median(probabilities))

    # Inputs first: a take whose measurements are unlike the others is suspect
    # whatever score it happened to produce. Then scores, on what survives.
    # A forced take is still *measured* by both checks -- so the output can say
    # it would have been dropped -- but is never acted on.
    forced_positions = {
        i for i, s in enumerate(scored) if is_forced(Path(s["meta"]["wav"]))
    }
    input_all, input_deviation = _find_input_outliers(scored, art)
    input_idx = [i for i in input_all if i not in forced_positions]
    score_idx = [
        i for i in _find_outliers(probabilities)
        if i not in set(input_idx) and i not in forced_positions
    ]
    # Capped jointly, so the two checks together cannot prune a batch down to
    # whichever takes happen to agree.
    combined = (input_idx + score_idx)[: int(len(scored) * MAX_EXCLUDED_FRACTION)]
    input_set = set(input_idx) & set(combined)
    score_set = set(score_idx) & set(combined)
    outlier_idx = set(combined)

    used = [s for i, s in enumerate(scored) if i not in outlier_idx]
    excluded = list(voicing_excluded)
    for i, s in enumerate(scored):
        if i in input_set:
            excluded.append(
                {
                    "kind": "input_outlier",
                    "wav": s["meta"]["wav"],
                    "calibrated_probability": probabilities[i],
                    "feature_deviation": round(input_deviation[i], 4),
                    "reason": (
                        "the measurements from this recording are unlike the other "
                        "takes; typically a cough, voice crack, or background noise"
                    ),
                }
            )
        elif i in score_set:
            excluded.append(
                {
                    "kind": "outlier",
                    "wav": s["meta"]["wav"],
                    "calibrated_probability": probabilities[i],
                    "deviation_from_median": round(abs(probabilities[i] - median_all), 4),
                    "reason": (
                        "calibrated probability too far from the other recordings "
                        "to combine"
                    ),
                }
            )

    used_probs = [s["internal"]["calibrated_probability"] for s in used]
    combined = float(np.median(used_probs))
    spread = confidence.spread_summary(used_probs)
    lr = confidence.likelihood_ratio(combined, art.training_prevalence)
    band = confidence.band_for(lr)
    evidence = confidence.evidence_strength(lr)
    score = confidence.confidence_score(lr, spread_std=spread["std"])

    # Concepts combined the same way as the probabilities, for the same reason.
    concept_names = [c["concept"] for c in used[0]["concept_breakdown"]]
    combined_concepts = {
        name: float(
            np.median([
                next(c["contribution"] for c in s["concept_breakdown"] if c["concept"] == name)
                for s in used
            ])
        )
        for name in concept_names
    }
    narratable = {k: v for k, v in combined_concepts.items() if k in art.narratable}

    takes_block = {
        "n_given": len(paths),
        "n_scored": len(scored),
        "n_used": len(used),
        "n_excluded": len(excluded),
        "n_failed": len(failed),
        "recommended": RECOMMENDED_TAKES,
        "used": [
            {"wav": s["meta"]["wav"], "calibrated_probability": p, "result": s["result"]}
            for s, p in zip(used, used_probs)
        ],
        "excluded": excluded,
        "failed": failed,
        # Named so the caller can show which quality checks were waived, and on
        # what. `would_have_been_excluded` records that the checks still ran.
        "forced": [
            {
                **f,
                "would_have_been_excluded": any(
                    i in input_all and Path(scored[i]["meta"]["wav"]) == Path(f["wav"])
                    for i in range(len(scored))
                ),
            }
            for f in forced_used
        ],
        "n_forced": len(forced_used),
    }

    facts = explain.build_facts(
        band=band,
        concept_contributions=narratable,
        confidence=score,
        evidence=evidence,
        takes=takes_block,
        spread=spread,
    )
    explanation_text = explain.phrase_explanation(facts, llm_fn=llm_fn)

    # A result that ignored its own quality checks must not read like one that
    # passed them, so the override is stated before anything else.
    if forced_used:
        names = ", ".join(Path(f["wav"]).name for f in forced_used)
        verb = "was" if len(forced_used) == 1 else "were"
        explanation_text = (
            f"QUALITY CHECKS OVERRIDDEN: {names} {verb} included by explicit request "
            "despite failing the recording-quality checks. This result is less "
            "trustworthy than one computed from clean recordings only.\n\n"
            + explanation_text
        )

    unreliable = [s for s in used if not s["input_quality"]["reliable"]]
    if unreliable:
        explanation_text = (
            f"UNRELIABLE INPUT: {len(unreliable)} of {len(used)} combined recordings "
            "had features well outside the training range.\n\n" + explanation_text
        )

    return {
        "result": facts["band"],
        "confidence_pct": score["confidence_pct"],
        "confidence_level": score["confidence_level"],
        "confidence_detail": score,
        "evidence": evidence,
        "adjusted_probability": _adjusted_block(lr),
        "explanation_text": explanation_text,
        "concept_breakdown": _breakdown(combined_concepts, art),
        "takes": takes_block,
        "spread": spread,
        "facts": facts,
        "internal": {
            **_internal_block(
                float(np.median([s["internal"]["raw_probability"] for s in used])), combined
            ),
            "combined_by": "median",
            "per_take_calibrated": used_probs,
        },
        "meta": {
            **used[0]["meta"],
            "wav": [str(p) for p in paths],
            "n_takes_used": len(used),
        },
    }
