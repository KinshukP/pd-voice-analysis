"""Turning the model's score into a decision, a strength, and an honest number.

The detector's calibrated probability answers a question nobody asked: "how
likely is Parkinson's *in a cohort that is 74.6% patients*" -- because that is
what it was trained and calibrated on. Its answer before hearing any voice is
0.795. Shown to a real user that overstates risk by roughly two orders of
magnitude.

Everything here is built on the quantity that does NOT carry that assumption:

    LR = [p/(1-p)] / [pi/(1-pi)]         pi = training prevalence

The likelihood ratio is how much more the recording resembles a patient's voice
than a control's. LR = 1 means the recording says nothing either way. It is a
property of the voice alone, so it is the right basis for both the decision and
the strength wording, and it is what makes the prevalence swap below a one-line
change rather than a recalibration.

The displayed probability is then rebuilt against a *stated* prevalence:

    adjusted_odds = LR x [q/(1-q)]       q = TARGET_PREVALENCE
    p_adjusted    = odds / (1 + odds)

One thing worth being precise about: because the transform is monotone, moving
to p_adjusted does not change *which side* of neutral a recording falls on --
that was already fixed by anchoring the decision at LR = 1. What it changes is
the number a person reads, and the scale the confidence is measured on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Assumed Parkinson's prevalence in the population being screened.
#:
#: DISCLOSED ASSUMPTION, not a measurement. Parkinson's affects roughly 1% of
#: adults over 60 and about 0.3% of the general population of all ages
#: (Parkinson's Foundation / GBD prevalence estimates). 1% is taken as a
#: middle-ground figure for a self-selected adult screening population -- people
#: who seek out a voice screening tool skew older and more symptomatic than the
#: general public, but are not a clinic population.
#:
#: This number is an input to the answer, not a fact about the user. It must be
#: displayed alongside any probability derived from it, and it is the first
#: thing to revisit if the tool is aimed at a different group (a movement-
#: disorders clinic would be far higher; a workplace health app lower).
TARGET_PREVALENCE = 0.01

#: The abstain zone, in likelihood-ratio terms. A recording this close to
#: neutral is not evidence of anything, so no call is made.
#:
#: Deliberately narrow: it fires on 2.1% of training recordings. A wide band was
#: the previous design's mistake -- it swallowed a fifth of all results and, worse,
#: reported a flat 0% confidence for every one of them, so "inconclusive" became
#: the answer for cases that were merely weak rather than genuinely ambiguous.
ABSTAIN_LR_LOW, ABSTAIN_LR_HIGH = 0.80, 1.25

#: Take-to-take standard deviation at which agreement is fully discounted.
SPREAD_REFERENCE = 0.25
#: Spread above which the recordings are called inconsistent to the user.
INCONSISTENT_STD = 0.15

#: |log10 LR| at which evidence counts as maximally strong -- 1.5 decades, i.e.
#: a likelihood ratio of ~32x either way. Set above the "large evidence" mark of
#: 10x on purpose: a 10x ratio saturating the scale had a single take reporting
#: 100% confidence, which no screening tool built on a 0.86-AUC model should say.
FULL_STRENGTH_DECADES = 1.5

#: Nothing is ever reported as certain.
MAX_CONFIDENCE_PCT = 95

#: Evidence strength by |LR|, following the usual diagnostic reading of
#: likelihood ratios (1-2 negligible, 2-5 small, 5-10 moderate, >10 large).
#: Expressed as the lower bound of the ratio-away-from-1, checked strong first.
_STRENGTH_BANDS = ((10.0, "very strong"), (5.0, "strong"), (2.0, "moderate"), (1.0, "weak"))

_EPS = 1e-6


def likelihood_ratio(p_trained: float, training_prevalence: float) -> float:
    """Strip the training set's prevalence out of a calibrated probability."""
    p = min(max(float(p_trained), _EPS), 1 - _EPS)
    prior_odds = training_prevalence / (1 - training_prevalence)
    return (p / (1 - p)) / prior_odds


def adjust_prevalence(lr: float, target_prevalence: float = TARGET_PREVALENCE) -> float:
    """Rebuild a probability against a stated real-world prevalence."""
    odds = lr * (target_prevalence / (1 - target_prevalence))
    return odds / (1 + odds)


def band_for(lr: float) -> str:
    """The decision, made on evidence rather than on the training set's mix."""
    if lr > ABSTAIN_LR_HIGH:
        return "likely_pd"
    if lr < ABSTAIN_LR_LOW:
        return "likely_healthy"
    return "uncertain"


def evidence_strength(lr: float) -> dict[str, Any]:
    """Which way the recording leans, and how far -- always both, never a bare number.

    A number alone ("4%") is unreadable: it cannot say whether 4% means a weak
    lean toward healthy or a weak lean toward disease. Direction and strength
    are always reported together, including when no call is made.
    """
    lr = max(float(lr), _EPS)
    # Phrased with "the evidence" as the subject so it reads correctly whether
    # one recording or seven produced it.
    if ABSTAIN_LR_LOW <= lr <= ABSTAIN_LR_HIGH:
        direction, toward = "neutral", "the evidence points neither way"
    elif lr > 1:
        direction, toward = "pd", "the evidence leans toward Parkinson's"
    else:
        direction, toward = "healthy", "the evidence leans healthy"

    ratio = lr if lr >= 1 else 1 / lr
    strength = (
        "negligible"
        if direction == "neutral"
        else next((name for floor, name in _STRENGTH_BANDS if ratio >= floor), "weak")
    )

    return {
        "direction": direction,
        "lean": toward,
        "strength": strength,
        "likelihood_ratio": round(float(lr), 4),
        "summary": f"{toward} — {strength} evidence",
    }


def level_for(confidence_pct: float) -> str:
    for floor, name in ((70, "high"), (40, "moderate"), (15, "low"), (0, "very low")):
        if confidence_pct >= floor:
            return name
    return "very low"


def confidence_score(
    lr: float,
    spread_std: float | None = None,
) -> dict[str, Any]:
    """How much to trust the call: strength of evidence x agreement between takes.

    Strength is measured in decades of likelihood ratio rather than as a
    distance in probability space. Probability space is compressed near 0 and 1,
    so the same evidence would score differently depending only on the assumed
    prevalence; |log10 LR| is prevalence-free, which is the point.
    """
    lr = max(float(lr), _EPS)
    decades = abs(np.log10(lr))
    strength_norm = float(min(decades / FULL_STRENGTH_DECADES, 1.0))

    if spread_std is None:
        spread_factor = 1.0
    else:
        spread_factor = float(max(0.0, 1.0 - min(spread_std / SPREAD_REFERENCE, 1.0)))

    pct = min(100.0 * strength_norm * spread_factor, MAX_CONFIDENCE_PCT)
    return {
        "confidence_pct": int(round(pct)),
        "confidence_level": level_for(pct),
        "evidence_strength_norm": round(strength_norm, 4),
        "spread_factor": round(spread_factor, 4),
        "spread_std": None if spread_std is None else round(float(spread_std), 4),
    }


def spread_summary(probabilities: list[float]) -> dict[str, Any]:
    """Describe how much the takes disagreed, and whether that is a problem."""
    values = np.asarray(probabilities, dtype=float)
    std = float(values.std(ddof=0))
    lo, hi = float(values.min()), float(values.max())
    inconsistent = std > INCONSISTENT_STD
    return {
        "std": round(std, 4),
        "range": round(hi - lo, 4),
        "min": round(lo, 4),
        "max": round(hi, 4),
        "inconsistent": inconsistent,
        "warning": (
            "Results were inconsistent across recordings; consider retrying in a "
            "quieter environment, holding a steady pitch and volume throughout."
            if inconsistent
            else None
        ),
    }
