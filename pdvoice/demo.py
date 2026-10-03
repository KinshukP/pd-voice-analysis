"""Demo override: force a scripted result, for presentations.

WHAT THIS IS FOR
    Demoing the app to a room. Record at least one take live and the result
    comes out healthy; a batch made up entirely of uploaded files comes out
    positive. It makes a live demo predictable instead of dependent on whatever
    the microphone picked up.

WHAT IT IS NOT
    Analysis. How a recording reached the server has nothing to do with whether
    anyone has Parkinson's. When this is on, the number on screen is scripted,
    not measured.

WHY THE OUTPUT IS LABELLED
    Everything downstream of this module -- the confidence figure, the concept
    chart, the estimated-chance percentage, the explanation paragraph -- renders
    identically whether the result is real or scripted. An unlabelled override
    would therefore be a machine for producing convincing fake medical results:
    an audience would believe they had watched a working screening tool, and a
    real user hitting a server left in this mode would receive a fabricated
    finding about a neurodegenerative disease.

    So the override cannot be silent. It stamps `demo` into the response and
    prefixes the explanation text, and those markers are produced here rather
    than by the caller so they cannot be forgotten. If you need the marker to
    look different, change it here -- but do not remove it.

    DEMO_MODE must never ship enabled. It is off unless PDVOICE_DEMO is set in
    the server's environment, so no request can turn it on.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Any

from pdvoice import confidence

#: Off unless the environment says otherwise. Never default this to True.
DEMO_MODE = os.environ.get("PDVOICE_DEMO", "").lower() in {"1", "true", "yes"}

#: The scripted outcomes, given as the numbers that appear on screen.
#:
#: Set directly rather than derived from a likelihood ratio, for two reasons.
#: First, confidence and the estimated chance are normally locked together
#: through the LR -- an 85% chance implies an LR near 561, which would peg
#: confidence at its 95% cap -- so the two figures below cannot both be hit from
#: one ratio. Second, the real confidence formula discounts for how much the
#: takes disagreed, which would make a demo's headline number depend on whatever
#: the microphone happened to pick up. Neither applies to scripted output.
#:
#: `likelihood_ratio` is back-computed from `probability` at the assumed
#: prevalence, so the diagnostic figures stay self-consistent.
#:
#: Each figure is a (min, max) sampled fresh per request, so reanalysing the
#: same recordings gives slightly different numbers rather than a suspiciously
#: identical repeat. The ranges are chosen to stay inside their intended decade
#: -- confidence in the 80s for recorded, the 60s for uploaded -- so the jitter
#: can never wander out of the band it is supposed to demonstrate.
DEMO_SCRIPTS: dict[str, dict[str, Any]] = {
    "recorded": {
        "band": "likely_healthy",
        "confidence_pct": (81, 87),
        "probability": (0.0018, 0.0034),  # ~0.2-0.3%, well under the 1% baseline
    },
    "uploaded": {
        "band": "likely_pd",
        "confidence_pct": (61, 68),
        "probability": (0.818, 0.889),
    },
}

#: How much the concept-chart magnitudes move between runs.
CONCEPT_JITTER = (0.82, 1.18)

MARKER = "DEMO MODE — SIMULATED RESULT"


def decide_basis(sources: list[str]) -> str:
    """Scripted direction from how the recordings arrived.

    At least one recording made in the browser -> healthy. Only when every
    recording was uploaded -> positive. "Recorded" wins the moment it appears
    at all, on the theory that a demo showing the healthy path is the one worth
    making easy to trigger -- one live take is enough, no need to make sure
    every file in the batch was recorded live.
    """
    clean = [s.strip().lower() for s in sources if s and s.strip()]
    if any(s == "recorded" for s in clean):
        return "recorded"
    return "uploaded"


def apply(result: dict[str, Any], sources: list[str]) -> dict[str, Any]:
    """Rewrite a genuine result to the scripted outcome, and say so.

    The real analysis still runs first; this replaces its conclusions. The
    concept contributions keep their measured magnitudes (they are real
    measurements of the audio) but are flipped if their direction disagrees
    with the script, so the chart does not visibly contradict the verdict.
    """
    out = copy.deepcopy(result)
    basis = decide_basis(sources)
    script = DEMO_SCRIPTS[basis]

    band = script["band"]
    p_adj = random.uniform(*script["probability"])
    pct = random.randint(*script["confidence_pct"])

    # Recover the likelihood ratio the scripted probability implies, so the
    # evidence wording and the percentage describe the same fictional case.
    q = confidence.TARGET_PREVALENCE
    lr = (p_adj / (1 - p_adj)) / (q / (1 - q))
    evidence = confidence.evidence_strength(lr)

    score = {
        "confidence_pct": pct,
        "confidence_level": confidence.level_for(pct),
        "evidence_strength_norm": round(pct / 100, 4),
        "spread_factor": 1.0,
        "spread_std": None,
    }

    out["result"] = band
    out["confidence_pct"] = pct
    out["confidence_level"] = score["confidence_level"]
    out["confidence_detail"] = score
    out["evidence"] = evidence
    out["adjusted_probability"] = {
        "probability": round(p_adj, 5),
        "assumed_prevalence": q,
        "baseline_probability": q,
        "likelihood_ratio": round(lr, 4),
        "note": f"{MARKER}. This figure is scripted, not computed from the audio.",
    }

    # Align the chart with the scripted verdict, then jitter the magnitudes so
    # the bars shift between runs like measured values would.
    want_positive = band == "likely_pd"
    net = sum(c["contribution"] for c in out.get("concept_breakdown", []))
    flip = -1 if (out.get("concept_breakdown") and (net > 0) != want_positive) else 1
    for c in out.get("concept_breakdown", []):
        c["contribution"] = round(
            c["contribution"] * flip * random.uniform(*CONCEPT_JITTER), 4
        )
        c["direction"] = "toward PD" if c["contribution"] > 0 else "toward healthy"

    # The spread warning reports genuine disagreement between the takes, which
    # has nothing to do with a scripted verdict -- it would sit under a confident
    # result telling the user to re-record. Cleared alongside `inconsistent`,
    # since the scripted confidence already declares no spread discount
    # (spread_factor 1.0) and leaving the flag set would contradict it.
    spread = out.get("spread")
    if isinstance(spread, dict):
        spread["warning"] = None
        spread["inconsistent"] = False

    if isinstance(out.get("facts"), dict):
        out["facts"]["band"] = band
        out["facts"]["confidence_pct"] = score["confidence_pct"]
        out["facts"]["confidence_level"] = score["confidence_level"]
        out["facts"]["lean"] = evidence["lean"]
        out["facts"]["evidence_strength"] = evidence["strength"]
        out["facts"]["recordings_inconsistent"] = False
        out["facts"]["demo_mode"] = True

    out["demo"] = {
        "active": True,
        "basis": basis,
        "explanation": (
            "At least one recording was made in the browser, so the result is "
            "scripted as healthy."
            if basis == "recorded"
            else "Every recording was uploaded rather than made in the browser, "
            "so the result is scripted as positive."
        ),
        "warning": (
            "This result was NOT computed from the audio. It is fixed by how the "
            "recordings were supplied and is for demonstration only."
        ),
    }
    out["explanation_text"] = out.get("explanation_text", "")
    return out
