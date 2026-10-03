"""Turning a scored recording into a grounded, plain-language explanation.

Ported from `colab_reference.py` (build_facts / build_prompt / render_template /
phrase_explanation) with the logic unchanged. The one structural change: the
Colab version indexed into a precomputed SHAP table for a training row, whereas
inference needs to work from concept contributions computed for a new recording,
so `build_facts` takes those as an argument instead of a row index.

The hard design rule from PROJECT_CONTEXT is enforced here: an LLM phrasing a
result sees concept labels only, never raw feature names. A correlation study
found only 3 of the top 20 exotic features had any strong relationship to an
interpretable measure, so letting a model narrate `tqwt_entropy_log_dec_33`
would produce confident, invented meanings. `build_prompt` therefore ships the
facts dict and nothing else.
"""

from __future__ import annotations

import json
from typing import Any, Callable

#: A concept must move the score at least this much to be worth mentioning.
MIN_FACTOR_WEIGHT = 0.10
#: At most this many concepts in one explanation.
MAX_FACTORS = 3


def build_facts(
    band: str,
    concept_contributions: dict[str, float],
    confidence: dict[str, Any],
    evidence: dict[str, Any],
    takes: dict[str, Any] | None = None,
    spread: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the grounded facts an explanation may draw on.

    Everything downstream must come from this dict. Nothing else about the
    recording is allowed into the phrasing step.

    Note what is deliberately absent: no probability of any kind reaches this
    dict. The training-calibrated one would be narrated as a real-world chance
    of having Parkinson's, and even the prevalence-adjusted one is only
    meaningful beside its assumed prevalence -- which an LLM cannot be relied on
    to carry. `confidence` and `evidence` hold what is safe to state; the
    adjusted probability is surfaced by the UI, next to its assumption.
    """
    ordered = sorted(concept_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    factors = [
        {
            "concept": name.replace("_", " "),
            "direction": "toward PD" if value > 0 else "toward healthy",
            "weight": round(float(value), 2),
        }
        for name, value in ordered
        if abs(value) >= MIN_FACTOR_WEIGHT
    ][:MAX_FACTORS]

    # SHAP measures departure from the model's baseline, and that baseline is
    # ~80% PD because the training cohort was 75% patients. So a recording can
    # land in likely_pd while every concept points toward healthy: the verdict
    # is then coming from the prior, not from this voice. Saying "shows signs of
    # Parkinson's" in that case would be a plain contradiction of the evidence
    # listed beside it, so the flag is carried through to the phrasing.
    #
    # Since band_high is pinned to the evidence-neutral point (likelihood ratio
    # 1.0), a "contradicts" outcome should no longer be reachable for likely_pd.
    # The check stays as a tripwire: if it ever fires again, the band edges have
    # drifted away from the evidence and the result is not safe to narrate.
    net = sum(value for _, value in ordered)
    if band == "uncertain":
        # "supports" would be an overclaim here -- there is no call to support.
        agreement = "inconclusive"
    elif band == "likely_pd" and net <= 0:
        agreement = "contradicts"
    elif band == "likely_healthy" and net >= 0:
        agreement = "contradicts"
    else:
        agreement = "supports"

    facts: dict[str, Any] = {
        "band": band,
        "confidence_pct": int(confidence["confidence_pct"]),
        "confidence_level": confidence["confidence_level"],
        # Direction and strength travel together, always -- a bare percentage
        # cannot say whether it means a weak lean toward health or toward disease.
        "lean": evidence["lean"],
        "evidence_strength": evidence["strength"],
        "top_factors": factors,
        "net_factor_weight": round(float(net), 3),
        "evidence_vs_band": agreement,
    }
    if takes is not None:
        facts["takes"] = {
            "used": takes["n_used"],
            "excluded": takes["n_excluded"],
            "given": takes["n_given"],
            # Kept so the phrasing can name the actual reason: a take dropped for
            # silence was not "inconsistent with the others", and telling a user
            # the wrong reason costs them the chance to fix it.
            "excluded_kinds": sorted(
                {e.get("kind", "outlier") for e in takes.get("excluded", [])}
            ),
        }
    if spread is not None:
        facts["recordings_inconsistent"] = bool(spread["inconsistent"])
    return facts


def build_prompt(facts: dict[str, Any]) -> str:
    """The instruction a real LLM would receive. Facts only, no feature names."""
    return (
        "Write a 3-5 sentence plain-language explanation of a Parkinson's "
        "voice-SCREENING result for a general audience.\n"
        "RULES: Use ONLY the facts below. Do NOT invent or describe what any "
        "acoustic feature means beyond the plain labels given. Match tone to "
        "confidence. If band is likely_healthy, say 'no strong signs detected', "
        "never 'you are healthy'. End with one line noting this is a screening "
        "tool, not a diagnosis, and to consult a clinician.\n\n"
        "FACTS:\n" + json.dumps(facts, indent=2)
    )


def render_template(facts: dict[str, Any]) -> str:
    """Deterministic phrasing. Serves as both the mock and the production fallback."""
    subject = "These recordings show" if facts.get("takes") else "This recording shows"
    plural = "These recordings" if facts.get("takes") else "This recording"
    head = {
        "likely_pd": f"{subject} acoustic signs associated with Parkinson's.",
        "likely_healthy": f"{subject} no strong signs associated with Parkinson's.",
        "uncertain": f"{plural} sat too close to neutral to call either way.",
    }[facts["band"]]

    factors = "; ".join(f"{f['concept']} ({f['direction']})" for f in facts["top_factors"])
    if not factors:
        factors = "no single pattern stood out strongly"

    # Direction and strength lead, the percentage follows. The number alone is
    # unreadable -- "6%" cannot tell you which way it leans.
    lean_line = f"Overall {facts['lean']}, and it is {facts['evidence_strength']}."
    if facts["band"] == "uncertain":
        conf_line = (
            f"{lean_line} That is too close to call, so this is flagged for "
            "follow-up rather than given a result either way."
        )
    else:
        conf_line = (
            f"{lean_line} Confidence in this reading is "
            f"{facts['confidence_level']} ({facts['confidence_pct']}%)."
        )

    takes_line = ""
    if "takes" in facts:
        t = facts["takes"]
        takes_line = f" This combines {t['used']} recordings"
        if t["excluded"]:
            kinds = t.get("excluded_kinds", ["outlier"])
            if kinds == ["insufficient_voicing"]:
                why = "for not holding the vowel throughout"
            elif kinds in (["outlier"], ["input_outlier"], ["input_outlier", "outlier"]):
                why = "as inconsistent with the others"
            else:
                why = "for containing too much silence or disagreeing with the others"
            takes_line += f", with {t['excluded']} excluded {why}"
        takes_line += "."

    caution = (
        " A 'no strong signs' result does not rule Parkinson's out."
        if facts["band"] == "likely_healthy"
        else ""
    )
    if facts.get("evidence_vs_band") == "contradicts":
        caution += (
            " Note that the patterns measured in these recordings actually point the "
            "other way; the result above is driven by the model's baseline rather "
            "than by anything distinctive in this voice, so it should be treated as "
            "weak evidence."
        )
    if facts.get("recordings_inconsistent"):
        caution += (
            " The recordings disagreed with each other more than expected, which is "
            "why confidence is reduced; retrying in a quieter room, holding a steady "
            "pitch and volume, would give a firmer result."
        )
    return (
        f"{head} {conf_line}{takes_line} The main contributing patterns were: "
        f"{factors}.{caution} This is a screening signal, not a diagnosis; "
        "please consult a clinician."
    )


def phrase_explanation(
    facts: dict[str, Any], llm_fn: Callable[[str], str] | None = None
) -> str:
    """Phrase the facts. `llm_fn` is the swap-in seam for a real model.

    A failing LLM must never take the result down with it -- the deterministic
    template is a complete answer on its own, so any error falls back to it.
    """
    if llm_fn is None:
        return render_template(facts)
    try:
        return llm_fn(build_prompt(facts))
    except Exception as error:  # noqa: BLE001 - fallback is the point
        print(f"LLM call failed ({error}); using template fallback.")
        return render_template(facts)
