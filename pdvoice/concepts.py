"""Feature -> human-readable concept mapping for the explanation layer.

Ported from `colab_reference.py` (the `concept_of` / `CONCEPT_LABEL` pair), with
three corrections that the move to the 107-feature model forced. The original
rules were written against 753 features where each issue affected a handful of
columns and was invisible; at 107 they distort the explanation.

1. `wavelet_complexity` existed only to hold TQWT features. The v2 model has
   none, so the concept is dropped rather than left as a bucket that can never
   contribute.
2. `b1`..`b4` (formant bandwidths) fell through to `other_spectral` because the
   rule tested for the substring "bandwidth" while the columns are named `b1`.
   They are vocal-tract measurements and now sit with `f1`..`f4`.
3. `gender` fell into `other_spectral`, i.e. it would have been described to a
   user as "other spectral detail". It is a demographic covariate, not a
   property of the voice.

With those fixed, `other_spectral` is empty on the v2 feature set: every feature
has a real home and nothing lands in a catch-all.

The hard rule from PROJECT_CONTEXT still holds: the LLM narrating a result sees
concept labels only, never raw feature names.
"""

from __future__ import annotations

import re

#: Concepts eligible to be narrated to a user, with their plain-language labels.
CONCEPT_LABEL: dict[str, str] = {
    "temporal_fluctuation": "moment-to-moment energy and spectral fluctuation",
    "spectral_shape": "overall tone and timbre of the voice",
    "vocal_tract": "vocal-tract resonance (formants and their bandwidths)",
    "voice_regularity": "steadiness of pitch and loudness (standard clinical markers)",
    "other_spectral": "other spectral detail",
}

#: Concepts that exist for bookkeeping but must never be narrated.
#: `gender` genuinely informs the model (PD prevalence differs by sex), but
#: "your sex pushed this score toward Parkinson's" is unactionable for a user
#: and inappropriate for a screening tool, so it is aggregated and suppressed.
SILENT_CONCEPT_LABEL: dict[str, str] = {
    "demographic": "reported demographic information",
}

ALL_CONCEPT_LABEL: dict[str, str] = {**CONCEPT_LABEL, **SILENT_CONCEPT_LABEL}

#: Concepts in a stable order, for indexing SHAP columns.
CONCEPTS: list[str] = list(ALL_CONCEPT_LABEL.keys())

#: Concepts safe to surface in a user-facing explanation.
NARRATABLE_CONCEPTS: list[str] = list(CONCEPT_LABEL.keys())


def concept_of(column: str) -> str:
    """Bucket one feature column into a concept.

    Order matters: the delta test must precede the MFCC test, because a column
    like `mean_3rd_delta` is a rate-of-change feature rather than a static
    spectral one, and both substrings can appear in the same name.
    """
    cl = column.lower()
    if cl == "gender":
        return "demographic"
    if any(
        k in cl
        for k in ["jitter", "shimmer", "harm", "rpde", "dfa", "ppe", "pulse", "period"]
    ):
        return "voice_regularity"
    if "tqwt" in cl:
        # Retained so the mapping stays correct if the 753-feature model is ever
        # re-examined; produces nothing on the v2 feature set.
        return "wavelet_complexity"
    if "delta" in cl or "log_energy" in cl:
        return "temporal_fluctuation"
    if "mfcc" in cl:
        return "spectral_shape"
    if re.fullmatch(r"[fb]\d+", cl) or any(
        k in cl
        for k in ["formant", "bandwidth", "intensity", "glottal", "gqe", "vfer", "vocal"]
    ):
        return "vocal_tract"
    return "other_spectral"


def build_feature_concept(feature_cols: list[str]) -> dict[str, str]:
    """Map every model input to its concept."""
    return {c: concept_of(c) for c in feature_cols}


def concept_index(feature_cols: list[str]) -> dict[str, list[int]]:
    """Column positions per concept, for summing SHAP values within a concept.

    Only concepts that actually claim at least one feature are returned, so the
    explanation layer cannot reference an empty bucket.
    """
    mapping = build_feature_concept(feature_cols)
    index: dict[str, list[int]] = {}
    for position, column in enumerate(feature_cols):
        index.setdefault(mapping[column], []).append(position)
    return index
