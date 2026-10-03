"""Single source of truth for what the detector's 753 features are.

The training CSV ships a two-row header: row 0 carries sparse family labels
("Baseline Features", "MFCC", "TQWT Features", ...) that apply until the next
non-empty cell, row 1 carries the actual column names. That grouping is the
dataset authors' own, so we read families from the file rather than inferring
them from name prefixes.

Nothing here computes features; it only describes them. `pdvoice.extract` is
checked against this module so a rename or reordering fails loudly instead of
silently feeding the model a shuffled vector.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_CSV = (
    REPO_ROOT
    / "data"
    / "parkinsons-disease-classification-dataset"
    / "pd_speech_features.csv"
)

#: Columns present in the CSV that are not model inputs.
NON_FEATURE_COLS = ("id", "class")

#: Family label -> our implementation status. Drives the inventory report.
FAMILY_STATUS = OrderedDict(
    [
        ("Baseline Features", "partial (18/21 via Praat; PPE/DFA/RPDE not implemented)"),
        ("Intensity Parameters", "implemented (Praat)"),
        ("Formant Frequencies", "implemented (Praat)"),
        ("Bandwidth Parameters", "implemented (Praat)"),
        ("Vocal Fold", "NOT IMPLEMENTED"),
        ("MFCC", "implemented (librosa)"),
        ("Wavelet Features", "NOT IMPLEMENTED"),
        ("TQWT Features", "NOT IMPLEMENTED"),
    ]
)

#: Baseline columns that are Praat voice-report fields (the confident subset).
PRAAT_BASELINE_COLS = (
    "numPulses",
    "numPeriodsPulses",
    "meanPeriodPulses",
    "stdDevPeriodPulses",
    "locPctJitter",
    "locAbsJitter",
    "rapJitter",
    "ppq5Jitter",
    "ddpJitter",
    "locShimmer",
    "locDbShimmer",
    "apq3Shimmer",
    "apq5Shimmer",
    "apq11Shimmer",
    "ddaShimmer",
    "meanAutoCorrHarmonicity",
    "meanNoiseToHarmHarmonicity",
    "meanHarmToNoiseHarmonicity",
)

#: Baseline columns that are Little et al. nonlinear measures, not Praat.
NONLINEAR_BASELINE_COLS = ("PPE", "DFA", "RPDE")


def _read_header(csv_path: Path | str = TRAINING_CSV) -> tuple[list[str], list[str]]:
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        family_row = next(reader)
        name_row = next(reader)
    return family_row, name_row


def load_families(csv_path: Path | str = TRAINING_CSV) -> "OrderedDict[str, list[str]]":
    """Map family label -> ordered column names, using the CSV's own grouping.

    `id`, `gender` and `class` sit under an empty family label in the file;
    `gender` is a real model input, so it is reported under "Demographic" while
    `id`/`class` are dropped.
    """
    family_row, name_row = _read_header(csv_path)

    current = ""
    families: "OrderedDict[str, list[str]]" = OrderedDict()
    for raw_family, name in zip(family_row, name_row):
        if raw_family.strip():
            current = raw_family.strip()
        if name in NON_FEATURE_COLS:
            continue
        label = current if current else "Demographic"
        families.setdefault(label, []).append(name)
    return families


def load_feature_cols(csv_path: Path | str = TRAINING_CSV) -> list[str]:
    """The 753 model input columns, in the order the detector expects."""
    _, name_row = _read_header(csv_path)
    return [n for n in name_row if n not in NON_FEATURE_COLS]


def mfcc_feature_names() -> list[str]:
    """The 84 MFCC-family names, regenerated in CSV order.

    Layout is 42 means then 42 stds; within each, three derivative orders
    (static, delta, delta-delta) of 14 channels (log energy + MFCC 0..12).
    The dataset's naming is irregular across those blocks -- e.g. the static
    block spells out `mean_MFCC_3rd_coef`, the delta block `mean_3rd_delta`,
    and the delta-delta block `mean_3rd_delta_delta` except for the 0th, which
    is `mean_delta_delta_0th`. These names are reproduced literally so the
    extractor's output keys match the training columns exactly.
    """

    def ordinal(i: int) -> str:
        return {0: "0th", 1: "1st", 2: "2nd", 3: "3rd"}.get(i, f"{i}th")

    names: list[str] = []
    for stat in ("mean", "std"):
        # static
        names.append(f"{stat}_Log_energy")
        names.extend(f"{stat}_MFCC_{ordinal(i)}_coef" for i in range(13))
        # delta
        names.append(f"{stat}_delta_log_energy")
        names.extend(f"{stat}_{ordinal(i)}_delta" for i in range(13))
        # delta-delta
        names.append(f"{stat}_delta_delta_log_energy")
        names.append(f"{stat}_delta_delta_0th")
        names.extend(f"{stat}_{ordinal(i)}_delta_delta" for i in range(1, 13))
    return names


def implemented_feature_names() -> list[str]:
    """Every feature the extractor currently produces, in CSV order."""
    families = load_families()
    implemented = (
        ["gender"]
        + list(PRAAT_BASELINE_COLS)
        + families["Intensity Parameters"]
        + families["Formant Frequencies"]
        + families["Bandwidth Parameters"]
        + mfcc_feature_names()
    )
    order = {name: i for i, name in enumerate(load_feature_cols())}
    return sorted(implemented, key=lambda n: order[n])


def unimplemented_feature_names() -> list[str]:
    """Every feature still stubbed out, in CSV order."""
    done = set(implemented_feature_names())
    return [n for n in load_feature_cols() if n not in done]
