"""Build step 1, validation: are our extracted features in the training data's ballpark?

Runs `extract_features` on a recording and, for every feature we implement,
compares the value against that column's distribution in the training CSV:
mean, std, min/max, the percentile the value lands at, and a robust z-score.

The question being answered is not "is this recording healthy or PD" — it is
"is our extraction method even measuring the same thing theirs did". A feature
that lands outside the training range, or at the 0th/100th percentile, means
the model would be extrapolating on input it has never seen.

    .venv/bin/python scripts/validate_extractor.py RECORDING.wav --gender 0

If you have no recording yet, run with no arguments for guidance on getting one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdvoice import feature_spec  # noqa: E402
from pdvoice.extract import EXTRACTION_PARAMS, extract_features  # noqa: E402

# A feature is "off" if it falls outside the training range, and "extreme" if it
# sits in the outer 1% of the training distribution.
EXTREME_PCT = 1.0

#: Families whose absolute values depend on microphone gain and distance, not
#: just on the voice. A mismatch here does not indict the extraction method.
MIC_DEPENDENT_FAMILIES = {"Intensity Parameters"}

NO_RECORDING_HELP = """\
No test recording supplied.

1. RECORD SOME YOURSELF (fastest, and matches the intended product flow)

       .venv/bin/python scripts/record_takes.py --takes 3

   Writes 44.1 kHz mono WAVs into recordings/, matching the training format.
   Record 3 takes: the training set has 3 per speaker, and how much your own
   numbers move between takes tells us the extractor's noise floor. Without
   that we cannot tell a real mismatch from ordinary variation.

   Then:
       .venv/bin/python scripts/validate_extractor.py recordings/ahhh_*.wav --gender 0

2. PULL A PUBLIC PD VOICE RECORDING (better for checking PD-vs-HC behaviour)
   The Sakar et al. CSV ships no audio, so the exact recordings are
   unavailable. Sources that do ship .wav sustained vowels:
     - Italian Parkinson's Voice and Speech (IEEE DataPort) - sustained /a/,
       PD and controls, closest match to this protocol.
     - PC-GITA (Colombian Spanish, on request) - sustained vowels, PD + HC.
     - mPower (Sage Bionetworks, on request) - phone-recorded sustained /a/;
       phone audio is a different recording channel to the training data.

READ THIS BEFORE INTERPRETING THE OUTPUT: your own recording is almost
certainly a healthy control, made on different hardware in a different room
to the training data. A mismatch can therefore mean either "our extraction
differs from theirs" or "our microphone differs from theirs". The intensity
features are absolute dB and will track your mic gain, so they are excluded
from the verdict. Jitter, shimmer, HNR and formants describe the shape of the
sound rather than its level, and are the honest signal.
"""


def _load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(feature_spec.TRAINING_CSV, header=1)
    return df


def _compare(takes: list[dict[str, float]], train: pd.DataFrame) -> pd.DataFrame:
    """Compare features (averaged over takes) against the training distribution.

    With more than one take we also report how far the feature moved between
    takes, expressed as a fraction of the training column's own spread. A
    feature that swings as much between two recordings of the same person as it
    does across the whole training set is too unstable to carry signal,
    regardless of whether it lands in range.
    """
    rows = []
    families = feature_spec.load_families()
    family_of = {name: fam for fam, names in families.items() for name in names}

    for name in takes[0]:
        values = np.array([t[name] for t in takes], dtype=float)
        value = float(np.mean(values))

        col = train[name].astype(float).to_numpy()
        col = col[np.isfinite(col)]
        mean, std = float(np.mean(col)), float(np.std(col))
        lo, hi = float(np.min(col)), float(np.max(col))
        median = float(np.median(col))
        # Robust spread: MAD scaled to be comparable to a standard deviation.
        mad = float(np.median(np.abs(col - median))) * 1.4826

        spread = float(np.max(values) - np.min(values)) if values.size > 1 else float("nan")
        # Take-to-take movement as a fraction of the training range.
        instability = spread / (hi - lo) if values.size > 1 and hi > lo else float("nan")

        if np.isnan(value):
            pct, z, flag = float("nan"), float("nan"), "NaN"
        else:
            pct = float((col < value).mean() * 100.0)
            z = (value - median) / mad if mad > 0 else float("inf")
            if value < lo or value > hi:
                flag = "OUT OF RANGE"
            elif pct < EXTREME_PCT or pct > 100 - EXTREME_PCT:
                flag = "extreme"
            else:
                flag = "ok"

        row = {
            "family": family_of.get(name, "?"),
            "feature": name,
            "extracted": value,
            "take_spread": spread,
            "instability": instability,
            "train_mean": mean,
            "train_std": std,
            "train_min": lo,
            "train_max": hi,
            "pct": pct,
            "robust_z": z,
            "flag": flag,
        }
        for i, v in enumerate(values, start=1):
            row[f"take_{i}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    a = abs(x)
    if a != 0 and (a < 1e-3 or a >= 1e6):
        return f"{x:.3e}"
    return f"{x:.4f}" if a < 1000 else f"{x:.1f}"


def _print_table(df: pd.DataFrame, verbose: bool) -> None:
    header = (
        f"{'feature':<30}{'extracted':>14}{'train mean':>14}"
        f"{'train range':>28}{'pct':>8}{'z':>9}  flag"
    )
    for family in df["family"].unique():
        sub = df[df["family"] == family]
        shown = sub if verbose else sub[sub["flag"] != "ok"]
        n_ok = int((sub["flag"] == "ok").sum())
        print(f"\n### {family} — {n_ok}/{len(sub)} in distribution")
        if shown.empty:
            print("    all features land inside the training distribution")
            continue
        if not verbose:
            print(f"    (showing {len(shown)} flagged of {len(sub)}; --verbose for all)")
        print("    " + header)
        for _, r in shown.iterrows():
            rng = f"[{_fmt(r.train_min)}, {_fmt(r.train_max)}]"
            print(
                f"    {r.feature:<30}{_fmt(r.extracted):>14}{_fmt(r.train_mean):>14}"
                f"{rng:>28}{_fmt(r.pct):>8}{_fmt(r.robust_z):>9}  {r.flag}"
            )


def _print_verdict(df: pd.DataFrame, n_takes: int) -> None:
    print("\n" + "=" * 100)
    print("VERDICT BY FAMILY")
    print("=" * 100)
    header = f"{'family':<26}{'ok':>6}{'extreme':>9}{'out of range':>14}{'NaN':>6}{'median |z|':>12}"
    if n_takes > 1:
        header += f"{'take drift':>12}"
    print(header)

    for family in df["family"].unique():
        sub = df[df["family"] == family]
        med_z = np.nanmedian(np.abs(sub["robust_z"].to_numpy(dtype=float)))
        line = (
            f"{family:<26}{int((sub.flag=='ok').sum()):>6}"
            f"{int((sub.flag=='extreme').sum()):>9}"
            f"{int((sub.flag=='OUT OF RANGE').sum()):>14}"
            f"{int((sub.flag=='NaN').sum()):>6}{_fmt(float(med_z)):>12}"
        )
        if n_takes > 1:
            drift = np.nanmedian(sub["instability"].to_numpy(dtype=float))
            line += f"{_fmt(float(drift)):>12}"
        if family in MIC_DEPENDENT_FAMILIES:
            line += "   (mic-dependent, excluded)"
        print(line)

    judged = df[~df["family"].isin(MIC_DEPENDENT_FAMILIES)]
    print("-" * 100)
    print(
        f"{'JUDGED (mic-dependent excluded)':<26}{int((judged.flag=='ok').sum()):>6}"
        f"{int((judged.flag!='ok').sum()):>9} not-ok of {len(judged)}"
    )

    print("\nHow to read this:")
    print("  'ok'           the value sits inside the range the training data covers.")
    print("  'extreme'      inside the range, but in its outermost 1% - plausible for")
    print("                 one speaker, suspicious if a whole family does it.")
    print("  'OUT OF RANGE' outside anything the training data contains. The model has")
    print("                 never seen input like this, so its output would be a guess.")
    print("  median |z|     typical distance from the training median, in units of the")
    print("                 training spread. Under ~2 is ordinary speaker variation.")
    if n_takes > 1:
        print("  take drift     how much a feature moved between your own takes, as a")
        print("                 fraction of the training range. Above ~0.2 means the")
        print("                 feature is too unstable to carry signal.")
    print("\n  A family mostly 'ok' is reproducing the original method closely enough to")
    print("  trust. A family mostly OUT OF RANGE is measuring something different in")
    print("  kind - a unit, scaling or formula difference - not just a different voice.")


def _print_assumptions() -> None:
    print("\n" + "=" * 100)
    print("PARAMETERS ASSUMED BY THIS EXTRACTOR")
    print("=" * 100)
    for key, spec in EXTRACTION_PARAMS.items():
        marker = {"confident": "  ", "assumed": " ?", "UNVERIFIED": " !"}.get(
            spec["confidence"], "  "
        )
        print(f"{marker} {key:<26} = {spec['value']}")
        print(f"     [{spec['confidence']}] {spec['note']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="*", help="one or more recordings of a sustained vowel")
    ap.add_argument(
        "--gender", type=int, choices=(0, 1), help="dataset gender code for the speaker"
    )
    ap.add_argument("--verbose", action="store_true", help="show every feature, not just flagged")
    ap.add_argument("--csv-out", help="write the full comparison table here")
    args = ap.parse_args()

    if not args.wav:
        print(NO_RECORDING_HELP)
        return 2
    if args.gender is None:
        ap.error("--gender is required (it is a model input and sets the formant ceiling)")

    takes = []
    for path in args.wav:
        print(f"Extracting from {path} ...")
        takes.append(extract_features(path, gender=args.gender))
    print(
        f"Got {len(takes[0])} of 753 features from {len(takes)} take(s) "
        f"({len(feature_spec.unimplemented_feature_names())} still stubbed)."
    )

    train = _load_training_frame()
    df = _compare(takes, train)

    print("\n" + "=" * 100)
    print(f"EXTRACTED (mean of {len(takes)} take(s)) vs TRAINING DISTRIBUTION")
    print("=" * 100)
    _print_table(df, args.verbose)
    _print_verdict(df, len(takes))
    _print_assumptions()

    if args.csv_out:
        df.to_csv(args.csv_out, index=False)
        print(f"\n[written] {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
