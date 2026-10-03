"""Run the full inference pipeline on a recording and print the result.

    .venv/bin/python scripts/run_inference.py recordings/ahhh_1.wav --gender 1
    .venv/bin/python scripts/run_inference.py recordings/*.wav --gender 1 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdvoice.pipeline import (  # noqa: E402
    RECOMMENDED_TAKES,
    InferenceError,
    run_inference,
    run_inference_multi,
)

BANNER = {
    "likely_pd": "LIKELY PD  (acoustic signs associated with Parkinson's)",
    "likely_healthy": "NO STRONG SIGNS DETECTED",
    "uncertain": "UNCERTAIN  (flagged for follow-up)",
}


def show(result: dict) -> None:
    print("=" * 78)
    print(f"  {BANNER[result['result']]}")
    print("=" * 78)
    print(f"  recording          {result['meta']['wav']}")
    print(f"  calibrated P(PD)   {result['calibrated_probability']:.4f}")
    print(f"  raw P(PD)          {result['raw_probability']:.4f}")
    print(f"  stated confidence  {result['confidence_pct']}%")
    edges = result["meta"]["band_edges"]
    print(f"  band edges         healthy < {edges[0]}  |  uncertain  |  PD > {edges[1]}")

    q = result["input_quality"]
    status = "OK" if q["reliable"] else "UNRELIABLE"
    print(
        f"\n  input quality      {status} - {q['n_out_of_range']}/{q['n_features']} "
        f"features outside training range ({q['fraction_out_of_range']:.1%})"
    )
    if q["out_of_range_features"]:
        shown = ", ".join(q["out_of_range_features"][:6])
        more = f" (+{len(q['out_of_range_features']) - 6} more)" if len(q["out_of_range_features"]) > 6 else ""
        print(f"                     {shown}{more}")

    print("\n  concept breakdown  (+ pushes toward PD)")
    for c in result["concept_breakdown"]:
        tag = "" if c["narrated"] else "   [silent, not narrated]"
        print(f"    {c['contribution']:+.4f}  {c['label']}{tag}")

    print("\n  explanation")
    for line in result["explanation_text"].split("\n"):
        print(f"    {line}" if line else "")
    print()


def show_multi(result: dict) -> None:
    print("=" * 78)
    print(f"  {BANNER[result['result']]}")
    print("=" * 78)
    t, s = result["takes"], result["spread"]
    print(
        f"  confidence         {result['confidence_pct']}%  ({result['confidence_level']})"
    )
    d = result["confidence_detail"]
    print(
        f"                     evidence strength {d['evidence_strength_norm']:.2f} "
        f"x agreement factor {d['spread_factor']:.2f}"
    )
    print(
        f"\n  takes              {t['n_given']} given, {t['n_scored']} scored, "
        f"{t['n_used']} used, {t['n_excluded']} excluded, {t['n_failed']} failed"
    )
    print(f"                     (recommended: {t['recommended']})")

    print("\n  used")
    for u in t["used"]:
        print(f"    {u['calibrated_probability']:.4f}  {u['result']:<15} {Path(u['wav']).name}")
    if t["excluded"]:
        print("\n  EXCLUDED")
        for e in t["excluded"]:
            if e.get("kind") == "insufficient_voicing":
                print(
                    f"    {Path(e['wav']).name:<20} {e['unvoiced_fraction']:.0%} unvoiced"
                )
            elif e.get("kind") == "input_outlier":
                print(
                    f"    {Path(e['wav']).name:<20} feature deviation "
                    f"{e['feature_deviation']:.3f} (p={e['calibrated_probability']:.4f})"
                )
            else:
                print(
                    f"    {Path(e['wav']).name:<20} p={e['calibrated_probability']:.4f} "
                    f"deviation {e['deviation_from_median']:.4f}"
                )
            print(f"      reason: {e['reason']}")
    if t["failed"]:
        print("\n  FAILED to score")
        for f in t["failed"]:
            print(f"    {Path(f['wav']).name}: {f['reason'][:100]}")

    print(
        f"\n  spread (used)      sd {s['std']:.4f}  range {s['range']:.4f} "
        f"[{s['min']:.4f}, {s['max']:.4f}]"
    )
    if s["warning"]:
        print(f"    !! {s['warning']}")

    print("\n  concept breakdown  (median across used takes, + pushes toward PD)")
    for c in result["concept_breakdown"]:
        tag = "" if c["narrated"] else "   [silent]"
        print(f"    {c['contribution']:+.4f}  {c['label']}{tag}")

    internal = result["internal"]
    print(f"\n  internal           combined P(PD) {internal['calibrated_probability']:.4f} "
          f"by {internal['combined_by']}  -- NOT a disease probability")
    print(f"                     {internal['note']}")

    print("\n  explanation")
    for line in result["explanation_text"].split("\n"):
        print(f"    {line}" if line else "")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="+", help="recording(s) of a sustained vowel")
    ap.add_argument("--gender", type=int, choices=(0, 1), required=True, help="0 female, 1 male")
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead")
    ap.add_argument(
        "--single",
        action="store_true",
        help="score each file independently instead of combining them",
    )
    args = ap.parse_args()

    if not args.single and len(args.wav) > 1:
        try:
            result = run_inference_multi(args.wav, gender=args.gender)
        except InferenceError as error:
            print(f"\n!! {error}\n", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, default=str)) if args.json else show_multi(result)
        return 0

    failures = 0
    for path in args.wav:
        try:
            result = run_inference(path, gender=args.gender)
        except InferenceError as error:
            print(f"\n!! {path}: {error}\n", file=sys.stderr)
            failures += 1
            continue
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            show(result)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
