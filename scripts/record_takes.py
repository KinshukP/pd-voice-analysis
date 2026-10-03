"""Record sustained-vowel takes for validating the feature extractor.

Writes 44.1 kHz mono WAV files into recordings/, matching the training data's
format. Records several takes in one go, because seeing how much your own
numbers move between takes tells us the extractor's noise floor — without that
we can't tell a real mismatch from ordinary take-to-take variation.

    .venv/bin/python scripts/record_takes.py --takes 3

The first run will make macOS ask for microphone permission for your terminal.
If no prompt appears and the recording comes out silent, grant it under
System Settings > Privacy & Security > Microphone.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / "recordings"

SAMPLE_RATE = 44100  # matches the training data
CHANNELS = 1

# Peak amplitude below this is too quiet to analyse; at or above 0.99 the
# waveform is clipping, which manufactures distortion the model would read as
# voice irregularity.
QUIET_PEAK = 0.02
CLIP_PEAK = 0.99


def _countdown(seconds: int = 3) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"  starting in {remaining}...", end="\r", flush=True)
        time.sleep(1)
    print("  RECORDING - say 'ahhh' now      ", flush=True)


def _report_level(y: np.ndarray) -> list[str]:
    """Flag the level problems that would invalidate a take."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
    print(f"  peak {peak:.3f}   rms {rms:.4f}")

    warnings: list[str] = []
    if peak >= CLIP_PEAK:
        warnings.append("CLIPPING - too loud. Move back from the mic and redo this take.")
    elif peak < QUIET_PEAK:
        warnings.append("VERY QUIET - move closer to the mic and redo this take.")
    for w in warnings:
        print(f"  !! {w}")
    return warnings


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # 7 matches pipeline.RECOMMENDED_TAKES: run_inference_multi needs at least 5
    # to combine, and a couple of spares absorb an outlier or a failed take.
    ap.add_argument("--takes", type=int, default=7, help="number of takes (default 7)")
    ap.add_argument("--seconds", type=float, default=5.0, help="length of each take (default 5)")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--prefix", default="ahhh", help="output filename prefix")
    ap.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    args = ap.parse_args()

    if args.list_devices:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"[{i}] {d['name']}")
        return 0

    RECORDINGS_DIR.mkdir(exist_ok=True)
    print(f"Input device: {sd.query_devices(args.device, 'input')['name']}")
    print(f"Recording {args.takes} take(s) of {args.seconds:g}s at {SAMPLE_RATE} Hz mono.")
    print(
        "\nHow to do this well:\n"
        "  - quiet room, no fan or music\n"
        "  - a steady 'ahhh' at comfortable, CONSTANT loudness\n"
        "  - hold one steady pitch; do not let it rise or fall\n"
        "  - start the vowel immediately and hold it to the end\n"
    )

    written: list[Path] = []
    flagged: list[str] = []
    for take in range(1, args.takes + 1):
        input(f"[take {take}/{args.takes}] press Enter when ready...")
        _countdown()
        audio = sd.rec(
            int(args.seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=args.device,
        )
        sd.wait()
        y = audio[:, 0]

        dest = RECORDINGS_DIR / f"{args.prefix}_{take}.wav"
        sf.write(dest, y, SAMPLE_RATE, subtype="PCM_16")
        print(f"  saved {dest.relative_to(REPO_ROOT)}")
        if _report_level(y):
            flagged.append(dest.name)
        written.append(dest)
        print()

    print("=" * 70)
    print(f"Wrote {len(written)} file(s) to {RECORDINGS_DIR.relative_to(REPO_ROOT)}/")
    if flagged:
        print(f"Re-record these before validating: {', '.join(flagged)}")
    print("\nNext:")
    print("  .venv/bin/python scripts/validate_extractor.py \\")
    print(f"      {RECORDINGS_DIR.relative_to(REPO_ROOT)}/{args.prefix}_1.wav --gender <0 or 1>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
