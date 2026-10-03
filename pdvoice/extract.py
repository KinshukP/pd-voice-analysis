"""Build step 1: turn a .wav recording into the detector's input features.

Scope note: this module deliberately implements only the blocks we are
confident about — the Praat voice-report baseline, intensity, formants and
bandwidths, and the MFCC family (114 of 753 features). The Vocal Fold, Wavelet
and TQWT blocks raise NotImplementedError rather than returning a guess. TQWT
alone carries ~50% of the model's importance, so a plausible-looking wrong
value there is worse than an obvious absence.

Every parameter that the source paper did not pin down is collected in
`EXTRACTION_PARAMS` and marked with its confidence, so the validation script
can report what was assumed. Where the paper is silent we use Praat's own
defaults, on the theory that the authors most likely did too.

Reference: Sakar et al. (2019), "A comparative analysis of speech signal
processing algorithms for Parkinson's disease classification and the use of the
tunable Q-factor wavelet transform", Applied Soft Computing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import parselmouth
import scipy.fftpack
from parselmouth.praat import call

from pdvoice import feature_spec

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
# "confident" = the dataset's column names or Praat's defaults pin this down.
# "assumed"   = the paper does not say; this is a documented guess to test.
EXTRACTION_PARAMS: dict[str, dict[str, Any]] = {
    "f0_min_hz": {"value": 75.0, "confidence": "confident", "note": "Praat default floor"},
    "f0_max_hz": {"value": 600.0, "confidence": "confident", "note": "Praat default ceiling"},
    "max_formant_hz_male": {
        "value": 5000.0,
        "confidence": "assumed",
        "note": "Praat's recommended value for adult male tracts",
    },
    "max_formant_hz_female": {
        "value": 5500.0,
        "confidence": "assumed",
        "note": "Praat's recommended value for adult female tracts",
    },
    "mfcc_n_coefficients": {
        "value": 13,
        "confidence": "confident",
        "note": "column names run MFCC_0th..MFCC_12th",
    },
    "mfcc_window_s": {"value": 0.025, "confidence": "assumed", "note": "standard 25 ms frame"},
    "mfcc_hop_s": {"value": 0.010, "confidence": "assumed", "note": "standard 10 ms hop"},
    "mfcc_n_mels": {
        "value": 26,
        "confidence": "derived",
        "note": (
            "Fitted against the training distribution, then re-confirmed after the "
            "delta edge-frame fix: 20/22/24/26/28 all put 66/66 level-independent "
            "MFCC features in range, so the exact count is not identifiable. 26 is "
            "kept as the classic HTK filterbank size; the tied alternatives differ "
            "by less median z than one speaker's takes vary, so switching would be "
            "fitting the extractor to whoever was last recorded. librosa's default "
            "of 128 is far worse - it is a spectrogram default, not an MFCC one."
        ),
    },
    "mfcc_log_base": {
        "value": "log10 of mel power",
        "confidence": "derived",
        "note": (
            "Scanned against the training columns. log10(power) beats natural log "
            "of power decisively (62/66 vs 48/66). log(amplitude) is within noise "
            "of log10(power) - they differ only by a constant 0.869 factor - so "
            "the base is pinned much better than the exact amplitude/power choice."
        ),
    },
    "mfcc_dct": {
        "value": "DCT-II, norm='ortho', no cepstral liftering",
        "confidence": "derived",
        "note": "Orthonormal beats unnormalised; lifter=22 (HTK style) made it worse.",
    },
    "mfcc_delta_width": {
        "value": 9,
        "confidence": "derived",
        "note": "Scanned 3/5/7/9; 9 (librosa's default) wins clearly.",
    },
    "mfcc_edge_frames_discarded": {
        "value": "width // 2 + 1 (= 5 per end)",
        "confidence": "derived",
        "note": (
            "librosa's delta emits fabricated constant values across the edge "
            "window (savgol mode='interp'). Measured directly: the first 5 and "
            "last 5 frames were each a single repeated constant ~10x interior "
            "magnitude. Discarded before computing the mean/std statistics."
        ),
    },
    "resample": {
        "value": "none (native rate preserved)",
        "confidence": "confident",
        "note": "training data was recorded at 44.1 kHz",
    },
    "analysis_window_s": {
        "value": 1.93,
        "confidence": "derived",
        "note": (
            "Not stated in the paper, but recoverable from the CSV: "
            "numPeriodsPulses * meanPeriodPulses gives the voiced duration each row "
            "was computed over, and 92% of rows land in 1.90-1.93 s with a hard "
            "ceiling at 1.9297 s. The authors clearly analysed a fixed window. "
            "Extracting over a longer recording would inflate numPulses and shift "
            "every frame-count-sensitive statistic."
        ),
    },
    "analysis_window_anchor": {
        "value": "centre",
        "confidence": "assumed",
        "note": (
            "The CSV fixes the window length but not where it was taken from. "
            "Centre-trimming avoids onset/offset instability in a sustained vowel."
        ),
    },
}

F0_MIN = EXTRACTION_PARAMS["f0_min_hz"]["value"]
F0_MAX = EXTRACTION_PARAMS["f0_max_hz"]["value"]

#: Praat voice-report label -> (our column name, scale factor).
#: Praat prints jitter/shimmer as percentages, but the training CSV stores them
#: as fractions (locPctJitter averages 0.0023, not 0.23), so those are /100.
#: Absolute jitter (seconds) and shimmer in dB are already in the CSV's units.
_VOICE_REPORT_MAP: dict[str, tuple[str, float]] = {
    "Number of pulses": ("numPulses", 1.0),
    "Number of periods": ("numPeriodsPulses", 1.0),
    "Mean period": ("meanPeriodPulses", 1.0),
    "Standard deviation of period": ("stdDevPeriodPulses", 1.0),
    "Jitter (local)": ("locPctJitter", 0.01),
    "Jitter (local, absolute)": ("locAbsJitter", 1.0),
    "Jitter (rap)": ("rapJitter", 0.01),
    "Jitter (ppq5)": ("ppq5Jitter", 0.01),
    "Jitter (ddp)": ("ddpJitter", 0.01),
    "Shimmer (local)": ("locShimmer", 0.01),
    "Shimmer (local, dB)": ("locDbShimmer", 1.0),
    "Shimmer (apq3)": ("apq3Shimmer", 0.01),
    "Shimmer (apq5)": ("apq5Shimmer", 0.01),
    "Shimmer (apq11)": ("apq11Shimmer", 0.01),
    "Shimmer (dda)": ("ddaShimmer", 0.01),
    "Mean autocorrelation": ("meanAutoCorrHarmonicity", 1.0),
    "Mean noise-to-harmonics ratio": ("meanNoiseToHarmHarmonicity", 1.0),
    "Mean harmonics-to-noise ratio": ("meanHarmToNoiseHarmonicity", 1.0),
}

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")


#: Maximum share of the analysis window that may be unvoiced before a recording
#: is refused. Set from measurement, not taste: seven clean sustained-vowel takes
#: measured 0.0% unvoiced, one take with a gap measured 21.9%, and recordings
#: with a real pause measured 52-59%. Anything in 5-20% separates those cleanly;
#: 10% keeps margin on both sides.
#:
#: This matters because silence corrupts the MFCC block badly -- the energy cliff
#: at a silence boundary inflates the delta and delta-delta statistics by orders
#: of magnitude (std_delta_delta_log_energy moved 0.0104 -> 11.7 on a recording
#: with a one-second pause) while Praat's voiced-only measures barely move. The
#: model would still return a confident-looking score from those corrupted values.
MAX_UNVOICED_FRACTION = 0.10


class FeatureExtractionError(RuntimeError):
    """Raised when a recording cannot be analysed (too short, silent, unvoiced)."""


class InsufficientVoicingError(FeatureExtractionError):
    """Raised when too much of the analysis window carries no voice.

    A subclass of FeatureExtractionError so existing handlers keep working, but
    distinguishable so callers can report it as its own, actionable reason
    ("hold the vowel for the whole recording") rather than a generic failure.
    """

    def __init__(self, message: str, unvoiced_fraction: float) -> None:
        super().__init__(message)
        self.unvoiced_fraction = unvoiced_fraction


# --------------------------------------------------------------------------
# Praat blocks
# --------------------------------------------------------------------------
def _parse_report_number(raw: str) -> float:
    """Pull a float out of a voice-report value like '7.142868E-3 seconds'."""
    if "--undefined--" in raw:
        return float("nan")
    match = _NUMBER_RE.search(raw.replace("%", " "))
    if match is None:
        return float("nan")
    return float(match.group(0))


def measure_voicing(
    wav_path: str | Path,
    window_s: float | None = None,
) -> dict[str, Any]:
    """How much of the analysis window actually carries voice.

    Uses Praat's own pitch tracker, which is the same machinery that decides
    which frames the jitter/shimmer/HNR features are computed over, so the
    answer is consistent with what extraction will do rather than a separate
    energy heuristic. (An RMS threshold disagrees noticeably: it called these
    same recordings 2-4% silent where Praat calls them 0% unvoiced.)

    Cheap enough to run as a pre-check: one pitch pass, no feature computation.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)
    if window_s is None:
        window_s = EXTRACTION_PARAMS["analysis_window_s"]["value"]

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if y.size == 0:
        raise FeatureExtractionError(f"{wav_path} contains no audio.")
    y = _centre_trim(y, sr, window_s)
    sound = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)

    try:
        pitch = call(
            sound, "To Pitch (cc)", 0, F0_MIN, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, F0_MAX
        )
        total = int(call(pitch, "Get number of frames"))
        voiced = int(call(pitch, "Count voiced frames"))
    except parselmouth.PraatError as error:
        raise FeatureExtractionError(
            f"Praat could not analyse {wav_path.name}: {error}"
        ) from error

    unvoiced_fraction = 1.0 - (voiced / total) if total else 1.0
    return {
        "n_frames": total,
        "n_voiced": voiced,
        "unvoiced_fraction": round(float(unvoiced_fraction), 4),
        "passes": bool(unvoiced_fraction <= MAX_UNVOICED_FRACTION),
        "threshold": MAX_UNVOICED_FRACTION,
    }


def require_voicing(wav_path: str | Path, window_s: float | None = None) -> dict[str, Any]:
    """`measure_voicing`, but raises if the recording is too quiet to score."""
    voicing = measure_voicing(wav_path, window_s=window_s)
    if not voicing["passes"]:
        raise InsufficientVoicingError(
            f"{voicing['unvoiced_fraction']:.0%} of the analysed audio carries no voice "
            f"(limit {MAX_UNVOICED_FRACTION:.0%}). Hold a steady 'ahhh' for the whole "
            "recording, without pausing for breath or trailing off at the end.",
            unvoiced_fraction=voicing["unvoiced_fraction"],
        )
    return voicing


def extract_baseline_praat(sound: parselmouth.Sound) -> dict[str, float]:
    """The 18 baseline features Praat's voice report reports directly.

    The dataset's column names line up one-for-one with the voice report's
    fields, which is strong evidence the authors used exactly this report, so
    we parse the report rather than issuing the queries individually.
    """
    pitch = call(
        sound, "To Pitch (cc)", 0, F0_MIN, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, F0_MAX
    )
    point_process = call(sound, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
    report = call(
        [sound, pitch, point_process],
        "Voice report",
        0,
        0,
        F0_MIN,
        F0_MAX,
        1.3,
        1.6,
        0.03,
        0.45,
    )

    parsed: dict[str, float] = {}
    for line in report.splitlines():
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        entry = _VOICE_REPORT_MAP.get(label.strip())
        if entry is None:
            continue
        column, scale = entry
        parsed[column] = _parse_report_number(value) * scale

    missing = [c for c, _ in _VOICE_REPORT_MAP.values() if c not in parsed]
    if missing:
        raise FeatureExtractionError(
            f"Praat voice report did not yield {missing}. The recording may be "
            "too short, silent, or entirely unvoiced."
        )
    return parsed


def extract_intensity(sound: parselmouth.Sound) -> dict[str, float]:
    """min/max/mean intensity in dB.

    Caveat for validation: these are absolute SPL-style values that scale with
    microphone gain and distance. Unless a recording is made at a comparable
    level to the training set, these three will drift no matter how correct the
    computation is.
    """
    intensity = call(sound, "To Intensity", F0_MIN, 0, "yes")
    return {
        "minIntensity": call(intensity, "Get minimum", 0, 0, "Parabolic"),
        "maxIntensity": call(intensity, "Get maximum", 0, 0, "Parabolic"),
        "meanIntensity": call(intensity, "Get mean", 0, 0, "energy"),
    }


def extract_formants(sound: parselmouth.Sound, gender: int) -> dict[str, float]:
    """First four formants and their bandwidths, averaged over defined frames.

    The paper does not state how it collapsed the per-frame formant track to a
    single number; we take the mean over frames where the formant is defined.
    The ceiling is chosen by gender per Praat's standard advice.
    """
    key = "max_formant_hz_female" if gender == _FEMALE else "max_formant_hz_male"
    max_formant = EXTRACTION_PARAMS[key]["value"]

    formant = call(sound, "To Formant (burg)", 0.0, 5, max_formant, 0.025, 50.0)
    n_frames = call(formant, "Get number of frames")

    out: dict[str, float] = {}
    for number in (1, 2, 3, 4):
        freqs: list[float] = []
        bandwidths: list[float] = []
        for frame in range(1, int(n_frames) + 1):
            t = call(formant, "Get time from frame number", frame)
            freq = call(formant, "Get value at time", number, t, "hertz", "Linear")
            bandwidth = call(formant, "Get bandwidth at time", number, t, "hertz", "Linear")
            if not np.isnan(freq):
                freqs.append(freq)
            if not np.isnan(bandwidth):
                bandwidths.append(bandwidth)
        out[f"f{number}"] = float(np.mean(freqs)) if freqs else float("nan")
        out[f"b{number}"] = float(np.mean(bandwidths)) if bandwidths else float("nan")
    return out


# --------------------------------------------------------------------------
# MFCC block
# --------------------------------------------------------------------------
def extract_mfcc(y: np.ndarray, sr: int) -> dict[str, float]:
    """The 84 MFCC-family features.

    The column names imply the structure: 13 cepstral coefficients plus frame
    log energy (14 channels), each with its delta and delta-delta, summarised
    by mean and std over frames -> 2 * 3 * 14 = 84.
    """
    win_length = int(round(EXTRACTION_PARAMS["mfcc_window_s"]["value"] * sr))
    hop_length = int(round(EXTRACTION_PARAMS["mfcc_hop_s"]["value"] * sr))
    n_fft = int(2 ** np.ceil(np.log2(win_length)))

    if y.size < win_length:
        raise FeatureExtractionError(
            f"Recording is shorter than one {win_length}-sample analysis window."
        )

    # Built explicitly rather than via librosa.feature.mfcc: that helper defaults
    # to a dB-scaled (10*log10) mel spectrogram and a 128-band filterbank, and
    # both choices put these features far outside the training distribution.
    # log10 of mel power with a 26-band filterbank reproduces it -- see
    # mfcc_n_mels / mfcc_log_base in EXTRACTION_PARAMS.
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=EXTRACTION_PARAMS["mfcc_n_mels"]["value"],
    )
    log_mel = np.log10(np.maximum(mel, 1e-30))
    mfcc = scipy.fftpack.dct(log_mel, axis=0, type=2, norm="ortho")[
        : EXTRACTION_PARAMS["mfcc_n_coefficients"]["value"]
    ]

    frames = librosa.util.frame(
        np.pad(y, n_fft // 2, mode="reflect"), frame_length=n_fft, hop_length=hop_length
    )
    energy = np.sum(frames.astype(np.float64) ** 2, axis=0)
    log_energy = np.log(np.maximum(energy, np.finfo(np.float64).tiny))
    log_energy = log_energy[: mfcc.shape[1]]

    # Channel order matches the column order: log energy first, then c0..c12.
    static = np.vstack([log_energy[np.newaxis, :], mfcc])
    if static.shape[1] < 9:
        raise FeatureExtractionError(
            "Recording yields too few frames for delta features (need >= 9)."
        )
    width = EXTRACTION_PARAMS["mfcc_delta_width"]["value"]
    delta = librosa.feature.delta(static, order=1, width=width)
    delta2 = librosa.feature.delta(static, order=2, width=width)

    # Discard the frames the derivative filter could not honestly compute.
    #
    # librosa's delta uses scipy's savgol_filter with mode='interp', which fits a
    # polynomial across the edge window and emits its derivative there. Those
    # values are fabricated, identical across the whole edge block, and roughly
    # 10x the magnitude of real interior values -- on a 194-frame recording the
    # 10 edge frames alone shifted mean_1st_delta_delta to -0.0099, outside the
    # training range of [-0.0042, +0.0040], when the interior mean was +0.0001
    # (training median +0.00003). Six features were out of range purely from this.
    #
    # Trimmed from the static block too, so all 84 statistics describe the same
    # stretch of audio rather than three slightly different ones.
    edge = width // 2 + 1
    if static.shape[1] <= 2 * edge + 1:
        raise FeatureExtractionError(
            f"Recording yields only {static.shape[1]} frames; need more than "
            f"{2 * edge + 1} once derivative edge frames are discarded."
        )
    static, delta, delta2 = (b[:, edge:-edge] for b in (static, delta, delta2))

    values = np.concatenate(
        [np.mean(block, axis=1) for block in (static, delta, delta2)]
        + [np.std(block, axis=1) for block in (static, delta, delta2)]
    )
    names = feature_spec.mfcc_feature_names()
    assert len(names) == values.size == 84, (len(names), values.size)
    return {name: float(value) for name, value in zip(names, values)}


# --------------------------------------------------------------------------
# Not yet reproduced
# --------------------------------------------------------------------------
_STUB_MESSAGE = (
    "{family} ({count} features) is not implemented. {detail}\n"
    "Deliberately left unimplemented rather than approximated: feeding the "
    "detector a guessed value here would produce a confident-looking score "
    "from out-of-distribution input."
)


def extract_nonlinear_baseline(sound: parselmouth.Sound) -> dict[str, float]:
    """PPE, DFA and RPDE — the Little et al. nonlinear measures. NOT IMPLEMENTED.

    These three sit in the Baseline family but are not Praat voice-report
    fields; they come from a separate line of work (pitch period entropy,
    detrended fluctuation analysis, recurrence period density entropy). DFA and
    PPE are tractable, RPDE less so, and all three have parameter choices the
    paper does not state.
    """
    raise NotImplementedError(
        _STUB_MESSAGE.format(
            family="Nonlinear baseline (PPE/DFA/RPDE)",
            count=3,
            detail=(
                "Little et al. measures, not Praat. Implementable, but the "
                "embedding/scaling parameters are unstated."
            ),
        )
    )


def extract_vocal_fold(sound: parselmouth.Sound) -> dict[str, float]:
    """GQ / GNE / VFER / IMF features. NOT IMPLEMENTED.

    Glottal quotient, glottis-to-noise excitation, vocal fold excitation ratio
    and empirical-mode-decomposition features. No mainstream Python
    implementation matches the paper's.
    """
    raise NotImplementedError(
        _STUB_MESSAGE.format(
            family="Vocal Fold",
            count=22,
            detail="GQ/GNE/VFER/IMF have no standard Python implementation.",
        )
    )


def extract_wavelet(y: np.ndarray, sr: int) -> dict[str, float]:
    """Discrete wavelet features on the F0 contour. NOT IMPLEMENTED.

    10-level decomposition (Ea, Ed_1..10 and entropy/TKEO statistics). The
    mother wavelet and the signal it is applied to are not stated in the paper.
    """
    raise NotImplementedError(
        _STUB_MESSAGE.format(
            family="Wavelet Features",
            count=182,
            detail="Mother wavelet and input signal (F0 contour vs. waveform) unstated.",
        )
    )


def extract_tqwt(y: np.ndarray, sr: int) -> dict[str, float]:
    """Tunable Q-factor wavelet transform features. NOT IMPLEMENTED.

    36 subbands x 12 statistics. This was the source paper's novel
    contribution, implemented with a MATLAB toolbox; Q, redundancy and level
    count are not recoverable from the CSV alone. ~50% of the detector's
    importance lives in this block, which is exactly why it must not be guessed.
    """
    raise NotImplementedError(
        _STUB_MESSAGE.format(
            family="TQWT Features",
            count=432,
            detail="Q / redundancy / level parameters unknown; MATLAB toolbox original.",
        )
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
# The dataset does not document its gender coding, but it is recoverable: the
# speaker counts per (gender, class) cell match Sakar et al.'s published cohort
# exactly -- gender=1 gives 23 HC + 107 PD = 130 (the paper's men), gender=0
# gives 41 HC + 81 PD = 122 (the paper's women). All four cells agree, so this
# is settled, not inferred. See docs/extraction_findings.md.
_FEMALE, _MALE = 0, 1


def _centre_trim(y: np.ndarray, sr: int, window_s: float | None) -> np.ndarray:
    """Take the middle `window_s` seconds, to match the training window length.

    Shorter recordings are returned untouched — padding them would fabricate
    silence and depress the very statistics we are trying to measure.
    """
    if window_s is None:
        return y
    want = int(round(window_s * sr))
    if y.size <= want:
        return y
    start = (y.size - want) // 2
    return y[start : start + want]


def extract_features(
    wav_path: str | Path,
    gender: int | None = None,
    strict: bool = False,
    window_s: float | None = EXTRACTION_PARAMS["analysis_window_s"]["value"],
) -> dict[str, float]:
    """Extract detector features from a recording of a sustained vowel.

    Args:
        wav_path: path to the recording. Read at its native sample rate; the
            training data was captured at 44.1 kHz.
        gender: the dataset's `gender` code — 0 for female, 1 for male (see the
            note on `_FEMALE`/`_MALE` above for how that was established). It is
            both a model input and the selector for the formant ceiling.
        strict: if True, raise NotImplementedError for the blocks we cannot yet
            reproduce. Default False returns only the implemented features, so
            the validation script can measure what exists today.
        window_s: analyse only the middle N seconds, matching the fixed window
            the training rows were computed over (see `analysis_window_s` in
            EXTRACTION_PARAMS). Pass None to analyse the whole recording.

    Returns:
        Mapping of feature column name -> value. With strict=False this holds
        114 of the model's 753 inputs, so it is NOT yet enough to run the
        detector.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)
    if gender not in (_MALE, _FEMALE):
        raise ValueError(f"gender must be 0 or 1 to match the dataset column, got {gender!r}")

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if y.size == 0:
        raise FeatureExtractionError(f"{wav_path} contains no audio.")
    y = _centre_trim(y, sr, window_s)
    # Praat analyses the same trimmed samples, not the file on disk, so the two
    # blocks cannot silently disagree about which audio they measured.
    sound = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)

    features: dict[str, float] = {"gender": float(gender)}
    try:
        features.update(extract_baseline_praat(sound))
        features.update(extract_intensity(sound))
        features.update(extract_formants(sound, gender))
    except parselmouth.PraatError as error:
        # Praat's own messages are informative but arrive as a foreign exception
        # type; callers should only ever have to handle FeatureExtractionError.
        raise FeatureExtractionError(
            f"Praat could not analyse {wav_path.name}: {error}"
        ) from error
    features.update(extract_mfcc(y, sr))

    if strict:
        extract_nonlinear_baseline(sound)
        extract_vocal_fold(sound)
        extract_wavelet(y, sr)
        extract_tqwt(y, sr)

    expected = feature_spec.implemented_feature_names()
    if set(features) != set(expected):
        raise AssertionError(
            "extractor output does not match the spec; "
            f"missing={sorted(set(expected) - set(features))} "
            f"unexpected={sorted(set(features) - set(expected))}"
        )
    return {name: features[name] for name in expected}
