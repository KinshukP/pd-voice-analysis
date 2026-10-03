"""Build step 3: the inference pipeline behind an HTTP boundary.

One endpoint, no auth, no persistence. The point is to prove the pipeline works
over HTTP before anything else is layered on.

    .venv/bin/uvicorn pdvoice.api:app --reload

    POST /screen   multipart: files=<5-7 wavs>, gender=<0|1>  -> JSON result
    GET  /health   readiness, including whether the model is loaded

Two things this does beyond calling the pipeline:

  * Uploads are written to a private temp directory under names this process
    chooses. A client-supplied filename is a string from a stranger, never a
    path -- it is kept only as a label to echo back.
  * Server paths are scrubbed from the response. The pipeline records which file
    each take came from, and those are temp paths that would leak the server's
    filesystem layout; they are mapped back to the client's own filenames.
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pdvoice import demo
from pdvoice.pipeline import (
    MIN_TAKES,
    RECOMMENDED_TAKES,
    InferenceError,
    _artifacts,
    run_inference_multi,
)

logger = logging.getLogger(__name__)

#: Generous for a few seconds of 44.1 kHz mono (a 5 s take is ~440 KB), while
#: still bounding what one request can make this process hold.
MAX_FILE_BYTES = 25 * 1024 * 1024
#: More than RECOMMENDED_TAKES, so a user can send spares, but bounded.
MAX_FILES = 12

ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}

#: Set PDVOICE_DEBUG=1 to include model probabilities in the response. Off by
#: default and controlled by the environment, not by a request parameter, so a
#: client cannot ask for them.
DEBUG = os.environ.get("PDVOICE_DEBUG", "").lower() in {"1", "true", "yes"}

#: Browser dev servers that may call this API. Override with a comma-separated
#: PDVOICE_CORS_ORIGINS. Deliberately an explicit list rather than "*": with
#: allow_credentials a wildcard is rejected by browsers anyway, and naming the
#: origins keeps a future deployment from silently inheriting a permissive rule.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "PDVOICE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup rather than on the first request: it takes a
    # few seconds, and a health check should fail immediately if the artifacts
    # are missing rather than the first user discovering it.
    try:
        _artifacts()
        app.state.model_ready = True
        logger.info("model artifacts loaded")
    except Exception:
        app.state.model_ready = False
        logger.exception("model artifacts failed to load")
    yield


app = FastAPI(
    title="Parkinson's Voice Screening",
    description="Screening, not diagnosis. See PROJECT_CONTEXT.md for limitations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    ready = bool(getattr(app.state, "model_ready", False))
    return {
        "status": "ok" if ready else "degraded",
        "model_loaded": ready,
        "min_takes": MIN_TAKES,
        "recommended_takes": RECOMMENDED_TAKES,
        "debug_probabilities_exposed": DEBUG,
        "demo_mode": demo.DEMO_MODE,
    }


def _scrub_message(message: str, names: dict[str, str]) -> str:
    """Rewrite temp paths and generated names in an error string as client filenames.

    Failure messages name the recordings that failed, which is the useful part --
    but they name them by the path this process invented, which tells the client
    nothing and exposes internal naming. Both the full path and the bare
    generated filename can appear, so both are replaced.
    """
    for path, original in names.items():
        message = message.replace(path, original).replace(Path(path).name, original)
    return message


def _scrub_paths(result: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    """Replace temp paths with the client's own filenames, everywhere they appear."""

    def label(path: Any) -> Any:
        return names.get(str(path), Path(str(path)).name)

    result["meta"]["wav"] = [label(p) for p in result["meta"].get("wav", [])]
    takes = result.get("takes", {})
    for key in ("used", "excluded", "failed", "forced"):
        for entry in takes.get(key, []):
            if "wav" in entry:
                entry["wav"] = label(entry["wav"])

    # The prose names files too -- the override warning lists them by name -- so
    # it needs the same treatment as the structured fields.
    if isinstance(result.get("explanation_text"), str):
        result["explanation_text"] = _scrub_message(result["explanation_text"], names)
    return result


def _public_view(result: dict[str, Any]) -> dict[str, Any]:
    """Remove the model's disease probability from the response entirely.

    A DO_NOT_DISPLAY marker in the payload does not stop anything rendering the
    number; not sending it does. The probability is calibrated to a 74.6%-PD
    training cohort, so shown to a general-population user it overstates risk by
    orders of magnitude.

    It appears in more places than the `internal` block, and all of them go:
      * internal.calibrated_probability / raw_probability / per_take_calibrated
      * takes.used[].calibrated_probability   (per-take score)
      * takes.excluded[].calibrated_probability
      * spread.min / spread.max               (the extreme per-take scores)

    `spread.std` and `spread.range` stay: they are differences between
    probabilities, which describe consistency and cannot be read as a personal
    risk figure. `confidence_pct` remains the number meant to be shown.
    """
    public = copy.deepcopy(result)
    public.pop("internal", None)

    takes = public.get("takes", {})
    for key in ("used", "excluded"):
        for entry in takes.get(key, []):
            entry.pop("calibrated_probability", None)

    spread = public.get("spread", {})
    spread.pop("min", None)
    spread.pop("max", None)
    return public


def _save_uploads(files: list[UploadFile], destination: Path) -> tuple[list[Path], dict[str, str]]:
    """Write uploads to disk under names we control. Returns paths and their labels."""
    paths: list[Path] = []
    names: dict[str, str] = {}

    for upload in files:
        original = Path(upload.filename or "recording.wav").name
        suffix = Path(original).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"{original!r} has unsupported extension {suffix!r}. "
                    f"Accepted: {', '.join(sorted(ALLOWED_SUFFIXES))}. "
                    "Record 44.1 kHz mono WAV to match the training data."
                ),
            )

        # Our name, not theirs -- the client filename never touches the path.
        target = destination / f"{uuid.uuid4().hex}{suffix}"
        size = 0
        with open(target, "wb") as handle:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{original!r} exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit.",
                    )
                handle.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail=f"{original!r} is empty.")

        paths.append(target)
        names[str(target)] = original
    return paths, names


@app.post("/screen")
def screen(
    files: list[UploadFile] = File(..., description=f"{MIN_TAKES}-{RECOMMENDED_TAKES} recordings"),
    gender: int = Form(..., description="0 female, 1 male"),
    force_include: list[str] = Form(
        default=[],
        description="Filenames to combine even if they fail the quality checks.",
    ),
    sources: list[str] = Form(
        default=[],
        description=(
            "Per-file origin, 'recorded' or 'uploaded', in the same order as files. "
            "Only consulted when the server is running in demo mode."
        ),
    ),
) -> JSONResponse:
    """Score a set of sustained-vowel recordings from one speaker.

    Defined with `def` rather than `async def` on purpose: inference is several
    seconds of CPU work, so FastAPI runs it in a worker thread instead of
    blocking the event loop for every other request.
    """
    if not getattr(app.state, "model_ready", False):
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")
    if gender not in (0, 1):
        raise HTTPException(
            status_code=422,
            detail="gender must be 0 (female) or 1 (male). It is a model input and "
            "sets the formant ceiling, so it cannot be inferred from the audio.",
        )
    if len(files) < MIN_TAKES:
        raise HTTPException(
            status_code=422,
            detail=f"Send at least {MIN_TAKES} recordings ({RECOMMENDED_TAKES} recommended); "
            f"got {len(files)}. A single take is not a reliable basis for a result.",
        )
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=422, detail=f"At most {MAX_FILES} recordings per request."
        )

    workdir = Path(tempfile.mkdtemp(prefix="pdvoice-"))
    try:
        paths, names = _save_uploads(files, workdir)

        # The client addresses takes by the filename it uploaded; translate to
        # the names we generated. Anything unrecognised is dropped rather than
        # errored -- a stale name from an earlier set should not fail the run.
        wanted = {n.strip() for n in force_include if n.strip()}
        forced_paths = {p for p, original in names.items() if original in wanted}

        try:
            result = run_inference_multi(paths, gender=gender, force_include=forced_paths)
        except InferenceError as error:
            # The recording set was unusable -- a client problem with an
            # actionable message, not a server fault. Scrubbed so it names the
            # client's own files rather than the ones we generated.
            raise HTTPException(
                status_code=422, detail=_scrub_message(str(error), names)
            ) from error

        result = _scrub_paths(result, names)
        # Applied last, over a genuine result, so the real analysis is what gets
        # overridden rather than skipped -- and so the marker cannot be lost by
        # a later transformation.
        if demo.DEMO_MODE:
            result = demo.apply(result, sources)
        return JSONResponse(result if DEBUG else _public_view(result))
    except HTTPException:
        raise
    except Exception:
        # Never surface internals; a stack trace would leak paths and versions.
        logger.exception("unexpected failure in /screen")
        raise HTTPException(status_code=500, detail="Internal error scoring the recordings.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
