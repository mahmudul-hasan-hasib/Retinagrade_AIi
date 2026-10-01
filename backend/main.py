"""RetinaGrade AI - FastAPI backend (CPU only).

============  =====================================================  =============
Endpoint      Purpose                                                  Model run?
============  =====================================================  =============
``GET  /``    Service banner                                           NO
``GET  /health``  Liveness + device/CUDA facts                         NO
``POST /predict``  Real DR grading of an uploaded image                **YES**
============  =====================================================  =============

``POST /predict`` validates the upload and then calls
``inference.predictor.predict()``, which runs the real CFP or UWF
EfficientNet-B0 checkpoint on the CPU. The response is the predictor's own
payload: predicted class, label, confidence and the full 5-class probability
distribution, plus provenance (``device``, ``checkpoint``, ``image_size`` and
the preprocessing block that says whether those values came from the checkpoint
or are a documented assumption).

Domain errors from the inference layer carry their own ``code`` and
``http_status`` (``inference/exceptions.py``) and are translated here into the
``{"success": false, "error": {...}}`` envelope. They never leak a traceback.

Still **not** implemented, by instruction:
  * Grad-CAM / saliency maps
  * Gemini report generation, PDF export, screening history
  * dataset evaluation or accuracy calculation

Nothing here selects a CUDA device. The models load lazily on the first request
for a modality, so that first request also pays the checkpoint load.

Run from ``backend/``::

    .venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000/docs for the interactive OpenAPI page.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Literal

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from inference.exceptions import InferenceError
from inference.predictor import predict
from schemas.api import (
    HealthResponse,
    RootResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("retinagrade")

APP_NAME = "RetinaGrade AI"
API_VERSION = "0.4.0-dev"

#: CPU is mandatory. CUDA is reported but never used.
DEVICE = "cpu"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

#: The two modalities this API accepts. Mirrors
#: ``inference.model_loader.SUPPORTED_MODALITIES``.
SUPPORTED_MODALITIES = ("cfp", "uwf")

#: Development origins. ``ALLOW_LOCALHOST_ANY_PORT`` additionally permits any
#: port on localhost, because a Flutter web dev server picks a free port.
ALLOW_LOCALHOST_ANY_PORT = True
LOCALHOST_ORIGINS: List[str] = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
LOCALHOST_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"

app = FastAPI(
    title=f"{APP_NAME} API",
    version=API_VERSION,
    description=(
        "Backend API for diabetic retinopathy screening (CFP / UWF), CPU only. "
        "POST /predict runs the real trained model; Grad-CAM and report "
        "generation are not implemented."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCALHOST_ORIGINS,
    allow_origin_regex=LOCALHOST_ORIGIN_REGEX if ALLOW_LOCALHOST_ANY_PORT else None,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


class PredictionModel(BaseModel):
    """A real DR prediction.

    Field-for-field mirror of ``inference.predictor.PredictionResult.to_dict()``.
    ``device`` is pinned to ``"cpu"``: nothing in this project may touch CUDA.
    ``probabilities`` is the full 5-class ICDR distribution, not just the winner.
    """

    modality: str = Field(examples=["CFP"])
    model: str = Field(examples=["EfficientNet-B0"])
    predicted_class: int = Field(ge=0, le=4, examples=[2])
    label: str = Field(examples=["Moderate"])
    description: str = Field(examples=["Moderate non-proliferative DR"])
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.703081])
    probabilities: Dict[str, float]
    image_size: int = Field(examples=[224])
    device: Literal["cpu"] = Field(default="cpu")
    checkpoint: str = Field(examples=["EfficientNetB0_CFP_final.pth"])
    #: Includes ``values_verified_by_checkpoint``. It is ``false`` for CFP,
    #: whose checkpoint stores no preprocessing config, so the input size used
    #: is a documented assumption rather than checkpoint data.
    preprocessing: Dict[str, Any]


class PredictResponse(BaseModel):
    """Body of a successful ``POST /predict``."""

    success: Literal[True] = True
    #: Wall-clock milliseconds spent inside ``predict()``, including the lazy
    #: checkpoint load on the first request for a modality.
    inference_ms: float = Field(examples=[812.44])
    prediction: PredictionModel


@app.exception_handler(Exception)
async def _unexpected_error_handler(_request, exc: Exception) -> JSONResponse:
    # Log the traceback server-side; never send it to the client.
    logger.exception("Unhandled error: %s", exc)
    return _error(
        500,
        "internal_error",
        "An unexpected internal error occurred. Check the server logs for details.",
    )


@app.get("/", response_model=RootResponse, tags=["system"])
def root() -> RootResponse:
    """Service banner. Does not touch any model."""
    return RootResponse(app=APP_NAME, status="running")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness plus device facts.

    ``cuda_available`` is reported for transparency only. Nothing in this
    project ever selects a CUDA device.
    """
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        cuda_available=torch.cuda.is_available(),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Grade an uploaded retinal image (real inference, CPU only)",
    responses={
        400: {"description": "Invalid modality"},
        413: {"description": "Image too large"},
        422: {"description": "Missing, empty or undecodable image"},
        500: {"description": "Inference failed"},
        503: {"description": "Model checkpoint unavailable"},
    },
)
async def predict_image(
    image: UploadFile = File(..., description="Retinal image (CFP or UWF)."),
    modality: str = Form(..., description="Imaging modality: 'cfp' or 'uwf'."),
) -> PredictResponse:
    """Validate the upload, then run the matching real model on the CPU.

    Validation, before anything is decoded or executed:
      1. ``modality`` is ``cfp`` or ``uwf`` (case-insensitive, trimmed);
      2. an ``image`` file field was actually sent;
      3. the file is not empty;
      4. the file is within the 25 MB size limit.

    Only then is the payload handed to ``inference.predictor.predict()``, which
    decodes, preprocesses and runs the checkpoint. That call is CPU-bound and
    would block the event loop, so it is dispatched to a worker thread; the
    event loop stays free to serve other requests while the network runs.

    Errors raised by the inference layer (a corrupt upload, a missing or
    mismatched checkpoint) are translated with their own ``code`` and
    ``http_status`` into the standard error envelope, without a traceback.
    """
    key = (modality or "").strip().lower()

    if key not in SUPPORTED_MODALITIES:
        return _error(
            400,
            "unsupported_modality",
            f"Unsupported modality {modality!r}. Allowed values: {', '.join(SUPPORTED_MODALITIES)}.",
        )

    if image is None or not image.filename:
        return _error(400, "invalid_image", "No image was uploaded. Send an 'image' file field.")

    payload = await image.read()
    if not payload:
        return _error(422, "invalid_image", "The uploaded image is empty (0 bytes).")
    if len(payload) > MAX_UPLOAD_BYTES:
        return _error(
            413,
            "image_too_large",
            f"The uploaded image is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    started = time.perf_counter()
    try:
        result = await run_in_threadpool(predict, payload, key)
    except InferenceError as exc:
        logger.warning("POST /predict failed (%s): %s", type(exc).__name__, exc)
        return _error(exc.http_status, exc.code, str(exc))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    logger.info(
        "POST /predict: modality=%s, file=%s, %d bytes -> %s (class %d, confidence %.6f) "
        "on %s in %.1f ms",
        key,
        image.filename,
        len(payload),
        result.label,
        result.predicted_class,
        result.confidence,
        result.device,
        elapsed_ms,
    )

    return PredictResponse(
        success=True,
        inference_ms=round(elapsed_ms, 2),
        prediction=PredictionModel(**result.to_dict()),
    )


__all__ = ["app", "PredictResponse", "PredictionModel"]
