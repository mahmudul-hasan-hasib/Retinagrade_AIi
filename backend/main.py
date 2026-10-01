"""RetinaGrade AI - FastAPI backend skeleton (CPU only).

THIS IS A DEVELOPMENT SKELETON. Only three endpoints exist and none of them
runs a neural network:

============  ======================================================  =============
Endpoint      Purpose                                                  Model run?
============  ======================================================  =============
``GET  /``    Service banner                                           NO
``GET  /health``  Liveness + device/CUDA facts                         NO
``POST /predict``  **Validates the request only** and returns a        **NO**
                 temporary placeholder payload
============  ======================================================  =============

Deliberately **not** implemented yet, by instruction:
  * real image prediction (the trained models are never called)
  * Grad-CAM / saliency maps
  * Gemini report generation
  * any Flutter or frontend code
  * dataset evaluation or accuracy calculation

Because no model is loaded, this app starts instantly and cannot touch CUDA.
``inference/`` is left untouched: the CFP and UWF checkpoints on disk are not
modified, read or executed by this file.

Run from ``backend/``::

    .venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000/docs for the interactive OpenAPI page.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas.api import (
    HealthResponse,
    PredictDevResponse,
    RootResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("retinagrade")

APP_NAME = "RetinaGrade AI"
API_VERSION = "0.3.0-dev"

#: CPU is mandatory. CUDA is reported but never used.
DEVICE = "cpu"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

#: The two modalities this API accepts. Mirrors
#: ``inference.model_loader.SUPPORTED_MODALITIES``.
SUPPORTED_MODALITIES = ("cfp", "uwf")

#: Lowercase modality key -> display name used in the response body.
MODALITY_DISPLAY_NAMES: Dict[str, str] = {"cfp": "CFP", "uwf": "UWF"}

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
        "Backend skeleton for diabetic retinopathy screening (CFP / UWF), CPU only. "
        "Image prediction is NOT implemented yet."
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
    response_model=PredictDevResponse,
    tags=["inference"],
    summary="Validate an upload only (TEMPORARY - does not run the model)",
    responses={
        400: {"description": "Invalid modality"},
        422: {"description": "Missing or empty image"},
        413: {"description": "Image too large"},
    },
)
async def predict_validation_only(
    image: UploadFile = File(..., description="Retinal image (not decoded in this skeleton)."),
    modality: str = Form(..., description="Imaging modality: 'cfp' or 'uwf'."),
) -> PredictDevResponse:
    """Validate the request, then return a placeholder. NO INFERENCE.

    Validation performed, and nothing more:
      1. an ``image`` file field was actually sent;
      2. the file has a filename;
      3. the file is not empty;
      4. the file is within the size limit;
      5. ``modality`` is ``cfp`` or ``uwf`` (case-insensitive, whitespace
         trimmed).

    The image bytes are read only to check emptiness and size. They are not
    decoded, not preprocessed and not passed to any network.
    """
    key = (modality or "").strip().lower()

    # --- validation only: no model is loaded or executed -------------------
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

    logger.info(
        "POST /predict validated: modality=%s, file=%s, %d bytes. Model NOT run (skeleton).",
        key,
        image.filename,
        len(payload),
    )
    # -----------------------------------------------------------------------

    return PredictDevResponse(
        success=True,
        message="Image received",
        modality=MODALITY_DISPLAY_NAMES[key],
    )


__all__ = ["app"]