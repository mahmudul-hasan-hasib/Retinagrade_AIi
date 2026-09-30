"""RetinaGrade AI - FastAPI application (CPU-only inference).

Endpoints
---------
``GET  /health``  liveness + CPU/CUDA status
``GET  /models``  verified per-modality model status
``POST /predict`` classify one retinal image (``cfp`` or ``uwf``)

Run from ``backend/``::

    .venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Dict

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from inference.exceptions import (
    ArchitectureMismatchError,
    CheckpointLoadError,
    InferenceError,
    InferenceFailedError,
    InvalidImageError,
    ModelNotFoundError,
    UnsupportedModalityError,
)
from inference.model_loader import ARCHITECTURE_NAME, CPU_DEVICE, NUM_CLASSES, manager
from inference.preprocessing import SUPPORTED_FORMATS, describe_preprocessing
from inference.predictor import DR_CLASSES, predict
from schemas.api import (
    HealthResponse,
    ModelsInfoResponse,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("retinagrade")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

NOTES = [
    "Inference runs on the CPU only; no CUDA or GPU packages are installed.",
    "Both checkpoints are 5-class DR graders verified from their own tensor shapes.",
    "Reported labels are diabetic retinopathy grades produced by the trained models.",
    "The CFP checkpoint also contains 7 unnamed auxiliary binary heads; their meaning is not "
    "stored in the model file, so they are never reported as clinical findings.",
    "Not a medical device. Results require review by a qualified clinician.",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models on %s ...", CPU_DEVICE)
    status = manager.describe()
    for modality, info in status.items():
        if info["loaded"]:
            logger.info("Model %s ready (%s)", modality.upper(), info["checkpoint"])
        else:
            logger.error("Model %s unavailable: %s", modality.upper(), info["error"])
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="RetinaGrade AI API",
    version="0.2.0",
    description="Diabetic retinopathy screening (CFP / UWF) - CPU-only inference.",
    lifespan=lifespan,
)


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


@app.exception_handler(InferenceError)
async def _inference_error_handler(_request, exc: InferenceError) -> JSONResponse:
    logger.warning("%s: %s", type(exc).__name__, exc)
    return _error(exc.http_status, exc.code, str(exc))


@app.exception_handler(Exception)
async def _unexpected_error_handler(_request, exc: Exception) -> JSONResponse:
    # Log the traceback server-side; never send it to the client.
    logger.exception("Unhandled error: %s", exc)
    return _error(
        500,
        "internal_error",
        "An unexpected internal error occurred. Check the server logs for details.",
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    manager.load_all()
    return HealthResponse(
        status="ok",
        service="retinagrade-ai",
        device=str(CPU_DEVICE),
        cuda_available=torch.cuda.is_available(),
        models_loaded=manager.is_ready(),
    )


@app.get("/models", response_model=ModelsInfoResponse, tags=["system"])
def models_info() -> ModelsInfoResponse:
    return ModelsInfoResponse(
        architecture=ARCHITECTURE_NAME,
        device=str(CPU_DEVICE),
        cuda_available=torch.cuda.is_available(),
        class_count=NUM_CLASSES,
        dr_classes=[dict(item) for item in DR_CLASSES],
        models=manager.describe(),
        preprocessing=describe_preprocessing(),
        notes=list(NOTES),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    responses={
        400: {"description": "Invalid modality or missing image"},
        415: {"description": "Unsupported image format"},
        422: {"description": "Invalid or corrupted image"},
        503: {"description": "Model unavailable"},
    },
)
async def predict_endpoint(
    image: UploadFile = File(..., description="Retinal fundus image (JPEG/PNG/BMP/TIFF/WEBP)."),
    modality: str = Form(..., description="Imaging modality: 'cfp' or 'uwf'."),
) -> PredictResponse:
    if not image or not image.filename:
        raise InvalidImageError("No image was uploaded. Send a 'image' file field.")
    if image.content_type and image.content_type.split(";")[0].strip().lower() not in {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/bmp",
        "image/tiff",
        "image/webp",
    }:
        raise InvalidImageError(
            f"Unsupported content type {image.content_type!r}. Expected an image file "
            f"({'/'.join(SUPPORTED_FORMATS)})."
        )

    payload = await image.read()
    if not payload:
        raise InvalidImageError("The uploaded image is empty (0 bytes).")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise InvalidImageError(
            f"The uploaded image is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )

    # Raises UnsupportedModalityError before any model work is done.
    result = predict(payload, modality)
    return PredictResponse(success=True, result=result.to_dict())


__all__ = ["app"]
