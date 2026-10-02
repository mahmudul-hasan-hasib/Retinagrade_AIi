"""RetinaGrade AI - FastAPI backend (CPU only).

============  =====================================================  =============
Endpoint      Purpose                                                  Model run?
============  =====================================================  =============
``GET  /``    Service banner                                           NO
``GET  /health``  Liveness + device/CUDA facts                         NO
``POST /predict``  Real DR grading of an uploaded image                **YES**
``POST /gradcam``  Grad-CAM heatmap for an uploaded image              **YES**
============  =====================================================  =============

``POST /predict`` validates the upload and then calls
``inference.predictor.predict()``, which runs the real CFP or UWF
EfficientNet-B0 checkpoint on the CPU. The response is the predictor's own
payload: predicted class, label, confidence and the full 5-class probability
distribution, plus provenance (``device``, ``checkpoint``, ``image_size`` and
the preprocessing block that says whether those values came from the checkpoint
or are a documented assumption).

``POST /gradcam`` validates the upload identically, then calls
``inference.gradcam.compute_gradcam()``. That runs a forward *and* a backward
pass through the same checkpoint and attributes the DR score back to the input
pixels that drove it, at the trunk's last convolutional stage ``features.8``.
The heatmap is returned as a base64-encoded PNG data URI inside the JSON body,
so the grade and the image that explains it arrive in a single response.

Domain errors from the inference layer carry their own ``code`` and
``http_status`` (``inference/exceptions.py``) and are translated here into the
``{"success": false, "error": {...}}`` envelope. They never leak a traceback.

Still **not** implemented, by instruction:
  * Gemini report generation, PDF export, screening history
  * dataset evaluation or accuracy calculation
  * Grad-CAM overlay compositing onto the source photograph: the heatmap is
    returned on its own, in the model-input frame, not blended with the image

Nothing here selects a CUDA device. The models load lazily on the first request
for a modality, so that first request also pays the checkpoint load.

Run from ``backend/``::

    .venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000/docs for the interactive OpenAPI page.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from inference.exceptions import InferenceError
from inference.gradcam import GRADCAM_TARGET_LAYER, compute_gradcam
from inference.predictor import DR_CLASSES, EXPECTED_CLASS_COUNT, predict
from schemas.api import (
    HealthResponse,
    RootResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("retinagrade")

APP_NAME = "RetinaGrade AI"
API_VERSION = "0.5.0-dev"

#: CPU is mandatory. CUDA is reported but never used.
DEVICE = "cpu"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

#: The two modalities this API accepts. Mirrors
#: ``inference.model_loader.SUPPORTED_MODALITIES``.
SUPPORTED_MODALITIES = ("cfp", "uwf")

#: DR grade labels and descriptions indexed by class. Taken from
#: ``inference.predictor.DR_CLASSES`` rather than restated here, so this endpoint
#: cannot drift from the single class map used by ``/predict``.
_DR_LABELS: tuple = tuple(entry["label"] for entry in DR_CLASSES)
_DR_DESCRIPTIONS: tuple = tuple(entry["description"] for entry in DR_CLASSES)

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
        "POST /predict runs the real trained model; POST /gradcam returns a "
        "Grad-CAM heatmap for the same model. Report generation is not implemented."
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


class GradCamImage(BaseModel):
    """The Grad-CAM heatmap, encoded as a PNG.

    The heatmap is returned as a ``data:`` URI rather than as a separate binary
    body so that the grade and the image explaining it stay in one JSON response
    that also carries the normal ``{"success": false, "error": {...}}`` envelope.
    ``width`` / ``height`` are the decoded pixel dimensions, which equal the model
    input size (224 for CFP, 512 for UWF), not the native Grad-CAM grid.
    """

    media_type: Literal["image/png"] = Field(default="image/png", examples=["image/png"])
    encoding: Literal["base64"] = Field(default="base64", examples=["base64"])
    width: int = Field(examples=[224])
    height: int = Field(examples=[224])
    #: ``data:image/png;base64,...`` - usable directly as an ``<img src>``.
    data_uri: str


class GradCamResponse(BaseModel):
    """Body of a successful ``POST /gradcam``.

    ``predicted_class`` / ``label`` / ``confidence`` describe the grade the
    heatmap explains. When the client does not ask for a specific
    ``target_class``, that grade *is* the model's prediction; when it does, these
    three fields describe the requested grade instead, and ``is_predicted_class``
    says whether the two happened to agree.

    The CFP model's seven auxiliary lesion heads are never read or reported.
    """

    success: Literal[True] = True
    modality: str = Field(examples=["CFP"])
    predicted_class: int = Field(ge=0, le=4, examples=[2])
    label: str = Field(examples=["Moderate"])
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.703081])
    description: str = Field(examples=["Moderate non-proliferative DR"])
    probabilities: Dict[str, float]
    #: True when the explained grade is also the model's top prediction.
    is_predicted_class: bool
    #: The trunk layer the heatmap was computed at.
    target_layer: str = Field(examples=[GRADCAM_TARGET_LAYER])
    #: Native Grad-CAM grid before upsampling: 7x7 for CFP, 16x16 for UWF.
    grid_shape: List[int] = Field(examples=[[7, 7]])
    image_size: int = Field(examples=[224])
    #: False would mean the backward pass never reached the network and the
    #: heatmap is meaningless.
    gradients_are_nonzero: bool
    inference_ms: float = Field(examples=[1450.9])
    device: Literal["cpu"] = Field(default="cpu")
    checkpoint: str = Field(examples=["EfficientNetB0_CFP_final.pth"])
    cam: GradCamImage


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


def _cam_to_png_data_uri(cam: torch.Tensor) -> tuple[str, int, int]:
    """Render a normalised Grad-CAM tensor as a base64 PNG data URI.

    ``cam`` is the ``(S, S)`` float32 tensor from ``inference.gradcam``,
    already normalised to ``[0, 1]``. It is mapped through a jet-style blue ->
    green -> yellow -> red ramp so that intensity is legible, then encoded as an
    8-bit RGB PNG.

    No plotting library is used: ``matplotlib`` is not a dependency of this
    project, so the colour ramp is applied directly in NumPy and encoded with
    Pillow, both of which are already required by the preprocessing layer.

    Returns ``(data_uri, width, height)``. The tensor is copied out of its graph
    first, so no autograd node is kept alive by the returned image.
    """
    values = cam.detach().to("cpu", torch.float32).clamp(0.0, 1.0).numpy()
    if values.ndim != 2:
        raise ValueError(f"Grad-CAM heatmap must be 2D, got shape {values.shape}")

    # Classic jet ramp, expressed so each channel is a tent function of the
    # normalised value: blue rises first, then green, then red.
    channels = [1.5 - np.abs(4.0 * values - centre) for centre in (3.0, 2.0, 1.0)]
    rgb = np.clip(np.stack(channels, axis=-1), 0.0, 1.0)
    rgb = (rgb * 255.0).round().astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    height, width = values.shape
    return f"data:image/png;base64,{encoded}", int(width), int(height)


@app.post(
    "/gradcam",
    response_model=GradCamResponse,
    tags=["inference"],
    summary="Grade an uploaded retinal image and return a Grad-CAM heatmap (CPU only)",
    responses={
        400: {"description": "Invalid modality or out-of-range target_class"},
        413: {"description": "Image too large"},
        422: {"description": "Missing, empty or undecodable image"},
        500: {"description": "Inference failed"},
        503: {"description": "Model checkpoint unavailable"},
    },
)
async def gradcam_image(
    image: UploadFile = File(..., description="Retinal image (CFP or UWF)."),
    modality: str = Form(..., description="Imaging modality: 'cfp' or 'uwf'."),
    target_class: Optional[int] = Form(
        None,
        description=(
            "DR grade (0-4) the heatmap should explain. Omit to explain the "
            "model's own prediction."
        ),
        ge=0,
        le=4,
    ),
) -> GradCamResponse:
    """Validate the upload, then produce a real Grad-CAM heatmap on the CPU.

    The upload is validated with exactly the same rules as ``POST /predict``:
    ``modality`` is ``cfp`` or ``uwf`` (case-insensitive, trimmed), an ``image``
    file field was actually sent, the file is non-empty, and it is within the
    25 MB limit. Each failure returns the same status and the same
    ``{"success": false, "error": {...}}`` envelope as ``/predict``.

    ``target_class`` is additionally range-checked here rather than inside the
    inference layer, so an out-of-range grade is a ``400`` (a client mistake)
    instead of a ``500``.

    The payload is then handed to ``inference.gradcam.compute_gradcam()``, which
    performs a forward *and* a backward pass through the same checkpoint used by
    ``/predict``. It must not run under ``torch.no_grad()``: that would sever the
    gradients the heatmap is built from. Like ``predict``, the call is CPU-bound
    and is dispatched to a worker thread so the event loop stays responsive.

    Only the DR grade is ever reported. The CFP model's seven auxiliary lesion
    heads are not read, because the checkpoints store no label map for them and
    they must not be presented as clinical findings.
    """
    key = (modality or "").strip().lower()

    if key not in SUPPORTED_MODALITIES:
        return _error(
            400,
            "unsupported_modality",
            f"Unsupported modality {modality!r}. Allowed values: {', '.join(SUPPORTED_MODALITIES)}.",
        )

    if target_class is not None and not 0 <= target_class < EXPECTED_CLASS_COUNT:
        return _error(
            400,
            "invalid_target_class",
            f"target_class must be between 0 and {EXPECTED_CLASS_COUNT - 1}, got {target_class}.",
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
        result = await run_in_threadpool(compute_gradcam, payload, key, target_class=target_class)
    except InferenceError as exc:
        logger.warning("POST /gradcam failed (%s): %s", type(exc).__name__, exc)
        return _error(exc.http_status, exc.code, str(exc))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not result.gradients_are_nonzero:
        # The heatmap would be meaningless, so this is reported rather than
        # returned as if it were real evidence.
        logger.error(
            "POST /gradcam: no gradient reached target layer %s for %s; refusing to return a CAM",
            GRADCAM_TARGET_LAYER,
            result.modality,
        )
        return _error(
            500,
            "gradcam_failed",
            "No gradient reached the Grad-CAM target layer; the heatmap could not be computed.",
        )

    try:
        data_uri, cam_width, cam_height = _cam_to_png_data_uri(result.cam)
    except Exception as exc:  # noqa: BLE001 - encoding failure is reported, not traced
        logger.error("POST /gradcam: PNG encoding failed: %s: %s", type(exc).__name__, exc)
        return _error(500, "gradcam_failed", "The Grad-CAM heatmap could not be encoded as PNG.")

    logger.info(
        "POST /gradcam: modality=%s, file=%s, %d bytes -> %s (class %d, confidence %.6f) "
        "at %s, grid %dx%d, cam %dx%d, on %s in %.1f ms",
        key,
        image.filename,
        len(payload),
        result.label,
        result.target_class,
        result.confidence,
        GRADCAM_TARGET_LAYER,
        result.grid_shape[0],
        result.grid_shape[1],
        cam_width,
        cam_height,
        result.device,
        elapsed_ms,
    )

    # ``probabilities`` is keyed in class order by ``compute_gradcam``, so the
    # argmax over the labelled keys is the model's own prediction. This lets the
    # response say whether an explicitly requested target_class agreed with it.
    predicted_index = max(
        range(EXPECTED_CLASS_COUNT), key=lambda i: result.probabilities[_DR_LABELS[i]]
    )

    return GradCamResponse(
        success=True,
        modality=result.modality,
        predicted_class=result.target_class,
        label=result.label,
        confidence=result.confidence,
        description=_DR_DESCRIPTIONS[result.target_class],
        probabilities=result.probabilities,
        is_predicted_class=result.target_class == predicted_index,
        target_layer=result.target_layer,
        grid_shape=[result.grid_shape[0], result.grid_shape[1]],
        image_size=result.image_size,
        gradients_are_nonzero=result.gradients_are_nonzero,
        inference_ms=round(elapsed_ms, 2),
        device=DEVICE,
        checkpoint=result.checkpoint,
        cam=GradCamImage(width=cam_width, height=cam_height, data_uri=data_uri),
    )


__all__ = [
    "app",
    "GradCamImage",
    "GradCamResponse",
    "PredictionModel",
    "PredictResponse",
]
