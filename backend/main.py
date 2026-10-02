"""RetinaGrade AI - FastAPI backend (CPU only).

============  =====================================================  =============
Endpoint      Purpose                                                  Model run?
============  =====================================================  =============
``GET  /``    Service banner                                           NO
``GET  /health``  Liveness + device/CUDA facts                         NO
``POST /predict``  Real DR grading of an uploaded image                **YES**
``POST /gradcam``  Grad-CAM heatmap for an uploaded image              **YES**
``POST /explain``  Gemini prose explanation of a prediction            NO
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

``POST /explain`` runs **no** model of ours. It takes the structured output the
caller already has - modality, grade, confidence, 5-class probabilities and
optional Grad-CAM facts - and asks Gemini to describe it in prose
(``services/gemini.py``). Gemini never sees the image and never sees a grade
it did not receive, and the caller's numbers are echoed back verbatim so the
explanation provably cannot have altered them. The key is read from the
environment and never leaves the server.

Domain errors from the inference layer carry their own ``code`` and
``http_status`` (``inference/exceptions.py``) and are translated here into the
``{"success": false, "error": {...}}`` envelope. ``services.gemini`` uses the
same ``code`` / ``http_status`` convention so its failures read identically.
Neither ever leaks a traceback.

Still **not** implemented, by instruction:
  * PDF export, screening history, report storage
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
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from dotenv import load_dotenv
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
from services import gemini

# Load backend/.env before anything reads GEMINI_API_KEY. Existing environment
# variables win, so a real deployment env is never silently overridden by a local
# file. The key stays in the process environment and is never returned or logged.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("retinagrade")

APP_NAME = "RetinaGrade AI"
API_VERSION = "0.6.0-dev"

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
        "Grad-CAM heatmap for the same model; POST /explain renders a prediction "
        "as prose via Gemini without re-running any model."
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


# --------------------------------------------------------------------------- #
# POST /explain
# --------------------------------------------------------------------------- #


class GradCamEvidence(BaseModel):
    """Optional Grad-CAM facts from a prior ``POST /gradcam`` call.

    Metadata only. The base64 heatmap is deliberately **not** accepted here:
    sending megabytes of pixels to a third-party API for a text summary would be
    a needless privacy cost, and Gemini never needs them - it is told only that a
    map exists and how strong its gradients were.
    """

    available: bool = Field(
        default=False,
        description="False when /gradcam failed or was not run for this image.",
    )
    target_layer: Optional[str] = Field(default=None, examples=["features.8"])
    grid_shape: Optional[List[int]] = Field(default=None, examples=[[7, 7]])
    image_size: Optional[int] = Field(default=None, examples=[224])
    gradients_are_nonzero: Optional[bool] = Field(default=None, examples=[True])
    is_predicted_class: Optional[bool] = Field(
        default=None,
        description="False when the heatmap explains a grade other than the prediction.",
    )
    summary: Optional[str] = Field(
        default=None,
        description=(
            "Optional backend-written sentence about the heatmap, e.g. that the "
            "gradients were strong and concentrated centrally."
        ),
    )


class ExplainRequest(BaseModel):
    """Body of ``POST /explain``: exactly what ``POST /predict`` already returned.

    Field types stay permissive on purpose so that range and content problems are
    reported through this API's own ``{"success": false, "error": {...}}``
    envelope with a useful ``code``, instead of FastAPI's bare
    ``{"detail": [...]}`` 422. Only a genuinely malformed type (a string where a
    number belongs) is left to Pydantic.

    ``modality`` accepts any casing or surrounding whitespace and is normalised
    by the handler, matching ``/predict`` and ``/gradcam``.
    """

    modality: str = Field(examples=["CFP"], description="Imaging modality: 'cfp' or 'uwf'.")
    predicted_class: int = Field(description="DR grade index 0-4 from /predict.")
    confidence: float = Field(description="Top-class probability, 0-1, from /predict.")
    probabilities: Dict[str, float] = Field(
        description="Full 5-class ICDR distribution keyed by grade label.",
    )
    gradcam: Optional[GradCamEvidence] = Field(
        default=None,
        description="Optional metadata from a prior /gradcam call.",
    )


class ExplainSource(BaseModel):
    """The caller's numbers, echoed back unchanged.

    This is the audit trail for the one rule this endpoint exists to enforce: the
    explanation is generated *from* these values and cannot have changed them.
    Compare ``source.confidence`` with the value sent in the request - they are
    the same object.
    """

    modality: str
    predicted_class: int
    label: str
    description: str
    confidence: float
    probabilities: Dict[str, float]
    gradcam_available: bool


class ExplainResponse(BaseModel):
    """Body of a successful ``POST /explain``.

    ``explanation`` is Gemini's prose. ``disclaimer`` is fixed server text that
    is appended regardless of what the model returned, so it cannot be dropped by
    truncation or a refusal. ``model`` names the Gemini model that produced the
    text, for provenance. ``latency_ms`` measures the Gemini call only - no model
    of ours runs here.
    """

    success: Literal[True] = True
    explanation: str
    #: Fixed, not model-generated. Always present.
    disclaimer: str
    model: str = Field(examples=[gemini.DEFAULT_GEMINI_MODEL])
    modality: str = Field(examples=["CFP"])
    latency_ms: float = Field(examples=[2410.7])
    source: ExplainSource


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

    ``gemini_configured`` is a bare boolean telling a client whether
    ``POST /explain`` is usable here, so the UI can say "not configured on this
    server" up front instead of letting the user click into a 503. No key, key
    length or key fragment is exposed.
    """
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        cuda_available=torch.cuda.is_available(),
        gemini_configured=gemini.is_available(),
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
    # No ``ge``/``le`` here on purpose. Constraining the form field would make
    # FastAPI reject an out-of-range grade during request validation and answer
    # with its own bare ``{"detail": [...]}`` 422, before this function ever
    # runs. Leaving the value unconstrained lets the explicit check below stay the
    # single source of truth and return the documented
    # ``400 invalid_target_class`` envelope instead.
    target_class: Optional[int] = Form(
        None,
        description=(
            "DR grade (0-4) the heatmap should explain. Omit to explain the "
            "model's own prediction."
        ),
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


def _normalise_gradcam_evidence(evidence: Optional[GradCamEvidence]) -> Optional[Dict[str, Any]]:
    """Reduce Grad-CAM metadata to the small dict the prompt builder expects."""
    if evidence is None:
        return None
    return {
        "available": bool(evidence.available),
        "target_layer": evidence.target_layer,
        "grid_shape": list(evidence.grid_shape) if evidence.grid_shape else None,
        "image_size": evidence.image_size,
        "gradients_are_nonzero": evidence.gradients_are_nonzero,
        "is_predicted_class": evidence.is_predicted_class,
        "summary": evidence.summary,
    }


@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["explanation"],
    summary="Explain a prediction in prose using Gemini (no model of ours is run)",
    responses={
        400: {"description": "Invalid modality, class, confidence or probability distribution"},
        422: {"description": "Malformed request body"},
        502: {"description": "Gemini refused the request or returned nothing usable"},
        503: {"description": "Gemini is not configured on this server"},
    },
)
async def explain_prediction(request: ExplainRequest) -> Any:
    """Turn the output of ``/predict`` into a short clinician-facing explanation.

    **No model of ours runs here.** The caller posts the numbers it already has
    and this endpoint only describes them, so it never disagrees with
    ``/predict``: the grade, confidence and distribution in the response are
    copies of the request, returned under ``source``.

    Gemini receives the structured numbers and optional Grad-CAM metadata - never
    the retinal image, and never a grade it did not receive. It is instructed to
    explain rather than diagnose; ``services/gemini.py`` holds that prompt. The
    heatmap bitmap is deliberately not accepted, only the fact that a map exists
    and how strong its gradients were, so no pixels leave the machine.

    The API key is read from the server's environment. When it is missing the
    endpoint answers ``503 gemini_not_configured`` and says so plainly, rather
    than failing obscurely or silently degrading to canned text. ``/predict`` and
    ``/gradcam`` are unaffected either way.

    Every returned explanation carries :data:`services.gemini.DISCLAIMER` as fixed
    server text, not model output.
    """
    key = (request.modality or "").strip().lower()
    if key not in SUPPORTED_MODALITIES:
        return _error(
            400,
            "unsupported_modality",
            f"Unsupported modality {request.modality!r}. Allowed values: {', '.join(SUPPORTED_MODALITIES)}.",
        )

    if not 0 <= request.predicted_class < EXPECTED_CLASS_COUNT:
        return _error(
            400,
            "invalid_predicted_class",
            f"predicted_class must be between 0 and {EXPECTED_CLASS_COUNT - 1}, "
            f"got {request.predicted_class}.",
        )

    # A confidence outside [0, 1] means the caller is not sending real /predict
    # output, and a probability outside [0, 1] means a corrupted payload. Both
    # would produce a confidently wrong explanation, so they are rejected here.
    if not 0.0 <= request.confidence <= 1.0:
        return _error(
            400,
            "invalid_confidence",
            f"confidence must be between 0.0 and 1.0, got {request.confidence}.",
        )

    probabilities = request.probabilities or {}
    expected_keys = set(_DR_LABELS)
    received_keys = set(probabilities)
    if received_keys != expected_keys:
        missing = sorted(expected_keys - received_keys)
        unexpected = sorted(received_keys - expected_keys)
        return _error(
            400,
            "invalid_probabilities",
            "probabilities must contain exactly the {} grade labels. Missing: {}. Unexpected: {}.".format(
                EXPECTED_CLASS_COUNT,
                ", ".join(missing) if missing else "none",
                ", ".join(unexpected) if unexpected else "none",
            ),
        )
    for label, value in probabilities.items():
        if not 0.0 <= value <= 1.0:
            return _error(
                400,
                "invalid_probabilities",
                f"Probability for {label!r} must be between 0.0 and 1.0, got {value}.",
            )

    index = request.predicted_class
    label = _DR_LABELS[index]
    description = _DR_DESCRIPTIONS[index]
    gradcam = _normalise_gradcam_evidence(request.gradcam)

    try:
        prompt = gemini.build_prompt(
            modality=key,
            predicted_class=index,
            label=label,
            description=description,
            confidence=request.confidence,
            probabilities=probabilities,
            class_order=list(_DR_LABELS),
            gradcam=gradcam,
        )
        settings = gemini.load_settings(require_key=True)
    except gemini.ExplanationError as exc:
        logger.warning("POST /explain refused: %s: %s", type(exc).__name__, exc)
        return _error(exc.http_status, exc.code, str(exc))

    started = time.perf_counter()
    try:
        # Network + model latency, so a worker thread keeps the event loop free.
        explanation = await run_in_threadpool(gemini.generate_explanation, prompt, settings)
    except gemini.ExplanationError as exc:
        logger.warning("POST /explain failed (%s): %s", type(exc).__name__, exc)
        return _error(exc.http_status, exc.code, str(exc))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not explanation.strip():
        return _error(
            502,
            "gemini_empty_response",
            "Gemini returned an empty explanation. The prediction itself is unaffected.",
        )

    logger.info(
        "POST /explain: modality=%s, class=%d (%s), confidence=%.6f, gradcam=%s, "
        "model=%s, %d chars in %.1f ms",
        key,
        index,
        label,
        request.confidence,
        "yes" if (gradcam and gradcam.get("available")) else "no",
        settings.model if settings else "unknown",
        len(explanation),
        elapsed_ms,
    )

    return ExplainResponse(
        success=True,
        explanation=explanation,
        # Fixed server text, appended whatever the model returned.
        disclaimer=gemini.DISCLAIMER,
        model=settings.model if settings else gemini.DEFAULT_GEMINI_MODEL,
        modality=key,
        latency_ms=round(elapsed_ms, 2),
        source=ExplainSource(
            modality=key,
            predicted_class=index,
            label=label,
            description=description,
            confidence=request.confidence,
            probabilities=probabilities,
            gradcam_available=bool(gradcam and gradcam.get("available")),
        ),
    )


__all__ = [
    "app",
    "ExplainRequest",
    "ExplainResponse",
    "ExplainSource",
    "GradCamEvidence",
    "GradCamImage",
    "GradCamResponse",
    "PredictionModel",
    "PredictResponse",
]
