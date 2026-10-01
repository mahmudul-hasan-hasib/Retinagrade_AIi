"""Pydantic request/response schemas for the RetinaGrade AI API.

Skeleton scope (Part 4A). Only the schemas used by the three endpoints in
``main.py`` are defined here. The DR prediction schemas (``PredictionResultModel``,
``PredictResponse``, ``ModelsInfoResponse``, ...) are deliberately **absent**:
no endpoint returns a prediction yet, and leaving unused prediction schemas
behind would imply inference that does not exist.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    """Body of ``GET /``."""

    app: str = Field(examples=["RetinaGrade AI"])
    status: str = Field(examples=["running"])


class HealthResponse(BaseModel):
    """Body of ``GET /health``.

    ``device`` is always ``"cpu"``. ``cuda_available`` is reported only so a
    client can see the machine's capability; this API never selects a CUDA
    device.
    """

    status: str = Field(examples=["healthy"])
    device: Literal["cpu"] = Field(default="cpu", examples=["cpu"])
    cuda_available: bool = Field(default=False, examples=[False])


class PredictDevResponse(BaseModel):
    """Body of ``POST /predict`` - validation only, no model is run."""

    success: Literal[True] = True
    message: str = Field(examples=["Image received"])
    modality: Literal["CFP", "UWF"] = Field(examples=["CFP"])


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorBody