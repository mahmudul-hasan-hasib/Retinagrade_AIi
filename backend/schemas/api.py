"""Pydantic request/response schemas for the RetinaGrade AI API."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Modality = Literal["cfp", "uwf"]


class PreprocessingInfo(BaseModel):
    """Preprocessing actually applied, so results are reproducible."""

    modality: str
    image_size: int
    mean: List[float]
    std: List[float]
    normalization: str
    color_space: str
    resize: str
    training_augmentation_applied: bool = False
    config_source: str = Field(
        description="'read from checkpoint config' when the checkpoint stored it, otherwise an assumption."
    )


class PredictionResultModel(BaseModel):
    """DR grade produced by the trained EfficientNet model."""

    modality: str = Field(examples=["CFP"])
    model: str = Field(examples=["EfficientNet-B0"])
    predicted_class: int = Field(ge=0, examples=[3])
    label: str = Field(examples=["Severe"])
    description: str
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.87])
    probabilities: Dict[str, float] = Field(
        description="Softmax probability per DR class; keys are the DR class names."
    )
    image_size: int
    device: str = Field(examples=["cpu"])
    checkpoint: str
    preprocessing: PreprocessingInfo


class PredictResponse(BaseModel):
    success: Literal[True] = True
    result: PredictionResultModel


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    service: str
    device: str
    cuda_available: bool
    models_loaded: bool


class ModelStatus(BaseModel):
    loaded: bool
    modality: str
    checkpoint: Optional[str] = None
    error: Optional[str] = None
    num_classes: Optional[int] = None
    num_binary_heads: Optional[int] = None
    device: Optional[str] = None
    eval_mode: Optional[bool] = None
    head_layout: Optional[str] = None


class ModelsInfoResponse(BaseModel):
    architecture: str
    device: str
    cuda_available: bool
    class_count: int
    dr_classes: List[Dict[str, object]]
    models: Dict[str, ModelStatus]
    preprocessing: Dict[str, Dict[str, object]]
    notes: List[str]
