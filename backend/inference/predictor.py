"""Prediction for RetinaGrade AI: DR grade, confidence, full distribution.

The trained networks are multi-task: a 5-class DR grading head (CFP also
carries 7 auxiliary binary heads). This module reports the **DR grade only**,
as produced by the models. The auxiliary binary heads are not named in the
checkpoints, so they are never surfaced as clinical findings; they can be
requested explicitly with ``include_raw_auxiliary=True`` as unnamed numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .exceptions import InferenceFailedError, UnsupportedModalityError
from .model_loader import (
    ARCHITECTURE_NAME,
    CPU_DEVICE,
    SUPPORTED_MODALITIES,
    ModelManager,
    manager as default_manager,
)
from .preprocessing import PreprocessConfig, get_config, preprocess_image

#: DR class mapping, fixed by the ICDR 5-grade scale. Both checkpoints are
#: verified to emit exactly 5 DR logits, so this mapping is exhaustive:
#:
#: ==========  ==================  =============================================
#: ``index``   ``key`` / ``label``  Meaning
#: ==========  ==================  =============================================
#: 0           ``No DR``           no apparent diabetic retinopathy
#: 1           ``Mild``            mild non-proliferative DR
#: 2           ``Moderate``        moderate non-proliferative DR
#: 3           ``Severe``          severe non-proliferative DR
#: 4           ``Proliferative DR`` proliferative DR
#: ==========  ==================  =============================================
#:
#: ``key`` and ``label`` are deliberately the *same* string so that
#: ``result.label == result.probabilities`` keys always line up and no
#: renaming step can drift away from the class index.
#:
#: The order is the logit order produced by ``dr_head`` (CFP) and
#: ``classifier.1`` (UWF), both verified as ``(5, 1280)`` in the real files.
DR_CLASSES = (
    {"index": 0, "key": "No DR", "label": "No DR", "description": "No apparent diabetic retinopathy"},
    {"index": 1, "key": "Mild", "label": "Mild", "description": "Mild non-proliferative DR"},
    {"index": 2, "key": "Moderate", "label": "Moderate", "description": "Moderate non-proliferative DR"},
    {"index": 3, "key": "Severe", "label": "Severe", "description": "Severe non-proliferative DR"},
    {"index": 4, "key": "Proliferative DR", "label": "Proliferative DR", "description": "Proliferative diabetic retinopathy"},
)

PROBABILITY_KEYS = tuple(c["key"] for c in DR_CLASSES)
LABELS_BY_KEY = {c["key"]: c for c in DR_CLASSES}
EXPECTED_CLASS_COUNT = len(DR_CLASSES)


@dataclass
class PredictionResult:
    """Full DR prediction for one image."""

    modality: str
    model: str
    predicted_class: int
    label: str
    description: str
    confidence: float
    probabilities: Dict[str, float]
    image_size: int
    device: str
    checkpoint: str
    preprocess_config: Dict[str, Any] = field(default_factory=dict)
    auxiliary_raw_outputs: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the prediction payload.

        The first six keys are the required output contract, in the required
        order. ``description``, ``image_size``, ``device``, ``checkpoint`` and
        ``preprocessing`` are additional provenance fields; ``device`` is always
        ``"cpu"`` because nothing in this project may touch CUDA.
        """
        payload: Dict[str, Any] = {
            "modality": self.modality,
            "model": self.model,
            "predicted_class": self.predicted_class,
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "description": self.description,
            "image_size": self.image_size,
            "device": self.device,
            "checkpoint": self.checkpoint,
            "preprocessing": self.preprocess_config,
        }
        if self.auxiliary_raw_outputs is not None:
            payload["auxiliary_raw_outputs"] = self.auxiliary_raw_outputs
            payload["auxiliary_note"] = (
                "Unnamed auxiliary binary heads from the checkpoint. Their meaning and order are "
                "NOT stored in the model file; do not interpret them as clinical findings."
            )
        return payload


def normalise_modality(modality: str) -> str:
    key = (modality or "").strip().lower()
    if key not in SUPPORTED_MODALITIES:
        raise UnsupportedModalityError(
            f"Unsupported modality {modality!r}. Allowed values: {', '.join(SUPPORTED_MODALITIES)}."
        )
    return key


def predict(
    image,
    modality: str,
    *,
    manager: Optional[ModelManager] = None,
    config: Optional[PreprocessConfig] = None,
    include_raw_auxiliary: bool = False,
) -> PredictionResult:
    """Classify one retinal image with the matching CFP or UWF model.

    All probabilities come from the actual model output via softmax; nothing
    is hardcoded. Runs entirely on the CPU.
    """
    key = normalise_modality(modality)
    active_manager = manager or default_manager
    loaded = active_manager.get(key)

    if loaded.num_classes != EXPECTED_CLASS_COUNT:
        raise InferenceFailedError(
            f"{loaded.display_name} model produces {loaded.num_classes} classes but the "
            f"DR class mapping defines {EXPECTED_CLASS_COUNT}. Refusing to guess the mapping."
        )

    preprocess = config or get_config(key, loaded.metadata.get("training_config"))
    tensor = preprocess_image(image, preprocess)

    try:
        logits = loaded.dr_logits(tensor)
    except Exception as exc:  # noqa: BLE001
        raise InferenceFailedError(f"Inference failed for the {loaded.display_name} model: {exc}") from exc

    if logits.dim() != 2 or logits.shape[1] != loaded.num_classes:
        raise InferenceFailedError(
            f"{loaded.display_name} model returned logits of shape {tuple(logits.shape)}, "
            f"expected (1, {loaded.num_classes})."
        )

    probabilities = torch.softmax(logits[0], dim=0)
    index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[index].item())
    class_info = LABELS_BY_KEY[PROBABILITY_KEYS[index]]

    auxiliary: Optional[Dict[str, float]] = None
    if include_raw_auxiliary and loaded.num_binary_heads:
        try:
            aux_logits = loaded.binary_logits(tensor)
        except Exception:  # noqa: BLE001 - auxiliary output is optional
            aux_logits = None
        if aux_logits is not None and aux_logits.numel() == loaded.num_binary_heads:
            aux_probs = torch.sigmoid(aux_logits[0])
            auxiliary = {
                f"aux_head_{i}": round(float(value), 6) for i, value in enumerate(aux_probs.tolist())
            }

    return PredictionResult(
        modality=loaded.display_name,
        model=ARCHITECTURE_NAME,
        predicted_class=index,
        label=class_info["label"],
        description=class_info["description"],
        confidence=round(confidence, 6),
        probabilities={key_name: round(float(value), 6) for key_name, value in zip(PROBABILITY_KEYS, probabilities.tolist())},
        image_size=preprocess.image_size,
        device=str(CPU_DEVICE),
        checkpoint=loaded.spec.checkpoint.name,
        preprocess_config=preprocess.describe(),
        auxiliary_raw_outputs=auxiliary,
    )


__all__ = [
    "DR_CLASSES",
    "PredictionResult",
    "get_config",
    "normalise_modality",
    "predict",
]
