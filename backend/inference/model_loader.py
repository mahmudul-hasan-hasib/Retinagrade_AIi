"""Model loading for RetinaGrade AI (CPU only).

Architecture of both checkpoints was determined by inspecting the real
checkpoint files (see ``inference/inspect_models.py``):

* ``EfficientNetB0_CFP_final.pth``  -> checkpoint dict with key
  ``model_state_dict`` holding a **multi-head** network:

  - ``features.*``          torchvision EfficientNet-B0 convolutional trunk
                            (stages 0-8, 1280 output channels, 3 input channels)
  - ``dr_head``             ``nn.Linear(1280, 5)``      -> DR grade, 5 classes
  - ``lesion_heads.0-6``    7 x ``nn.Linear(1280, 1)``  -> binary auxiliary heads

* ``EfficientNetB0_UWF_final.pt``  -> checkpoint dict with key
  ``model_state_dict`` holding a **single-head** network, i.e. exactly
  ``torchvision.models.efficientnet_b0(num_classes=5)``:

  - ``features.*``          same EfficientNet-B0 trunk
  - ``classifier.1``        ``nn.Linear(1280, 5)``      -> DR grade, 5 classes

Both load into the architectures below with ``strict=True`` and report zero
missing / unexpected keys, which proves the reconstruction is exact.
No weights are downloaded (``weights=None``); the checkpoints are the only
source of parameters.

The model was trained with PyTorch ``2.10.0+cu128`` / ``torchvision
0.25.0+cu128`` on CUDA. Those CUDA builds are **not** required and are not
installed here: inference runs on ``torch.device("cpu")`` with a CPU-only
PyTorch build.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.serialization import safe_globals

from .exceptions import (
    ArchitectureMismatchError,
    CheckpointLoadError,
    InferenceError,
    ModelNotFoundError,
    UnsupportedModalityError,
)
from .model_architecture import (  # single source of truth for the architecture
    BACKBONE_CHANNELS,
    INPUT_CHANNELS,
    NUM_BINARY_HEADS,
    NUM_CLASSES,
    CfpEfficientNetB0,
    create_cfp_model,
    create_uwf_model,
)

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "models"

CFP_CHECKPOINT = MODELS_DIR / "EfficientNetB0_CFP_final.pth"
UWF_CHECKPOINT = MODELS_DIR / "EfficientNetB0_UWF_final.pt"

CPU_DEVICE = torch.device("cpu")

# Architecture constants are owned by ``model_architecture`` and re-exported
# here for the existing callers of this module.
NUM_LESION_HEADS = NUM_BINARY_HEADS
ARCHITECTURE_NAME = "EfficientNet-B0"

MODALITY_CFP = "cfp"
MODALITY_UWF = "uwf"
SUPPORTED_MODALITIES: Tuple[str, str] = (MODALITY_CFP, MODALITY_UWF)

MULTI_HEAD = "multi_head"
SINGLE_HEAD = "single_head"

# Globals allow-listed inside ``torch.load(weights_only=True)``. The real
# checkpoints store NumPy scalars (``best_val_qwk``) and a ``TorchVersion``
# string; everything else is a plain tensor. Unsafe unpickling is never used.
_SAFE_GLOBALS = [
    np._core.multiarray.scalar,
    np.dtype,
    np.ndarray,
    np.dtypes.Float32DType,
    np.dtypes.Float64DType,
    np.dtypes.Int64DType,
    np.dtypes.UInt8DType,
    np.dtypes.BoolDType,
    torch.torch_version.TorchVersion,
]

# Keys that may hold the weights inside a checkpoint dictionary.
_STATE_DICT_KEYS = ("model_state_dict", "state_dict", "model", "net", "weights")


class MultiTaskEfficientNetB0(CfpEfficientNetB0):
    """Backwards-compatible alias for :class:`CfpEfficientNetB0`.

    The architecture itself now lives in ``model_architecture.py`` so that it
    is defined exactly once and can be verified on its own.
    """


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one screening model."""

    modality: str
    display_name: str
    checkpoint: Path
    head_layout: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    MODALITY_CFP: ModelSpec(
        modality=MODALITY_CFP,
        display_name="CFP",
        checkpoint=CFP_CHECKPOINT,
        head_layout=MULTI_HEAD,
    ),
    MODALITY_UWF: ModelSpec(
        modality=MODALITY_UWF,
        display_name="UWF",
        checkpoint=UWF_CHECKPOINT,
        head_layout=SINGLE_HEAD,
    ),
}


@dataclass
class LoadedModel:
    """A checkpoint that has been built, validated and parked on the CPU."""

    spec: ModelSpec
    model: nn.Module
    num_classes: int
    num_binary_heads: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def modality(self) -> str:
        return self.spec.modality

    @property
    def display_name(self) -> str:
        return self.spec.display_name

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def dr_logits(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return DR grade logits ``(N, num_classes)`` for a CPU tensor."""
        with torch.no_grad():
            tensor = tensor.to(CPU_DEVICE)
            output = self.model(tensor)
        if isinstance(output, tuple):
            return output[0]
        return output

    def binary_logits(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the auxiliary binary head logits, if the model has any."""
        with torch.no_grad():
            output = self.model(tensor.to(CPU_DEVICE))
        if isinstance(output, tuple):
            return output[1]
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            "modality": self.spec.display_name,
            "model": ARCHITECTURE_NAME,
            "checkpoint": self.spec.checkpoint.name,
            "head_layout": self.spec.head_layout,
            "num_classes": self.num_classes,
            "num_binary_heads": self.num_binary_heads,
            "device": str(self.device),
            "eval_mode": not self.model.training,
            "checkpoint_metadata": self.metadata,
        }


def safe_load_checkpoint(path: Path) -> Dict[str, Any]:
    """Deserialise a checkpoint onto the CPU with ``weights_only=True``."""
    if not path.is_file():
        raise ModelNotFoundError(
            f"Checkpoint not found: {path.name}. Expected it in '{path.parent}'."
        )
    try:
        with safe_globals(_SAFE_GLOBALS):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except ModelNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported verbatim, no traceback leak
        raise CheckpointLoadError(
            f"Could not deserialise '{path.name}' safely on CPU: {type(exc).__name__}: {exc}"
        ) from exc

    if isinstance(checkpoint, nn.Module):
        # A full pickled module would be architecture-uncontrolled; refuse it
        # rather than silently accepting an unverified network.
        raise ArchitectureMismatchError(
            f"'{path.name}' contains a complete pickled nn.Module. This project loads "
            "state_dict checkpoints only, so the architecture cannot be verified."
        )
    if not isinstance(checkpoint, dict):
        raise ArchitectureMismatchError(
            f"'{path.name}' is neither a state_dict nor a checkpoint dict "
            f"(found {type(checkpoint).__name__})."
        )
    return checkpoint


def extract_state_dict(checkpoint: Dict[str, Any], path: Path) -> Dict[str, torch.Tensor]:
    """Pull the tensor state_dict out of a checkpoint dictionary."""
    for key in _STATE_DICT_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value
    raise ArchitectureMismatchError(
        f"'{path.name}' has no usable state_dict (top-level keys: {sorted(checkpoint)})."
    )


def detect_head_layout(state_dict: Dict[str, torch.Tensor], path: Path) -> str:
    """Decide which architecture a state_dict belongs to, from its own keys."""
    has_dr_head = "dr_head.weight" in state_dict
    has_classifier = any(k.startswith("classifier.") for k in state_dict)
    binary_heads = sorted(
        int(k.split(".")[1]) for k in state_dict if k.startswith("lesion_heads.") and k.endswith("weight")
    )
    if has_dr_head and binary_heads:
        return MULTI_HEAD
    if has_classifier and not has_dr_head:
        return SINGLE_HEAD
    raise ArchitectureMismatchError(
        f"'{path.name}' state_dict matches no known architecture "
        f"(dr_head={has_dr_head}, classifier={has_classifier}, binary heads={binary_heads})."
    )


def build_model(head_layout: str, num_classes: int, num_binary_heads: int) -> nn.Module:
    """Instantiate the architecture without downloading any pretrained weights.

    Delegates to ``model_architecture``, which owns the construction.
    """
    if head_layout == MULTI_HEAD:
        return create_cfp_model(num_classes=num_classes, num_binary_heads=num_binary_heads)
    if head_layout == SINGLE_HEAD:
        return create_uwf_model(num_classes=num_classes)
    raise ArchitectureMismatchError(f"Unknown head layout: {head_layout!r}")


def _validate_state_dict_shapes(
    state_dict: Dict[str, torch.Tensor], spec: ModelSpec, num_classes: int, num_binary_heads: int
) -> None:
    """Check the parts of the architecture that the class mapping depends on."""
    first_conv = state_dict.get("features.0.0.weight")
    if first_conv is None or first_conv.dim() != 4:
        raise ArchitectureMismatchError(
            f"'{spec.checkpoint.name}' has no 'features.0.0.weight' stem convolution."
        )
    if first_conv.shape[1] != INPUT_CHANNELS:
        raise ArchitectureMismatchError(
            f"'{spec.checkpoint.name}' expects {first_conv.shape[1]} input channels, "
            f"this project is configured for {INPUT_CHANNELS} (RGB)."
        )
    if spec.head_layout == MULTI_HEAD:
        head = state_dict.get("dr_head.weight")
        if head is None or tuple(head.shape) != (num_classes, BACKBONE_CHANNELS):
            raise ArchitectureMismatchError(
                f"'{spec.checkpoint.name}' dr_head shape {tuple(head.shape) if head is not None else None} "
                f"!= ({num_classes}, {BACKBONE_CHANNELS})."
            )
        heads = [k for k in state_dict if k.startswith("lesion_heads.") and k.endswith("weight")]
        if len(heads) != num_binary_heads:
            raise ArchitectureMismatchError(
                f"'{spec.checkpoint.name}' has {len(heads)} auxiliary binary heads, expected {num_binary_heads}."
            )
    else:
        linear = state_dict.get("classifier.1.weight")
        if linear is None or tuple(linear.shape) != (num_classes, BACKBONE_CHANNELS):
            raise ArchitectureMismatchError(
                f"'{spec.checkpoint.name}' classifier shape "
                f"{tuple(linear.shape) if linear is not None else None} != ({num_classes}, {BACKBONE_CHANNELS})."
            )


def _extract_metadata(checkpoint: Dict[str, Any], spec: ModelSpec) -> Dict[str, Any]:
    """Collect human-readable training metadata (no tensors) from a checkpoint."""
    metadata: Dict[str, Any] = {
        "epoch": checkpoint.get("epoch"),
        "best_qwk": _as_float(
            checkpoint.get("best_val_qwk", checkpoint.get("best_qwk"))
        ),
    }
    config = checkpoint.get("config")
    if isinstance(config, dict):
        metadata["training_config"] = {
            key: _as_plain(value)
            for key, value in config.items()
            if key
            in {
                "experiment",
                "dataset",
                "model",
                "image_size",
                "num_classes",
                "normalization",
                "pretrained",
                "augmentation",
                "loss",
                "optimizer",
                "pytorch_version",
                "torchvision_version",
                "cuda_version",
                "device",
            }
        }
    if spec.head_layout == MULTI_HEAD and "history" in checkpoint:
        history = checkpoint["history"]
        if isinstance(history, list) and history and isinstance(history[0], dict):
            metadata["training_history_fields"] = sorted(history[0].keys())
    if "history" in checkpoint and isinstance(checkpoint["history"], dict):
        history = checkpoint["history"]
        best = _as_float(history.get("val_qwk"))
        if best is not None:
            metadata["best_qwk"] = best
    return metadata


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
        return float(value)
    return None


def _as_plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def load_model(spec: ModelSpec) -> LoadedModel:
    """Load, validate and evaluate one checkpoint on the CPU."""
    if not spec.checkpoint.is_file():
        raise ModelNotFoundError(
            f"{spec.display_name} model is missing: expected '{spec.checkpoint.name}' "
            f"in '{spec.checkpoint.parent}'."
        )

    checkpoint = safe_load_checkpoint(spec.checkpoint)
    state_dict = extract_state_dict(checkpoint, spec.checkpoint)
    head_layout = detect_head_layout(state_dict, spec.checkpoint)
    if head_layout != spec.head_layout:
        raise ArchitectureMismatchError(
            f"'{spec.checkpoint.name}' is a {head_layout} checkpoint but "
            f"{spec.display_name} is registered as {spec.head_layout}."
        )

    num_classes = _infer_num_classes(state_dict, spec, head_layout)
    num_binary_heads = _infer_num_binary_heads(state_dict, spec, head_layout)
    _validate_state_dict_shapes(state_dict, spec, num_classes, num_binary_heads)

    model = build_model(head_layout, num_classes, num_binary_heads)
    try:
        result = model.load_state_dict(state_dict, strict=True)
    except Exception as exc:  # noqa: BLE001
        raise ArchitectureMismatchError(
            f"State dict of '{spec.checkpoint.name}' does not match "
            f"{ARCHITECTURE_NAME}/{head_layout}: {type(exc).__name__}: {exc}"
        ) from exc
    if result.missing_keys or result.unexpected_keys:
        raise ArchitectureMismatchError(
            f"State dict of '{spec.checkpoint.name}' mismatch - "
            f"missing: {list(result.missing_keys)}, unexpected: {list(result.unexpected_keys)}"
        )

    model.to(CPU_DEVICE)
    model.eval()
    for parameter in model.parameters():
        if parameter.device.type != "cpu":  # pragma: no cover - defensive
            raise ArchitectureMismatchError(
                f"Parameter '{parameter.shape}' did not move to CPU (on {parameter.device})."
            )

    _validate_forward(model, spec, num_classes, num_binary_heads)

    loaded = LoadedModel(
        spec=spec,
        model=model,
        num_classes=num_classes,
        num_binary_heads=num_binary_heads,
        metadata=_extract_metadata(checkpoint, spec),
    )
    logger.info(
        "Loaded %s model from %s (%s, %d classes, %d params)",
        spec.display_name,
        spec.checkpoint.name,
        head_layout,
        num_classes,
        sum(p.numel() for p in model.parameters()),
    )
    return loaded


def _validate_forward(
    model: nn.Module, spec: ModelSpec, num_classes: int, num_binary_heads: int
) -> None:
    """Run one small CPU forward pass to confirm the real output dimensions.

    A 64x64 probe is used instead of the training size: EfficientNet is fully
    convolutional with global pooling, so the output width is size independent,
    and this keeps startup fast on CPU.
    """
    probe = torch.zeros(1, INPUT_CHANNELS, 64, 64, device=CPU_DEVICE)
    model.eval()
    try:
        with torch.no_grad():
            output = model(probe)
    except Exception as exc:  # noqa: BLE001
        raise ArchitectureMismatchError(
            f"Forward pass failed for '{spec.checkpoint.name}': {type(exc).__name__}: {exc}"
        ) from exc

    if spec.head_layout == MULTI_HEAD:
        if not isinstance(output, tuple) or len(output) != 2:
            raise ArchitectureMismatchError(
                f"'{spec.checkpoint.name}' forward must return (dr_logits, auxiliary_logits)."
            )
        dr_logits, aux_logits = output
        if tuple(dr_logits.shape) != (1, num_classes):
            raise ArchitectureMismatchError(
                f"'{spec.checkpoint.name}' produced DR logits {tuple(dr_logits.shape)}, expected (1, {num_classes})."
            )
        if tuple(aux_logits.shape) != (1, num_binary_heads):
            raise ArchitectureMismatchError(
                f"'{spec.checkpoint.name}' produced auxiliary logits {tuple(aux_logits.shape)}, "
                f"expected (1, {num_binary_heads})."
            )
    else:
        if isinstance(output, tuple) or not torch.is_tensor(output):
            raise ArchitectureMismatchError(
                f"'{spec.checkpoint.name}' forward must return a single logits tensor."
            )
        if tuple(output.shape) != (1, num_classes):
            raise ArchitectureMismatchError(
                f"'{spec.checkpoint.name}' produced logits {tuple(output.shape)}, expected (1, {num_classes})."
            )


def _infer_num_classes(state_dict: Dict[str, torch.Tensor], spec: ModelSpec, head_layout: str) -> int:
    """Read the output class count from the checkpoint, never assume 5."""
    if head_layout == MULTI_HEAD:
        head = state_dict["dr_head.weight"]
    else:
        head = state_dict["classifier.1.weight"]
    return int(head.shape[0])


def _infer_num_binary_heads(state_dict: Dict[str, torch.Tensor], spec: ModelSpec, head_layout: str) -> int:
    if head_layout != MULTI_HEAD:
        return 0
    return sum(1 for k in state_dict if k.startswith("lesion_heads.") and k.endswith("weight"))


class ModelManager:
    """Owns the CFP and UWF models: loads each exactly once, keeps them on CPU.

    CFP and UWF stay completely separate. A failure on one modality never
    causes a silent fallback to the other model.
    """

    def __init__(self, specs: Optional[Dict[str, ModelSpec]] = None) -> None:
        self._specs = dict(specs if specs is not None else MODEL_SPECS)
        self._models: Dict[str, LoadedModel] = {}
        self._errors: Dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def modalities(self) -> Tuple[str, ...]:
        return tuple(self._specs)

    def load_all(self) -> Dict[str, LoadedModel]:
        """Load every registered checkpoint once. Re-entrant and idempotent."""
        with self._lock:
            for modality, spec in self._specs.items():
                if modality in self._models:
                    continue
                try:
                    self._models[modality] = load_model(spec)
                except (ModelNotFoundError, CheckpointLoadError, ArchitectureMismatchError) as exc:
                    self._errors[modality] = str(exc)
                    logger.error("Failed to load %s model: %s", spec.display_name, exc)
            return dict(self._models)

    def is_loaded(self, modality: str) -> bool:
        return modality in self._models

    def get(self, modality: str) -> LoadedModel:
        """Return a loaded model, or raise a precise error."""
        key = (modality or "").strip().lower()
        if key not in self._specs:
            raise UnsupportedModalityError(
                f"Unsupported modality {modality!r}. Supported modalities: {', '.join(self._specs)}."
            )
        if key not in self._models:
            self.load_all()
        if key not in self._models:
            raise self._resolve_error(key)
        return self._models[key]

    def _resolve_error(self, modality: str) -> InferenceError:
        spec = self._specs[modality]
        message = self._errors.get(
            modality, f"{spec.display_name} model could not be loaded from '{spec.checkpoint.name}'."
        )
        if not spec.checkpoint.is_file():
            return ModelNotFoundError(message)
        if "missing" in message or "unexpected" in message or "does not match" in message:
            return ArchitectureMismatchError(message)
        return CheckpointLoadError(message)

    def describe(self) -> Dict[str, Any]:
        """Machine-readable status of every registered model."""
        self.load_all()
        status: Dict[str, Any] = {}
        for modality, spec in self._specs.items():
            if modality in self._models:
                status[modality] = {"loaded": True, **self._models[modality].describe()}
            else:
                status[modality] = {
                    "loaded": False,
                    "modality": spec.display_name,
                    "checkpoint": spec.checkpoint.name,
                    "error": self._errors.get(modality, "not loaded"),
                }
        return status

    def is_ready(self) -> bool:
        return all(modality in self._models for modality in self._specs)


manager = ModelManager()

__all__ = [
    "ARCHITECTURE_NAME",
    "CPU_DEVICE",
    "MODEL_SPECS",
    "ModelManager",
    "ModelSpec",
    "MultiTaskEfficientNetB0",
    "LoadedModel",
    "SUPPORTED_MODALITIES",
    "detect_head_layout",
    "load_model",
    "manager",
    "safe_load_checkpoint",
]
