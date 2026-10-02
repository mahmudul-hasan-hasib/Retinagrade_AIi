"""Grad-CAM evidence for the two real RetinaGrade AI checkpoints (CPU only).

Grad-CAM attributes a model's class score back to the image regions that drove
it: the activations of one convolutional layer are weighted by the gradient of
the target class with respect to those activations, then reduced to a single
heatmap. Nothing here changes the network: the architecture, the weights and the
preprocessing pipeline are read-only inputs, and this module never writes to a
parameter, a buffer or ``eval()`` state.

Target layer
------------
The target is ``features.8``, the final ``Conv2dNormActivation``
(``Conv2d(320, 1280, 1, 1)`` -> ``BatchNorm2d(1280)`` -> ``SiLU``). It is the
last convolutional stage in the trunk, and it feeds ``avgpool`` directly:

    features.8 -> avgpool -> flatten -> dr_head        (CFP, multi-head)
    features.8 -> avgpool -> flatten -> classifier.1   (UWF, single-head)

Because the only thing between the target layer and the score is a
parameter-free global average pool, the Grad-CAM weights - the spatial mean of
the class gradient - are *exactly* the class-head weights. The standard Grad-CAM
formulation therefore applies without approximation, and the layer carries the
largest receptive field in the network, which is what makes it useful for
localising small lesions.

Both modalities share the trunk factory ``create_efficientnet_b0_trunk()``, so
the target path is the same string for CFP and UWF. Only the head differs.

CAM grid
--------
The grid size is read from the captured activations, never hardcoded, because it
follows the preprocessing size rather than the architecture: the trunk is fully
convolutional and downsamples by 32, so CFP at 224 yields a 7x7 grid and UWF at
512 yields 16x16. The heatmap is bilinearly resized back to the model input size
so the caller always receives one pixel of CAM per pixel of model input.

What is deliberately absent
--------------------------
* The CFP auxiliary ``lesion_heads`` are never read. The CFP forward returns a
  ``(dr_logits, auxiliary_logits)`` tuple and only element ``0`` is used here.
  The checkpoints store no label map for those seven heads, so their outputs must
  never be surfaced as named clinical findings.
* No FastAPI route, no report generation, no persistence: this module is the
  core only.
* ``torch.no_grad()`` is never used on the CAM forward pass - it would sever
  exactly the gradients this module exists to compute.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .exceptions import ArchitectureMismatchError, InferenceFailedError
from .model_loader import (
    ARCHITECTURE_NAME,
    CPU_DEVICE,
    LoadedModel,
    ModelManager,
    manager as default_manager,
)
from .preprocessing import PreprocessConfig, get_config, preprocess_image
from .predictor import DR_CLASSES, EXPECTED_CLASS_COUNT, normalise_modality

#: Dotted path of the Grad-CAM target layer, shared by both modalities.
GRADCAM_TARGET_LAYER = "features.8"

#: Human-readable description of why that layer is the target.
GRADCAM_TARGET_LAYER_REASON = (
    "Last convolutional stage of the EfficientNet-B0 trunk (Conv2d 320->1280, "
    "BatchNorm2d, SiLU), feeding a parameter-free global average pool directly."
)

#: ``index -> label`` for the 5 ICDR grades, taken from ``predictor`` so the
#: label map is defined in exactly one place.
_LABELS = tuple(entry["label"] for entry in DR_CLASSES)

# One lock per modality. ``ModelManager`` hands out a single cached
# ``LoadedModel`` per modality, so two concurrent Grad-CAM requests would share
# that one instance. The captured activations are per-forward tensors and the
# parameters are never written, but serialising per modality keeps the captured
# state and the gradient read unambiguously paired with their request.
_CAM_LOCKS: Dict[str, threading.Lock] = {}
_CAM_LOCKS_GUARD = threading.Lock()


def _cam_lock(modality: str) -> threading.Lock:
    """Return the per-modality Grad-CAM lock, creating it on first use."""
    with _CAM_LOCKS_GUARD:
        lock = _CAM_LOCKS.get(modality)
        if lock is None:
            lock = threading.Lock()
            _CAM_LOCKS[modality] = lock
        return lock


@dataclass
class GradCamResult:
    """A Grad-CAM heatmap for one image, alongside the score it explains.

    ``cam`` is a ``(image_size, image_size)`` float32 tensor normalised to
    ``[0, 1]``: ``0`` is the least contributing input pixel in the model-input
    frame and ``1`` the most. ``grid_shape`` records the native resolution the
    heatmap was computed at *before* upsampling, which is what differs between
    modalities (7x7 for CFP at 224, 16x16 for UWF at 512).
    """

    modality: str
    model: str
    target_layer: str
    #: Class index the heatmap explains; the argmax unless one was requested.
    target_class: int
    label: str
    confidence: float
    probabilities: Dict[str, float]
    cam: torch.Tensor
    grid_shape: Tuple[int, int]
    image_size: int
    device: str
    checkpoint: str
    #: L2 norm of the target-class gradients captured at the target layer.
    #: ``> 0`` proves the backward pass actually reached the network; ``0``
    #: would mean the heatmap is meaningless.
    gradient_norm: float

    @property
    def cam_shape(self) -> Tuple[int, int]:
        return tuple(self.cam.shape)  # type: ignore[return-value]

    @property
    def cam_min(self) -> float:
        return float(self.cam.min().item())

    @property
    def cam_max(self) -> float:
        return float(self.cam.max().item())

    @property
    def gradients_are_nonzero(self) -> bool:
        return self.gradient_norm > 0.0

    def describe(self) -> Dict[str, Any]:
        """Return the heatmap statistics without the tensor itself."""
        return {
            "modality": self.modality,
            "model": self.model,
            "target_layer": self.target_layer,
            "target_class": self.target_class,
            "label": self.label,
            "confidence": self.confidence,
            "cam_shape": list(self.cam_shape),
            "grid_shape": list(self.grid_shape),
            "cam_min": self.cam_min,
            "cam_max": self.cam_max,
            "gradient_norm": self.gradient_norm,
            "gradients_are_nonzero": self.gradients_are_nonzero,
            "image_size": self.image_size,
            "device": self.device,
            "checkpoint": self.checkpoint,
        }


def resolve_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the Grad-CAM target layer of ``model``.

    Raises :class:`ArchitectureMismatchError` when the module does not carry the
    expected trunk, rather than silently producing a heatmap from a substitute
    layer.
    """
    try:
        layer = model.get_submodule(GRADCAM_TARGET_LAYER)
    except AttributeError as exc:
        raise ArchitectureMismatchError(
            f"Grad-CAM target layer '{GRADCAM_TARGET_LAYER}' is not present on "
            f"{type(model).__name__}. The trunk must be the torchvision "
            f"EfficientNet-B0 features; do not substitute another layer."
        ) from exc
    return layer


def _select_logits(output: Any, modality: str) -> torch.Tensor:
    """Pick the DR-grade logits out of a model output.

    The CFP network returns a ``(dr_logits, auxiliary_logits)`` tuple because it
    carries seven auxiliary lesion heads; only element ``0`` is the DR grade and
    only element ``0`` is read. The UWF network returns a single tensor. The
    auxiliary tuple element is deliberately never touched.
    """
    logits = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(logits):
        raise InferenceFailedError(
            f"{modality} model returned {type(output).__name__} instead of DR logits."
        )
    return logits


def _normalise_cam(cam: torch.Tensor) -> torch.Tensor:
    """Scale a raw CAM to ``[0, 1]``.

    A constant CAM (every location equally important, e.g. an input the model
    ignored) has ``max == min`` and no meaningful contrast; it is returned as
    all-zeros rather than dividing by zero and emitting NaN.
    """
    cam_min = cam.min()
    cam_max = cam.max()
    span = cam_max - cam_min
    if not torch.isfinite(span) or span.item() <= 0.0:
        return torch.zeros_like(cam)
    return (cam - cam_min) / span


class _ActivationCapture:
    """Capture one target-layer activation and its class gradient.

    A single forward hook is used rather than a paired
    forward/backward hook pair: the hook calls ``retain_grad()`` on the output it
    receives, which is the exact tensor that flows into ``avgpool``, so its
    ``.grad`` is populated by a later ``backward()``. That avoids the
    ``register_full_backward_hook`` edge cases around modules with non-tensor or
    multiple outputs.

    ``remove()`` is idempotent so it is safe to call from a ``finally`` block and
    again on the error path.
    """

    def __init__(self, layer: torch.nn.Module) -> None:
        self._activations: Optional[torch.Tensor] = None
        self._handle = layer.register_forward_hook(self._hook)
        self.removed = False

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        if not torch.is_tensor(output):
            raise InferenceFailedError(
                f"Grad-CAM target layer '{GRADCAM_TARGET_LAYER}' returned "
                f"{type(output).__name__}; a 4D feature map was expected."
            )
        if output.dim() != 4:
            raise InferenceFailedError(
                f"Grad-CAM target layer produced a {output.dim()}D tensor "
                f"{tuple(output.shape)}; a 4D (N, C, H, W) feature map is required."
            )
        if output.shape[2] < 2 or output.shape[3] < 2:
            raise InferenceFailedError(
                f"Grad-CAM target layer produced a degenerate {output.shape[2]}x"
                f"{output.shape[3]} spatial grid; global average pooling already "
                f"removed all spatial information."
            )
        output.retain_grad()
        self._activations = output

    @property
    def activations(self) -> torch.Tensor:
        if self._activations is None:
            raise InferenceFailedError(
                f"Grad-CAM target layer '{GRADCAM_TARGET_LAYER}' never ran during "
                f"the forward pass; the trunk may have been bypassed."
            )
        return self._activations

    @property
    def gradients(self) -> torch.Tensor:
        grads = self.activations.grad
        if grads is None:
            raise InferenceFailedError(
                f"No gradient reached the Grad-CAM target layer '{GRADCAM_TARGET_LAYER}'. "
                f"The backward pass was severed - check that the forward pass ran "
                f"under grad mode and not inside torch.no_grad()."
            )
        return grads

    def remove(self) -> None:
        """Detach the hook. Safe to call more than once."""
        if not self.removed:
            self._handle.remove()
            self.removed = True


def _run_gradcam(
    loaded: LoadedModel,
    tensor: torch.Tensor,
    target_class: Optional[int],
) -> Tuple[torch.Tensor, Tuple[int, int], int, float, Dict[str, float], float]:
    """Run one Grad-CAM forward/backward on ``tensor``.

    Returns ``(cam, grid_shape, target_class, confidence, probabilities,
    gradient_norm)``. ``cam`` is normalised to ``[0, 1]`` at the model-input
    resolution and ``grid_shape`` is the native ``(H, W)`` of the target-layer
    activations *before* upsampling. The model is forced into ``eval()`` and its
    prior mode restored afterwards: in ``train()`` mode BatchNorm would both use
    batch statistics and update its running estimates, silently mutating the
    loaded checkpoint's state.
    """
    model = loaded.model
    if tensor.dim() != 4 or tensor.shape[0] != 1:
        raise InferenceFailedError(
            f"Grad-CAM expects a single preprocessed image (1, 3, S, S), got "
            f"{tuple(tensor.shape)}."
        )

    was_training = model.training
    model.eval()
    capture = _ActivationCapture(resolve_target_layer(model))
    try:
        tensor = tensor.to(CPU_DEVICE).detach().requires_grad_(False)

        # Grad mode is required here. model_loader.dr_logits() cannot be reused
        # because it wraps the forward in torch.no_grad(), which would zero the
        # gradients this method reads.
        with torch.enable_grad():
            output = model(tensor)
            logits = _select_logits(output, loaded.display_name)
            if logits.dim() != 2 or logits.shape[1] != loaded.num_classes:
                raise InferenceFailedError(
                    f"{loaded.display_name} model returned DR logits of shape "
                    f"{tuple(logits.shape)}, expected (1, {loaded.num_classes})."
                )
            if logits.shape[1] != EXPECTED_CLASS_COUNT:
                raise InferenceFailedError(
                    f"{loaded.display_name} model produces {logits.shape[1]} classes but "
                    f"the DR class mapping defines {EXPECTED_CLASS_COUNT}."
                )

            probabilities = torch.softmax(logits[0], dim=0)
            if target_class is None:
                index = int(torch.argmax(probabilities).item())
            else:
                index = int(target_class)
                if not 0 <= index < logits.shape[1]:
                    raise InferenceFailedError(
                        f"target_class {index} is outside the 5 DR grades "
                        f"(0..{logits.shape[1] - 1})."
                    )

            score = logits[0, index]
            model.zero_grad(set_to_none=True)
            score.backward()

            activations = capture.activations
            gradients = capture.gradients
            gradient_norm = float(gradients.norm().item())

            if activations.shape != gradients.shape:
                raise InferenceFailedError(
                    f"Grad-CAM activation {tuple(activations.shape)} and gradient "
                    f"{tuple(gradients.shape)} shapes disagree."
                )

            # Grad-CAM: per-channel weights are the spatial mean of the class
            # gradient, then the channels are recombined and rectified. Both
            # operands are read-only activations, so this allocates nothing new
            # inside the network and touches no weight.
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))

            grid_height, grid_width = int(activations.shape[2]), int(activations.shape[3])
            cam = F.interpolate(
                cam,
                size=(tensor.shape[2], tensor.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            cam = _normalise_cam(cam[0, 0])

        return (
            cam,
            (grid_height, grid_width),
            index,
            float(probabilities[index].item()),
            {name: round(float(value), 6) for name, value in zip(_LABELS, probabilities.tolist())},
            gradient_norm,
        )
    finally:
        # Always detach the hook, including when the forward or backward raised,
        # so a failed request cannot leak a hook onto the cached model.
        capture.remove()
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()


def compute_gradcam(
    image: Any,
    modality: str,
    *,
    target_class: Optional[int] = None,
    manager: Optional[ModelManager] = None,
    config: Optional[PreprocessConfig] = None,
) -> GradCamResult:
    """Compute a Grad-CAM heatmap for one retinal image.

    ``image`` is anything :func:`preprocessing.preprocess_image` accepts: a PIL
    image, ``bytes`` or a binary file object, i.e. exactly what ``main.py``
    receives from an upload. Preprocessing is not duplicated here: it is delegated
    to the same deterministic :func:`get_config` / :func:`preprocess_image` pair
    that ``predictor.predict`` uses, so the heatmap always explains a prediction
    computed from an identically prepared tensor.

    ``target_class`` selects which DR grade the heatmap explains; ``None``
    (default) explains the predicted grade. Only the DR grade is ever exposed:
    the CFP model's seven auxiliary lesion heads are not read or reported.

    Returns a :class:`GradCamResult` whose ``cam`` is a ``(S, S)`` float32
    tensor in ``[0, 1]``.
    """
    key = normalise_modality(modality)
    active_manager = manager or default_manager
    loaded = active_manager.get(key)

    if loaded.num_classes != EXPECTED_CLASS_COUNT:
        raise InferenceFailedError(
            f"{loaded.display_name} model produces {loaded.num_classes} classes but "
            f"the DR class mapping defines {EXPECTED_CLASS_COUNT}. Refusing to guess."
        )

    preprocess = config or get_config(key, loaded.metadata.get("training_config"))
    tensor = preprocess_image(image, preprocess)

    with _cam_lock(key):
        cam, grid_shape, index, confidence, probabilities, gradient_norm = _run_gradcam(
            loaded, tensor, target_class
        )

    return GradCamResult(
        modality=loaded.display_name,
        model=ARCHITECTURE_NAME,
        target_layer=GRADCAM_TARGET_LAYER,
        target_class=index,
        label=_LABELS[index],
        confidence=round(confidence, 6),
        probabilities=probabilities,
        cam=cam,
        grid_shape=grid_shape,
        image_size=preprocess.image_size,
        device=str(CPU_DEVICE),
        checkpoint=loaded.spec.checkpoint.name,
        gradient_norm=gradient_norm,
    )


__all__ = [
    "GRADCAM_TARGET_LAYER",
    "GRADCAM_TARGET_LAYER_REASON",
    "GradCamResult",
    "compute_gradcam",
    "resolve_target_layer",
]