"""Image preprocessing for RetinaGrade AI (CPU).

What the checkpoints actually store
-----------------------------------
Verified by reading both real checkpoint files (see the table below); nothing
here is assumed unless marked as an assumption.

============================  ========  ====================================
Checkpoint                   ``config`` What it stores about preprocessing
============================  ========  ====================================
``EfficientNetB0_UWF_final``  present    ``image_size = 512``,
``.pt``                                   ``normalization = 'ImageNet mean/std'``,
                                         ``pretrained = 'ImageNet'``
``EfficientNetB0_CFP_final``  **absent**  nothing. Top-level keys are exactly
``.pth``                                  ``best_val_qwk``, ``epoch``,
                                         ``history``, ``model_state_dict``.
                                         There is no image size, no
                                         normalization and no colour mode.
============================  ========  ====================================

ASSUMPTION - CFP input size and normalization
---------------------------------------------
Because the CFP checkpoint stores no preprocessing configuration, both values
below are **assumptions, not facts read from the checkpoint**:

* normalization -> ImageNet mean/std. Justified, but only indirectly: the same
  training project stored ``normalization = 'ImageNet mean/std'`` and
  ``pretrained = 'ImageNet'`` for the UWF run, so ImageNet statistics are the
  documented default. If the CFP run used a different scheme, its logits are
  meaningless.
* input size -> ``224``, the canonical EfficientNet-B0 input, used here *only*
  because the checkpoint does not say otherwise. Note that the sibling UWF
  checkpoint says ``512``; if the CFP run also trained at 512 then 224 is the
  wrong size and accuracy will suffer. EfficientNet-B0 is fully convolutional
  with global average pooling, so any size runs without erroring -- a wrong
  size fails silently, which is exactly why this is reported rather than
  hidden. Override it with :func:`with_image_size` (the CLI exposes
  ``--image-size``).

For UWF the size is not an assumption: 512 is read out of the checkpoint at load
time by :func:`get_config`.

Caveat on the UWF config, for the record
----------------------------------------
The UWF checkpoint's ``config`` dict contains ``experiment = 'E1_CFP_EfficientNetB0'``
and ``dataset = 'CFP'`` even though the file is the UWF model. Those two fields
are stale labels written by the training script, so the dict is not internally
consistent. ``image_size`` / ``normalization`` are read from it anyway because
they are the only preprocessing evidence in either file, but the inconsistency
is why the CFP defaults are treated as assumptions rather than inferred from
the UWF config.

Inference-time transforms are deliberately minimal. The checkpoints were trained
with ``RandomHorizontalFlip(p=0.5) + RandomRotation(10)``, which is
**training-only** augmentation and must not be applied at inference time.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.transforms import functional as TF

from .exceptions import InvalidImageError, UnsupportedModalityError

logger = logging.getLogger(__name__)

ImageSource = Union[Image.Image, bytes, bytearray, BinaryIO]

#: Formats accepted from the API client.
SUPPORTED_FORMATS = ("JPEG", "PNG", "BMP", "TIFF", "WEBP")

#: ImageNet channel statistics (torchvision EfficientNet-B0 was initialised
#: from ImageNet weights, confirmed by ``config['pretrained'] = 'ImageNet'``).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Size that IS confirmed by a checkpoint (``EfficientNetB0_UWF_final.pt``).
CHECKPOINT_IMAGE_SIZE = 512

#: Size used when the checkpoint is silent. NOT confirmed by any checkpoint.
DEFAULT_IMAGE_SIZE = 224
IMAGENET_INPUT_SIZE = 224

#: Per-modality defaults. ``source`` documents where the values came from and
#: ``verified`` says whether they were read out of the checkpoint or assumed.
_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "cfp": {
        "image_size": DEFAULT_IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "source": (
            "ASSUMED - EfficientNetB0_CFP_final.pth has no 'config' key, so the "
            "image size and normalization below are not read from the checkpoint. "
            "ImageNet statistics match the sibling UWF run; 224 is the canonical "
            "EfficientNet-B0 input. The UWF checkpoint used 512 - if CFP also "
            "trained at 512 this default is wrong and accuracy will suffer."
        ),
        "verified": False,
    },
    "uwf": {
        "image_size": CHECKPOINT_IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "source": (
            "read from EfficientNetB0_UWF_final.pt config: "
            "image_size=512, normalization='ImageNet mean/std', pretrained='ImageNet'"
        ),
        "verified": True,
    },
}


@dataclass(frozen=True)
class PreprocessConfig:
    """Deterministic, side-effect-free preprocessing parameters."""

    image_size: int
    mean: tuple
    std: tuple
    source: str
    modality: str
    verified: bool = False

    @property
    def is_assumed(self) -> bool:
        """True when the checkpoint did not state these values."""
        return not self.verified

    def describe(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "image_size": self.image_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "normalization": "ImageNet mean/std",
            "normalization_source": (
                "checkpoint config" if self.verified else "ASSUMED (default, not in checkpoint)"
            ),
            "color_space": "RGB",
            "resize": f"bilinear resize to ({self.image_size}, {self.image_size})",
            "range_before_normalize": "[0, 1]",
            "training_augmentation_applied": False,
            "config_source": self.source,
            "values_verified_by_checkpoint": self.verified,
        }


def _resolve_base(modality: str) -> Dict[str, Any]:
    key = (modality or "").strip().lower()
    if key not in _DEFAULTS:
        raise UnsupportedModalityError(
            f"Unsupported modality {modality!r}. Supported modalities: {', '.join(_DEFAULTS)}."
        )
    return _DEFAULTS[key]


def get_config(modality: str, checkpoint_config: Optional[Dict[str, Any]] = None) -> PreprocessConfig:
    """Build the preprocessing config for ``modality``.

    ``checkpoint_config`` is the checkpoint's own ``config`` dict when one
    exists; a positive ``image_size`` inside it takes precedence over the
    documented default and flips ``verified`` to True. When the checkpoint has
    no ``config`` (the CFP case) the documented assumption is returned with
    ``verified=False`` so the caller can report it as an assumption.
    """
    key = (modality or "").strip().lower()
    base = dict(_resolve_base(key))
    source = base["source"]
    verified = bool(base["verified"])

    if checkpoint_config:
        stored_size = checkpoint_config.get("image_size")
        if isinstance(stored_size, int) and stored_size > 0:
            base["image_size"] = int(stored_size)
            source = "read from checkpoint config (image_size=%d)" % stored_size
            verified = True
        normalization = str(checkpoint_config.get("normalization", "")).lower()
        if normalization and ("imagenet" not in normalization):
            logger.warning(
                "Checkpoint %s reports normalisation %r which is not ImageNet; "
                "falling back to ImageNet statistics and this may be wrong.",
                key,
                normalization,
            )
            verified = False
        elif normalization and not verified:
            source += "; normalization read from checkpoint, image size assumed"

    if not verified:
        logger.warning(
            "Preprocessing for %s is an ASSUMPTION, not checkpoint data: %s",
            key,
            source,
        )

    return PreprocessConfig(
        image_size=int(base["image_size"]),
        mean=tuple(base["mean"]),
        std=tuple(base["std"]),
        source=source,
        modality=key,
        verified=verified,
    )


def with_image_size(config: PreprocessConfig, image_size: int) -> PreprocessConfig:
    """Return a copy of ``config`` with a different input size.

    Overriding always clears ``verified``: the checkpoint-confirmed size has
    just been replaced by a human-supplied one.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return replace(
        config,
        image_size=image_size,
        source=config.source + " (image_size overridden on the command line)",
        verified=False,
    )


def describe_preprocessing() -> Dict[str, Dict[str, Any]]:
    """Return the effective config for every modality, for API/reporting use."""
    return {key: get_config(key).describe() for key in _DEFAULTS}


def decode_image(source: ImageSource) -> Image.Image:
    """Decode an uploaded image into an RGB :class:`PIL.Image.Image`.

    Accepts a PIL image, raw bytes, or a file-like object.
    """
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB")

    if isinstance(source, (bytes, bytearray)):
        if not source:
            raise InvalidImageError("The uploaded image is empty (0 bytes).")
        data: bytes = bytes(source)
    elif isinstance(source, (BinaryIO, io.IOBase)) or hasattr(source, "read"):
        try:
            source.seek(0)
        except (AttributeError, OSError):  # pragma: no cover - non-seekable stream
            pass
        data = source.read()
        if not data:
            raise InvalidImageError("The uploaded image is empty (0 bytes).")
        if not isinstance(data, (bytes, bytearray)):
            raise InvalidImageError("The uploaded image could not be read as binary data.")
        data = bytes(data)
    else:
        raise InvalidImageError(
            f"Unsupported image input type: {type(source).__name__}. Expected a file upload."
        )

    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            fmt = opened.format
            if fmt not in SUPPORTED_FORMATS:
                raise InvalidImageError(
                    f"Unsupported image format {fmt or 'unknown'}. "
                    f"Supported formats: {', '.join(SUPPORTED_FORMATS)}."
                )
            return ImageOps.exif_transpose(opened).convert("RGB")
    except InvalidImageError:
        raise
    except UnidentifiedImageError as exc:
        raise InvalidImageError(
            "The uploaded file is not a readable image. It may be corrupted or renamed."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"The uploaded image could not be decoded: {exc}") from exc


def preprocess_image(source: ImageSource, config: PreprocessConfig) -> torch.Tensor:
    """Decode and normalise an image into a ``(1, 3, S, S)`` CPU tensor."""
    image = decode_image(source)
    image = TF.resize(image, [config.image_size, config.image_size], antialias=True)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, config.mean, config.std)
    return tensor.unsqueeze(0)


def load_image_file(path: Union[str, Path], modality: str) -> torch.Tensor:
    """Preprocess an image from disk (used by the CLI test script)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise InvalidImageError(f"Image file not found: {file_path}")
    return preprocess_image(file_path.read_bytes(), get_config(modality))


__all__ = [
    "CHECKPOINT_IMAGE_SIZE",
    "DEFAULT_IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PreprocessConfig",
    "SUPPORTED_FORMATS",
    "decode_image",
    "describe_preprocessing",
    "get_config",
    "load_image_file",
    "preprocess_image",
    "with_image_size",
]
