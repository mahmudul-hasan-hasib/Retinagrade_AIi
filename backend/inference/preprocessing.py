"""Image preprocessing for RetinaGrade AI (CPU).

Where the values come from
--------------------------
* ``EfficientNetB0_UWF_final.pt`` stores its own training ``config``:
  ``image_size = 512``, ``normalization = 'ImageNet mean/std'``,
  ``pretrained = 'ImageNet'``. Those values are read from the checkpoint at
  load time and are used as-is.
* ``EfficientNetB0_CFP_final.pth`` stores **no** preprocessing configuration.
  The documented default below is therefore an assumption, not a fact read
  from the checkpoint. The sibling UWF checkpoint shows the same project used
  ImageNet normalisation with ImageNet-pretrained weights, so ImageNet
  statistics are assumed; the CFP input size defaults to 224 (the canonical
  EfficientNet-B0 input) because the CFP checkpoint does not record a size.
  Both are overridable per modality and are reported in
  :func:`describe_preprocessing`.

Inference-time transforms are deliberately minimal: the checkpoints were
trained with ``RandomHorizontalFlip(p=0.5) + RandomRotation(10)``, which is
**training-only** augmentation and must not be applied here.
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

from .exceptions import InvalidImageError, ModelNotFoundError, UnsupportedModalityError

logger = logging.getLogger(__name__)

ImageSource = Union[Image.Image, bytes, bytearray, BinaryIO]

#: Formats accepted from the API client.
SUPPORTED_FORMATS = ("JPEG", "PNG", "BMP", "TIFF", "WEBP")

#: ImageNet channel statistics (torchvision EfficientNet-B0 was initialised
#: from ImageNet weights, confirmed by ``config['pretrained'] = 'ImageNet'``).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 224
IMAGENET_INPUT_SIZE = 224

#: Per-modality defaults. ``source`` documents where the values came from.
_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "cfp": {
        "image_size": DEFAULT_IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "source": "assumed (CFP checkpoint stores no preprocessing config)",
    },
    "uwf": {
        "image_size": 512,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "source": "read from checkpoint config",
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

    def describe(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "image_size": self.image_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "normalization": "ImageNet mean/std",
            "color_space": "RGB",
            "resize": f"bilinear resize to ({self.image_size}, {self.image_size})",
            "training_augmentation_applied": False,
            "config_source": self.source,
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

    ``checkpoint_config`` is the checkpoint's stored ``config`` dict when one
    exists; it takes precedence over the documented defaults.
    """
    key = (modality or "").strip().lower()
    base = dict(_resolve_base(key))
    source = base["source"]

    if checkpoint_config:
        stored_size = checkpoint_config.get("image_size")
        if isinstance(stored_size, int) and stored_size > 0:
            base["image_size"] = int(stored_size)
            source = "read from checkpoint config (image_size=%d)" % stored_size
        normalization = str(checkpoint_config.get("normalization", "")).lower()
        if normalization and ("imagenet" not in normalization):
            logger.warning(
                "Checkpoint %s reports normalisation %r which is not ImageNet; "
                "falling back to ImageNet statistics and this may be wrong.",
                key,
                normalization,
            )

    return PreprocessConfig(
        image_size=int(base["image_size"]),
        mean=tuple(base["mean"]),
        std=tuple(base["std"]),
        source=source,
        modality=key,
    )


def with_image_size(config: PreprocessConfig, image_size: int) -> PreprocessConfig:
    """Return a copy of ``config`` with a different input size."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return replace(config, image_size=image_size, source=config.source + " (image_size overridden)")


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
