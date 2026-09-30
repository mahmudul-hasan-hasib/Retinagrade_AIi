"""Domain exceptions for the RetinaGrade AI inference layer.

These are translated into HTTP responses in ``main.py``; they never carry
tracebacks or other internal details to the API client.
"""


class InferenceError(Exception):
    """Base class for every error raised by the inference layer."""

    code = "inference_error"
    http_status = 500


class ModelNotFoundError(InferenceError):
    """A checkpoint file is missing from ``backend/models``."""

    code = "model_not_found"
    http_status = 503


class CheckpointLoadError(InferenceError):
    """A checkpoint exists but cannot be deserialised safely."""

    code = "checkpoint_load_error"
    http_status = 500


class ArchitectureMismatchError(InferenceError):
    """A checkpoint does not match the architecture required by this project."""

    code = "architecture_mismatch"
    http_status = 500


class UnsupportedModalityError(InferenceError):
    """The requested modality is not one of the supported imaging modalities."""

    code = "unsupported_modality"
    http_status = 400


class InvalidImageError(InferenceError):
    """The uploaded image is empty, unsupported, or cannot be decoded."""

    code = "invalid_image"
    http_status = 422


class InferenceFailedError(InferenceError):
    """An unexpected failure occurred while running the network."""

    code = "inference_failed"
    http_status = 500
