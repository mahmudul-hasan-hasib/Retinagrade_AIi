"""Run the real RetinaGrade AI models on a real image and verify the output.

This is the Part 4A inference test. It does exactly one thing: load a real
image, run the matching real checkpoint on the CPU, print the prediction and
assert the output contract. It deliberately implements no API, no Grad-CAM and
no metrics.

Usage (from ``backend/``)::

    .venv\\Scripts\\python.exe test_prediction.py --image "test_images/cfp_test.jpg"  --modality cfp
    .venv\\Scripts\\python.exe test_prediction.py --image "test_images/uwf_test.jpg"  --modality uwf

No fake or synthetic image is ever generated here. If ``test_images`` is empty
the script says so and exits non-zero without claiming inference was tested.

What it verifies (TASK 4)
--------------------------
1. exactly 5 probabilities
2. every probability is in [0, 1]
3. the probabilities sum to ~1
4. ``predicted_class`` indexes the highest probability
5. ``confidence`` equals that highest probability
6. the device is CPU
7. CUDA is not used
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.exceptions import InferenceError  # noqa: E402
from inference.model_loader import (  # noqa: E402
    ARCHITECTURE_NAME,
    CPU_DEVICE,
    MODEL_SPECS,
    SUPPORTED_MODALITIES,
    manager,
)
from inference.preprocessing import (  # noqa: E402
    CHECKPOINT_IMAGE_SIZE,
    PreprocessConfig,
    get_config,
    with_image_size,
)
from inference.predictor import EXPECTED_CLASS_COUNT, PROBABILITY_KEYS, predict  # noqa: E402

LINE = "=" * 72
THIN = "-" * 72

#: Sum-of-probabilities tolerance. ``predict`` rounds each value to 6 decimals,
#: so five values can drift by at most 2.5e-6 from an exact 1.0. 1e-4 is a
#: comfortable margin that still catches any real error.
SUM_TOLERANCE = 1e-4

TEST_IMAGES_DIR = Path(__file__).resolve().parent / "test_images"


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _header(label: str) -> None:
    print()
    print(label)
    print(THIN)


def _check(name: str, passed: bool, detail: str = "") -> bool:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    return passed


def _is_cuda_free() -> Tuple[bool, str]:
    """True when this process cannot and did not use CUDA."""
    if torch.cuda.is_available():
        return False, "torch.cuda.is_available() is True on this machine"
    return True, f"torch.cuda.is_available()=False, torch.version.cuda={torch.version.cuda}"


# ---------------------------------------------------------------------------
# The seven TASK 4 assertions
# ---------------------------------------------------------------------------


def verify_result(payload: Dict[str, Any]) -> bool:
    """Assert the output contract on a prediction payload.

    Returns True only when all seven checks pass.
    """
    probabilities: Dict[str, float] = payload["probabilities"]
    values: List[float] = list(probabilities.values())
    predicted_class: int = payload["predicted_class"]
    confidence: float = payload["confidence"]

    print()
    print("OUTPUT CONTRACT CHECKS")
    print(THIN)

    results = [
        _check(
            "exactly 5 probabilities",
            len(probabilities) == EXPECTED_CLASS_COUNT,
            f"got {len(probabilities)}: {list(probabilities)}",
        ),
        _check(
            "probability keys match the DR class mapping",
            tuple(probabilities) == PROBABILITY_KEYS,
            f"expected {list(PROBABILITY_KEYS)}",
        ),
        _check(
            "every probability in [0, 1]",
            all(0.0 <= value <= 1.0 for value in values),
            f"min={min(values):.6f} max={max(values):.6f}" if values else "no values",
        ),
        _check(
            "probabilities sum to ~1",
            abs(sum(values) - 1.0) <= SUM_TOLERANCE,
            f"sum={sum(values):.9f} (tolerance {SUM_TOLERANCE})",
        ),
        _check(
            "predicted_class is in range",
            0 <= predicted_class < len(values),
            f"predicted_class={predicted_class}",
        ),
        _check(
            "predicted_class has the highest probability",
            bool(values) and max(range(len(values)), key=lambda i: values[i]) == predicted_class,
            f"argmax index={max(range(len(values)), key=lambda i: values[i]) if values else None}, "
            f"predicted_class={predicted_class}",
        ),
        _check(
            "confidence equals the highest probability",
            bool(values) and abs(confidence - max(values)) <= 1e-9,
            f"confidence={confidence}, max={max(values) if values else None}",
        ),
        _check(
            "label matches predicted_class",
            payload["label"] == PROBABILITY_KEYS[predicted_class],
            f"label={payload['label']!r}, class {predicted_class} -> {PROBABILITY_KEYS[predicted_class]!r}",
        ),
        _check(
            "device is CPU",
            payload["device"] == "cpu",
            f"device={payload['device']}",
        ),
    ]

    cuda_ok, cuda_detail = _is_cuda_free()
    results.append(
        _check("CUDA is not used", cuda_ok and payload["device"] == "cpu", cuda_detail)
    )

    print()
    print(f"CONTRACT: {'ALL CHECKS PASSED' if all(results) else 'ONE OR MORE CHECKS FAILED'}")
    return all(results)


# ---------------------------------------------------------------------------
# Environment / preprocessing report
# ---------------------------------------------------------------------------


def _print_environment() -> None:
    cuda_ok, cuda_detail = _is_cuda_free()
    print(f"Python:        {sys.version.split()[0]}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"TorchVision:   {__import__('torchvision').__version__}")
    print(f"Forced device: {CPU_DEVICE}")
    print(f"CUDA build:    {torch.version.cuda}")
    print(f"CUDA used:     NO ({cuda_detail})")


def _print_preprocessing(modality: str, config: "PreprocessConfig") -> None:
    spec = MODEL_SPECS[modality]

    print(f"Modality:      {spec.display_name}")
    print(f"Checkpoint:    {spec.checkpoint.name}")
    print(f"Architecture:  {ARCHITECTURE_NAME}")
    print(f"DR head:       {spec.head_layout}")
    print(f"Image size:    {config.image_size}")
    print(f"  confirmed by checkpoint: {config.verified}")
    print(f"Normalisation: mean={tuple(config.mean)} std={tuple(config.std)}")
    print("  (ImageNet mean/std)")
    print(f"Config source: {config.source}")

    if config.is_assumed:
        print()
        print("  ** ASSUMPTION WARNING **")
        print(f"  The {spec.display_name} checkpoint does not state these values, so they are")
        print("  a documented default, not checkpoint data. The sibling UWF checkpoint")
        print(f"  records image_size={CHECKPOINT_IMAGE_SIZE}. If this model was trained at a")
        print("  different size the logits are still produced but accuracy is not")
        print("  trustworthy. Re-run with --image-size to test another size.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _no_real_image(modality: str, image_arg: Optional[str]) -> int:
    print()
    print(THIN)
    print("INFERENCE NOT TESTED")
    print(THIN)
    if image_arg:
        print(f"No such file: {image_arg}")
    else:
        print("No --image was supplied.")
    print(f"Drop a real {modality.upper()} retinal image into:")
    print(f"  {TEST_IMAGES_DIR}")
    print("then re-run with --image <path> --modality " + modality)
    print()
    print("No image was invented or synthesised, so no prediction is reported.")
    return 2


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run a real RetinaGrade AI model on a real image (CPU only).",
    )
    parser.add_argument("--image", default=None, help="path to a real retinal image")
    parser.add_argument(
        "--modality",
        default="cfp",
        choices=sorted(SUPPORTED_MODALITIES),
        help="imaging modality of --image (default: cfp)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="override the model input size (needed for CFP, whose checkpoint does not record one)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print only the JSON prediction payload",
    )
    args = parser.parse_args(argv)

    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.image) if args.image else None

    # Resolve the model and the preprocessing config first: these work without
    # an image, and reporting them is useful even when no image was supplied.
    try:
        loaded = manager.get(args.modality)
        config = get_config(args.modality, loaded.metadata.get("training_config"))
        if args.image_size is not None:
            config = with_image_size(config, args.image_size)
    except InferenceError as exc:
        print(f"\nERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    if not args.json:
        print(LINE)
        print("RETINAGRADE AI - REAL IMAGE INFERENCE (CPU ONLY)")
        print(LINE)
        _print_environment()
        _header("MODEL AND PREPROCESSING")
        _print_preprocessing(args.modality, config)

    if image_path is None or not image_path.is_file():
        return _no_real_image(args.modality, args.image)

    try:
        payload = predict(image_path.read_bytes(), args.modality, config=config).to_dict()
    except InferenceError as exc:
        print(f"\nERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if verify_result(payload) else 1

    _header("PREDICTION")
    print(json.dumps(payload, indent=2))

    print()
    print(THIN)
    print(f"Image:      {image_path}")
    print(f"Result:     {payload['label']} (class {payload['predicted_class']}), "
          f"confidence {payload['confidence']}")
    print(f"Modality:   {payload['modality']}  Model: {payload['model']}")
    print(f"Device:     {payload['device']}  Image size: {payload['image_size']}")

    return 0 if verify_result(payload) else 1


if __name__ == "__main__":
    raise SystemExit(main())