"""Verification script for the real RetinaGrade AI checkpoints (CPU only).

Loads both models, prints verified facts, and does NOT fabricate predictions.
Image inference is only attempted when you pass a real image path.

Usage (from ``backend/``)::

    .venv\\Scripts\\python.exe test_models.py
    .venv\\Scripts\\python.exe test_models.py --image ..\\path\\to\\retina.jpg --modality cfp
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.exceptions import InferenceError  # noqa: E402
from inference.model_loader import (  # noqa: E402
    ARCHITECTURE_NAME,
    CPU_DEVICE,
    MODEL_SPECS,
    manager,
)
from inference.preprocessing import get_config  # noqa: E402
from inference.predictor import predict  # noqa: E402

LINE = "=" * 70
THIN = "-" * 70


def _header(label: str) -> None:
    print()
    print(label)
    print(THIN)


def _print_model(label: str, modality: str) -> bool:
    spec = MODEL_SPECS[modality]
    print(f"Checkpoint file: {spec.checkpoint.name}")
    print(f"Checkpoint exists: {spec.checkpoint.is_file()}")
    if not spec.checkpoint.is_file():
        print(f"ERROR: {spec.display_name} checkpoint missing at {spec.checkpoint}")
        print("Loaded successfully: NO")
        return False

    try:
        loaded = manager.get(modality)
    except InferenceError as exc:
        print(f"Checkpoint type: {type(exc).__name__}")
        print(f"Load error: {exc}")
        print("Loaded successfully: NO")
        return False

    model = loaded.model
    num_binary = loaded.num_binary_heads
    print(f"Checkpoint type: {type(loaded.model).__name__} (state_dict checkpoint, strict=True)")
    print(f"Head layout: {spec.head_layout}")
    print(f"Architecture: {ARCHITECTURE_NAME} + {model.__class__.__name__}")
    print(f"Output classes: {loaded.num_classes}")
    if num_binary:
        print(f"Classifier: dr_head = {model.dr_head}")
        print(f"Auxiliary binary heads: {num_binary} x nn.Linear(1280, 1) (unnamed in checkpoint)")
    else:
        print(f"Classifier: classifier.1 = {model.classifier[1]}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print(f"Device: {loaded.device}")
    print(f"Eval mode: {not model.training}")
    print(f"A training flag on: {model.training}")
    print(f"Parameters not on CPU: {sum(1 for p in model.parameters() if p.device.type != 'cpu')}")
    meta = loaded.metadata
    config = meta.get("training_config", {})
    print(f"Checkpoint metadata: epoch={meta.get('epoch')} best_qwk={meta.get('best_qwk')}")
    if config:
        print(
            "Training config: "
            f"image_size={config.get('image_size')} num_classes={config.get('num_classes')} "
            f"normalization={config.get('normalization')}"
        )
    print("Loaded successfully: YES")
    return True


def _run_image(image: Optional[Path], modality: str) -> None:
    _header("REAL IMAGE INFERENCE")
    if image is None:
        print("No --image supplied: image inference NOT tested.")
        print("Loading and CPU placement were verified above; a real retinal image")
        print("is still required to verify end-to-end inference.")
        return
    if not image.is_file():
        print(f"Image not found: {image}")
        print("Image inference NOT tested.")
        return
    try:
        result = predict(image.read_bytes(), modality)
    except InferenceError as exc:
        print(f"Inference failed: {exc}")
        return
    print(f"Image: {image}")
    print(f"Modality: {result.modality}")
    print(f"Label: {result.label} (class {result.predicted_class})")
    print(f"Confidence: {result.confidence}")
    print(f"Probabilities: {result.probabilities}")
    print(f"Device: {result.device}")


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Verify the real RetinaGrade AI models on CPU.")
    parser.add_argument("--image", type=Path, default=None, help="real retinal image to test (optional)")
    parser.add_argument(
        "--modality",
        default="cfp",
        choices=sorted(MODEL_SPECS),
        help="modality of --image (default: cfp)",
    )
    args = parser.parse_args(argv)

    print(LINE)
    print("RETINAGRADE AI - MODEL VERIFICATION (CPU ONLY)")
    print(LINE)
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    print(f"PyTorch: {torch.__version__}")
    print(f"TorchVision: {__import__('torchvision').__version__}")
    print(f"Device used: {CPU_DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA build in this PyTorch: {torch.version.cuda}")

    _header("CFP MODEL")
    cfp_ok = _print_model("CFP", "cfp")

    _header("UWF MODEL")
    uwf_ok = _print_model("UWF", "uwf")

    _header("PREPROCESSING IN EFFECT")
    for modality, spec in MODEL_SPECS.items():
        config = get_config(modality)
        print(f"{spec.display_name}: size={config.image_size} mean={config.mean} std={config.std}")
        print(f"     source: {config.source}")

    _header("CUDA CHECK")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device in use: {torch.device('cpu')}")
    print(f"Models loaded on CPU: {manager.is_ready()}")

    _run_image(args.image, args.modality)

    print()
    print(THIN)
    print(f"RESULT: CFP={'OK' if cfp_ok else 'FAILED'}  UWF={'OK' if uwf_ok else 'FAILED'}")
    return 0 if (cfp_ok and uwf_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
