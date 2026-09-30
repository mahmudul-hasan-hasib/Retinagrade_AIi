"""Load the real checkpoints through ModelManager and verify the result.

This is the weight-loading verification step. It answers: *do the real
checkpoint files actually load into the reconstructed architectures, on the CPU,
with ``strict=True``, and do the heads have the expected widths?*

It performs no image prediction. No image is read, no real input is decoded, and
the only forward pass is the zero-filled structural probe the loader itself
runs to confirm output dimensions.

Checks performed per modality:

1. the checkpoint is loaded via :class:`ModelManager`
2. the result is an ``nn.Module``
3. ``model.training`` is ``False`` (eval mode)
4. the device is CPU - parameters *and* buffers
5. ``load_state_dict(strict=True)`` succeeded with no missing/unexpected keys
6. the number of parameters is printed
7. the final classification output width is printed and asserted

Usage (from ``backend/``)::

    .venv\\Scripts\\python.exe -m inference.test_model_loading

Exit code 0 means every check passed. Any failure prints the exact error and
exits non-zero without attempting a workaround.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # allow `python inference/test_model_loading.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from inference.exceptions import InferenceError  # noqa: E402
from inference.model_loader import (  # noqa: E402
    CPU_DEVICE,
    LoadedModel,
    MODEL_SPECS,
    ModelManager,
)

EXPECTED_NUM_CLASSES = 5
EXPECTED_CFP_LESION_HEADS = 7

LINE = "-" * 29
WIDE = "=" * 78


class CheckFailed(RuntimeError):
    """A verification assertion did not hold."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def _all_tensors_on_cpu(model: nn.Module) -> bool:
    tensors = [p for p in model.parameters()] + [b for b in model.buffers()]
    return all(t.device.type == "cpu" for t in tensors)


def _classifier_width(loaded: LoadedModel) -> Optional[int]:
    """Read the classification width straight off the head that owns it."""
    heads = {
        name: module
        for name, module in loaded.model.named_modules()
        if isinstance(module, nn.Linear) and module.out_features > 1
    }
    widths = {module.out_features for module in heads.values()}
    if len(widths) != 1:
        raise CheckFailed(
            f"expected exactly one multi-class head width, found {sorted(widths)} "
            f"across {sorted(heads)}"
        )
    return widths.pop()


def verify_cfp(loaded: LoadedModel) -> Dict[str, Any]:
    """TASK 4: assert the CFP head widths, including the auxiliary heads."""
    model = loaded.model

    # The DR classification head.
    _require(
        hasattr(model, "dr_head"),
        "CFP model has no 'dr_head'; this is not the multi-head architecture",
    )
    _require(
        isinstance(model.dr_head, nn.Linear),
        f"CFP dr_head should be a bare nn.Linear, got {type(model.dr_head).__name__}",
    )
    _require(
        model.dr_head.out_features == EXPECTED_NUM_CLASSES,
        f"CFP dr_head.out_features == {model.dr_head.out_features}, "
        f"expected {EXPECTED_NUM_CLASSES}",
    )
    _require(
        model.dr_head.in_features == 1280,
        f"CFP dr_head.in_features == {model.dr_head.in_features}, expected 1280",
    )

    # The auxiliary binary heads: count is verified, meaning is NOT.
    lesion_heads = getattr(model, "lesion_heads", None)
    _require(
        isinstance(lesion_heads, nn.ModuleList),
        f"CFP lesion_heads should be an nn.ModuleList, got {type(lesion_heads).__name__}",
    )
    count = len(lesion_heads)
    _require(
        count == EXPECTED_CFP_LESION_HEADS,
        f"CFP lesion_heads contains {count} heads, expected {EXPECTED_CFP_LESION_HEADS}",
    )
    for index, head in enumerate(lesion_heads):
        _require(
            isinstance(head, nn.Linear) and head.out_features == 1 and head.in_features == 1280,
            f"CFP lesion_heads.{index} is not nn.Linear(1280 -> 1)",
        )

    return {
        "head": f"dr_head = nn.Linear({model.dr_head.in_features}, {model.dr_head.out_features})",
        "head_out_features": model.dr_head.out_features,
        "lesion_heads_count": count,
    }


def verify_uwf(loaded: LoadedModel) -> Dict[str, Any]:
    """TASK 4: assert the UWF classifier width at index 1."""
    model = loaded.model
    classifier = getattr(model, "classifier", None)
    _require(
        isinstance(classifier, nn.Sequential),
        f"UWF classifier should be an nn.Sequential, got {type(classifier).__name__}",
    )
    _require(
        len(classifier) == 2,
        f"UWF classifier should have 2 slots (Dropout, Linear), has {len(classifier)}",
    )
    _require(
        isinstance(classifier[0], nn.Dropout),
        f"UWF classifier[0] should be nn.Dropout, got {type(classifier[0]).__name__}",
    )
    linear = classifier[1]
    _require(
        isinstance(linear, nn.Linear),
        f"UWF classifier.1 should be an nn.Linear, got {type(linear).__name__}",
    )
    _require(
        linear.out_features == EXPECTED_NUM_CLASSES,
        f"UWF classifier.1.out_features == {linear.out_features}, "
        f"expected {EXPECTED_NUM_CLASSES}",
    )
    _require(
        linear.in_features == 1280,
        f"UWF classifier.1.in_features == {linear.in_features}, expected 1280",
    )

    return {
        "head": (
            f"classifier = Sequential({type(classifier[0]).__name__}"
            f"(p={classifier[0].p}), Linear({linear.in_features}, {linear.out_features}))"
        ),
        "head_out_features": linear.out_features,
    }


def verify_loaded(label: str, loaded: LoadedModel) -> Dict[str, Any]:
    """Run every per-modality check and print the block for that modality."""
    model = loaded.model

    print()
    print(f"{label} MODEL")
    print(LINE)

    # 2. it is an nn.Module
    _require(isinstance(model, nn.Module), f"{label}: not an nn.Module")
    print(f"Loaded: {'YES' if isinstance(model, nn.Module) else 'NO'}")

    # 3. eval mode
    _require(not model.training, f"{label}: model is still in training mode")
    eval_mode = not model.training
    print(f"Eval mode: {eval_mode}")

    # 4. CPU, parameters and buffers
    _require(str(loaded.device) == "cpu", f"{label}: device is {loaded.device}, expected cpu")
    _require(_all_tensors_on_cpu(model), f"{label}: some parameter or buffer is not on cpu")
    print(f"Device: {loaded.device}")

    # 5. strict state_dict load
    report = loaded.load_report
    _require(
        loaded.strict_load_succeeded,
        f"{label}: strict load_state_dict did not report a clean result "
        f"(missing={report.get('missing_keys')}, unexpected={report.get('unexpected_keys')})",
    )
    print(
        f"Strict load: True (strict={report.get('strict')}, "
        f"{report.get('loaded_tensors')} tensors from '{report.get('source_key')}', "
        f"0 missing, 0 unexpected)"
    )

    # 6. parameter count
    print(f"Parameters: {loaded.num_parameters:,}")

    # 7. classification output width
    width = _classifier_width(loaded)
    _require(
        width == EXPECTED_NUM_CLASSES,
        f"{label}: classification width {width}, expected {EXPECTED_NUM_CLASSES}",
    )
    _require(
        loaded.num_classes == EXPECTED_NUM_CLASSES,
        f"{label}: loader reported num_classes {loaded.num_classes}, "
        f"expected {EXPECTED_NUM_CLASSES}",
    )
    print(f"Classification outputs: {width}")

    # Head-specific assertions.
    details = verify_cfp(loaded) if loaded.spec.head_layout == "multi_head" else verify_uwf(loaded)
    print(f"Classifier: {details['head']}")
    if "lesion_heads_count" in details:
        count = details["lesion_heads_count"]
        print(f"Lesion heads: {count}")
        print(
            "  NOTE: these are unnamed auxiliary outputs. The checkpoint stores no\n"
            "        label mapping for them, so they are NOT clinical findings and\n"
            "        must not be reported or named."
        )

    print(f"Head layout: {loaded.spec.head_layout}")
    print(f"Checkpoint: {loaded.spec.checkpoint.name}")

    return {
        "label": label,
        "device": str(loaded.device),
        "eval_mode": eval_mode,
        "num_parameters": loaded.num_parameters,
        "out_features": width,
        **details,
    }


def verify_cpu_only() -> Dict[str, Any]:
    """TASK 3: confirm CUDA is unavailable and nothing is placed on a GPU."""
    cuda_available = torch.cuda.is_available()
    print()
    print("DEVICE")
    print(LINE)
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"device = {CPU_DEVICE}")

    _require(cuda_available is False, f"CUDA is available ({cuda_available}); CPU-only violated")
    _require(CPU_DEVICE.type == "cpu", f"CPU_DEVICE is {CPU_DEVICE}, expected cpu")
    # Recorded, not enforced: a CPU-only build reports torch.version.cuda = None,
    # but the build string is metadata and is never used to place a tensor.
    print(f"torch.version.cuda: {torch.version.cuda} (build metadata, unused)")

    return {"cuda_available": cuda_available, "device": str(CPU_DEVICE)}


def main(argv: Optional[List[str]] = None) -> int:
    del argv  # no options; this script is a fixed verification sequence

    print(WIDE)
    print("RETINAGRADE AI - REAL WEIGHT LOADING VERIFICATION (CPU ONLY)")
    print(WIDE)
    print(f"  Python          {sys.version.split()[0]}")
    print(f"  torch           {torch.__version__}")
    print(f"  device          {CPU_DEVICE}")
    print(f"  checkpoints     {', '.join(spec.checkpoint.name for spec in MODEL_SPECS.values())}")

    try:
        cpu_info = verify_cpu_only()

        # 1. load both models through ModelManager.
        manager = ModelManager()
        manager.load_all()
        _require(manager.is_ready(), "ModelManager did not load every modality")

        results = [
            verify_loaded("CFP", manager.get("cfp")),
            verify_loaded("UWF", manager.get("uwf")),
        ]
    except CheckFailed as exc:
        print()
        print(WIDE)
        print("VERIFICATION FAILED")
        print(WIDE)
        print(f"  {type(exc).__name__}: {exc}")
        print()
        print("STOPPED. No workaround was attempted and nothing was faked.")
        return 1
    except InferenceError as exc:
        print()
        print(WIDE)
        print("VERIFICATION FAILED - MODEL LOADING ERROR")
        print(WIDE)
        print(f"  {type(exc).__name__}: {exc}")
        print()
        print("STOPPED. Report this error verbatim; do not relax strict=True.")
        traceback.print_exc()
        return 1
    except Exception as exc:  # noqa: BLE001
        print()
        print(WIDE)
        print("VERIFICATION FAILED - UNEXPECTED ERROR")
        print(WIDE)
        print(f"  {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    cfp, uwf = results
    print()
    print(WIDE)
    print("SUMMARY")
    print(WIDE)
    print(f"{'':22}{'CFP':<28}UWF")
    print(f"{'Loaded':22}{'YES':<28}YES")
    print(f"{'Device':22}{cfp['device']:<28}{uwf['device']}")
    print(f"{'Eval mode':22}{str(cfp['eval_mode']):<28}{uwf['eval_mode']}")
    print(f"{'Parameters':22}{cfp['num_parameters']:>13,}    {uwf['num_parameters']:>13,}")
    print(f"{'Classification outputs':22}{cfp['out_features']:>13}    {uwf['out_features']:>13}")
    print(f"{'Lesion heads':22}{cfp.get('lesion_heads_count', 'n/a'):>13}    {'n/a':>13}")
    print()
    print(f"CUDA available: {cpu_info['cuda_available']}")

    print()
    print("FINAL REPORT")
    print(LINE)
    # Reaching this point means every assertion above held.
    print("- CFP weights loaded successfully: YES")
    print("- UWF weights loaded successfully: YES")
    print(f"- CFP device: {cfp['device']}")
    print(f"- UWF device: {uwf['device']}")
    print(f"- CFP output classes: {cfp['out_features']}")
    print(f"- UWF output classes: {uwf['out_features']}")
    print(f"- CFP lesion heads count: {cfp.get('lesion_heads_count', 'n/a')}")
    print(f"- Eval mode status: CFP={cfp['eval_mode']}, UWF={uwf['eval_mode']}")
    print(f"- CUDA availability: {cpu_info['cuda_available']}")
    print("- Any error: none")

    print()
    print("Both real checkpoints loaded into the reconstructed architectures with")
    print("strict=True, are in eval mode, and every parameter and buffer is on the CPU.")
    print("No image was read and no prediction was made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
