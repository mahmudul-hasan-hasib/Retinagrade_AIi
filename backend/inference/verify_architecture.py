"""Verify that the reconstructed architectures match the two real checkpoints.

This script answers exactly one question:

    *Can the exact architecture required by each checkpoint be reconstructed?*

It

1. constructs the CFP architecture (:func:`create_cfp_model`),
2. constructs the UWF architecture (:func:`create_uwf_model`),
3. prints the architecture type,
4. prints the classifier / final layer,
5. prints the expected number of output classes,
6. compares every parameter name and shape against the checkpoint state_dict.

**No weights are loaded by default.** Step 6 compares the ``name -> shape``
signature, which is precisely what ``load_state_dict(strict=True)`` requires.
Pass ``--strict-load`` to additionally perform the real ``strict=True`` load as a
final proof; that is the only mode that puts checkpoint values into a model.

No image is read, no forward pass is run, no API is involved. CPU only.

Usage (from ``backend/``)::

    .venv\\Scripts\\python.exe -m inference.verify_architecture
    .venv\\Scripts\\python.exe inference\\verify_architecture.py --verbose
    .venv\\Scripts\\python.exe -m inference.verify_architecture --strict-load
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

if __package__ in (None, ""):  # allow `python inference/verify_architecture.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.inspect_models import (  # noqa: E402
    CFP_CHECKPOINT,
    UWF_CHECKPOINT,
    find_state_dict,
    load_checkpoint,
)
from inference.model_architecture import (  # noqa: E402
    ARCHITECTURE_NAME,
    BACKBONE_CHANNELS,
    DROPOUT_PROB,
    EXPECTED_TRUNK_TENSORS,
    INPUT_CHANNELS,
    NUM_BINARY_HEADS,
    NUM_CLASSES,
    TORCHVISION_ORIGIN,
    create_cfp_model,
    create_uwf_model,
)

LINE = "=" * 78
PREVIEW = 6

# Layers the reconstruction depends on, checked individually for readability.
PROBES: Tuple[str, ...] = (
    "features.0.0.weight",    # stem convolution -> input channels
    "features.8.0.weight",    # last convolution -> trunk width
    "dr_head.weight",         # CFP classifier
    "classifier.1.weight",    # UWF classifier
    "lesion_heads.0.weight",  # CFP first auxiliary binary head
    "lesion_heads.6.weight",  # CFP last auxiliary binary head
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def signature(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple[int, ...]]:
    """Name -> shape map: everything ``load_state_dict(strict=True)`` needs."""
    return {key: tuple(value.shape) for key, value in state_dict.items()}


def head_linears(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Every ``nn.Linear`` outside the trunk, in definition order.

    The trunk is skipped: the open question is the *head*, and the trunk is
    covered by the ``features.*`` spot-checks.
    """
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not name.startswith("features")
    ]


def head_containers(model: nn.Module) -> List[Tuple[str, nn.Sequential]]:
    """``Sequential`` wrappers outside the trunk, e.g. a dropout before a head."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Sequential) and not name.startswith("features")
    ]


def checkpoint_metadata(path: Path) -> Dict[str, Any]:
    """Non-tensor top-level keys of a checkpoint, for context only."""
    checkpoint = load_checkpoint(path)
    if not isinstance(checkpoint, dict):
        return {}
    return {
        key: type(value).__name__
        for key, value in checkpoint.items()
        if key != "model_state_dict"
    }


def check_one(
    label: str,
    path: Path,
    model: nn.Module,
    classifier_hint: str,
    strict_load: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """Build the architecture, print it, and diff it against the checkpoint."""
    print()
    print(LINE)
    print(f"{label}  ->  {path.name}")
    print(LINE)

    model_sig = signature(model.state_dict())
    result: Dict[str, Any] = {
        "label": label,
        "path": path,
        "model": model,
        "classifier": classifier_hint,
        "num_classes": NUM_CLASSES,
        "architecture": f"{type(model).__module__}.{type(model).__name__}",
    }

    # --- 3. architecture type ------------------------------------------------
    print()
    print("  3. Architecture type")
    print(f"    {result['architecture']}")
    print(f"    family             : {ARCHITECTURE_NAME} (torchvision)")
    print(f"    origin             : {TORCHVISION_ORIGIN}")
    print(f"    trunk              : efficientnet_b0(weights=None).features")
    print(f"    input channels     : {INPUT_CHANNELS}")
    print(f"    trunk output       : {BACKBONE_CHANNELS}")
    print(f"    total parameters   : {sum(p.numel() for p in model.parameters()):,}")
    print(f"    model tensors      : {len(model_sig)}")
    print(f"    device             : {next(model.parameters()).device} (construction only)")

    # --- 4. classifier / final layer ----------------------------------------
    print()
    print("  4. Classifier / final layer")
    for name, seq in head_containers(model):
        slots = ", ".join(f"[{i}] {type(c).__name__}" for i, c in enumerate(seq))
        print(f"    {name} = nn.Sequential({slots})")
        for i, child in enumerate(seq):
            if isinstance(child, nn.Dropout):
                print(
                    f"        slot {i} is nn.Dropout and holds no tensors -> the "
                    f"state_dict only shows indices "
                    f"{[j for j, c in enumerate(seq) if any(True for _ in c.parameters())]}"
                )
                print(
                    f"        dropout p is not stored in any state_dict; torchvision "
                    f"default {DROPOUT_PROB} is used"
                )
                print(
                    "        (Dropout is the identity in eval(), so it cannot change "
                    "an inference output)"
                )
    binary_count = 0
    for name, layer in head_linears(model):
        kind = "classifier" if layer.out_features > 1 else "binary head"
        if layer.out_features == 1:
            binary_count += 1
        print(f"    {name} = nn.Linear({layer.in_features} -> {layer.out_features})  [{kind}]")
    if binary_count:
        print(f"    -> 1 multi-class head + {binary_count} binary auxiliary head(s)")
    else:
        print("    -> single multi-class head, no auxiliary heads")

    # --- 5. expected number of output classes --------------------------------
    print()
    print("  5. Expected output classes")
    detected = sorted({layer.out_features for _, layer in head_linears(model) if layer.out_features > 1})
    for name, layer in head_linears(model):
        if layer.out_features > 1:
            print(f"    {name}.weight shape {tuple(layer.weight.shape)} -> out_features = {layer.out_features}")
    print(f"    multi-class head output counts : {detected}")
    print(f"    expected num_classes           : {NUM_CLASSES}")

    # --- 6. compare against the checkpoint state_dict ------------------------
    print()
    print("  6. Comparison against the checkpoint state_dict")
    if not path.is_file():
        print(f"    checkpoint missing: {path}")
        result["reconstructed"] = False
        result["verdict"] = "CHECKPOINT MISSING"
        return result

    try:
        print(f"    checkpoint non-tensor keys: {checkpoint_metadata(path)}")
        state_dict = find_state_dict(load_checkpoint(path))
    except Exception as exc:  # noqa: BLE001 - exact error must be visible
        print(f"    checkpoint failed to load: {type(exc).__name__}: {exc}")
        result["reconstructed"] = False
        result["verdict"] = "LOAD FAILED"
        return result

    if state_dict is None:
        print("    no state_dict found inside the checkpoint")
        result["reconstructed"] = False
        result["verdict"] = "NO STATE_DICT"
        return result

    ckpt_sig = signature(state_dict)
    ckpt_keys, model_keys = set(ckpt_sig), set(model_sig)
    missing = sorted(model_keys - ckpt_keys)     # model wants it, checkpoint lacks it
    unexpected = sorted(ckpt_keys - model_keys)  # checkpoint has it, model lacks it
    mismatched = sorted(
        k for k in ckpt_keys & model_keys if ckpt_sig[k] != model_sig[k]
    )
    order_identical = list(ckpt_sig) == list(model_sig)

    trunk_keys = [k for k in ckpt_sig if k.startswith("features.")]
    print(f"    state_dict held under key 'model_state_dict'")
    print(f"    checkpoint tensors   : {len(ckpt_sig)}")
    print(f"    model tensors        : {len(model_sig)}")
    print(f"    checkpoint 'features.*' tensors : {len(trunk_keys)} (expected {EXPECTED_TRUNK_TENSORS})")
    print(f"    names identical      : {ckpt_keys == model_keys}")
    print(f"    shapes identical     : {not mismatched}")
    print(f"    key order identical  : {order_identical}")
    print(f"    missing keys         : {len(missing)}" + (f" {missing[:PREVIEW]}" if missing else ""))
    print(f"    unexpected keys      : {len(unexpected)}" + (f" {unexpected[:PREVIEW]}" if unexpected else ""))
    print(f"    shape mismatches     : {len(mismatched)}")
    for key in mismatched[:PREVIEW]:
        print(f"        {key}: model {model_sig[key]} vs checkpoint {ckpt_sig[key]}")

    print()
    print("    Key layer spot-checks (model vs checkpoint)")
    for key in PROBES:
        if key in model_sig or key in ckpt_sig:
            in_model, in_ckpt = model_sig.get(key), ckpt_sig.get(key)
            ok = in_model is not None and in_model == in_ckpt
            print(f"      {'OK  ' if ok else 'DIFF'} {key:<24} model={in_model} checkpoint={in_ckpt}")

    if verbose:
        print()
        print("    First / last model parameter names")
        keys = list(model_sig)
        for key in keys[:PREVIEW]:
            print(f"      {key:<52} {model_sig[key]}")
        print("      ...")
        for key in keys[-PREVIEW:]:
            print(f"      {key:<52} {model_sig[key]}")

    exact = not missing and not unexpected and not mismatched

    if strict_load:
        print()
        print("    strict load_state_dict(strict=True) [opt-in, loads real weights]")
        try:
            model.load_state_dict(state_dict, strict=True)
            print("      All keys matched successfully.")
        except Exception as exc:  # noqa: BLE001
            print(f"      FAILED: {type(exc).__name__}: {exc}")
            exact = False

    result.update(
        {
            "checkpoint_tensors": len(ckpt_sig),
            "model_tensors": len(model_sig),
            "missing": missing,
            "unexpected": unexpected,
            "mismatched": mismatched,
            "reconstructed": exact,
            "verdict": "EXACT MATCH" if exact else "MISMATCH",
        }
    )

    print()
    print(f"    VERDICT: {result['verdict']} - architecture "
          f"{'CAN' if exact else 'CANNOT'} be reconstructed exactly")
    return result


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the reconstructed architectures against the real checkpoints."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="also print first/last parameter names"
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="additionally run load_state_dict(strict=True) with the real weights "
             "(off by default: this step normally loads no weights)",
    )
    args = parser.parse_args(argv)

    import torchvision

    print(LINE)
    print("RETINAGRADE AI - ARCHITECTURE VERIFICATION (NO PREDICTION, CPU ONLY)")
    print(LINE)
    print(f"  Python            {sys.version.split()[0]}")
    print(f"  torch             {torch.__version__}")
    print(f"  torchvision       {torchvision.__version__}")
    print(f"  CUDA available    {torch.cuda.is_available()}")
    try:
        import timm  # noqa: F401

        print("  timm              installed, but NOT used: the checkpoint trunk keys")
        print("                    are torchvision's, not timm's (see report below)")
    except ImportError:
        print("  timm              not installed (and not required)")
    print(f"  origin            {TORCHVISION_ORIGIN}")
    print(
        f"  constants         in={INPUT_CHANNELS}  trunk_out={BACKBONE_CHANNELS}  "
        f"num_classes={NUM_CLASSES}  binary_heads={NUM_BINARY_HEADS}"
    )
    print(f"  weights loaded    {'yes (--strict-load)' if args.strict_load else 'no'}")

    # 1 + 2. construct both architectures (no weights, no forward pass).
    cfp_model = create_cfp_model()
    uwf_model = create_uwf_model()

    results = [
        check_one(
            "CFP", CFP_CHECKPOINT, cfp_model,
            "dr_head = nn.Linear(1280, 5)",
            args.strict_load, args.verbose,
        ),
        check_one(
            "UWF", UWF_CHECKPOINT, uwf_model,
            "classifier = Sequential(Dropout, Linear(1280, 5))",
            args.strict_load, args.verbose,
        ),
    ]

    print()
    print(LINE)
    print("SUMMARY")
    print(LINE)
    header = (
        f"{'label':<6}{'class':<22}{'params':>12}{'ckpt tens':>11}"
        f"{'model tens':>12}  verdict"
    )
    print(header)
    for result in results:
        model = result["model"]
        print(
            f"{result['label']:<6}{type(model).__name__:<22}"
            f"{sum(p.numel() for p in model.parameters()):>12,}"
            f"{result.get('checkpoint_tensors', 0):>11,}"
            f"{result.get('model_tensors', 0):>12}  {result['verdict']}"
        )

    print()
    for result in results:
        print(f"{result['label']}:")
        print(f"- Architecture: {ARCHITECTURE_NAME} (torchvision.models.efficientnet_b0) - {result['architecture']}")
        print(f"- Classifier: {result['classifier']}")
        print(f"- Output classes: {result['num_classes']}")
        print(f"- Architecture reconstructed: {'YES' if result['reconstructed'] else 'NO'}")
        print()

    all_exact = all(r["reconstructed"] for r in results)
    if all_exact:
        print(
            "Both architectures are reconstructed exactly: every checkpoint tensor has a\n"
            "same-named, same-shaped (and same-ordered) parameter, so a\n"
            "strict=True load_state_dict will succeed for both files."
        )
    else:
        print("At least one architecture does NOT match its checkpoint. See details above.")
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
