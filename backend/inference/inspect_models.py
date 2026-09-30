"""Read-only structural inspection of the two real RetinaGrade AI checkpoints.

This script only *looks at* the files. It never reconstructs a network, never
guesses an architecture, never runs a forward pass and never writes, converts
or renames anything. Every number it prints is read out of the checkpoint.

Each checkpoint is opened read-only with::

    torch.load(path, map_location="cpu", weights_only=True)

``weights_only=True`` is PyTorch's safe unpickler. Both real checkpoints store
NumPy scalars and a ``TorchVersion`` string next to the tensors, so those exact
globals are allow-listed (see ``_SAFE_GLOBALS``). Unsafe loading
(``weights_only=False``) is never used. If a checkpoint cannot be loaded the
exact error is printed and the run stops.

Usage (from ``backend/``)::

    .venv\\Scripts\\python.exe -m inference.inspect_models
    .venv\\Scripts\\python.exe inference\\inspect_models.py --all-keys
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.serialization import safe_globals

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "models"

CFP_CHECKPOINT = MODELS_DIR / "EfficientNetB0_CFP_final.pth"
UWF_CHECKPOINT = MODELS_DIR / "EfficientNetB0_UWF_final.pt"

TARGETS: Tuple[Tuple[str, Path], ...] = (
    ("CFP", CFP_CHECKPOINT),
    ("UWF", UWF_CHECKPOINT),
)

# Keys that may hold the tensors when the file is a checkpoint dictionary.
STATE_DICT_KEYS: Tuple[str, ...] = (
    "model_state_dict",
    "state_dict",
    "model",
    "net",
    "network",
    "weights",
)

PREVIEW = 8
LINE = "=" * 78
THIN = "-" * 78


def _build_safe_globals() -> List[Any]:
    """Allow-list only the exact extra globals the real checkpoints need.

    These are the NumPy scalar/dtype reconstructors used by ``np.float64`` and
    friends, plus ``TorchVersion``. Everything else in the files is a tensor.
    """
    candidates: List[Any] = [
        getattr(np, "_core", np).multiarray.scalar,
        np.dtype,
        np.ndarray,
        torch.torch_version.TorchVersion,
    ]
    dtypes = getattr(np, "dtypes", None)
    if dtypes is not None:
        for name in (
            "Float16DType",
            "Float32DType",
            "Float64DType",
            "Int8DType",
            "Int16DType",
            "Int32DType",
            "Int64DType",
            "UInt8DType",
            "BoolDType",
            "StrDType",
        ):
            obj = getattr(dtypes, name, None)
            if obj is not None:
                candidates.append(obj)
    return candidates


SAFE_GLOBALS = _build_safe_globals()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_checkpoint(path: Path) -> Any:
    """Load one checkpoint onto the CPU with the safe unpickler.

    The original exception is allowed to propagate unchanged so that
    ``main`` can print the exact error and stop.
    """
    with safe_globals(SAFE_GLOBALS):
        return torch.load(path, map_location="cpu", weights_only=True)


# --------------------------------------------------------------------------
# generic structural analysis (no architecture assumptions)
# --------------------------------------------------------------------------


def is_state_dict(obj: Any) -> bool:
    """True if ``obj`` is itself a flat mapping of name -> tensor."""
    return (
        isinstance(obj, dict)
        and len(obj) > 0
        and all(torch.is_tensor(v) for v in obj.values())
    )


def find_state_dict(checkpoint: Any) -> Optional[Dict[str, torch.Tensor]]:
    """Return the tensor mapping, whether the file is a bare state_dict or a
    checkpoint dictionary that contains one. ``None`` if there is none."""
    if is_state_dict(checkpoint):
        return checkpoint
    if isinstance(checkpoint, dict):
        for key in STATE_DICT_KEYS:
            value = checkpoint.get(key)
            if is_state_dict(value):
                return value
    return None


def describe_type(obj: Any) -> str:
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def format_value(value: Any, limit: int = 160) -> str:
    """Readable one-line description of a non-tensor checkpoint value."""
    if torch.is_tensor(value):
        return f"Tensor{tuple(value.shape)} {value.dtype}"
    if isinstance(value, dict):
        return f"dict(len={len(value)})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def group_by_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
    """Group state_dict keys by their first dotted segment."""
    groups: Dict[str, List[str]] = {}
    for key in state_dict:
        groups.setdefault(key.split(".", 1)[0], []).append(key)
    return groups


def find_leaf_linears(state_dict: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    """Every ``nn.Linear``-shaped leaf, found without assuming any architecture.

    A 2-D ``<path>.weight`` is treated as a leaf Linear layer when nothing else
    in the state_dict lives under ``<path>.`` apart from its own bias - i.e. no
    sub-module is nested inside it. Convolution (4-D) and BatchNorm (1-D)
    tensors are skipped, and the same rule works for a plain head
    (``dr_head``) as well as for a Linear inside a Sequential/ModuleList
    (``classifier.1``, ``lesion_heads.3``).
    """
    leaves: List[Dict[str, Any]] = []
    for key, value in state_dict.items():
        if not key.endswith(".weight") or not torch.is_tensor(value) or value.dim() != 2:
            continue
        path = key[: -len(".weight")]
        own_keys = (key, path + ".bias")
        if any(
            other.startswith(path + ".") and other not in own_keys
            for other in state_dict
        ):
            continue
        bias = state_dict.get(path + ".bias")
        segments = path.split(".")
        leaves.append(
            {
                "path": path,
                "weight": tuple(value.shape),
                "bias": tuple(bias.shape) if torch.is_tensor(bias) else None,
                "in_features": int(value.shape[1]),
                "out_features": int(value.shape[0]),
                "top_prefix": segments[0],
                "container": ".".join(segments[:-1]) or None,
                "index": int(segments[-1]) if segments[-1].isdigit() else None,
            }
        )
    return leaves


def describe_container(state_dict: Dict[str, torch.Tensor], container: str) -> Dict[str, Any]:
    """Observed structure of a module container, from key names only.

    ``ModuleList`` indices are always contiguous from 0; a container whose
    observed indices do not start at 0 therefore holds a parameter-free module
    (e.g. a dropout) before its first tensor-bearing slot. That is reported as
    an observation only - no layer type is asserted.
    """
    indices = sorted(
        {
            int(m.group(1))
            for m in (
                re.match(rf"^{re.escape(container)}\.(\d+)\.", key) for key in state_dict
            )
            if m
        }
    )
    contiguous = bool(indices) and indices == list(range(len(indices)))
    return {
        "container": container,
        "indices": indices,
        "contiguous_from_zero": contiguous,
        "note": (
            "contiguous indices 0..n-1 (ModuleList/ModuleDict pattern)"
            if contiguous
            else "indices do not start at 0, so slot(s) before them hold no tensors"
        ),
    }


def conv_summary(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """First and last 4-D convolution in key order, plus the widest one."""
    convs: List[Tuple[str, Tuple[int, ...]]] = [
        (k, tuple(v.shape))
        for k, v in state_dict.items()
        if k.endswith(".weight") and torch.is_tensor(v) and v.dim() == 4
    ]
    if not convs:
        return {"first": None, "last": None, "widest": None, "count": 0}
    return {
        "first": convs[0],
        "last": convs[-1],
        "widest": max(convs, key=lambda item: item[1][1]),
        "count": len(convs),
    }


def stage_indices(state_dict: Dict[str, torch.Tensor], prefix: str) -> List[int]:
    """Numeric second-level indices found under ``<prefix>.<n>.``."""
    found = set()
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    for key in state_dict:
        match = pattern.match(key)
        if match:
            found.add(int(match.group(1)))
    return sorted(found)


def detect_classes(heads: List[Dict[str, Any]]) -> Tuple[Optional[int], str]:
    """Read the class count off the Linear layers that sit outside the trunk.

    A class classifier must have more than one output. If exactly one such
    layer exists its output count is the class count; if none or several
    exist, nothing is assumed and the reason is returned instead.
    """
    if not heads:
        return None, "no Linear layer outside the trunk was found"

    multi = [h for h in heads if h["out_features"] > 1]
    binary = [h for h in heads if h["out_features"] == 1]

    if len(multi) == 1:
        head = multi[0]
        basis = f"'{head['path']}.weight' is the only Linear outside the trunk with >1 output: {head['weight']}"
        if binary:
            basis += f"; plus {len(binary)} single-output (binary) Linear layer(s)"
        return head["out_features"], basis

    if not multi:
        return None, (
            f"only {len(binary)} single-output (binary) Linear layer(s) outside the "
            "trunk; no multi-class classifier detected"
        )

    return None, (
        f"ambiguous: {len(multi)} Linear layers outside the trunk have >1 output "
        f"({[h['out_features'] for h in multi]}); not assuming which is the classifier"
    )


def analyse(path: Path) -> Dict[str, Any]:
    """Load and structurally describe one checkpoint. Read-only."""
    before = path.stat()
    report: Dict[str, Any] = {
        "path": path,
        "size_bytes": before.st_size,
        "mtime": before.st_mtime,
    }

    checkpoint = load_checkpoint(path)

    report["checkpoint_type"] = describe_type(checkpoint)
    report["is_nn_module"] = isinstance(checkpoint, torch.nn.Module)
    report["is_tensor"] = torch.is_tensor(checkpoint)
    report["is_bare_state_dict"] = is_state_dict(checkpoint)
    report["is_checkpoint_dict"] = isinstance(checkpoint, dict) and not report["is_bare_state_dict"]

    if isinstance(checkpoint, dict):
        report["top_level_keys"] = list(checkpoint.keys())
        report["top_level_summary"] = {k: format_value(v) for k, v in checkpoint.items()}
    else:
        report["top_level_keys"] = None
        report["top_level_summary"] = {}

    state_dict = find_state_dict(checkpoint)
    if state_dict is None:
        report["state_dict_found"] = False
        after = path.stat()
        report["unchanged"] = (
            after.st_size == before.st_size and after.st_mtime == before.st_mtime
        )
        return report

    report["state_dict_found"] = True
    report["state_dict_source_key"] = (
        "the file itself" if report["is_bare_state_dict"] else _source_key(checkpoint, state_dict)
    )
    report["num_tensors"] = len(state_dict)
    report["num_parameters"] = int(sum(v.numel() for v in state_dict.values()))
    report["dtypes"] = sorted({str(v.dtype) for v in state_dict.values()})
    report["devices_after_load"] = sorted({str(v.device) for v in state_dict.values()})

    keys = list(state_dict)
    report["first_keys"] = [(k, tuple(state_dict[k].shape)) for k in keys[:PREVIEW]]
    report["last_keys"] = [(k, tuple(state_dict[k].shape)) for k in keys[-PREVIEW:]]
    report["all_keys"] = [(k, tuple(state_dict[k].shape)) for k in keys]

    groups = group_by_prefix(state_dict)
    report["groups"] = {p: len(k) for p, k in groups.items()}

    convs = conv_summary(state_dict)
    report["conv"] = convs

    stem = convs["first"]
    report["input_channels"] = int(stem[1][1]) if stem else None
    report["trunk_output_channels"] = int(convs["last"][1][0]) if convs["last"] else None
    report["trunk_prefix"] = stem[0].split(".")[0] if stem else None
    report["trunk_stages"] = stage_indices(state_dict, report["trunk_prefix"] or "")

    # Every nn.Linear-shaped leaf, split into "inside the trunk" and "heads".
    all_leaves = find_leaf_linears(state_dict)
    report["leaf_linear_count"] = len(all_leaves)
    heads = [leaf for leaf in all_leaves if leaf["top_prefix"] != report["trunk_prefix"]]
    report["heads"] = heads
    report["containers"] = [
        describe_container(state_dict, container)
        for container in sorted({h["container"] for h in heads if h["container"]})
    ]
    report["num_classes"], report["num_classes_basis"] = detect_classes(heads)

    after = path.stat()
    report["unchanged"] = after.st_size == before.st_size and after.st_mtime == before.st_mtime

    return report


def _source_key(checkpoint: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> str:
    for key in STATE_DICT_KEYS:
        if checkpoint.get(key) is state_dict:
            return key
    return "unknown"


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _q(label: str, value: Any) -> None:
    print(f"  {label:<28} {value}")


def print_report(label: str, report: Dict[str, Any], show_all: bool) -> None:
    print()
    print(LINE)
    print(f"{label}  ->  {report['path'].name}")
    print(LINE)
    _q("file size", f"{report['size_bytes']:,} bytes")

    ctype = report["checkpoint_type"]
    print()
    print("  1. Python type of loaded checkpoint")
    _q("type", ctype)
    _q("is torch.nn.Module", report["is_nn_module"])
    _q("is a single tensor", report["is_tensor"])

    print()
    print("  2. Is it a state_dict?")
    _q("bare state_dict", report["is_bare_state_dict"])

    print()
    print("  3. Is it a checkpoint dictionary?")
    _q("checkpoint dict", report["is_checkpoint_dict"])

    print()
    print("  4. Available top-level keys")
    if report["top_level_keys"] is None:
        _q("keys", "n/a (not a mapping)")
    elif not report["top_level_keys"]:
        _q("keys", "(empty dict)")
    else:
        for key in report["top_level_keys"]:
            _q(f"  '{key}'", report["top_level_summary"].get(key, ""))

    print()
    print("  5. state_dict availability")
    _q("state_dict found", report["state_dict_found"])
    if not report["state_dict_found"]:
        print()
        print("  (items 6-10 need a state_dict; stopping report for this file)")
        _q("file unchanged by read", report["unchanged"])
        return
    _q("held under key", report["state_dict_source_key"])
    _q("tensors in state_dict", report["num_tensors"])
    _q("total parameters", f"{report['num_parameters']:,}")
    _q("dtypes", report["dtypes"])
    _q("devices after map_location", report["devices_after_load"])

    print()
    print("  6. First few parameter names")
    for name, shape in report["first_keys"]:
        print(f"      {name:<58} {shape}")

    print()
    print("  7. Last few parameter names")
    for name, shape in report["last_keys"]:
        print(f"      {name:<58} {shape}")

    if show_all:
        print()
        print("  -- every state_dict key (name -> shape) --")
        for name, shape in report["all_keys"]:
            print(f"      {name:<58} {shape}")

    print()
    print("  8. Shapes of important layers (read from the checkpoint)")
    conv = report["conv"]
    _q("4-D conv layers", conv["count"])
    if conv["first"]:
        _q("first conv (stem)", f"{conv['first'][0]}  {conv['first'][1]}")
        _q("  -> input channels", report["input_channels"])
    if conv["widest"]:
        _q("widest conv", f"{conv['widest'][0]}  {conv['widest'][1]}")
    if conv["last"]:
        _q("last conv", f"{conv['last'][0]}  {conv['last'][1]}")
        _q("  -> output channels", report["trunk_output_channels"])
    _q("trunk key prefix", report["trunk_prefix"])
    _q("trunk stages present", report["trunk_stages"])
    _q("tensors per key prefix", report["groups"])

    print()
    print("  9. Classifier / final layer information")
    _q("nn.Linear leaves total", f"{report['leaf_linear_count']} (in trunk + heads)")
    if not report["heads"]:
        print("      no Linear layer outside the trunk was detected")
    for head in report["heads"]:
        bias = head["bias"] if head["bias"] is not None else "absent"
        print(
            f"      {head['path']}"
            f"   weight{head['weight']}  bias{bias}"
            f"   (in={head['in_features']} -> out={head['out_features']})"
        )
    for container in report["containers"]:
        print(
            f"      container '{container['container']}'"
            f"  tensor-bearing indices {container['indices']}"
        )
        print(f"          {container['note']}")

    print()
    print("  10. Number of output classes")
    _q("classes", report["num_classes"] if report["num_classes"] is not None else "NOT DETERMINABLE")
    _q("basis", report["num_classes_basis"])

    print()
    _q("file unchanged by read", report["unchanged"])


def print_summary(reports: "Dict[str, Dict[str, Any]]") -> None:
    print()
    print(LINE)
    print("SUMMARY")
    print(LINE)
    print(f"{'label':<6} {'type':<12} {'state_dict':<11} {'ckpt dict':<10} {'tensors':>8}  classes")
    for label, report in reports.items():
        classes = report["num_classes"] if report.get("state_dict_found") else "n/a"
        if classes is None:
            classes = "not determinable"
        print(
            f"{label:<6} {report['checkpoint_type'].split('.')[-1]:<12} "
            f"{str(report['is_bare_state_dict']):<11} {str(report['is_checkpoint_dict']):<10} "
            f"{report.get('num_tensors', 0):>8}  {classes}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only inspection of the RetinaGrade AI checkpoints (CPU only)."
    )
    parser.add_argument(
        "--all-keys",
        action="store_true",
        help="also print every state_dict key, not just the first/last few",
    )
    args = parser.parse_args(argv)

    print(LINE)
    print("RETINAGRADE AI - CHECKPOINT INSPECTION (READ-ONLY, CPU ONLY)")
    print(LINE)
    print(f"  Python                {sys.version.split()[0]}")
    print(f"  torch                 {torch.__version__}")
    print(f"  numpy                 {np.__version__}")
    print(f"  torch.load call       map_location='cpu', weights_only=True")
    print(f"  CUDA available        {torch.cuda.is_available()}")
    print(f"  CUDA build in torch   {torch.version.cuda}")
    print(f"  models directory      {MODELS_DIR}")

    reports: Dict[str, Dict[str, Any]] = {}
    for label, path in TARGETS:
        print()
        if not path.is_file():
            print(LINE)
            print(f"{label}  ->  {path.name}")
            print(LINE)
            print(f"  ERROR: file not found: {path}")
            print("  STOP: cannot inspect a missing checkpoint.")
            return 2

        try:
            report = analyse(path)
        except Exception as exc:  # noqa: BLE001 - exact error must be shown
            print(LINE)
            print(f"{label}  ->  {path.name}")
            print(LINE)
            print(f"  CHECKPOINT FAILED TO LOAD: {type(exc).__name__}: {exc}")
            print()
            print("  Exact traceback:")
            traceback.print_exc()
            print()
            print("  STOP: no further inspection is performed.")
            return 1

        reports[label] = report
        print_report(label, report, args.all_keys)

    print_summary(reports)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
