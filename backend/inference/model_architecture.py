"""Exact model construction required by the two real RetinaGrade checkpoints.

Model *construction only*. This module does not load weights, does not
preprocess images and does not predict: weight loading belongs to a later step,
and every dimension below was read out of the checkpoint files rather than
assumed. ``inference/verify_architecture.py`` re-checks each claim below against
the real files.

==========================================================================
1. Where the architecture comes from: torchvision, not timm
==========================================================================
Both checkpoints store the convolutional trunk under the key prefix
``features`` as **358** tensors. That trunk is a torchvision EfficientNet-B0,
identified from the keys themselves:

===========================  =============================================
Observation in state_dict    What it pins down
===========================  =============================================
``features.0.0.weight``       stem ``Conv2d(3, 32, kernel=3, stride=2,
                             padding=1)`` -> 3-channel RGB input
``(32, 3, 3, 3)``
``features.0.1.*``            ``BatchNorm2d(32)`` directly after the stem
``features.0`` ..             9 top-level stages (0-8), the MBConv layout
``features.8``
``features.<i>.0.block.<j>.*``   MBConv / ConvBNActivation blocks
``features.<i>.1.block.<j>.*``   fused stage (``features.1`` .. ``features.7``)
``features.*.block.2.fc1/.fc2``  Squeeze-Excitation, 64 such tensors
``num_batches_tracked``       49 in both files, i.e. 49 ``BatchNorm2d``,
                             exactly torchvision B0's count
``features.8.0.weight``       final ``Conv2d(320, 1280, 1)`` -> 1280-d
``(1280, 320, 1, 1)``          pooled features
===========================  =============================================

timm is ruled out by those keys. A timm ``efficientnet_b0`` names the trunk
``blocks.<i>.<j>.conv_dw`` / ``.se.fc1`` and puts the classifier in
``conv_head`` / ``head.fc``; a checkpoint from it would contain ``blocks.*``,
``conv_head.*`` or ``bn1.*`` keys. Both real files contain **zero** such keys
and 358/358 of their trunk keys match
``torchvision.models.efficientnet_b0(weights=None).features`` in name, shape
*and order*.

==========================================================================
2. CFP - ``EfficientNetB0_CFP_final.pth`` (374 tensors, 4,064,985 params)
==========================================================================
=============  ====  =====================================================
Key prefix      #     Meaning
=============  ====  =====================================================
``features``    358   torchvision EfficientNet-B0 trunk, unchanged
``dr_head``     2     ``nn.Linear(1280, 5)``   -> 5 DR grade classes
``lesion_heads`` 14   ``lesion_heads.0`` .. ``lesion_heads.6``,
                     7 x ``nn.Linear(1280, 1)`` -> binary auxiliary heads
=============  ====  =====================================================

The key is ``dr_head.weight``, **not** ``dr_head.1.weight``, so ``dr_head`` is a
bare ``nn.Linear`` and not a ``Sequential``: the CFP head has no dropout
wrapper. ``lesion_heads`` indices are contiguous from 0, i.e. an
``nn.ModuleList``.

The checkpoint stores no label map for the 7 auxiliary heads, so their
ordering and meaning are **unknown**. They stay unnamed raw outputs and must
never be presented as named clinical findings.

==========================================================================
3. UWF - ``EfficientNetB0_UWF_final.pt`` (360 tensors, 4,056,018 params)
==========================================================================
=============  ====  =====================================================
Key prefix      #     Meaning
=============  ====  =====================================================
``features``    358   the same torchvision EfficientNet-B0 trunk
``classifier``  2     ``classifier.1.weight (5, 1280)``, ``.bias (5,)``
=============  ====  =====================================================

``classifier`` has exactly one tensor-bearing index, ``1``. Index ``0`` holds no
tensors, so slot 0 is a parameter-free module: this is the stock torchvision
``nn.Sequential(nn.Dropout(p), nn.Linear(...))`` layout, i.e. the whole file is
exactly ``torchvision.models.efficientnet_b0(num_classes=5)``. The checkpoint's
own ``config`` dict corroborates ``num_classes: 5`` and ``model:
"EfficientNet-B0"``.

==========================================================================
4. What the checkpoint does NOT store, and how it is handled
==========================================================================
* ``num_classes`` is read from the head's ``out_features`` (5). It is never
  widened or narrowed to make a load succeed.
* BatchNorm ``eps`` / ``momentum`` are Python hyperparameters, not buffers, so
  no state_dict can store them. torchvision's defaults are used. They do not
  affect whether weights load.
* The dropout probability is likewise not stored; torchvision's default
  ``0.2`` is used. ``nn.Dropout`` is the identity in ``eval()``, so it cannot
  change an inference output.
* The checkpoint stores no pooling parameters. Global average pooling is
  parameter-free and is included because a 1280-d head can only consume pooled
  features; it adds no keys.
* The checkpoints were trained on CUDA (torch 2.10.0+cu128 /
  torchvision 0.25.0+cu128 per the UWF ``config``). Inference here runs on a
  CPU-only build; the tensors are ``float32`` and load with
  ``map_location="cpu"``.

==========================================================================
5. One trunk, two heads
==========================================================================
Both models are built from the **same** trunk factory
(:func:`create_efficientnet_b0_trunk`); only the head differs. The trunk is
therefore defined exactly once - there is no duplicated backbone code. The two
construction functions return two separate instances; nothing is shared or
aliased between the CFP and UWF models.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import EfficientNet, efficientnet_b0

__all__ = [
    "ARCHITECTURE_NAME",
    "BACKBONE_CHANNELS",
    "CfpEfficientNetB0",
    "DROPOUT_PROB",
    "EXPECTED_TRUNK_TENSORS",
    "INPUT_CHANNELS",
    "NUM_BINARY_HEADS",
    "NUM_CLASSES",
    "TORCHVISION_ORIGIN",
    "create_cfp_model",
    "create_efficientnet_b0_trunk",
    "create_uwf_model",
]

# ---------------------------------------------------------------------------
# Dimensions read out of the real checkpoints (evidence in the module docstring).
# ---------------------------------------------------------------------------

INPUT_CHANNELS = 3          # features.0.0.weight     -> (32, 3, 3, 3)
BACKBONE_CHANNELS = 1280    # features.8.0.weight     -> (1280, 320, 1, 1)
NUM_CLASSES = 5             # dr_head.weight          -> (5, 1280)
                            # classifier.1.weight     -> (5, 1280)
NUM_BINARY_HEADS = 7        # lesion_heads.0 .. lesion_heads.6
EXPECTED_TRUNK_TENSORS = 358  # features.* in BOTH checkpoints

# Provenance of the trunk, established from the checkpoint keys (see docstring).
ARCHITECTURE_NAME = "EfficientNet-B0"
TORCHVISION_ORIGIN = "torchvision.models.efficientnet_b0"

# torchvision's default EfficientNet dropout. NOT recoverable from a
# state_dict, and irrelevant in eval() because Dropout is the identity there.
DROPOUT_PROB = 0.2


def create_efficientnet_b0_trunk() -> nn.Sequential:
    """Return a fresh torchvision EfficientNet-B0 convolutional trunk.

    ``weights=None`` is mandatory: no pretrained download is attempted and the
    checkpoint stays the only source of parameters. The returned module carries
    exactly the ``EXPECTED_TRUNK_TENSORS`` ``features.*`` tensors that both real
    checkpoints contain.

    This is the single trunk factory shared by both modalities.
    """
    return efficientnet_b0(weights=None).features


class CfpEfficientNetB0(nn.Module):
    """EfficientNet-B0 trunk + a 5-class DR head + 7 binary auxiliary heads.

    Module names are dictated by the CFP state_dict so that a
    ``strict=True`` load is exact:

    ``features.<n>...``   torchvision trunk, unchanged
    ``dr_head``           ``nn.Linear(1280, 5)``
    ``lesion_heads.<i>``  ``nn.Linear(1280, 1)`` for ``i`` in ``0..6``

    ``forward`` states the module's output contract only. Preprocessing,
    softmax, label mapping and result reporting are later concerns and are
    deliberately absent here.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_binary_heads: int = NUM_BINARY_HEADS,
    ) -> None:
        super().__init__()
        self.features = create_efficientnet_b0_trunk()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dr_head = nn.Linear(BACKBONE_CHANNELS, num_classes)
        self.lesion_heads = nn.ModuleList(
            [nn.Linear(BACKBONE_CHANNELS, 1) for _ in range(num_binary_heads)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(dr_logits, auxiliary_binary_logits)`` for a batch ``x``."""
        pooled = torch.flatten(self.avgpool(self.features(x)), 1)
        dr_logits = self.dr_head(pooled)
        binary_logits = torch.cat([head(pooled) for head in self.lesion_heads], dim=1)
        return dr_logits, binary_logits


def create_cfp_model(
    num_classes: int = NUM_CLASSES,
    num_binary_heads: int = NUM_BINARY_HEADS,
) -> CfpEfficientNetB0:
    """Build the architecture required by ``EfficientNetB0_CFP_final.pth``.

    Randomly initialised: this constructs the required *structure* only. It does
    not read, convert or write any checkpoint. Default arguments reproduce the
    checkpoint exactly; overriding them is only meaningful for tests and would
    break a ``strict=True`` load of the real file.
    """
    return CfpEfficientNetB0(
        num_classes=num_classes, num_binary_heads=num_binary_heads
    )


def create_uwf_model(num_classes: int = NUM_CLASSES) -> EfficientNet:
    """Build the architecture required by ``EfficientNetB0_UWF_final.pt``.

    This is stock ``torchvision.models.efficientnet_b0(num_classes=5)``: the
    trunk plus ``classifier = Sequential(Dropout, Linear(1280, 5))``. It matches
    the UWF checkpoint key-for-key, shape-for-shape and order-for-order.
    """
    return efficientnet_b0(weights=None, num_classes=num_classes)
