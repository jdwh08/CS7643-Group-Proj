# training/unet/losses.py

from __future__ import annotations

import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss


def build_loss(config, device: torch.device) -> nn.Module | callable:
    """
    Build loss function from config.train.loss.
    Supports:
      - "cross_entropy"
      - "ce_dice"
      - "focal"
    """
    loss_name = config.train.loss.lower()

    if loss_name == "cross_entropy":
        weight = torch.tensor(
            config.train.class_weights, device=device
        ).float()
        return nn.CrossEntropyLoss(weight=weight, ignore_index=255)

    elif loss_name == "ce_dice":
        weight = torch.tensor(
            config.train.class_weights, device=device
        ).float()
        ce = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        dice = DiceLoss(ignore_index=255, mode="multiclass")

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return ce(logits, targets) + config.train.dice_weight * dice(logits, targets)

        return loss_fn

    elif loss_name == "focal":
        return FocalLoss(
            mode="multiclass",
            alpha=config.train.alpha,
            gamma=config.train.gamma,
            ignore_index=255,
        )

    else:
        msg = f"Unknown loss function: {loss_name}"
        raise ValueError(msg)