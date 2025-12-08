# training/unet/viz.py

from __future__ import annotations
import os
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.data.io import clean_hand_mask
from src.data.s1augmentations import S1_MEAN, S1_STD


def denorm(band: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.clip(band * std + mean, 0.0, 1.0)


@torch.no_grad()
def pick_water_rich_examples(
    model: torch.nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    max_examples: int = 3,
    min_water_frac: float = 0.40,
    type: str = "",
    post_processor=None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Scan the validation set, pick up to `max_examples` chips where
    >= min_water_frac of valid pixels are water (label == 1).

    Returns list of (img_np, mask_np, pred_np) with:
        img_np: (2, H, W) float32
        mask_np: (H, W) int
        pred_np: (H, W) int
    """
    model.eval()
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for batch in val_loader:
        imgs: torch.Tensor = batch["image"].to(device)
        masks: torch.Tensor = batch["mask"].to(device)
        if type == "segformer":
            out = model(imgs)
            logits = out.logits
        else:
            logits = model(imgs)

        if post_processor and type == "segformer":
            rescale_sizes = rescale_sizes = [(256, 256)] * imgs.shape[0]
            pred_masks = post_processor.post_process_semantic_segmentation(
                out, rescale_sizes
            )
            preds = torch.stack(pred_masks, dim=0)
        else:
            preds = torch.argmax(logits, dim=1)

        imgs_np = imgs.cpu().numpy()
        masks_np = masks.cpu().numpy()
        preds_np = preds.cpu().numpy()

        b = imgs_np.shape[0]
        for i in range(b):
            mask_np = masks_np[i]  # (H, W)
            valid = mask_np != 255
            if valid.sum() == 0:
                continue
            water_frac = (mask_np[valid] == 1).mean()
            if water_frac >= min_water_frac:
                examples.append((imgs_np[i], mask_np, preds_np[i]))
                if len(examples) >= max_examples:
                    return examples

    return examples


@torch.no_grad()
def save_example_plot(
    model: nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    save_path: str,
    type: str = "",
    post_processor=None,
) -> None:
    """
    Save VV/VH, hand mask, and predicted mask for up to 3 validation samples
    with >= 40% water pixels.

    Layout: rows = examples, cols = [VV, VH, Ground Truth, Pred].
    """
    if not val_loader:
        return
    examples = pick_water_rich_examples(
        model,
        device,
        val_loader,
        max_examples=3,
        min_water_frac=0.40,
        type=type,
        post_processor=post_processor,
    )
    if len(examples) == 0:
        print(
            "[UNet] No validation chips found with >= 40% water; "
            "falling back to first batch example."
        )
        # fallback: original single-example behavior
        model.eval()
        batch = next(iter(val_loader))
        imgs: torch.Tensor = batch["image"].to(device)
        masks: torch.Tensor = batch["mask"].to(device)
        if type == "segformer":
            out = model(imgs[:1])
            logits = out.logits
        else:
            logits = model(imgs[:1])
        preds = torch.argmax(logits, dim=1)

        img_np = imgs[0].cpu().numpy()
        mask_np = masks[0].cpu().numpy()
        pred_np = preds[0].cpu().numpy()
        examples = [(img_np, mask_np, pred_np)]

    n = len(examples)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=4,
        figsize=(4 * 4, 4 * n),
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=0)  # make it (1,4) for uniform indexing

    for row_idx, (img_np, mask_np, pred_np) in enumerate(examples):
        vv = img_np[0]
        vh = img_np[1]

        vv_denorm = denorm(vv, S1_MEAN[0], S1_STD[0])
        vh_denorm = denorm(vh, S1_MEAN[1], S1_STD[1])
        mask_clean = clean_hand_mask(mask_np)
        mask_vis = mask_clean.astype(float)
        mask_vis[mask_vis == 255] = np.nan

        pred_clean = clean_hand_mask(pred_np)
        pred_vis = pred_clean.astype(float)
        pred_vis[pred_vis == 255] = np.nan

        ax_row = axes[row_idx]

        ax_row[0].imshow(vv_denorm, cmap="gray")
        ax_row[0].set_title(f"VV (example {row_idx + 1})")

        ax_row[1].imshow(vh_denorm, cmap="gray")
        ax_row[1].set_title("VH")

        ax_row[2].imshow(mask_vis, cmap="Reds", vmin=0, vmax=1)
        ax_row[2].set_title("Hand Labeled Mask")

        ax_row[3].imshow(pred_vis, cmap="Reds", vmin=0, vmax=1)
        ax_row[3].set_title("Predicted Mask")

        for a in ax_row:
            a.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "example_predictions.png"))
    plt.close()


def save_learning_curves(
    train_loss_history: list[float],
    train_iou_history: list[float],
    val_loss_history: list[float],
    val_iou_history: list[float],
    save_path: str,
) -> None:
    # Loss curve
    plt.figure()
    plt.plot(train_loss_history, label="Train")
    plt.plot(val_loss_history, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

    # IoU curve
    plt.figure()
    plt.plot(train_iou_history, label="Train")
    plt.plot(val_iou_history, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.title("Mean IoU vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "iou_curve.png"))
    plt.close()
