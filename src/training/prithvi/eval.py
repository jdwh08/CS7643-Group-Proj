# src/training/transformers_transfer_prithvi/eval.py
#####################################################################
# Flood Segmentation Project [Vision Transformer Transfer Learning with Prithvi]
#####################################################################

### ABOUT
"""Evaluating a Vision Transformer model on Sen1Floods11 via Terrastack."""

#####################################################################
### BOARD:
# TODO(jdwh08): ...

#####################################################################
### IMPORTS
import gc
import sys
from operator import itemgetter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from terratorch.tasks import SemanticSegmentationTask
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

### OWN MODULES
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from src.data.s2hand import S2HandDataset
from src.training.transformers_transfer_prithvi.sen1floods11 import (
    Sen1Floods11S2WeakDataModule,
)

#####################################################################
### SETTINGS
OUTPUT_PATH = project_root / "outputs" / "transformers_transfer_prithvi"
DATA_PATH = project_root / "data"

#####################################################################
### CODE


def load_model(model_path: Path) -> tuple[nn.Module, int]:
    """Loads a trained model checkpoint from a path folder.

    Returns:
        tuple[nn.Module, int]: The model and its corresponding epoch.
    """
    # 1. Find the .ckpt file
    try:
        ckpt_file = next(iter(model_path.glob("**/*epoch=*.ckpt")))
        config_file = next(iter(model_path.glob("**/hparams.yaml")))
    except StopIteration as e:
        msg = f"No .pt file found in {model_path}"
        raise FileNotFoundError(msg) from e

    # 2. Load the model
    config = yaml.safe_load(config_file.read_text())
    model = SemanticSegmentationTask.load_from_checkpoint(
        ckpt_file,
        model_factory="EncoderDecoderFactory",
        model_args=config["model_args"],
    )
    # 3. Handle model settings
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # 4. Get the epoch from the checkpoint name (model-epoch=X.pt)
    print(f"Checkpoint File: {ckpt_file.stem}")
    epoch = int(ckpt_file.stem.split("epoch=")[1].split(".ckpt")[0])
    return model, epoch


def load_params(model_path: Path) -> pd.DataFrame:
    """Loads the parameters for a model."""
    # 1. Find the .yaml file
    yaml_file = next(iter(model_path.glob("hparams.yaml")))
    # 2. Load the parameters
    params_all = yaml.safe_load(yaml_file.read_text())
    # 3. Obtain only parameters of interest
    params: dict[str, Any] = {}
    params.update(**params_all["model_args"])
    params["loss"] = params_all["loss"]
    params["lr"] = params_all["lr"]
    params["optimizer"] = params_all["optimizer"]
    params.update(**params_all["optimizer_hparams"])
    params["scheduler"] = params_all["scheduler"]
    params.update(**params_all["scheduler_hparams"])

    # 4. Remove sub-parameters that are not needed
    params.pop("necks")

    # 5. Convert to pandas DataFrame
    params_df = pd.DataFrame.from_dict(params, orient="index").T
    return params_df


def get_log_data(model_path: Path) -> pd.DataFrame:
    """Returns a DataFrame of the log data for a model."""
    # 1. Find the log file
    log_file = next(iter(model_path.glob("metrics.csv")))
    # 2. Load the log data
    log_data = pd.read_csv(log_file)

    # 3. Clean the log data -- columns
    # NOTE(jdwh08): Format has columns for epoch, lr-AdamW, step, etc.
    # We want to plot:
    # A: train/loss, val/loss, lr-AdamW
    # B: train/mIOU, val/mIOU
    # C: train/F1_Score, val/F1_Score
    # along with vertical bars for each epoch.

    log_data = log_data.loc[
        :,
        [
            "step",
            "epoch",
            "lr-AdamW",
            "train/loss",
            "val/loss",
            "train/mIoU",
            "val/mIoU",
            "train/F1_Score",
            "val/F1_Score",
        ],
    ]

    # 4. Clean the log data -- rows
    # NOTE(jdwh08): Combine rows together if they have the same step and epoch value
    # by replacing the NaN values with non-NaN values in other rows.
    # First, forward-fill learning rate since it only appears at certain steps
    log_data["lr-AdamW"] = pd.to_numeric(log_data["lr-AdamW"], errors="coerce")
    log_data = log_data.sort_values("step")
    log_data["lr-AdamW"] = log_data["lr-AdamW"].ffill()

    # Group by step and epoch, taking first non-null value for each column
    log_data = log_data.groupby(["step", "epoch"], dropna=False).first().reset_index()
    return log_data


def plot_log_data(
    log_data: pd.DataFrame, model_path: Path, model_epoch: int | None = None
) -> None:
    """Plot the log data.

    There are 3 plots:
    A: train/loss, val/loss, lr-AdamW
    B: train/mIOU, val/mIOU
    C: train/F1_Score, val/F1_Score
    along with vertical bars for each epoch.
    """
    # Filter out rows with NaN step values
    log_data = log_data.dropna(subset=["step"]).copy()

    # Convert step to numeric (in case it's not)
    log_data["step"] = pd.to_numeric(log_data["step"], errors="coerce")
    log_data = log_data.dropna(subset=["step"])

    # Sort by step to ensure proper ordering
    log_data = log_data.sort_values("step").reset_index(drop=True)

    # Identify epoch boundaries
    # Find where epoch changes (excluding NaN epochs)
    # Convert epoch to numeric
    log_data["epoch"] = pd.to_numeric(log_data["epoch"], errors="coerce")

    # Find epoch transitions: where a new epoch number first appears
    epoch_boundary_steps = []
    seen_epochs = set()

    for _, row in log_data.iterrows():
        epoch_val = row["epoch"]
        step = row["step"]

        # If epoch is not NaN and we haven't seen this epoch before
        # Check if epoch_val is a valid number
        try:
            epoch_int = int(float(epoch_val))
            if epoch_int not in seen_epochs:
                epoch_boundary_steps.append(step)
                seen_epochs.add(epoch_int)
        except (ValueError, TypeError):
            # Skip NaN or invalid values
            continue

    # Sort to ensure proper order
    epoch_boundary_steps = sorted(epoch_boundary_steps)

    # Create step-to-epoch mapping
    # Forward-fill epoch values based on step order to map each step to its epoch
    log_data_sorted = log_data.sort_values("step").copy()
    log_data_sorted["epoch_mapped"] = log_data_sorted["epoch"].ffill()

    # Create a mapping dictionary from step to epoch
    # Also create a sorted list for interpolation
    step_to_epoch = {}
    step_epoch_pairs = []
    for _, row in log_data_sorted.iterrows():
        step = row["step"]
        epoch_mapped = row["epoch_mapped"]

        # Check if epoch_mapped is not NaN
        try:
            has_data = bool(pd.notna(epoch_mapped))
        except (ValueError, TypeError):
            has_data = False
        if has_data:
            try:
                epoch_int = int(float(epoch_mapped))
                step_to_epoch[step] = epoch_int
                step_epoch_pairs.append((step, epoch_int))
            except (ValueError, TypeError):
                pass

    # Sort pairs for interpolation
    step_epoch_pairs = sorted(step_epoch_pairs, key=lambda x: float(x[0]))

    # Create epoch-to-step-range mapping for alignment
    # Epoch boundaries mark where a new epoch starts
    # Epoch 0: from step 0 to (second_boundary - 1) if there's a second boundary
    # Epoch 1: from second_boundary to (third_boundary - 1)
    # etc.
    epoch_to_step_range = {}
    if epoch_boundary_steps:
        max_step = log_data["step"].max()

        # Epoch 0: from 0 to (first non-zero boundary - 1), or to max_step if only one boundary
        if len(epoch_boundary_steps) == 1:
            # Only epoch 0 exists
            epoch_to_step_range[0] = (0, max_step)
        else:
            # Epoch 0: from 0 to (second boundary - 1)
            epoch_to_step_range[0] = (
                0,
                epoch_boundary_steps[1] - 1
                if len(epoch_boundary_steps) > 1
                else max_step,
            )

            # Middle epochs: from boundary[i] to (boundary[i+1] - 1)
            for i in range(1, len(epoch_boundary_steps)):
                epoch_num = i
                start_step = epoch_boundary_steps[i]
                end_step = (
                    epoch_boundary_steps[i + 1] - 1
                    if i + 1 < len(epoch_boundary_steps)
                    else max_step
                )
                epoch_to_step_range[epoch_num] = (start_step, end_step)
    else:
        # Fallback: single epoch
        max_step = log_data["step"].max()
        epoch_to_step_range[0] = (0, max_step)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

    # Plot 1: Loss and Learning Rate
    ax1 = axes[0]

    # Plot train/loss
    train_loss_data = log_data.dropna(subset=["train/loss"])
    if len(train_loss_data) > 0:
        ax1.plot(
            train_loss_data["step"],
            train_loss_data["train/loss"],
            label="Train Loss",
            color="blue",
            linewidth=1.5,
            alpha=0.8,
        )

    # Plot val/loss
    val_loss_data = log_data.dropna(subset=["val/loss"])
    if len(val_loss_data) > 0:
        ax1.plot(
            val_loss_data["step"],
            val_loss_data["val/loss"],
            label="Val Loss",
            color="red",
            linewidth=1.5,
            alpha=0.8,
        )

    # Plot learning rate on secondary y-axis
    ax1_lr = ax1.twinx()

    # Ensure learning rate is numeric and forward-filled
    log_data["lr-AdamW"] = pd.to_numeric(log_data["lr-AdamW"], errors="coerce")
    log_data["lr-AdamW"] = log_data["lr-AdamW"].ffill()

    lr_data = log_data.dropna(subset=["lr-AdamW"]).copy()
    if len(lr_data) > 0:
        ax1_lr.plot(
            lr_data["step"],
            lr_data["lr-AdamW"],
            label="Learning Rate",
            color="green",
            linewidth=1.5,
            alpha=0.6,
            linestyle="--",
        )

    # Add epoch boundaries
    for step in epoch_boundary_steps:
        ax1.axvline(x=step, color="gray", alpha=0.3, linestyle="-", linewidth=1)
        ax1_lr.axvline(x=step, color="gray", alpha=0.3, linestyle="-", linewidth=1)

    # Add model epoch marker (thick vertical dashed line)
    # Find the step where validation metrics were logged for model_epoch
    model_epoch_step = None
    if model_epoch is not None:
        # Find validation data at the end of model_epoch
        val_data_at_epoch = log_data[
            (log_data["epoch"] == model_epoch) & (log_data["val/loss"].notna())
        ]
        if len(val_data_at_epoch) > 0:
            # Get the step where validation was logged for this epoch
            model_epoch_step = val_data_at_epoch["step"].max()
        elif model_epoch < len(epoch_boundary_steps):
            # Fallback: use the epoch boundary step
            model_epoch_step = (
                epoch_boundary_steps[model_epoch]
                if model_epoch < len(epoch_boundary_steps)
                else None
            )

        if model_epoch_step is not None:
            ax1.axvline(
                x=model_epoch_step,
                color="purple",
                alpha=0.7,
                linestyle="--",
                linewidth=2.5,
                label="Model Epoch",
            )
            ax1_lr.axvline(
                x=model_epoch_step,
                color="purple",
                alpha=0.7,
                linestyle="--",
                linewidth=2.5,
                label="",  # No label to avoid duplicate in legend
            )

    ax1.set_xlabel("Step", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11, color="black")
    ax1_lr.set_ylabel("Learning Rate", fontsize=11, color="green")
    ax1.set_title("Loss and Learning Rate", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax1_lr.legend(loc="upper right")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1_lr.tick_params(axis="y", labelcolor="green")

    # Plot 2: mIOU (using epoch on x-axis)
    ax2 = axes[1]

    # Convert step to epoch for plotting
    # Validation metrics are logged at the END of each epoch
    # Training metrics are logged throughout each epoch
    def step_to_epoch_for_train(step_val: float) -> int:
        """Convert step to epoch for training metrics (current epoch)."""
        step_val_float = float(step_val)
        # Find which epoch this step belongs to based on epoch boundaries
        if epoch_boundary_steps:
            for i, boundary in enumerate(epoch_boundary_steps):
                if step_val_float < boundary:
                    return max(0, i - 1)
            # If step is >= last boundary, it's the last epoch
            return len(epoch_boundary_steps) - 1
        return 0

    def step_to_epoch_for_val(step_val: float, epoch_val: float) -> int:
        """Convert step to epoch for validation metrics (epoch that just completed)."""
        # Validation metrics are logged at the end of an epoch
        # Use the epoch value from the data directly if available
        if pd.notna(epoch_val):
            try:
                return int(float(epoch_val))
            except (ValueError, TypeError):
                pass
        # Fallback: find the epoch that just completed before the next boundary
        step_val_float = float(step_val)
        if epoch_boundary_steps:
            for i, boundary in enumerate(epoch_boundary_steps):
                if step_val_float < boundary:
                    return max(0, i - 1)
            # If step is >= last boundary, it's the last epoch
            return len(epoch_boundary_steps) - 1
        return 0

    # Plot train/mIOU
    train_miou_data = log_data.dropna(subset=["train/mIoU"]).copy()
    if len(train_miou_data) > 0:
        # Shift epoch by +1 to align with validation metrics
        def get_train_epoch(step: float) -> int:
            return step_to_epoch_for_train(step) + 1

        train_miou_data["epoch_x"] = train_miou_data["step"].apply(get_train_epoch)
        ax2.plot(
            train_miou_data["epoch_x"],
            train_miou_data["train/mIoU"],
            label="Train mIOU",
            color="blue",
            linewidth=1.5,
            alpha=0.8,
        )

    # Plot val/mIOU
    val_miou_data = log_data.dropna(subset=["val/mIoU"]).copy()
    if len(val_miou_data) > 0:
        # For validation, shift epoch by +1 (metrics at end of epoch N are plotted at epoch N+1)
        def get_val_epoch(row: pd.Series) -> int:
            step_val = float(row["step"])
            epoch_val = row["epoch"]
            # Get the epoch value and add 1
            try:
                epoch_val_float = float(epoch_val)
                epoch_int = int(epoch_val_float)
                # If conversion succeeded, return epoch + 1
                return epoch_int + 1
            except (ValueError, TypeError, AttributeError):
                pass
            # Fallback: use step-based mapping and add 1
            return step_to_epoch_for_val(step_val, float("nan")) + 1

        val_miou_data["epoch_x"] = val_miou_data.apply(get_val_epoch, axis=1)
        ax2.plot(
            val_miou_data["epoch_x"],
            val_miou_data["val/mIoU"],
            label="Val mIOU",
            color="red",
            linewidth=1.5,
            alpha=0.8,
        )

    # Add epoch boundaries (vertical lines at integer epoch values)
    # Shift boundaries by +1 to align with shifted data
    max_epoch = max(epoch_to_step_range.keys()) if epoch_to_step_range else 0
    for epoch in range(int(max_epoch) + 2):  # Start at 1, end at max_epoch + 1
        ax2.axvline(x=epoch, color="gray", alpha=0.3, linestyle="-", linewidth=1)

    # Add model epoch marker (thick vertical dashed line)
    # Shift by +1 to align with shifted data
    if model_epoch is not None:
        model_epoch_shifted = model_epoch + 1
        ax2.axvline(
            x=model_epoch_shifted,
            color="purple",
            alpha=0.7,
            linestyle="--",
            linewidth=2.5,
            label="Model Epoch",
        )

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("mIoU", fontsize=11)
    ax2.set_title("Mean Intersection over Union (mIOU)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    # Plot 3: F1 Score (using epoch on x-axis)
    ax3 = axes[2]

    # Plot train/F1_Score
    train_f1_data = log_data.dropna(subset=["train/F1_Score"]).copy()
    if len(train_f1_data) > 0:
        # Shift epoch by +1 to align with validation metrics
        def get_train_epoch_f1(step: float) -> int:
            return step_to_epoch_for_train(step) + 1

        train_f1_data["epoch_x"] = train_f1_data["step"].apply(get_train_epoch_f1)
        ax3.plot(
            train_f1_data["epoch_x"],
            train_f1_data["train/F1_Score"],
            label="Train F1",
            color="blue",
            linewidth=1.5,
            alpha=0.8,
        )

    # Plot val/F1_Score
    val_f1_data = log_data.dropna(subset=["val/F1_Score"]).copy()
    if len(val_f1_data) > 0:
        # For validation, shift epoch by +1 (metrics at end of epoch N are plotted at epoch N+1)
        def get_val_epoch_f1(row: pd.Series) -> int:
            step_val = float(row["step"])
            epoch_val = row["epoch"]
            # Get the epoch value and add 1
            try:
                epoch_val_float = float(epoch_val)
                epoch_int = int(epoch_val_float)
                # If conversion succeeded, return epoch + 1
                return epoch_int + 1
            except (ValueError, TypeError, AttributeError):
                pass
            # Fallback: use step-based mapping and add 1
            return step_to_epoch_for_val(step_val, float("nan")) + 1

        val_f1_data["epoch_x"] = val_f1_data.apply(get_val_epoch_f1, axis=1)
        ax3.plot(
            val_f1_data["epoch_x"],
            val_f1_data["val/F1_Score"],
            label="Val F1",
            color="red",
            linewidth=1.5,
            alpha=0.8,
        )

    # Add epoch boundaries (vertical lines at integer epoch values)
    # Shift boundaries by +1 to align with shifted data
    max_epoch = max(epoch_to_step_range.keys()) if epoch_to_step_range else 0
    for epoch in range(int(max_epoch) + 2):  # Start at 1, end at max_epoch + 1
        ax3.axvline(x=epoch, color="gray", alpha=0.3, linestyle="-", linewidth=1)

    # Add model epoch marker (thick vertical dashed line)
    # Shift by +1 to align with shifted data
    if model_epoch is not None:
        model_epoch_shifted = model_epoch + 1
        ax3.axvline(
            x=model_epoch_shifted,
            color="purple",
            alpha=0.7,
            linestyle="--",
            linewidth=2.5,
            label="Model Epoch",
        )

    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("F1", fontsize=11)
    ax3.set_title("F1", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")

    plt.tight_layout()
    plt.savefig(model_path / "log_data.png")


def get_val_data(bands: list[str]) -> DataLoader[dict[str, Tensor]]:
    """Returns a DataLoader for the validation set."""
    val_data = Sen1Floods11S2WeakDataModule(
        data_root=str(DATA_PATH),
        batch_size=16,
        num_workers=4,
        bands=bands,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_data.setup("validate")
    return val_data.val_dataloader()


def get_test_data(bands: list[str]) -> DataLoader[dict[str, Tensor]]:
    """Returns a DataLoader for the validation set."""
    val_data = Sen1Floods11S2WeakDataModule(
        data_root=str(DATA_PATH),
        batch_size=16,
        num_workers=4,
        bands=bands,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_data.setup("test")
    return val_data.test_dataloader()


def plot_masks(
    val_data: DataLoader[dict[str, Tensor]],
    model_path: Path,
    model: nn.Module,
    bands: list[str],
    num_samples: int = 3,
    min_water_frac: float = 0.40,
) -> None:
    """Plot the test data."""
    output_path = model_path / "test_data_plots"
    output_path.mkdir(parents=True, exist_ok=True)

    # Ensure model is in evaluation mode
    model.eval()

    torch.manual_seed(31415926)
    with torch.no_grad():
        # Determine the actual number of batches available
        dataloader_batch_size: int = (
            val_data.batch_size if val_data.batch_size is not None else 1
        )

        progress_bar = tqdm(total=num_samples, desc="Plotting masks")
        for i, batch in enumerate(val_data):
            if i * dataloader_batch_size >= num_samples:
                break

            image = batch["image"].to(model.device)
            mask = batch["mask"].to(model.device)
            prediction = model(image).output

            for j in range(image.shape[0]):
                data = {
                    "image": image[j],
                    "mask": mask[j],
                    "prediction": prediction[j],
                }

                # Check if we have enough water in the mask
                mask_np = mask[j].cpu().numpy()
                water_frac = (mask_np[mask_np != 255] == 1).mean()
                if water_frac < min_water_frac:
                    continue
                del mask_np, water_frac

                fig = S2HandDataset.plot(data, bands=bands)
                fig.savefig(
                    output_path / f"val_data_{i * dataloader_batch_size + j}.png"
                )
                plt.close(fig)
                del data, fig
                torch.cuda.empty_cache()
                gc.collect()

                progress_bar.update(1)
                if i * dataloader_batch_size + j >= num_samples:
                    break

        progress_bar.close()


def test_model(model_path: Path) -> None:
    """Test the model."""
    model, epoch = load_model(model_path)
    params = load_params(model_path)
    val_data = get_val_data(params["backbone_bands"][0])
    test_data = get_test_data(params["backbone_bands"][0])
    log_data = get_log_data(model_path)
    test_metrics_standardized(test_data, model, model_path)
    plot_log_data(log_data, model_path, epoch)
    plot_masks(val_data, model_path, model, params["backbone_bands"][0])


def test_all_models() -> None:
    """Test all models."""
    # Get all top level folders in the OUTPUT_PATH
    data_paths = [f for f in OUTPUT_PATH.iterdir() if f.is_dir()]
    for data_path in data_paths:
        model_paths = [f for f in data_path.iterdir() if f.is_dir()]
        for model_path in model_paths:
            test_model(model_path)


def test_metrics_standardized(
    test_loader: DataLoader[dict[str, Tensor]],
    model: nn.Module,
    model_path: Path,
) -> None:
    """Use the metric calculations from UNet/Test to have standardized results."""
    model.eval()

    def confusion(
        preds: np.ndarray,
        refs: np.ndarray,
        ignore_index: int = 255,
    ) -> tuple[int, int, int, int]:
        """Compute TP, FP, FN, TN for the WATER class (label=1), ignoring 255.
        preds, refs: numpy arrays of shape (N, H, W) or (N,).
        """
        # flatten
        preds_f = preds.reshape(-1)
        refs_f = refs.reshape(-1)

        # ignore 255
        mask = refs_f != ignore_index
        preds_f = preds_f[mask]
        refs_f = refs_f[mask]

        # water = 1, land = 0
        tp = np.sum((preds_f == 1) & (refs_f == 1))
        fp = np.sum((preds_f == 1) & (refs_f == 0))
        fn = np.sum((preds_f == 0) & (refs_f == 1))
        tn = np.sum((preds_f == 0) & (refs_f == 0))
        return int(tp), int(fp), int(fn), int(tn)

    def derived_stats(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
        eps = 1e-6
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        omission = fn / (tp + fn + eps)
        commission = fp / (tp + fp + eps)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "omission": float(omission),
            "commission": float(commission),
        }

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    pbar = tqdm(test_loader, desc="Test", leave=False)
    for batch in pbar:
        imgs: torch.Tensor = batch["image"].to(model.device, non_blocking=True)
        masks: torch.Tensor = batch["mask"].to(model.device, non_blocking=True)

        logits = model(imgs).output

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

        refs = masks.detach().cpu().numpy()

        tp, fp, fn, tn = confusion(preds, refs, ignore_index=255)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    stats = derived_stats(total_tp, total_fp, total_fn, total_tn)
    precision, recall, f1, omission, commission = itemgetter(
        "precision", "recall", "f1", "omission", "commission"
    )(stats)

    print(f"Test precision (water):  {precision:.4f}")
    print(f"Test recall (water):     {recall:.4f}")
    print(f"Test F1 (water):         {f1:.4f}")
    print(f"Test omission rate:      {omission:.4f}")
    print(f"Test commission rate:    {commission:.4f}")

    metrics_dict = {
        "Test_precision_water": float(precision),
        "Test_recall_water": float(recall),
        "Test_F1_water": float(f1),
        "Test_omission": float(omission),
        "Test_commission": float(commission),
    }
    out_path = model_path / "test-metrics.yml"
    with out_path.open("w") as f:
        yaml.safe_dump(metrics_dict, f, default_flow_style=False, sort_keys=False)
