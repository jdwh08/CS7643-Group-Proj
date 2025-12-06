# training/unet/train_unet.py

import os
import sys
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import evaluate
import yaml
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss

# ---------------------------
# Project / path setup
# ---------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))          # training/unet
LOG_DIR = os.path.join(BASE_DIR, "logs")
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../.."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config  
from src.data.loaders import make_s1hand_loaders, make_s1weak_loader  
from src.data.io import clean_hand_mask  
from src.data.s1augmentations import S1_MEAN, S1_STD  
from src.training.unet.model import FloodUNet  

torch.set_float32_matmul_precision("high")


class UNetTrainer:
    """
    Train / eval / visualize U-Net (ResNet-50 encoder, no pretraining)
    on Sen1Floods11 S1 data.
    """

    def __init__(self) -> None:
        # ---------------------------
        # Config
        # ---------------------------
        config_path = os.path.join(BASE_DIR, "config_unet.yml")
        with open(config_path, "r") as f:
            self.config_dict = yaml.safe_load(f)
            self.config = Config(config_dict=self.config_dict)
        self.batch_size = int(self.config.train.batch_size)
        self.num_workers = int(self.config.train.num_workers)
        self.lr = float(self.config.train.lr)
        self.n_epochs = int(self.config.train.n_epochs)
        self.weight_decay = float(self.config.optimizer.weight_decay)
        self.image_size = int(self.config.data.image_size)
        self.dataset = self.config.train.dataset
        # ---------------------------
        # Model
        # ---------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FloodUNet(self.config).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[UNet] Trainable parameters: {n_params:,}")

        # ---------------------------
        # Data loaders
        # ---------------------------
        (
            self.hand_train_loader,
            self.val_loader,
            self.test_loader,
        ) = make_s1hand_loaders(
            data_root=DATA_ROOT,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=self.image_size,
        )

        if self.dataset == "weak":
            print("[UNet] Training on S1Weak, validating/testing on S1Hand.")
            self.train_loader = make_s1weak_loader(
                data_root=DATA_ROOT,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                image_size=self.image_size,
                max_samples=None,
            )
        else:
            print("[UNet] Training on S1Hand.")
            self.train_loader = self.hand_train_loader

        print(
            f"[UNet] Loader sizes — train: {len(self.train_loader)}, "
            f"val: {len(self.val_loader)}, test: {len(self.test_loader)}"
        )

        # ---------------------------
        # Optimizer / scheduler / loss
        # ---------------------------
        self.criterion = self._build_loss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(self.train_loader) * 10,
            T_mult=2,
            eta_min=0.0,
        )

        # ---------------------------
        # Metrics + history
        # ---------------------------
        self.best_val_iou = 0.0
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.train_iou_history: list[float] = []
        self.val_iou_history: list[float] = []
        self.val_omission_history: list[float] = []
        self.val_commission_history: list[float] = []
        self.val_precision_history: list[float] = []
        self.val_recall_history: list[float] = []
        self.val_f1_history: list[float] = []

        # will be set in save_model()
        self.save_path: str = ""

    # ---------------------------
    # Training loop
    # ---------------------------

    def training(self) -> None:
        for epoch in range(self.n_epochs):
            print(f"\n[UNet] Epoch {epoch + 1}/{self.n_epochs}")
            #train
            train_loss, train_iou = self._train_one_epoch(epoch)
            #evaluate on validation data
            val_loss, val_iou, val_stats = self._evaluate(epoch)

            self.train_loss_history.append(train_loss)
            self.train_iou_history.append(train_iou)
            self.val_loss_history.append(val_loss)
            self.val_iou_history.append(val_iou)

            self.val_precision_history.append(val_stats["precision"])
            self.val_recall_history.append(val_stats["recall"])
            self.val_f1_history.append(val_stats["f1"])
            self.val_omission_history.append(val_stats["omission"])
            self.val_commission_history.append(val_stats["commission"])
            
            print(
                f"  train_loss={train_loss:.4f}, train_mIoU={train_iou:.4f}, "
                f"val_loss={val_loss:.4f}, val_mIoU={val_iou:.4f}, "
                f"val_precision={val_stats['precision']:.4f}, "
                f"val_recall={val_stats['recall']:.4f}, "
                f"val_F1={val_stats['f1']:.4f}, "
                f"val_omission={val_stats['omission']:.4f}, "
                f"val_commission={val_stats['commission']:.4f}"
            )

            # checkpointing
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                print(f"  [UNet] New best mIoU: {val_iou:.4f} — saving model.")
                self.save_model()

        print(f"\n[UNet] Finished training. Best val mIoU: {self.best_val_iou:.4f}")
        self.test()

    def _train_one_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        metric = evaluate.load("mean_iou")

        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for batch in pbar:
            imgs: torch.Tensor = batch["image"].to(self.device, non_blocking=True)
            masks: torch.Tensor = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            logits = self.model(imgs)  # (B,2,H,W)
            loss = self.criterion(logits, masks.long())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()
            n_batches += 1

            # metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                refs = masks.detach().cpu().numpy()
                metric.add_batch(predictions=preds, references=refs)

            pbar.set_postfix(loss=loss.item())

        metrics = metric.compute(num_labels=2, ignore_index=255)
        mean_iou = float(metrics["mean_iou"])
        avg_loss = running_loss / max(1, n_batches)
        return avg_loss, mean_iou

    @torch.no_grad()
    def _evaluate(self, epoch: int) -> tuple[float, float, dict[str, float]]:
        self.model.eval()
        metric = evaluate.load("mean_iou")

        running_loss = 0.0
        n_batches = 0

        # confusion accumulators for water class
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        pbar = tqdm(self.val_loader, desc=f"Val {epoch}", leave=False)
        for batch in pbar:
            imgs: torch.Tensor = batch["image"].to(self.device, non_blocking=True)
            masks: torch.Tensor = batch["mask"].to(self.device, non_blocking=True)

            logits = self.model(imgs)
            loss = self.criterion(logits, masks.long())

            running_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            refs = masks.detach().cpu().numpy()
            metric.add_batch(predictions=preds, references=refs)

            tp, fp, fn, tn = self._confusion(preds, refs, ignore_index=255)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        metrics = metric.compute(num_labels=2, ignore_index=255)
        mean_iou = float(metrics["mean_iou"])
        avg_loss = running_loss / max(1, n_batches)
        
        # derived stats (water class)
        eps = 1e-6
        precision = total_tp / (total_tp + total_fp + eps)
        recall = total_tp / (total_tp + total_fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        omission = total_fn / (total_tp + total_fn + eps)      # FN rate
        commission = total_fp / (total_tp + total_fp + eps)    # FP rate

        stats = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "omission": float(omission),
            "commission": float(commission),
        }
        return avg_loss, mean_iou, stats

    @torch.no_grad()
    def test(self) -> None:
        self.model.eval()
        metric = evaluate.load("mean_iou")

        running_loss = 0.0
        n_batches = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        pbar = tqdm(self.test_loader, desc="Test", leave=False)
        for batch in pbar:
            imgs: torch.Tensor = batch["image"].to(self.device, non_blocking=True)
            masks: torch.Tensor = batch["mask"].to(self.device, non_blocking=True)

            logits = self.model(imgs)
            loss = self.criterion(logits, masks.long())

            running_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            refs = masks.detach().cpu().numpy()
            metric.add_batch(predictions=preds, references=refs)
            
            tp, fp, fn, tn = self._confusion(preds, refs, ignore_index=255)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        metrics = metric.compute(num_labels=2, ignore_index=255)
        mean_iou = float(metrics["mean_iou"])
        avg_loss = running_loss / max(1, n_batches)

        eps = 1e-6
        precision = total_tp / (total_tp + total_fp + eps)
        recall = total_tp / (total_tp + total_fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        omission = total_fn / (total_tp + total_fn + eps)
        commission = total_fp / (total_tp + total_fp + eps)

        print(f"\n[UNet] Test loss: {avg_loss:.4f}")
        print(f"[UNet] Test mIoU: {mean_iou:.4f}")
        print(f"[UNet] Test precision (water):  {precision:.4f}")
        print(f"[UNet] Test recall (water):     {recall:.4f}")
        print(f"[UNet] Test F1 (water):         {f1:.4f}")
        print(f"[UNet] Test omission rate:      {omission:.4f}")
        print(f"[UNet] Test commission rate:    {commission:.4f}")

    # ---------------------------
    # Saving / logging
    # ---------------------------

    def save_model(self) -> None:
        os.makedirs(LOG_DIR, exist_ok=True)

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"unet_resnet50_{self.dataset}_{now}"

        self.save_path = os.path.join(LOG_DIR, run_name)
        os.makedirs(self.save_path, exist_ok=True)

        weights_path = os.path.join(self.save_path, "weights.pth")
        torch.save(self.model.state_dict(), weights_path)

        self._save_learning_curves()
        self._save_metrics()
        self._save_example_plot()
        print(f"[UNet] Saved run to {self.save_path}")

    def _save_learning_curves(self) -> None:
        # Loss curve
        plt.figure()
        plt.plot(self.train_loss_history, label="Train")
        plt.plot(self.val_loss_history, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "loss_curve.png"))
        plt.close()

        # IoU curve
        plt.figure()
        plt.plot(self.train_iou_history, label="Train")
        plt.plot(self.val_iou_history, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Mean IoU")
        plt.title("Mean IoU vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "iou_curve.png"))
        plt.close()

    def _save_metrics(self) -> None:
        out_path = os.path.join(self.save_path, "h_params_config.yml")
        metric_dict = {
            **self.config_dict,
            "Loss": float(self.val_loss_history[-1]),
            "IoU": float(self.val_iou_history[-1]),
            "Val_precision": float(self.val_precision_history[-1]),
            "Val_recall": float(self.val_recall_history[-1]),
            "Val_F1": float(self.val_f1_history[-1]),
            "Val_omission": float(self.val_omission_history[-1]),
            "Val_commission": float(self.val_commission_history[-1]),
        }
        with open(out_path, "w") as f:
            yaml.dump(metric_dict, f, default_flow_style=False, sort_keys=False)

    def _build_loss(self):
        config = self.config
        loss_name = config.train.loss.lower()

        if loss_name == "cross_entropy":
            weight = torch.tensor(
                config.train.class_weights, device=self.device
            ).float()
            return nn.CrossEntropyLoss(weight=weight, ignore_index=255)

        elif loss_name == "ce_dice":
            weight = torch.tensor(
                config.train.class_weights, device=self.device
            ).float()
            ce = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
            dice = DiceLoss(ignore_index=255,mode="multiclass")
            def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return ce(logits, targets) + config.train.dice_weight * dice(logits, targets)
            return loss_fn

        elif loss_name == "focal":
            return FocalLoss(
                mode="multiclass",
                alpha=config.train.alpha,
                gamma=config.train.gamma,
                ignore_index=255
            )

        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
        
    @staticmethod
    def _denorm(band: np.ndarray, mean: float, std: float) -> np.ndarray:
        return np.clip(band * std + mean, 0.0, 1.0)

    @staticmethod
    def _confusion(
        preds: np.ndarray,
        refs: np.ndarray,
        ignore_index: int = 255,
    ) -> tuple[int, int, int, int]:
        """
        Compute TP, FP, FN, TN for the WATER class (label=1), ignoring 255.
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

    @torch.no_grad()
    def _save_example_plot(self) -> None:
        """
        Save VV/VH, hand mask, and predicted mask for a single val sample.
        """
        if not self.val_loader:
            return

        self.model.eval()
        batch = next(iter(self.val_loader))
        imgs: torch.Tensor = batch["image"].to(self.device)
        masks: torch.Tensor = batch["mask"].to(self.device)

        idx = 0
        img = imgs[idx : idx + 1]  # (1,2,H,W)
        mask = masks[idx]

        logits = self.model(img)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

        vv = img[0, 0].cpu().numpy()
        vh = img[0, 1].cpu().numpy()

        vv_denorm = self._denorm(vv, S1_MEAN[0], S1_STD[0])
        vh_denorm = self._denorm(vh, S1_MEAN[1], S1_STD[1])

        mask_np = clean_hand_mask(mask.cpu().numpy())
        mask_vis = mask_np.astype(float)
        mask_vis[mask_vis == 255] = np.nan

        pred_clean = clean_hand_mask(pred)
        pred_vis = pred_clean.astype(float)
        pred_vis[pred_vis == 255] = np.nan

        fig, ax = plt.subplots(1, 4, figsize=(18, 5))
        ax[0].imshow(vv_denorm, cmap="gray")
        ax[0].set_title("VV")
        ax[1].imshow(vh_denorm, cmap="gray")
        ax[1].set_title("VH")
        ax[2].imshow(mask_vis, cmap="Reds", vmin=0, vmax=1)
        ax[2].set_title("Hand Mask (cleaned)")
        ax[3].imshow(pred_vis, cmap="Reds", vmin=0, vmax=1)
        ax[3].set_title("Predicted Mask (cleaned)")

        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "example_prediction.png"))
        plt.close()


if __name__ == "__main__":
    trainer = UNetTrainer()
    trainer.training()