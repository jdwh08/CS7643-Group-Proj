# training/unet/train_unet.py

import os
import sys
import datetime
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import evaluate
import yaml
import copy
from operator import itemgetter
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
from src.training.unet.model import FloodUNet 
from src.training.unet.viz import save_example_plot,save_learning_curves 
from src.training.unet.loss import build_loss
from src.training.unet.metrics import confusion,derived_stats

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
        self.patience = getattr(self.config.train, "early_stopping_patience", 5)
        self.best_state_dict = None
        self.best_epoch = -1
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
        self.criterion = build_loss(self.config,self.device)

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
        epochs_no_improve = 0
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
                f"train_loss={train_loss:.4f}, train_mIoU={train_iou:.4f}, "
                f"val_loss={val_loss:.4f}, val_mIoU={val_iou:.4f}, "
                f"val_precision={val_stats['precision']:.4f}, "
                f"val_recall={val_stats['recall']:.4f}, "
                f"val_F1={val_stats['f1']:.4f}, "
                f"val_omission={val_stats['omission']:.4f}, "
                f"val_commission={val_stats['commission']:.4f}"
            )

            # early stopping
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self.best_epoch = epoch
                epochs_no_improve = 0
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                print(f"[UNet] New best mIoU: {val_iou:.4f} (epoch {epoch + 1})")
            else:
                epochs_no_improve += 1

            if self.patience is not None and epochs_no_improve >= self.patience:
                print(
                    f"[UNet] Early stopping at epoch {epoch + 1}: "
                    f"no improvement for {self.patience} epochs."
                )
                break

        if self.best_state_dict is not None:
            print(
                f"\n[UNet] Finished training. Best val mIoU: "
                f"{self.best_val_iou:.4f} at epoch {self.best_epoch + 1}"
            )
            # restore best weights before saving & test
            self.model.load_state_dict(self.best_state_dict)
        else:
            print("\n[UNet] Finished training without improvement on validation set.")
        self.save_model()
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

            tp, fp, fn, tn = confusion(preds, refs, ignore_index=255)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        metrics = metric.compute(num_labels=2, ignore_index=255)
        mean_iou = float(metrics["mean_iou"])
        avg_loss = running_loss / max(1, n_batches)
        stats=derived_stats(total_tp,total_fp,total_fn,total_tn)
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
            
            tp, fp, fn, tn = confusion(preds, refs, ignore_index=255)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        metrics = metric.compute(num_labels=2, ignore_index=255)
        mean_iou = float(metrics["mean_iou"])
        avg_loss = running_loss / max(1, n_batches)        
        stats=derived_stats(total_tp,total_fp,total_fn,total_tn)
        precision, recall, f1, omission, commission = itemgetter(
            "precision", "recall", "f1", "omission", "commission"
        )(stats)

        print(f"\n[UNet] Test loss: {avg_loss:.4f}")
        print(f"[UNet] Test mIoU: {mean_iou:.4f}")
        print(f"[UNet] Test precision (water):  {precision:.4f}")
        print(f"[UNet] Test recall (water):     {recall:.4f}")
        print(f"[UNet] Test F1 (water):         {f1:.4f}")
        print(f"[UNet] Test omission rate:      {omission:.4f}")
        print(f"[UNet] Test commission rate:    {commission:.4f}")

        metrics_dict = {
                "Test_loss": float(avg_loss),
                "Test_mIoU": float(mean_iou),
                "Test_precision_water": float(precision),
                "Test_recall_water": float(recall),
                "Test_F1_water": float(f1),
                "Test_omission": float(omission),
                "Test_commission": float(commission),
            }
        out_path = os.path.join(self.save_path, "hparams-test-metrics.yml")
        with open(out_path, "w") as f:
            yaml.dump(metrics_dict, f, default_flow_style=False, sort_keys=False)

    # ---------------------------
    # Saving / logging
    # ---------------------------
    def save_model(self) -> None:
        os.makedirs(LOG_DIR, exist_ok=True)
        loss_name = self.config.train.loss
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"unet_resnet50_{self.dataset}_{loss_name}_{now}"
        
        self.save_path = os.path.join(LOG_DIR, run_name)
        os.makedirs(self.save_path, exist_ok=True)

        weights_path = os.path.join(self.save_path, "weights.pth")
        torch.save(self.model.state_dict(), weights_path)

        save_learning_curves(
            self.train_loss_history,
            self.train_iou_history,
            self.val_loss_history,
            self.val_iou_history,
            self.save_path,
        )
        save_example_plot(
            self.model,
            self.device,
            self.val_loader,
            self.save_path
        )
        
        out_path = os.path.join(self.save_path, 'hparams-best-val.yml')
        metrics = {
            **self.config_dict,            
            "Loss": float(self.val_loss_history[self.best_epoch]),
            "IoU": float(self.val_iou_history[self.best_epoch]),
            "Val_precision": float(self.val_precision_history[self.best_epoch]),
            "Val_recall": float(self.val_recall_history[self.best_epoch]),
            "Val_F1": float(self.val_f1_history[self.best_epoch]),
            "Val_omission": float(self.val_omission_history[self.best_epoch]),
            "Val_commission": float(self.val_commission_history[self.best_epoch]),
        }

        with open(out_path, "w") as f:
            yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
            
        print(f"[UNet] Saved run to {self.save_path}")


if __name__ == "__main__":
    trainer = UNetTrainer()
    trainer.training()