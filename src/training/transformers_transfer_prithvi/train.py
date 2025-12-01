# src/training/transformers_transfer_prithvi/train.py
#####################################################################
# Flood Segmentation Project [Vision Transformer Transfer Learning with Prithvi-100M]
#####################################################################

### ABOUT
"""Training a Vision Transformer model on Sen1Floods11 via Terrastack."""

#####################################################################
### BOARD:
# TODO(jdwh08): Add both hand and weak

#####################################################################
### IMPORTS

import logging
import sys
from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from terratorch.tasks import SemanticSegmentationTask

### OWN MODULES
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from src.data.io import IGNORE_INDEX
from src.training.transformers_transfer_prithvi.sen1floods11 import (
    Sen1Floods11S2HandDataModule,
)

#####################################################################
### SETTINGS
DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"
OUTPUT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "outputs"
    / "transformers_transfer_prithvi"
)

TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 16
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

logger = logging.getLogger(__name__)
# Also log to console
logger.addHandler(logging.StreamHandler(sys.stdout))

MAIN_METRIC: str = "val/mIoU_Micro"  # Micro-average IoU

# Set seed for reproducibility
# This seeds Python, NumPy, PyTorch, and sets deterministic CUDNN
seed_everything(seed=31415926, workers=True)

#####################################################################
### CODE

data = Sen1Floods11S2HandDataModule(
    data_root=str(DATA_PATH),
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    bands=BANDS,
)

task = SemanticSegmentationTask(
    model_factory="EncoderDecoderFactory",
    model_args={
        "backbone": "prithvi_eo_v2_300_tl",  # "prithvi_eo_v1_100",  # "prithvi_eo_v2_300",
        "backbone_pretrained": False,
        "backbone_bands": BANDS,
        "decoder": "UperNetDecoder",
        "decoder_channels": 256,
        "decoder_scale_modules": False,
        "num_classes": 2,
        "rescale": True,
        "head_dropout": 0.1,
        "necks": [
            # {"name": "SelectIndices", "indices": [2, 5, 8, 11]},  # 100M model
            {"name": "SelectIndices", "indices": [5, 11, 17, 23]},  # 300M model
            {"name": "ReshapeTokensToImage"},
            # {"name": "LearnedInterpolateToPyramidal"},
        ],
    },
    loss="ce",
    # NOTE(jdwh08): src/data/io.py hard-coded 255 as our ignore index (-1 -> 255)
    ignore_index=IGNORE_INDEX,
    freeze_backbone=False,
    freeze_decoder=False,
    optimizer="AdamW",
    optimizer_hparams={
        "weight_decay": 0.005,
    },
    lr=1.0e-5,
    scheduler="CosineAnnealingLR",
    scheduler_hparams={
        "T_max": TRAIN_EPOCHS,
        "eta_min": 0,
    },
    plot_on_val=10,
)

### Loggers
# CSVLogger saves all metrics to CSV for easy plotting and hyperparameter analysis
csv_logger = CSVLogger(
    save_dir=str(OUTPUT_PATH),
    name="lightning_logs",
    version=None,  # Auto-increment version numbers
)

# TensorBoardLogger for interactive visualization
tb_logger = TensorBoardLogger(
    save_dir=str(OUTPUT_PATH),
    name="lightning_logs",
    version=None,  # Auto-increment version numbers
)

### Callbacks
# ModelCheckpoint to save only the best model based on validation IoU

# Metrics:
# 'lr-AdamW', 'train/loss', 'val/loss', 'val/Accuracy', 'val/Boundary_mIoU',
# 'val/Class_Accuracy_0', 'val/Class_Accuracy_1', 'val/F1_Score',
# 'val/IoU_0', 'val/IoU_1', 'val/Pixel_Accuracy', 'val/mIoU', 'val/mIoU_Micro',
# 'train/Accuracy', 'train/Boundary_mIoU', 'train/Class_Accuracy_0', 'train/Class_Accuracy_1',
# 'train/F1_Score', 'train/IoU_0', 'train/IoU_1', 'train/Pixel_Accuracy',
# 'train/mIoU', 'train/mIoU_Micro',
# 'epoch', 'step'

checkpoint_callback = ModelCheckpoint(
    monitor=MAIN_METRIC,  # Monitor validation macro-average IoU
    mode="max",  # Save model with maximum IoU
    save_top_k=1,  # Save only the best model
    filename="best-{epoch:02d}",  # Simple filename with epoch
    # Lightning will add metric name and value automatically
    auto_insert_metric_name=True,
    save_last=False,  # Don't save last checkpoint, only best
    verbose=True,  # Print when a better model is found
)

# EarlyStopping to prevent overfitting
# Stops training if validation IoU doesn't improve for 'patience' epochs
early_stopping = EarlyStopping(
    monitor="val/loss",
    mode="max",  # Stop when metric stops increasing
    patience=10,  # Number of epochs to wait before stopping
    min_delta=0.0001,  # Minimum change to qualify as an improvement
    verbose=True,  # Print when early stopping is triggered
    check_finite=True,  # Stop if metric becomes NaN or infinite
)

### Trainer
trainer = Trainer(
    accelerator="auto",
    strategy="auto",
    devices="auto",
    num_nodes=1,
    precision="16-mixed",
    logger=[csv_logger, tb_logger],  # Use both loggers
    callbacks=[
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        early_stopping,  # Stop training early if validation metric stops improving
        checkpoint_callback,  # Save best model based on validation IoU
    ],
    max_epochs=TRAIN_EPOCHS,
    log_every_n_steps=1,
    enable_checkpointing=True,
    default_root_dir=OUTPUT_PATH,
)

### Train
trainer.fit(task, datamodule=data)
