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

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar
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

TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 16
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

logger = logging.getLogger(__name__)
# Also log to console
logger.addHandler(logging.StreamHandler(sys.stdout))

torch.manual_seed(31415926)
torch.cuda.manual_seed(3141592)
torch.cuda.manual_seed_all(3141592)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#####################################################################
### CODE

data = Sen1Floods11S2HandDataModule(
    data_root=str(DATA_PATH),
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    bands=BANDS,
)

task = SemanticSegmentationTask(
    model_factory="EncoderDecoderFactory",
    model_args={
        "backbone": "prithvi_eo_v2_300",  # "prithvi_eo_v1_100",  # "prithvi_eo_v2_300",
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
        "weight_decay": 5e-3,
    },
    lr=5.0e-5,
    scheduler="CosineAnnealingLR",
    scheduler_hparams={
        "T_max": TRAIN_EPOCHS,
        "eta_min": 0,
    },
)

trainer = Trainer(
    accelerator="auto",
    strategy="auto",
    devices="auto",
    num_nodes=1,
    precision="16-mixed",
    logger=True,
    callbacks=[
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    max_epochs=TRAIN_EPOCHS,
    log_every_n_steps=1,
    enable_checkpointing=True,
    default_root_dir=OUTPUT_PATH,
)


trainer.fit(task, datamodule=data)
