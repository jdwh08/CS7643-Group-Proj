# src/training/prithvi/train_weak.py
#####################################################################
# Flood Segmentation Project [Vision Transformer Transfer Learning with Prithvi]
#####################################################################

### ABOUT
"""Training a Prithvi Vision Transformer model on Sen1Floods11 (weak labels)."""


#####################################################################
### IMPORTS

import logging
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from terratorch.tasks import SemanticSegmentationTask

### OWN MODULES
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from src.data.io import IGNORE_INDEX
from src.training.transformers_transfer_prithvi.eval import test_model
from src.training.transformers_transfer_prithvi.sen1floods11 import (
    Sen1Floods11S2WeakDataModule,
)

#####################################################################
### SETTINGS
SEED = 31415926
DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"
OUTPUT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "outputs"
    / "transformers_transfer_prithvi"
)

ALL_BAND_NAMES = (
    "COASTAL_AEROSOL",
    "BLUE",
    "GREEN",
    "RED",
    "RED_EDGE_1",
    "RED_EDGE_2",
    "RED_EDGE_3",
    "NIR_BROAD",
    "NIR_NARROW",
    "WATER_VAPOR",
    "CIRRUS",
    "SWIR_1",
    "SWIR_2",
)

ALL_BACKBONE_NAMES = (
    "prithvi_eo_v1_100",
    "prithvi_eo_v2_300_tl",
)  # prithvi_eo_v2_600_tl is overkill and too big for normal vram
ALL_BACKBONE_INDICES = {
    "prithvi_eo_v1_100": {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
    "prithvi_eo_v2_300_tl": {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
}
ALL_DECODERS = (
    "UNetDecoder",  # embed_dim, channels
    "UperNetDecoder",  # embed_dim, channels
)
ALL_SCHEDULERS = (
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
)

logger = logging.getLogger(__name__)
# Also log to console
logger.addHandler(logging.StreamHandler(sys.stdout))


def _deep_merge_dict(
    base: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Recursively merge override keys into a copy of base."""
    result = deepcopy(base)
    if not overrides:
        return result
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


### MODEL PARAMETERS (Defaults)
MAIN_METRIC: str = "val/mIoU"  # Micro-average IoU
LOSS_METRIC: (
    Literal["ce", "focal", "dice", "lovasz"]
    | dict[Literal["ce", "focal", "dice", "lovasz"], float]
) = "ce"
CLASS_WEIGHTS: list[float] | None = [0.35, 0.65]  # 1 is flood
# dice / lovasz are competitive but not quite as good as ce pre-tuning

TRAIN_BATCH_SIZE = 64  # not enough vram for 128 unless a100
BANDS = [
    # "COASTAL_AEROSOL",  # ocean water quality stuff, atmosphere
    "BLUE",  # water penetration and turbidity
    "GREEN",  # modified normalized difference water index
    "RED",  # vegetation (NDVI)
    # "RED_EDGE_1",  # vegetation
    "RED_EDGE_2",  # vegetation
    # "RED_EDGE_3",  # vegetation
    "NIR_BROAD",  # water absorbtion, vegetation
    # "NIR_NARROW",  #  water absorbtion
    # "WATER_VAPOR",  # cloud, atmosphere
    # "CIRRUS",  # cloud, atmosphere
    "SWIR_1",  # absorbtion, modified normalized difference water index
    "SWIR_2",  # absorbtion, MNDWI
]
BACKBONE = ALL_BACKBONE_NAMES[0]
BACKBONE_INDICES = ALL_BACKBONE_INDICES[BACKBONE]
DECODER = ALL_DECODERS[0]  # unet, upernet, fcn
DECODER_CHANNELS = [1024, 512, 256, 128] if ALL_DECODERS[0] == "UNetDecoder" else 512
HEAD_DROPOUT = 0.1
WEIGHT_DECAY = 0.1
LR = 1.0e-4

TRAIN_EPOCHS = 50
SCHEDULER = ALL_SCHEDULERS[0]  # 1 is cyclic
SCHEDULER_HPARAMS: dict[str, Any] = {
    "T_max": TRAIN_EPOCHS,
    "eta_min": LR / 1e6,
}
# SCHEDULER_HPARAMS = {
#     "T_0": 10,
#     "T_mult": 2,
#     "eta_min": LR/1e4,
# }
ES_PATIENCE = 10  # NOTE(jdwh08): ES auto disabled if using cyclic lr scheduler
ES_THRESH = 1e-4


@dataclass
class PrithviWeakConfig:
    data_path: Path = Path(__file__).parent.parent.parent.parent / "data"
    output_path: Path = (
        Path(__file__).parent.parent.parent.parent
        / "outputs"
        / "transformers_transfer_prithvi"
    )
    main_metric: str = "val/mIoU"  # Micro-average IoU
    loss_metric: (
        Literal["ce", "focal", "dice", "lovasz"]
        | dict[Literal["ce", "focal", "dice", "lovasz"], float]
    ) = field(default_factory=lambda: LOSS_METRIC)
    class_weights: list[float] | None = field(default_factory=lambda: CLASS_WEIGHTS)
    train_batch_size: int = 64  # not enough vram for 128 unless a100
    bands: list[str] = field(default_factory=lambda: BANDS)
    backbone: str = BACKBONE
    backbone_indices: dict[str, Any] | None = None
    decoder: str = DECODER  # unet, upernet, fcn
    decoder_channels: Any = field(default_factory=lambda: DECODER_CHANNELS)
    decoder_scale_modules: bool = False
    head_dropout: float = 0.1
    weight_decay: float = 0.1
    lr: float = LR
    train_epochs: int = TRAIN_EPOCHS
    scheduler: str = SCHEDULER  # 1 is cyclic
    scheduler_hparams: dict[str, Any] = field(default_factory=lambda: SCHEDULER_HPARAMS)
    es_patience: int = (
        ES_PATIENCE  # NOTE(jdwh08): ES auto disabled if using cyclic lr scheduler
    )
    es_thresh: float = ES_THRESH
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    precision: str = "16-mixed"
    log_every_n_steps: int = 1
    plot_on_val: int = 10
    freeze_backbone: bool = False
    freeze_decoder: bool = False
    logger_name: str = "logs_weak"
    logger_version: int | None = None
    save_last: bool = True
    seed: int = SEED

    def __post_init__(self) -> None:
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

    @classmethod
    def from_overrides(
        cls, overrides: dict[str, Any] | None = None
    ) -> "PrithviWeakConfig":
        base = asdict(cls())
        merged = _deep_merge_dict(base, overrides)
        if not (overrides or {}).get("scheduler_hparams"):
            # Recompute scheduler hparams so they follow updated lr/train_epochs.
            merged["scheduler_hparams"] = {}
        return cls(**merged)


#####################################################################
### CODE


class PrithviWeakTrainer:
    """End-to-end training pipeline for weakly-labeled Sen1Floods11."""

    def __init__(
        self, config: PrithviWeakConfig, config_path: Path | None = None
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def _resolve_backbone_indices(self) -> dict[str, Any]:
        if self.config.backbone_indices:
            return self.config.backbone_indices
        if self.config.backbone not in ALL_BACKBONE_INDICES:
            message = f"Backbone '{self.config.backbone}' not supported."
            raise ValueError(message)
        return ALL_BACKBONE_INDICES[self.config.backbone]

    def _build_datamodule(self) -> Sen1Floods11S2WeakDataModule:
        return Sen1Floods11S2WeakDataModule(
            data_root=str(self.config.data_path),
            batch_size=self.config.train_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
            bands=self.config.bands,
        )

    def _build_logger(self) -> CSVLogger:
        csv_logger = CSVLogger(
            save_dir=str(self.config.output_path),
            name=self.config.logger_name,
            version=self.config.logger_version,
        )
        model_output_path = (
            self.config.output_path
            / f"{csv_logger.name}"
            / f"version_{csv_logger.version}"
        )
        model_output_path.mkdir(parents=True, exist_ok=True)
        return csv_logger

    def _build_task(self) -> SemanticSegmentationTask:
        backbone_indices = self._resolve_backbone_indices()
        model_args = {
            "backbone": self.config.backbone,
            "backbone_pretrained": False,
            "backbone_bands": self.config.bands,
            "decoder": self.config.decoder,
            "decoder_channels": self.config.decoder_channels,
            "decoder_scale_modules": self.config.decoder_scale_modules,
            "num_classes": 2,
            "rescale": True,
            "head_dropout": self.config.head_dropout,
            "necks": [
                backbone_indices,
                {"name": "ReshapeTokensToImage"},
            ],
        }
        if (
            ALL_DECODERS[0] == self.config.decoder
            and "decoder_scale_modules" in model_args
        ):
            # UNet doesn't have scale_modules
            model_args.pop("decoder_scale_modules", None)

        return SemanticSegmentationTask(
            model_factory="EncoderDecoderFactory",
            model_args=model_args,
            loss=self.config.loss_metric,
            ignore_index=IGNORE_INDEX,
            freeze_backbone=self.config.freeze_backbone,
            freeze_decoder=self.config.freeze_decoder,
            optimizer="AdamW",
            optimizer_hparams={
                "weight_decay": self.config.weight_decay,
            },
            lr=self.config.lr,
            scheduler=self.config.scheduler,
            scheduler_hparams=self.config.scheduler_hparams,
            class_weights=self.config.class_weights,
        )

    def _build_callbacks(
        self,
    ) -> tuple[list[Any], ModelCheckpoint, EarlyStopping | None]:
        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.main_metric,
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}",
            auto_insert_metric_name=True,
            save_last=self.config.save_last,
            verbose=True,
        )
        early_stopping = EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=self.config.es_patience,
            min_delta=self.config.es_thresh,
            verbose=True,
            check_finite=True,
        )
        callbacks: list[Any] = [
            RichProgressBar(),
            LearningRateMonitor(logging_interval="epoch"),
            early_stopping,
            checkpoint_callback,
        ]
        if ALL_SCHEDULERS[1] == self.config.scheduler:
            # Don't do early stopping with cyclic
            callbacks.remove(early_stopping)
            early_stopping = None
        return callbacks, checkpoint_callback, early_stopping

    def run(self) -> Path:
        seed_everything(seed=self.config.seed, workers=True)

        data = self._build_datamodule()
        csv_logger = self._build_logger()
        model_output_path = (
            self.config.output_path
            / f"{csv_logger.name}"
            / f"version_{csv_logger.version}"
        )
        task = self._build_task()
        callbacks, checkpoint_callback, _ = self._build_callbacks()

        trainer = Trainer(
            accelerator="auto",
            strategy="auto",
            devices="auto",
            num_nodes=1,
            precision=self.config.precision,
            logger=[csv_logger],
            callbacks=callbacks,
            max_epochs=self.config.train_epochs,
            log_every_n_steps=self.config.log_every_n_steps,
            enable_checkpointing=True,
            default_root_dir=self.config.output_path,
        )

        trainer.fit(task, datamodule=data)
        ckpt_path = checkpoint_callback.best_model_path or "best"
        trainer.test(ckpt_path=ckpt_path, datamodule=data)

        test_model(model_output_path)
        return model_output_path


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration for the weak trainer."""
    if config_path is None:
        return {}
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)
    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}


def main(config_path: Path | None = None) -> Path:
    raw_config = load_config(config_path)
    trainer_config = PrithviWeakConfig.from_overrides(raw_config.get("settings"))
    trainer = PrithviWeakTrainer(trainer_config, config_path=config_path)
    return trainer.run()


if __name__ == "__main__":
    main()
