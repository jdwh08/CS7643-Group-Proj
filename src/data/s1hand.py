# src/data/s1hand.py

#################################################
# Sentinel-1 Hand-labeled dataset
#################################################

# IMPORTS
import csv
import logging
from pathlib import Path

import albumentations as A  # noqa: N812
import numpy as np
import torch
from torch.utils.data import Dataset

### OWN MODULES
from .io import clean_hand_mask, load_tiff, normalize_s1

#################################################
# SETTINGS

logger = logging.getLogger(__name__)
MIN_CSV_COLUMNS = 2
NUM_CLASSES = 2

#################################################
# CODE


class S1HandDataset(Dataset[dict[str, torch.Tensor]]):
    """Hand-labeled Sentinel-1 dataset.

    Expected structure under data_root:

        HandLabeled/
            S1Hand/
                <prefix>_S1Hand.tif
            LabelHand/
                <prefix>_LabelHand.tif

        splits/flood_handlabeled/
            flood_train_data.csv
            flood_valid_data.csv
            flood_test_data.csv

    Each CSV row:
        Ghana_103272_S1Hand.tif,Ghana_103272_LabelHand.tif
    """

    def __init__(
        self,
        data_root: str | None = None,
        split: str | None = None,  # "train", "valid", "test", or None
        split_csv: str | None = None,  # overrides split
        transform: A.Compose | None = None,
        max_samples: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_root: The root directory of the dataset.
            split: The split to load.
            split_csv: The CSV file to load.
            transform: The transform to apply to the data.
            max_samples: The maximum number of samples to load.
        """
        if data_root is None:
            self.data_root = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_root = Path(data_root)

        self.s1_dir = self.data_root / "HandLabeled" / "S1Hand"
        self.label_dir = self.data_root / "HandLabeled" / "LabelHand"
        self.transform = transform

        # determine CSV path
        csv_path: Path | None
        if split_csv is not None:
            csv_path = Path(split_csv)
        elif split is not None:
            csv_path = (
                self.data_root
                / "splits"
                / "flood_handlabeled"
                / f"flood_{split}_data.csv"
            )
        else:
            csv_path = None  # NOTE(schen602): fallback -- list all available files

        self.samples = []

        # load from CSV
        if csv_path is not None:
            if not csv_path.exists():
                msg = f"Missing split CSV file: {csv_path}"
                raise FileNotFoundError(msg)

            with csv_path.open() as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < MIN_CSV_COLUMNS:
                        continue
                    img_name = row[0].strip()
                    mask_name = row[1].strip()
                    img_path = self.s1_dir / img_name
                    mask_path = self.label_dir / mask_name
                    if img_path.exists() and mask_path.exists():
                        self.samples.append((img_path, mask_path))

        # else: match all available *_S1Hand.tif + *_LabelHand.tif
        else:
            paths = sorted(self.s1_dir.glob("*_S1Hand.tif"))
            for p in paths:
                stem = p.stem.replace("_S1Hand.tif", "")
                mask = self.label_dir / f"{stem}_LabelHand.tif"
                if mask.exists():
                    self.samples.append((p, mask))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            msg = f"No S1Hand samples found in {self.s1_dir}"
            raise RuntimeError(msg)

        logger.info(f"[S1HandDataset] Loaded {len(self.samples)} pairs (split={split})")

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: The index of the sample to get.

        Returns:
            A dictionary with "image" and "mask" keys.
        """
        img_path, mask_path = self.samples[idx]

        # load raw arrays via rasterio
        img = load_tiff(img_path)  # (2, H, W)
        img = normalize_s1(img)

        mask = load_tiff(mask_path)[0]  # (H, W)
        mask = clean_hand_mask(mask)

        if self.transform is None:
            # Return raw normalized S1 + cleaned mask
            # convert to tensors manually so loader works consistently
            img_t = torch.from_numpy(img).float()  # (2,H,W)
            mask_t = torch.from_numpy(mask).long()  # (H,W)
            return {"image": img_t, "mask": mask_t}

        # albumentations expects HWC image
        img_hwc = np.transpose(img, (1, 2, 0))  # (H, W, 2)

        augmented = self.transform(image=img_hwc, mask=mask)
        img_t = augmented["image"]  # (C,H,W) float
        mask_t = augmented["mask"].long()  # (H,W) long

        return {"image": img_t, "mask": mask_t}
