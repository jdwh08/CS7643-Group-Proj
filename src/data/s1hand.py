# src/data/s1hand.py

import os
import csv
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_tiff, normalize_s1, clean_hand_mask


class S1HandDataset(Dataset):
    """
    Hand-labeled Sentinel-1 dataset.

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
        data_root: str,
        split: str = None,           # "train", "valid", "test", or None
        split_csv: str = None,       # overrides split
        transform=None,
        max_samples: int = None,
    ):
        self.data_root = data_root
        self.s1_dir = os.path.join(data_root, "HandLabeled", "S1Hand")
        self.label_dir = os.path.join(data_root, "HandLabeled", "LabelHand")
        self.transform=transform

        # determine CSV path
        if split_csv is not None:
            csv_path = split_csv
        elif split is not None:
            csv_path = os.path.join(
                data_root,
                "splits",
                "flood_handlabeled",
                f"flood_{split}_data.csv",
            )
        else:
            csv_path = None  # fallback: list all available files

        self.samples = []

        # load from CSV
        if csv_path is not None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Missing split CSV: {csv_path}")

            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    img_name = row[0].strip()
                    mask_name = row[1].strip()
                    img_path = os.path.join(self.s1_dir, img_name)
                    mask_path = os.path.join(self.label_dir, mask_name)
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))

        # else: match all available *_S1Hand.tif + *_LabelHand.tif
        else:
            paths = sorted(glob(os.path.join(self.s1_dir, "*_S1Hand.tif")))
            for p in paths:
                stem = os.path.basename(p).replace("_S1Hand.tif", "")
                mask = os.path.join(self.label_dir, f"{stem}_LabelHand.tif")
                if os.path.exists(mask):
                    self.samples.append((p, mask))
                    
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
            
        if len(self.samples) == 0:
            raise RuntimeError(f"No S1Hand samples found in {self.s1_dir}")

        print(
            f"[S1HandDataset] Loaded {len(self.samples)} pairs "
            f"(split={split})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # load raw arrays via rasterio
        img = load_tiff(img_path)             # (2, H, W)
        img = normalize_s1(img)

        mask = load_tiff(mask_path)[0]        # (H, W)
        mask = clean_hand_mask(mask)
        
        if self.transform is None:
            # Return raw normalized S1 + cleaned mask
            # convert to tensors manually so loader works consistently
            img_t = torch.from_numpy(img).float()      # (2,H,W)
            mask_t = torch.from_numpy(mask).long()     # (H,W)
            return img_t, mask_t

        # albumentations expects HWC image
        img_hwc = np.transpose(img, (1, 2, 0))  # (H, W, 2)

        augmented = self.transform(image=img_hwc, mask=mask)
        img_t = augmented["image"]            # (C,H,W) float
        mask_t = augmented["mask"].long()     # (H,W) long

        return img_t, mask_t