# src/data/s1weak.py

import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_tiff, normalize_s1, clean_weak_mask


class S1WeakDataset(Dataset):
    """
    Weakly-labeled Otsu-based Sentinel-1 dataset.

    Expected structure under data_root:

        WeaklyLabeled/
            S1Weak/
                <prefix>_S1Weak.tif
            S1OtsuLabelWeak/
                <prefix>_S1OtsuLabelWeak.tif

    File pairing rule:
        Replace "S1Weak" → "S1OtsuLabelWeak"
    """

    def __init__(
        self,
        data_root: str,
        max_samples: int = None,       # limit for local dev
        transform=None,
    ):
        self.data_root = data_root
        self.s1_dir = os.path.join(data_root, "WeaklyLabeled", "S1Weak")
        self.label_dir = os.path.join(data_root, "WeaklyLabeled", "S1OtsuLabelWeak")

        self.transform = transform

        # gather S1Weak files
        s1_files = sorted(glob(os.path.join(self.s1_dir, "*.tif")))
        if max_samples:
            s1_files = s1_files[:max_samples]

        self.samples = []
        for f in s1_files:
            fname = os.path.basename(f)
            label_name = fname.replace("S1Weak", "S1OtsuLabelWeak")
            label_path = os.path.join(self.label_dir, label_name)

            if os.path.exists(label_path):
                self.samples.append((f, label_path))
            else:
                print(f"[WARN] Missing label for {fname}")

        if len(self.samples) == 0:
            raise RuntimeError("No S1Weak samples found.")

        print(
            f"[S1WeakDataset] Loaded {len(self.samples)} weak tiles "
            f"(max_samples={max_samples})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = normalize_s1(load_tiff(img_path))
        mask = clean_weak_mask(load_tiff(mask_path)[0])

        if self.transform is None:
            # Return raw normalized S1 + cleaned mask
            # convert to tensors manually so loader works consistently
            img_t = torch.from_numpy(img).float()      # (2,H,W)
            mask_t = torch.from_numpy(mask).long()     # (H,W)
            return img_t, mask_t
        
        img_hwc = np.transpose(img, (1, 2, 0))

        augmented = self.transform(image=img_hwc, mask=mask)
        img_t = augmented["image"]
        mask_t = augmented["mask"].long()

        return img_t, mask_t