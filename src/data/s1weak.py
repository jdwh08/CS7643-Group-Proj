# src/data/s1weak.py

#################################################
# Sentinel-1 Weakly-labeled dataset
#################################################

# IMPORTS
import logging
from pathlib import Path

import albumentations as A  # noqa: N812
import numpy as np
import torch
from torch.utils.data import Dataset

### OWN MODULES
from .io import clean_weak_mask, load_tiff, normalize_s1

#################################################
# SETTINGS

logger = logging.getLogger(__name__)

#################################################
# CODE


class S1WeakDataset(Dataset[dict[str, torch.Tensor]]):
    """Weakly-labeled Otsu-based Sentinel-1 dataset.

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
        data_root: str | None = None,
        max_samples: int | None = None,  # limit for local dev
        transform: A.Compose | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_root: The root directory of the dataset.
            max_samples: The maximum number of samples to load.
            transform: The transform to apply to the data.
        """
        if data_root is None:
            self.data_root = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_root = Path(data_root)
        self.s1_dir = self.data_root / "WeaklyLabeled" / "S1Weak"
        self.label_dir = self.data_root / "WeaklyLabeled" / "S1OtsuLabelWeak"

        self.transform = transform

        # gather S1Weak files
        s1_files = sorted(self.s1_dir.glob("*.tif"))
        if max_samples:
            s1_files = s1_files[:max_samples]

        self.samples = []
        for f in s1_files:
            fname = f.name
            label_name = fname.replace("S1Weak", "S1OtsuLabelWeak")
            label_path = self.label_dir / label_name

            if label_path.exists():
                self.samples.append((f, label_path))
            else:
                logger.warning(f"Missing label for {fname}")

        if len(self.samples) == 0:
            msg = f"No S1Weak samples found in {self.s1_dir}"
            raise RuntimeError(msg)

        logger.info(
            f"[S1WeakDataset] Loaded {len(self.samples)} weak tiles "
            f"(max_samples={max_samples})"
        )

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

        img = normalize_s1(load_tiff(img_path))
        mask = clean_weak_mask(load_tiff(mask_path)[0])

        if self.transform is None:
            # Return raw normalized S1 + cleaned mask
            # convert to tensors manually so loader works consistently
            img_t = torch.from_numpy(img).float()  # (2,H,W)
            mask_t = torch.from_numpy(mask).long()  # (H,W)
            return {"image": img_t, "mask": mask_t}

        img_hwc = np.transpose(img, (1, 2, 0))

        augmented = self.transform(image=img_hwc, mask=mask)
        img_t = augmented["image"]
        mask_t = augmented["mask"].long()

        return {"image": img_t, "mask": mask_t}
