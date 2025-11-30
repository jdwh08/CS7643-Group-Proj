# src/data/s2weak.py

#################################################
# Sentinel-2 Weakly-labeled dataset
#################################################
# NOTE: Plot and coordinates / item interface from TerraTorch

# IMPORTS
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A  # noqa: N812
import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset

### OWN MODULES
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from src.data.io import IGNORE_INDEX, clean_weak_mask, load_tiff, normalize_s2

#################################################
# SETTINGS


logger = logging.getLogger(__name__)

# Sentinel-2 band names in standard order
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

RGB_BANDS = ("RED", "GREEN", "BLUE")

MIN_CSV_COLUMNS = 2
NUM_CLASSES = 2

#################################################
# CODE


class S2WeakDataset(Dataset[dict[str, torch.Tensor]]):
    """Weakly-labeled Index-based Sentinel-2 dataset.

    Expected structure under data_root:

        WeaklyLabeled/
            S2Weak/
                <prefix>_S2Weak.tif
            S2IndexLabelWeak/
                <prefix>_S2IndexLabelWeak.tif

    File pairing rule:
        Replace "S2Weak" → "S2IndexLabelWeak"
    """

    def __init__(
        self,
        data_root: str | None = None,
        max_samples: int | None = None,  # limit for local dev
        transform: A.Compose | None = None,
        constant_scale: float = 0.0001,
        bands: Sequence[str] = ALL_BAND_NAMES,
        no_data_replace: float | None = 0.0,
        use_metadata: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_root: The root directory of the dataset.
                If None, uses the default data root.
            max_samples: The maximum number of samples to load.
            transform: The transform to apply to the data.
            constant_scale (float): Factor to multiply image values by.
                Defaults to 0.0001.
            bands: List of band names to use. Must be a subset of ALL_BAND_NAMES.
                   Defaults to all bands.
            no_data_replace: Replace NaN values in input images with this value.
                            If None, does no replacement. Defaults to 0.0.
            use_metadata: Whether to return metadata info (time and location).
                         Requires Sen1Floods11_Metadata.geojson file.
        """
        if data_root is None:
            self.data_root = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_root = Path(data_root)
        self.s2_dir = self.data_root / "WeaklyLabeled" / "S2Weak"
        self.label_dir = self.data_root / "WeaklyLabeled" / "S2IndexLabelWeak"

        self.transform = transform
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.use_metadata = use_metadata

        # Validate and set up bands
        if not all(band in ALL_BAND_NAMES for band in bands):
            invalid = [b for b in bands if b not in ALL_BAND_NAMES]
            msg = f"Invalid band names: {invalid}. Must be from {ALL_BAND_NAMES}"
            raise ValueError(msg)

        self.bands = tuple(bands)
        # Get indices of selected bands in the full band order
        self.band_indices = np.array([ALL_BAND_NAMES.index(b) for b in bands])

        # gather S2Weak files
        s2_files = sorted(self.s2_dir.glob("*.tif"))
        if max_samples:
            s2_files = s2_files[:max_samples]

        self.samples = []
        for f in s2_files:
            fname = f.name
            label_name = fname.replace("S2Weak", "S2IndexLabelWeak")
            label_path = self.label_dir / label_name

            if label_path.exists():
                self.samples.append((f, label_path))
            else:
                logger.warning(f"Missing label for {fname}")

        if len(self.samples) == 0:
            msg = f"No S2Weak samples found in {self.s2_dir}"
            raise RuntimeError(msg)

        # Load metadata if requested
        self.metadata = None
        if self.use_metadata:
            metadata_file = self.data_root / "Sen1Floods11_Metadata.geojson"
            if metadata_file.exists():
                self.metadata = geopandas.read_file(metadata_file)
            else:
                logger.warning(
                    f"Metadata file not found: {metadata_file}. "
                    "Metadata will not be available."
                )
                self.use_metadata = False

        logger.info(
            f"[S2WeakDataset] Loaded {len(self.samples)} weak tiles "
            f"(max_samples={max_samples}, bands={len(self.bands)})"
        )

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.samples)

    def _get_date(self, index: int) -> torch.Tensor:
        """Get temporal coordinates from metadata.

        Matches the TerraTorch interface.

        Args:
            index: Sample index.

        Returns:
            Tensor of shape (1, 2) with [year, day_of_year-1].
        """
        fallback_date = "1998-10-13"
        if self.metadata is None:
            # Fallback date if metadata not available
            date = pd.to_datetime(fallback_date)
        else:
            file_name = self.samples[index][0]
            location = file_name.name.split("_")[0]
            matches = self.metadata[self.metadata["location"] == location]
            if matches.shape[0] != 1:
                date = pd.to_datetime(fallback_date)
            else:
                date = pd.to_datetime(matches["s1_date"].item())

        return torch.tensor(
            [[date.year, date.dayofyear - 1]], dtype=torch.float32
        )  # (n_timesteps, coords)

    def _get_coords(self, img_path: Path) -> torch.Tensor:
        """Get location coordinates (lat, lon) from image center.

        Matches the TerraTorch interface.

        Args:
            img_path: Path to the image file.

        Returns:
            Tensor of shape (2,) with [lat, lon].
        """
        with rasterio.open(img_path) as src:
            # Get center coordinates
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2.0
            center_lat = (bounds.bottom + bounds.top) / 2.0

        return torch.tensor([center_lat, center_lon], dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: The index of the sample to get.

        Returns:
            A dictionary with "image" and "mask" keys. If use_metadata is True,
            also includes "location_coords" and "temporal_coords" keys.
        """
        img_path, mask_path = self.samples[idx]

        # Load raw arrays via rasterio
        img = load_tiff(str(img_path))  # (C, H, W) - all 13 bands

        # Filter to selected bands
        img = img[self.band_indices, ...]  # (num_selected_bands, H, W)

        # Apply normalization: no_data replacement
        img = normalize_s2(img, no_data_replace=self.no_data_replace)

        mask = clean_weak_mask(load_tiff(str(mask_path))[0])
        # Clamp any invalid values: keep 255 (ignore), 0, 1 as-is; clamp others to [0, 1]
        valid_mask = (mask == IGNORE_INDEX) | (mask == 0) | (mask == 1)
        mask = np.where(valid_mask, mask, np.clip(mask, 0, 1))

        # Get metadata if requested
        location_coords = None
        temporal_coords = None
        if self.use_metadata:
            location_coords = self._get_coords(img_path)
            temporal_coords = self._get_date(idx)

        if self.transform is None:
            # Return raw normalized S2 + cleaned mask
            # convert to tensors manually so loader works consistently
            img_t = torch.from_numpy(img * self.constant_scale).float()  # (C, H, W)
            mask_t = torch.from_numpy(mask).long()  # (H, W)

            result = {"image": img_t, "mask": mask_t}
            if self.use_metadata:
                # Type narrowing: if use_metadata is True, coords are not None
                if location_coords is None or temporal_coords is None:
                    msg = (
                        "Missing location & temporal coordinates when using metadata."
                        " Check your dataset configuration."
                    )
                    raise ValueError(msg)
                result["location_coords"] = location_coords.float()
                result["temporal_coords"] = temporal_coords.float()
            return result

        # albumentations expects HWC image
        img_hwc = np.transpose(img, (1, 2, 0))  # (H, W, C)

        augmented = self.transform(image=img_hwc, mask=mask)
        img_t = augmented["image"]  # (C, H, W) float
        mask_t = augmented["mask"].long()  # (H, W) long

        result = {"image": img_t, "mask": mask_t}
        if self.use_metadata:
            # Type narrowing: if use_metadata is True, coords are not None
            if location_coords is None or temporal_coords is None:
                msg = (
                    "Location & temporal coordinates are required when using metadata."
                    " Check your dataset configuration."
                )
                raise ValueError(msg)
            result["location_coords"] = location_coords.float()
            result["temporal_coords"] = temporal_coords.float()
        return result

    def clip_image(
        self, image: np.ndarray[np.float32, Any]
    ) -> np.ndarray[np.float32, Any]:
        """Clip image values to [0, 1].

        Args:
            image: The image to clip.

        Returns:
            The clipped image.
        """
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)
        return image

    def plot(
        self, sample: dict[str, torch.Tensor], suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset. Code adapted from TerraTorch.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        num_images = 4

        rgb_indices = [self.bands.index(band) for band in RGB_BANDS]
        if len(rgb_indices) != 3:  # noqa: PLR2004
            msg = "Dataset missing some of the RGB bands"
            raise ValueError(msg)

        # RGB -> channels-last
        image = sample["image"][rgb_indices, ...].permute(1, 2, 0).numpy()
        image = self.clip_image(image)

        mask = sample["mask"].numpy().squeeze()

        if "prediction" in sample:
            prediction = sample["prediction"]
            num_images += 1
        else:
            prediction = None

        fig, ax = plt.subplots(1, num_images, figsize=(12, 5), layout="compressed")

        ax[0].axis("off")

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        ax[1].axis("off")
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        ax[2].axis("off")
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(mask, cmap="jet", norm=norm)

        ax[3].axis("off")
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(mask, cmap="jet", alpha=0.3, norm=norm)

        if "prediction" in sample:
            ax[4].title.set_text("Predicted Mask")
            ax[4].imshow(prediction, cmap="jet", norm=norm)

        cmap = plt.get_cmap("jet")
        legend_data = [[i, cmap(norm(i)), str(i)] for i in range(2)]
        handles = [
            Rectangle((0, 0), 1, 1, color=tuple(v for v in c))
            for _, c, _ in legend_data
        ]
        labels = [n for _, _, n in legend_data]
        ax[0].legend(handles, labels, loc="center")
        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
