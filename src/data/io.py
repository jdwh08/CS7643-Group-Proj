# src/data/io.py

#################################################
# I/O utilities for data loading and normalization
#################################################

# IMPORTS
from typing import Any

import numpy as np
import rasterio

#################################################
# SETTINGS

IGNORE_INDEX = 255  # NOTE(jdwh08): Explicitly mark ignored index (-1 -> 255)

#################################################
# CODE


def load_tiff(path: str) -> np.ndarray[np.float32, Any]:
    """Load a GeoTIFF as (C,H,W).

    Args:
        path: The path to the GeoTIFF file.

    Returns:
        The array loaded from the GeoTIFF file.
    """
    with rasterio.open(path) as src:
        arr = src.read()
    return arr


def normalize_s1(arr: np.ndarray[np.float32, Any]) -> np.ndarray[np.float32, Any]:
    """Normalize S1 to match Sen1Floods11 notebook example code.

    Args:
        arr: The array to normalize.

    Returns:
        The normalized array.

    Note:
        - Replace NaNs
        - Clip dB to [-50, 1]
        - Scale to [0,1]
    """
    arr = np.nan_to_num(arr)
    arr = np.clip(arr, -50.0, 1.0)
    arr = (arr + 50.0) / 51.0
    return arr.astype(np.float32)


def normalize_s2(
    arr: np.ndarray[np.float32, Any],
    no_data_replace: float | None = 0.0,
) -> np.ndarray[np.float32, Any]:
    """Normalize S2 data to match Sen1Floods11 processing.

    Args:
        arr: The array to normalize, shape (C, H, W).
        constant_scale: Scale factor to multiply image values by. Defaults to 0.0001.
        no_data_replace: Replace NaN values with this value. If None, no replacement.
                        Defaults to 0.0.

    Returns:
        The normalized array with same shape (C, H, W).

    Note:
        - Applies constant scaling (typically 0.0001 for Sentinel-2 reflectance)
        - Optionally replaces NaN values
        - Returns float32 array
    """
    arr = arr.astype(np.float32)
    if no_data_replace is not None:
        arr = np.nan_to_num(arr, nan=no_data_replace)
    return arr


def clean_hand_mask(mask: np.ndarray[np.int16, Any]) -> np.ndarray[np.uint8, Any]:
    """Clean hand-labeled mask.

    Args:
        mask: The mask to clean.

    Returns:
        The cleaned mask.

    Note:
        -1 = NoData  → 255 (ignore)
        0 = Land
        1 = Water
    """
    mask = mask.astype(np.int16)
    # NOTE(schen602): 255 is a special reserved class index used in segmentation
    # to mean "Ignore this pixel when computing loss".
    mask = np.where(mask == -1, IGNORE_INDEX, mask)
    return mask.astype(np.uint8)


def clean_weak_mask(mask: np.ndarray[np.int16, Any]) -> np.ndarray[np.uint8, Any]:
    """Clean weakly-labeled Otsu-based mask.

    Args:
        mask: The mask to clean.

    Returns:
        The cleaned mask.

    Note:
        -1 = NoData → 255 (ignore)
        0 = Land
        1 = Water
    """
    mask = mask.astype(np.int16)
    mask = np.where(mask == -1, IGNORE_INDEX, mask)
    return mask.astype(np.uint8)
