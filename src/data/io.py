import numpy as np
import rasterio

def load_tiff(path: str) -> np.ndarray:
    """Load a GeoTIFF as (C,H,W)"""
    with rasterio.open(path) as src:
        arr = src.read()
    return arr

def normalize_s1(arr: np.ndarray) -> np.ndarray:
    """
    Match Sen1Floods11 notebook normalization:
    - Replace NaNs
    - Clip dB to [-50, 1]
    - Scale to [0,1]
    """
    arr = np.nan_to_num(arr)
    arr = np.clip(arr, -50.0, 1.0)
    arr = (arr + 50.0) / 51.0
    return arr.astype(np.float32)

def clean_hand_mask(mask: np.ndarray) -> np.ndarray:
    """
    HAND-LABELED MASK:
      -1 = NoData  → 255 (ignore)
       0 = Not water
       1 = Water
    """
    mask = mask.astype(np.int16)
    #The number 255 is a special reserved class index used in segmentation to mean:Ignore this pixel when computing loss.
    mask = np.where(mask == -1, 255, mask)
    return mask.astype(np.uint8)

def clean_weak_mask(mask: np.ndarray) -> np.ndarray:
    """
    WEAKLY-LABELED OTSU MASK:
      -1 = NoData → 255
       0 = Not water
       1 = Water
    """
    mask = mask.astype(np.int16)
    mask = np.where(mask == -1, 255, mask)
    return mask.astype(np.uint8)