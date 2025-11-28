# src/data/augmentations.py

#################################################
# Data augmentation transforms
#################################################

# IMPORTS
import albumentations as A  # noqa: N812
from albumentations.pytorch import ToTensorV2

#################################################
# SETTINGS

MEANS: dict[str, float] = {
    "COASTAL_AEROSOL": 0.16450718,
    "BLUE": 0.1412956,
    "GREEN": 0.13795798,
    "RED": 0.12353792,
    "RED_EDGE_1": 0.1481099,
    "RED_EDGE_2": 0.23991728,
    "RED_EDGE_3": 0.28587557,
    "NIR_BROAD": 0.26345379,
    "NIR_NARROW": 0.30902815,
    "WATER_VAPOR": 0.04911151,
    "CIRRUS": 0.00652506,
    "SWIR_1": 0.2044958,
    "SWIR_2": 0.11912015,
}

STDS: dict[str, float] = {
    "COASTAL_AEROSOL": 0.06977374,
    "BLUE": 0.07406382,
    "GREEN": 0.07370365,
    "RED": 0.08692279,
    "RED_EDGE_1": 0.07778555,
    "RED_EDGE_2": 0.09105416,
    "RED_EDGE_3": 0.10690993,
    "NIR_BROAD": 0.10096586,
    "NIR_NARROW": 0.11798815,
    "WATER_VAPOR": 0.03380113,
    "CIRRUS": 0.01463465,
    "SWIR_1": 0.09772074,
    "SWIR_2": 0.07659938,
}


#################################################
# CODE


def get_s2_normalization_stats(
    bands: list[str] | None = None,
) -> tuple[list[float], list[float]]:
    """Get normalization statistics for specified Sentinel-2 bands.

    Args:
        bands: List of band names to use. If None, uses all bands in
               MEANS/STDS order. Must match the channel order of your
               input images.

    Returns:
        Tuple of (means, stds) as lists, one value per channel.

    Example:
        >>> bands = ["BLUE", "GREEN", "RED", "NIR_BROAD"]
        >>> means, stds = get_s2_normalization_stats(bands)
        >>> # Returns: ([0.1412956, 0.13795798, 0.12353792, 0.26345379], ...)
    """
    if bands is None:
        # Use all bands in dictionary order (Python 3.7+ preserves insertion order)
        bands = list(MEANS.keys())

    means = [MEANS[band] for band in bands]
    stds = [STDS[band] for band in bands]
    return means, stds


def get_s2_train_transform(
    image_size: int = 256,
    bands: list[str] | None = None,
) -> A.Compose:
    """Albumentations pipeline for training data on Sentinel-2.

    Args:
        image_size: The size of the image to crop to.
        bands: List of band names in the order they appear in your images.
               If None, uses all bands in MEANS/STDS dictionary order.
               Must match the channel order of your input images.

    Returns:
        A Compose object that applies the following transformations:
        - RandomCrop(image_size, image_size)
        - Random horizontal / vertical flips
        - Normalize with Sentinel-2 mean/std (band-specific)
        - Convert to torch tensors
    """
    means, stds = get_s2_normalization_stats(bands)
    return A.Compose(
        [
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=means, std=stds, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )


def get_s2_val_transform(
    image_size: int = 256,
    bands: list[str] | None = None,
) -> A.Compose:
    """Albumentations pipeline for validation data on Sentinel-2.

    Args:
        image_size: The size of the image to crop to.
        bands: List of band names in the order they appear in your images.
               If None, uses all bands in MEANS/STDS dictionary order.
               Must match the channel order of your input images.

    Returns:
        A Compose object that applies the following transformations:
        - CenterCrop(image_size, image_size)
        - Normalize with Sentinel-2 mean/std (band-specific)
        - Convert to torch tensors
    """
    means, stds = get_s2_normalization_stats(bands)
    return A.Compose(
        [
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=means, std=stds, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )
