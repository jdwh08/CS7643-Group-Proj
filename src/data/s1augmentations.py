# src/data/augmentations.py

#################################################
# Data augmentation transforms
#################################################

# IMPORTS
import albumentations as A  # noqa: N812
from albumentations.pytorch import ToTensorV2

#################################################
# SETTINGS

# Same stats as original notebook NORM
S1_MEAN = [0.6851, 0.5235]
S1_STD = [0.0820, 0.1102]

#################################################
# CODE


def get_s1_train_transform(image_size: int = 256) -> A.Compose:
    """Albumentations pipeline for training.

    Args:
        image_size: The size of the image to crop to.

    Returns:
        A Compose object that applies the following transformations:
        - RandomCrop(image_size, image_size)
        - Random horizontal / vertical flips
        - Normalize with S1 mean/std
        - Convert to torch tensors
    """
    return A.Compose(
        [
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=S1_MEAN, std=S1_STD, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )


def get_s1_val_transform(image_size: int = 256) -> A.Compose:
    """Albumentations pipeline for validation.

    Args:
        image_size: The size of the image to crop to.

    Returns:
        A Compose object that applies the following transformations:
        - CenterCrop(image_size, image_size)
        - Normalize with S1 mean/std
        - Convert to torch tensors
    """
    return A.Compose(
        [
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=S1_MEAN, std=S1_STD, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )
