# src/data/augmentations.py

import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Same stats as original notebook NORM
S1_MEAN = [0.6851, 0.5235]
S1_STD  = [0.0820, 0.1102]

def get_train_transform(image_size: int = 256):
    """
    Albumentations pipeline that replicates:
      - RandomCrop(256,256)
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
    
def get_val_transform(image_size: int = 256):
    """
    Center crop + normalize + tensor, no random aug.
    """
    return A.Compose(
        [
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=S1_MEAN, std=S1_STD, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )