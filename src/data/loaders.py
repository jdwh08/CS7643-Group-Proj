# src/data/loaders.py

from typing import Optional, Tuple

from torch.utils.data import DataLoader

from .s1hand import S1HandDataset
from .s1weak import S1WeakDataset
from .augmentations import get_train_transform, get_val_transform


def make_s1hand_loaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 256,
    use_splits: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test loaders for hand-labeled chips.

    - Train  → random augmentation
    - Val    → deterministic center crop + normalize
    - Test   → SAME AS VAL (NO augmentation!)
    """

    if use_splits:
        train_ds = S1HandDataset(
            data_root=data_root,
            split="train",
            transform=get_train_transform(image_size),   
        )
        val_ds = S1HandDataset(
            data_root=data_root,
            split="valid",
            transform=get_val_transform(image_size),   
        )
        test_ds = S1HandDataset(
            data_root=data_root,
            split="test",
            transform=get_val_transform(image_size),    
        )

    else:
        # debugging mode: load all tiles without split
        train_ds = S1HandDataset(
            data_root=data_root,
            split=None,
            transform=get_train_transform(image_size),
        )
        # WARNING: same dataset used for val/test in debug mode
        val_ds = S1HandDataset(
            data_root=data_root,
            split=None,
            transform=get_val_transform(image_size),
        )
        test_ds = S1HandDataset(
            data_root=data_root,
            split=None,
            transform=get_val_transform(image_size),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader



def make_s1weak_loader(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 256,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for S1 Weakly-labeled dataset.

    ALWAYS uses train augmentation (weak labels => train only).
    """

    ds = S1WeakDataset(
        data_root=data_root,
        transform=get_train_transform(image_size),  
        max_samples=max_samples,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader