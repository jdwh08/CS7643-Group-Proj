# src/data/loaders.py

#################################################
# DataLoader factory functions
#################################################

# IMPORTS
import torch
from torch.utils.data import DataLoader

### OWN MODULES
from .s1augmentations import get_s1_train_transform, get_s1_val_transform
from .s1hand import S1HandDataset
from .s1weak import S1WeakDataset
from .s2augmentations import get_s2_train_transform, get_s2_val_transform
from .s2hand import ALL_BAND_NAMES, S2HandDataset
from .s2weak import S2WeakDataset

#################################################
# SETTINGS

#################################################
# CODE


def make_s1hand_loaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 256,
    use_splits: bool = True,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
]:
    """Create train/val/test loaders for hand-labeled chips.

    - Train  → random augmentation
    - Val    → deterministic center crop + normalize
    - Test   → SAME AS VAL (NO augmentation!)
    """
    if use_splits:
        train_ds = S1HandDataset(
            data_root=data_root,
            split="train",
            transform=get_s1_train_transform(image_size),
        )
        val_ds = S1HandDataset(
            data_root=data_root,
            split="valid",
            transform=get_s1_val_transform(image_size),
        )
        test_ds = S1HandDataset(
            data_root=data_root,
            split="test",
            transform=get_s1_val_transform(image_size),
        )

    else:
        # debugging mode: load all tiles without split
        train_ds = S1HandDataset(
            data_root=data_root,
            split=None,
            transform=get_s1_train_transform(image_size),
        )
        # WARNING: same dataset used for val/test in debug mode
        val_ds = S1HandDataset(
            data_root=data_root,
            split=None,
            transform=get_s1_val_transform(image_size),
        )
        test_ds = S1HandDataset(
            data_root=data_root,
            split=None,
            transform=get_s1_val_transform(image_size),
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
    max_samples: int | None = None,
    shuffle: bool = True,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Create a DataLoader for S1 Weakly-labeled dataset.

    Args:
        data_root: The root directory of the dataset.
        batch_size: The batch size.
        num_workers: The number of workers to use.
        image_size: The size of the image to crop to.
        max_samples: The maximum number of samples to load.
        shuffle: Whether to shuffle the dataset.

    Returns:
        A DataLoader for the S1 Weakly-labeled dataset.

    Note:
        ALWAYS uses train augmentation (weak labels => train only).
    """
    ds = S1WeakDataset(
        data_root=data_root,
        transform=get_s1_train_transform(image_size),
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


def make_s2weak_loader(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 256,
    max_samples: int | None = None,
    shuffle: bool = True,
    bands: list[str] | None = None,
    no_data_replace: float | None = 0.0,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Create a DataLoader for S2 Weakly-labeled dataset.

    Args:
        data_root: The root directory of the dataset.
        batch_size: The batch size.
        num_workers: The number of workers to use.
        image_size: The size of the image to crop to.
        max_samples: The maximum number of samples to load.
        shuffle: Whether to shuffle the dataset.
        bands: List of band names to use. If None, uses all bands.
        no_data_replace: Replace NaN values in input images with this value.

    Returns:
        A DataLoader for the S2 Weakly-labeled dataset.

    Note:
        ALWAYS uses train augmentation (weak labels => train only).
    """
    if bands is None:
        bands = list(ALL_BAND_NAMES)

    # Create transform with band-specific normalization
    train_transform = get_s2_train_transform(image_size=image_size, bands=bands)

    ds = S2WeakDataset(
        data_root=data_root,
        transform=train_transform,
        max_samples=max_samples,
        bands=bands,
        no_data_replace=no_data_replace,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def make_s2hand_loaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 256,
    use_splits: bool = True,
    bands: list[str] | None = None,
    no_data_replace: float | None = 0.0,
    no_label_replace: int | None = -1,
    use_metadata: bool = False,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
]:
    """Create train/val/test loaders for hand-labeled Sentinel-2 chips.

    Args:
        data_root: The root directory of the dataset.
        batch_size: The batch size.
        num_workers: The number of workers to use.
        image_size: The size of the image to crop to.
        use_splits: Whether to use train/valid/test splits.
        bands: List of band names to use. If None, uses all bands.
        no_data_replace: Replace NaN values in input images with this value.
        no_label_replace: Replace NaN values in label with this value.
        use_metadata: Whether to return metadata info (time and location).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Note:
        - Train  → random augmentation
        - Val    → deterministic center crop + normalize
        - Test   → SAME AS VAL (NO augmentation!)
    """
    if bands is None:
        bands = list(ALL_BAND_NAMES)

    # Create transforms with band-specific normalization
    train_transform = get_s2_train_transform(image_size=image_size, bands=bands)
    val_transform = get_s2_val_transform(image_size=image_size, bands=bands)

    if use_splits:
        train_ds = S2HandDataset(
            data_root=data_root,
            split="train",
            transform=train_transform,
            bands=bands,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            use_metadata=use_metadata,
        )
        val_ds = S2HandDataset(
            data_root=data_root,
            split="valid",
            transform=val_transform,
            bands=bands,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            use_metadata=use_metadata,
        )
        test_ds = S2HandDataset(
            data_root=data_root,
            split="test",
            transform=val_transform,
            bands=bands,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            use_metadata=use_metadata,
        )

    else:
        # debugging mode: load all tiles without split
        train_ds = S2HandDataset(
            data_root=data_root,
            split=None,
            transform=train_transform,
            bands=bands,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            use_metadata=use_metadata,
        )
        # WARNING: same dataset used for val/test in debug mode
        val_ds = S2HandDataset(
            data_root=data_root,
            split=None,
            transform=val_transform,
            bands=bands,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            use_metadata=use_metadata,
        )
        test_ds = S2HandDataset(
            data_root=data_root,
            split=None,
            transform=val_transform,
            bands=bands,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            use_metadata=use_metadata,
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
