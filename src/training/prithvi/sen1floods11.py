# src/training/prithvi/sen1floods11.py

#################################################
# Sen1Floods data adaptor
#################################################

# NOTE: Interface adapted from the Terratorch project
# Uses datasets and augmentations from src/data/loaders.py

#################################################
### IMPORTS

from collections.abc import Sequence
from typing import Any

### OWN MODULES
from src.data.s2augmentations import (
    get_s2_train_transform,
    get_s2_val_transform,
)
from src.data.s2hand import ALL_BAND_NAMES, S2HandDataset
from src.data.s2weak import S2WeakDataset

### EXTERNAL IMPORTS
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule


#################################################
### CODE
class Sen1Floods11S2HandDataModule(NonGeoDataModule):
    """NonGeo DataModule implementation for Sen1Floods11 Sentinel-2 Hand-labeled data.

    Uses S2HandDataset and augmentations from src/data/loaders.py.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] | None = None,
        drop_last: bool = True,
        image_size: int = 256,
        no_data_replace: float | None = 0.0,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the Sen1Floods11S2HandDataModule.

        Args:
            data_root (str): Root directory of the dataset.
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 4.
            num_workers (int, optional): Number of workers for data loading.
                Defaults to 0.
            bands (Sequence[str], optional): List of bands to use.
                Defaults to ALL_BAND_NAMES.
            drop_last (bool, optional): Whether to drop the last incomplete batch.
                Defaults to True.
            image_size (int, optional): Size of the image to crop to.
                Defaults to 256.
            no_data_replace (float | None, optional): Replacement value for
                missing data. Defaults to 0.0.
            no_label_replace (int | None, optional): Replacement value for
                missing labels. Defaults to -1.
            use_metadata (bool): Whether to return metadata info
                (time and location). Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        # Don't pass dataset_class since we create datasets directly in setup()
        super().__init__(None, batch_size, num_workers, **kwargs)  # type: ignore[arg-type]
        self.data_root = data_root

        if bands is None:
            bands = list(ALL_BAND_NAMES)
        self.bands = list(bands)
        self.image_size = image_size
        self.drop_last = drop_last
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.use_metadata = use_metadata

        # Create transforms using the same functions as make_s2hand_loaders
        self.train_transform = get_s2_train_transform(
            image_size=image_size, bands=self.bands
        )
        self.val_transform = get_s2_val_transform(
            image_size=image_size, bands=self.bands
        )
        self.test_transform = get_s2_val_transform(
            image_size=image_size, bands=self.bands
        )
        self.predict_transform = get_s2_val_transform(
            image_size=image_size, bands=self.bands
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either fit, validate, test, or predict.
        """
        if stage in ["fit"]:
            self.train_dataset = S2HandDataset(
                data_root=self.data_root,
                split="train",
                transform=self.train_transform,
                bands=self.bands,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = S2HandDataset(
                data_root=self.data_root,
                split="valid",
                transform=self.val_transform,
                bands=self.bands,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["test"]:
            self.test_dataset = S2HandDataset(
                data_root=self.data_root,
                split="test",
                transform=self.test_transform,
                bands=self.bands,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["predict"]:
            self.predict_dataset = S2HandDataset(
                data_root=self.data_root,
                split="test",
                transform=self.predict_transform,
                bands=self.bands,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A DataLoader for the specified split.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
            pin_memory=True,
        )


class Sen1Floods11S2WeakDataModule(NonGeoDataModule):
    """NonGeo DataModule implementation for Sen1Floods11 Sentinel-2 Weakly-labeled data.

    Uses S2WeakDataset and augmentations from src/data/loaders.py.
    Note: Weakly-labeled data always uses train augmentation.

    We also change this to use hand data for validation and test
    in order to align with the rest of the project.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] | None = None,
        drop_last: bool = True,
        image_size: int = 256,
        max_samples: int | None = None,
        no_data_replace: float | None = 0.0,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the Sen1Floods11S2WeakDataModule.

        Args:
            data_root (str): Root directory of the dataset.
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 4.
            num_workers (int, optional): Number of workers for data loading.
                Defaults to 0.
            bands (Sequence[str], optional): List of bands to use.
                Defaults to ALL_BAND_NAMES.
            drop_last (bool, optional): Whether to drop the last incomplete batch.
                Defaults to True.
            image_size (int, optional): Size of the image to crop to.
                Defaults to 256.
            max_samples (int | None, optional): Maximum number of samples to load.
                Defaults to None.
            no_data_replace (float | None, optional): Replacement value for
                missing data. Defaults to 0.0.
            use_metadata (bool): Whether to return metadata info
                (time and location). Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(None, batch_size, num_workers, **kwargs)  # type: ignore[arg-type]
        self.data_root = data_root

        if bands is None:
            bands = list(ALL_BAND_NAMES)
        self.bands = list(bands)
        self.image_size = image_size
        self.drop_last = drop_last
        self.max_samples = max_samples
        self.no_data_replace = no_data_replace
        self.use_metadata = use_metadata

        # Create transforms using the same functions as make_s2hand_loaders
        self.train_transform = get_s2_train_transform(
            image_size=image_size, bands=self.bands
        )
        self.val_transform = get_s2_val_transform(
            image_size=image_size, bands=self.bands
        )
        self.test_transform = get_s2_val_transform(
            image_size=image_size, bands=self.bands
        )
        self.predict_transform = get_s2_val_transform(
            image_size=image_size, bands=self.bands
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either fit, validate, test, or predict.

        Note:
            To align, we use the hand data for validation and test...
        """
        if stage in ["fit"]:
            self.train_dataset = S2WeakDataset(
                data_root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                max_samples=self.max_samples,
                no_data_replace=self.no_data_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["fit", "validate"]:
            # self.val_dataset = S2WeakDataset(
            self.val_dataset = S2HandDataset(
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                max_samples=self.max_samples,
                no_data_replace=self.no_data_replace,
                use_metadata=self.use_metadata,
                split="valid",
            )
        if stage in ["test"]:
            # self.test_dataset = S2WeakDataset(
            self.test_dataset = S2HandDataset(
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                max_samples=self.max_samples,
                no_data_replace=self.no_data_replace,
                use_metadata=self.use_metadata,
                split="test",
            )
        if stage in ["predict"]:
            # self.predict_dataset = S2WeakDataset(
            self.predict_dataset = S2HandDataset(
                data_root=self.data_root,
                transform=self.predict_transform,
                bands=self.bands,
                max_samples=self.max_samples,
                no_data_replace=self.no_data_replace,
                use_metadata=self.use_metadata,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A DataLoader for the specified split.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
            pin_memory=True,
        )
