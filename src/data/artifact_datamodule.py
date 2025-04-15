import glob
import os
from typing import List, Optional, Tuple

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.components.artifact_dataset import ArtifactImageDataset


class ArtifactImageDataModule(LightningDataModule):
    """
    LightningDataModule for the artifact image classification task.

    This module handles dataset loading, transformation, and splitting
    for training, validation, and testing.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_split: Tuple[int, int] = (1440, 360),
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        """
        Initialize the ArtifactImageDataModule.

        :param data_dir: Path to the root data directory.
        :param train_val_split: Tuple indicating the train/validation split sizes.
        :param batch_size: Batch size for all dataloaders.
        :param num_workers: Number of subprocesses to use for data loading.
        :param pin_memory: Whether to pin memory in dataloaders.
        """
        super().__init__()
        self.save_hyperparameters()

        # Data augmentation and preprocessing for training set
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Preprocessing for validation and test sets
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self) -> None:
        """
        Placeholder for data preparation logic (e.g., download).
        Not used in this case since data is assumed to be already present locally.
        """
        super().prepare_data()
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for the specified stage.

        :param stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in (None, "fit"):
            train_dir = os.path.join(self.hparams.data_dir, "train")
            all_train_paths: List[str] = sorted(glob.glob(os.path.join(train_dir, "*.png")))
            total_required = self.hparams.train_val_split[0] + self.hparams.train_val_split[1]
            if total_required > len(all_train_paths):
                raise ValueError("Not enough training images for the specified train/validation split.")

            # Shuffle indices for random splitting while ensuring reproducibility
            generator = torch.Generator().manual_seed(42)
            perm = torch.randperm(len(all_train_paths), generator=generator).tolist()
            train_count, val_count = self.hparams.train_val_split

            # Select indices and paths for training and validation sets
            train_indices = perm[:train_count]
            val_indices = perm[train_count : train_count + val_count]
            train_paths = [all_train_paths[i] for i in train_indices]
            val_paths = [all_train_paths[i] for i in val_indices]

            self.data_train = ArtifactImageDataset(train_paths, transform=self.train_transforms)
            self.data_val = ArtifactImageDataset(val_paths, transform=self.val_transforms)

        if stage in (None, "test", "predict"):
            test_dir = os.path.join(self.hparams.data_dir, "test")
            test_paths: List[str] = sorted(glob.glob(os.path.join(test_dir, "*.png")))
            self.data_test = ArtifactImageDataset(test_paths, transform=self.val_transforms)

    def train_dataloader(self) -> DataLoader:
        """
        Create and return the training dataloader.

        :return: Dataloader for the training set.
        """
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation dataloader.

        :return: Dataloader for the validation set.
        """
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create and return the test dataloader.

        :return: Dataloader for the test set.
        """
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
