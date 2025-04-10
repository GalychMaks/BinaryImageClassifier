import glob
import os
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
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
    ):
        """
        Initialize the ArtifactImageDataModule.

        :param data_dir: Path to the root data directory.
        :param train_val_split: Tuple indicating the train/val split sizes.
        :param batch_size: Batch size for all dataloaders.
        :param num_workers: Number of subprocesses to use for data loading.
        :param pin_memory: Whether to pin memory in dataloaders.
        """
        super().__init__()
        self.save_hyperparameters()

        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for the specified stage.

        :param stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in (None, "fit"):
            all_train_paths = sorted(glob.glob(os.path.join(self.hparams.data_dir, "train", "*.png")))
            full_dataset = ArtifactImageDataset(all_train_paths, transform=self.transforms)
            self.data_train, self.data_val = random_split(
                full_dataset, self.hparams.train_val_split, generator=torch.Generator().manual_seed(42)
            )

        if stage in (None, "test", "predict"):
            test_paths = sorted(glob.glob(os.path.join(self.hparams.data_dir, "test", "*.png")))
            self.data_test = ArtifactImageDataset(test_paths, transform=self.transforms)

    def train_dataloader(self):
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

    def val_dataloader(self):
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

    def test_dataloader(self):
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
