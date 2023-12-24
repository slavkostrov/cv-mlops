"""Module with data preparation logic & custom lightning DataModule."""
import pathlib

import lightning as L
import torch
import torchvision
from torchvision import transforms

DatasetPath = str | pathlib.Path | None


class ImageDataModule(L.LightningDataModule):
    """DataModule for image classification."""

    def __init__(
        self,
        train_data_path: DatasetPath = None,
        validation_data_path: DatasetPath = None,
        test_data_path: DatasetPath = None,
        batch_size: int = 50,
        num_workers: int | None = None,
        transformer: transforms.Compose | None = None,
    ):
        """Constructor of ImageDataModule.

        Args:
            train_data_path: path to train images and labels (default None)
            validation_data_path: path to validation set (default None)
            test_data_path: path to test set (default None)
            batch_size: batch size for use in dataloaders (default 50)
            num_workers: num workers fo use in dataloaders (default None, use torch default)
            transformer: transforms for data (default None)
        """
        if (train_data_path is None or validation_data_path is None) and test_data_path is None:
            raise ValueError("Train or test data path must be provided.")

        super().__init__()
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformer = transformer

    def setup(self, stage: str):
        """Setup datasets for current stage (`fit` and `test` implemeted).

        Fit stage needs both train and val datasets. Test only needs test.
        """
        if stage == "fit":
            if self.train_data_path is None or self.validation_data_path is None:
                raise RuntimeError("Need both train and validation sets.")

            self.train_dataset = torchvision.datasets.ImageFolder(self.train_data_path, transform=self.transformer)
            self.validation_dataset = torchvision.datasets.ImageFolder(
                self.validation_data_path, transform=self.transformer
            )
        elif stage == "test":
            if self.test_data_path is None:
                raise RuntimeError("Test set must be given for test stage.")
            self.test_dataset = torchvision.datasets.ImageFolder(self.test_data_path, transform=self.transformer)
        else:
            raise NotImplementedError(f"Not implemented for stage {stage}.")

    def train_dataloader(self):
        """Dataloader for train, shuffle enabled."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Dataloader for validation, shuffle disabled."""
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """Dataloader for test, shuffle disabled."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
