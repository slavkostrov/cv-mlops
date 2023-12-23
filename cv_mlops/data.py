import pathlib

import lightning as L
import torch
import torchvision
from torchvision import transforms

DatasetPath = str | pathlib.Path | None


class ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_path: DatasetPath = None,
        validation_data_path: DatasetPath = None,
        test_data_path: DatasetPath = None,
        batch_size: int = 50,
        num_workers: int | None = None,
        transformer: transforms.Compose | None = None,
    ):
        if train_data_path is None and test_data_path is None:
            raise ValueError("Train or test data path must be provided.")

        super().__init__()
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transformer = transformer

    def prepare_data(self):
        # TODO: dvc pull??? no
        pass

    def setup(self, stage: str):
        if self.train_data_path:
            self.train_dataset = torchvision.datasets.ImageFolder(self.train_data_path, transform=self.transformer)
            self.validation_dataset = torchvision.datasets.ImageFolder(
                self.validation_data_path, transform=self.transformer
            )

        if self.test_data_path:
            self.test_dataset = torchvision.datasets.ImageFolder(self.test_data_path, transform=self.transformer)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
