"""Module with implemetation of fully connected NN model."""
from torch import nn
from torchmetrics import Accuracy, F1Score
from torchvision import transforms

from ..constants import IMAGE_MEAN, IMAGE_STD, SIZE_H, SIZE_W
from ..utils import load_object_from_path
from .base import BaseModel


class _FCModel(nn.Module):
    """Implentation of FC model on PyTorch (withount lightning usage)."""

    def __init__(
        self,
        embedding_size: int,
        size_h: int = SIZE_H,
        size_w: int = SIZE_W,
        num_classes: int = 2,
    ) -> None:
        """Constructor of FC model.

        Args:
            embedding_size: size of embedding on last layer
            size_h: height of images (default SIZE_H)
            size_w: width of images (default SIZE_W)
            num_classes: number of classes to predict (default 2)
        """
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * size_h * size_w, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, embedding_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(embedding_size, num_classes)

    def forward(self, input_data):
        """Forward step of NN."""
        data = self.flatten(input_data)
        data = self.dropout1(self.relu1(self.fc1(data)))
        data = self.dropout2(self.relu2(self.fc2(data)))
        predict = self.fc3(data)
        return predict


class FCModel(BaseModel):
    """Lightning module based on _FCModel."""

    def __init__(
        self,
        embedding_size: int,
        size_h: int = SIZE_H,
        size_w: int = SIZE_W,
        num_classes: int = 2,
        optimizer_config: dict | None = None,
    ) -> None:
        """Constructor of FC model (with Lightning).

        Save all params, add metric modules (accuracy and f1 for now).

        Args:
            embedding_size: size of embedding on last layer
            size_h: height of images (default SIZE_H)
            size_w: width of images (default SIZE_W)
            num_classes: number of classes to predict (default 2)
            optimizer_config: specific config for optimizer (default None)
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = _FCModel(
            embedding_size=embedding_size,
            size_h=size_h,
            size_w=size_w,
            num_classes=num_classes,
        )

        task = "multiclass" if num_classes > 2 else "binary"
        self.accuracy = Accuracy(task, num_classes=num_classes)
        self.f1_score = F1Score(task, num_classes=num_classes)

        self.optimizer_config = optimizer_config or dict(name="torch.optim.Adam", params=dict())

    def default_step(self, batch, name: str):
        """Default step for train and validation."""
        # split batch on images and labels
        X_batch, y_batch = batch

        # make predictions
        logits = self.model(X_batch)
        predictions = logits.argmax(dim=1)

        # calculate loss and metrics
        loss = nn.functional.cross_entropy(logits, y_batch)
        accuracy = self.accuracy(predictions, y_batch)
        f1_score = self.f1_score(predictions, y_batch)

        # logging to loggers
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log(f"{name}_f1", f1_score, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        """Training step of module."""
        # split batch on images and labels
        loss = self.default_step(batch, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Validation loop."""
        loss = self.default_step(batch, "val")
        return {"val_loss": loss}

    def configure_optimizers(self):
        """Configure optimizers, based on provided optimizer config."""
        # TODO: maybe not safe ...

        # load specified optimizer
        optimizer = load_object_from_path(
            path=self.optimizer_config.name,
            params=self.parameters(),
            **self.optimizer_config.params,
        )

        if self.optimizer_config.scheduler.enable:
            # add scheduler if enabled
            scheduler = load_object_from_path(
                self.optimizer_config.scheduler.name,
                optimizer=optimizer,
                **self.optimizer_config.scheduler.params,
            )
            # return both
            return {"optimizer": optimizer, "scheduler": scheduler}

        # return only optimizer if sheduler not specified
        return optimizer

    def forward(self, input_data):
        """Forward step of model, need to predict."""
        if isinstance(input_data, (tuple, list)) and len(input_data) == 2:
            # FIXME: sometimes both image and label provided
            # so we need to save only x
            input_data, _ = input_data
        return self.model(input_data)

    def get_transformer(self):
        """Specific transforms for model. Resize, to tensor and normalization."""
        transformer = transforms.Compose(
            [transforms.Resize((SIZE_H, SIZE_W)), transforms.ToTensor(), transforms.Normalize(IMAGE_MEAN, IMAGE_STD)]
        )
        return transformer
