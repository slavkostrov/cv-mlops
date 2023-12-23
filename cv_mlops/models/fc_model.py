from torch import nn
from torchmetrics import Accuracy, F1Score
from torchvision import transforms

from ..constants import IMAGE_MEAN, IMAGE_STD, SIZE_H, SIZE_W
from ..utils import load_object_from_path
from .base import BaseModel


class _FCModel(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        size_h: int = SIZE_H,
        size_w: int = SIZE_W,
        num_classses: int = 2,
    ) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * size_h * size_w, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, embedding_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(embedding_size, num_classses)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))

        x = self.fc3(x)
        return x


class FCModel(BaseModel):
    def __init__(
        self,
        embedding_size: int,
        size_h: int = SIZE_H,
        size_w: int = SIZE_W,
        num_classes: int = 2,
        optimizer_config: dict | None = None,
    ) -> None:
        super().__init__()

        self.model = _FCModel(
            embedding_size=embedding_size,
            size_h=size_h,
            size_w=size_w,
            num_classses=num_classes,
        )

        task = "multiclass" if num_classes > 2 else "binary"
        self.accuracy = Accuracy(task, num_classes=num_classes)
        self.f1_score = F1Score(task, num_classes=num_classes)

        self.optimizer_config = optimizer_config or dict()

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch

        logits = self.model(X_batch)
        predictions = logits.argmax(dim=1)
        loss = nn.functional.cross_entropy(logits, y_batch)
        accuracy = self.accuracy(predictions, y_batch)
        f1_score = self.f1_score(predictions, y_batch)

        self.log("train_loss", loss, on_epoch=True)
        self.log("accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("f1_score", f1_score, on_epoch=True)
        # TODO: ADD logs
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        X_batch, y_batch = batch

        logits = self.model(X_batch)
        predictions = logits.argmax(dim=1)
        loss = nn.functional.cross_entropy(logits, y_batch)
        accuracy = self.accuracy(predictions, y_batch)
        f1_score = self.f1_score(predictions, y_batch)

        self.log("val_loss", loss, on_epoch=True)
        self.log("accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("f1_score", f1_score, on_epoch=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        # TODO: возможно имеет смысл переписать более гибко/безопасно
        optimizer = load_object_from_path(
            path=self.optimizer_config.name,
            params=self.parameters(),
            **self.optimizer_config.params,
        )

        if self.optimizer_config.scheduler.enable:
            scheduler = load_object_from_path(
                self.optimizer_config.scheduler.name,
                optimizer=optimizer,
                **self.optimizer_config.scheduler.params,
            )
            return {"optimizer": optimizer, "scheduler": scheduler}

        return optimizer

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))

        x = self.fc3(x)
        return x

    def get_transformer(self):
        transformer = transforms.Compose(
            [transforms.Resize((SIZE_H, SIZE_W)), transforms.ToTensor(), transforms.Normalize(IMAGE_MEAN, IMAGE_STD)]
        )
        return transformer
