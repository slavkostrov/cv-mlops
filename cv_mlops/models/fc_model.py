from torch import nn
from torchvision import transforms

from ..constants import IMAGE_MEAN, IMAGE_STD, SIZE_H, SIZE_W
from .base import BaseModel


class FCModel(BaseModel):
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

    def get_transformer(self):
        transformer = transforms.Compose(
            [transforms.Resize((SIZE_H, SIZE_W)), transforms.ToTensor(), transforms.Normalize(IMAGE_MEAN, IMAGE_STD)]
        )
        return transformer
