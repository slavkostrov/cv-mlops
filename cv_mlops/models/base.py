import torch


class BaseModel(torch.nn.Module):
    def get_transformer(self):
        return None
