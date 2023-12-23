import lightning as L


class BaseModel(L.LightningModule):
    def get_transformer(self):
        return None
