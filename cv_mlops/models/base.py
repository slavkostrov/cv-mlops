"""Module with base model implementation."""
import lightning as L


class BaseModel(L.LightningModule):
    """Base model for all models."""

    def get_transformer(self):
        """Create transforms for specific model."""
        return None
