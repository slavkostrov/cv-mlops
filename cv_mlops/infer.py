"""Module with inference functions."""

import logging
import pathlib
import subprocess

import lightning as L
import pandas as pd
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from .data import ImageDataModule
from .utils import load_object_from_path, prepare_overrides

# TODO: create yaml config for logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def infer(
    config_name: str = "infer",
    config_path: str | None = None,
    **config_override_params,
):
    """Infer model with Hydra usage.

    Args:
        config_name: name of config to use
        config_path: path to config (default None)
        **config_override_params: params for override config
    """
    config_path = config_path or "../configs"
    overrides = prepare_overrides(config_override_params)
    with initialize(version_base=None, config_path=config_path):
        cfg: DictConfig = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        LOGGER.info("Current inference config:\n%s", OmegaConf.to_yaml(cfg))

        # Load data from DVC
        subprocess.run(["dvc", "pull"], check=True)

        # Load model from checkpoint (TODO: ONNX?)
        model_class: L.LightningModule = load_object_from_path(cfg.model.name, return_cls=True)
        model = model_class.load_from_checkpoint(cfg.params.checkpoint_path)

        # Prepare test dataset, create lightning Module
        test_data_path = pathlib.Path(cfg.data.path) / cfg.data.folders.test
        test_datamodule: L.LightningDataModule = ImageDataModule(
            test_data_path=test_data_path,
            transformer=model.get_transformer(),
            **cfg.data.loader.params,
        )

        # Predict with trainer
        trainer = L.Trainer()
        predictions = trainer.predict(model, datamodule=test_datamodule)
        predictions = torch.softmax(torch.vstack(predictions), -1)

        # Prepare output dataframe
        # TODO: refactor
        class_to_idx = test_datamodule.test_dataset.class_to_idx
        classes = test_datamodule.test_dataset.classes
        df = pd.DataFrame(
            {
                "file": [x[0] for x in test_datamodule.test_dataset.imgs],
                "label": [classes[x[1]] for x in test_datamodule.test_dataset.imgs],
                "cat_prob": predictions[:, class_to_idx.get("cat")],
                "dog_prob": predictions[:, class_to_idx.get("dog")],
            }
        )
        df.to_csv(cfg.params.output_path, index=False)
        LOGGER.info("Save prediction to %s.", cfg.params.output_path)
