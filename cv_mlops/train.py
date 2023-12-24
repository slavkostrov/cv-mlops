"""Module with model's train."""
import logging
import logging.config
import pathlib
import subprocess
from typing import NoReturn

import lightning as L
import lightning.pytorch as pl
import mlflow
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from .data import ImageDataModule
from .utils import get_current_commit, load_object_from_path, prepare_overrides

# TODO: create yaml config for logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def setup_callbacks(callbacks_cfg: DictConfig) -> list[L.Callback]:
    """Create and setup lightning callbacks.

    By default, create LearningRateMonitor, DeviceStatsMonitor and RichModelSummary.
    Also model checkpointing can be enabled.
    """
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(**callbacks_cfg.rich_model_summary.params),
    ]

    if callbacks_cfg.model_checkpoint.enable:
        LOGGER.info("Model checkpointing enable, params - %s.", callbacks_cfg.model_checkpoint.params)
        callbacks.append(pl.callbacks.ModelCheckpoint(**callbacks_cfg.model_checkpoint.params))

    return callbacks


def train(
    config_name: str = "train",
    config_path: str | None = None,
    **config_override_params,
) -> NoReturn:
    """Train model with Hydra usage.

    Args:
        config_name: name of config to use
        config_path: path to config (default None)
        **config_override_params: params for override config
    """
    # TODO: maybe need to fix default config path (it is relative to runner (__file__))
    config_path = config_path or "../configs"
    overrides = prepare_overrides(config_override_params)
    with initialize(version_base=None, config_path=config_path):
        # Load config
        # we need return_hydra_config=True for resolve hydra.runtime.cwd etc
        cfg: DictConfig = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        LOGGER.info("Current training config:\n%s", OmegaConf.to_yaml(cfg))

        # Pull train data from DVC
        subprocess.run(["dvc", "pull"], check=True)

        # Create model
        model: L.LightningModule = load_object_from_path(
            path=cfg.model.name,
            optimizer_config=cfg.optimizer,
            **cfg.model.params,
        )
        LOGGER.info("Successfuly Loaded model %s.", cfg.model.name)

        # Create data module
        train_data_path = pathlib.Path(cfg.data.path) / cfg.data.folders.train
        validation_data_path = pathlib.Path(cfg.data.path) / cfg.data.folders.validation
        datamodule: L.LightningDataModule = ImageDataModule(
            train_data_path=train_data_path,
            validation_data_path=validation_data_path,
            transformer=model.get_transformer(),
            **cfg.data.loader.params,
        )

        LOGGER.info(
            "Created data module with train_data_path=%s and validation_data_path=%s.",
            train_data_path,
            validation_data_path,
        )

        # Setup loggers and callbacks
        callbacks = setup_callbacks(cfg.callbacks)
        mlflow_logger = pl.loggers.MLFlowLogger(
            experiment_name=cfg.params.exp_name, tracking_uri=cfg.mlflow.tracking_uri
        )
        loggers = [mlflow_logger, pl.loggers.TensorBoardLogger(cfg.tensorboard.save_dir, name=cfg.params.exp_name)]

        # Create trainer
        trainer = L.Trainer(**cfg.trainer.params, callbacks=callbacks, logger=loggers)

        # Setup MLflow run params
        mlflow.set_tracking_uri(uri=cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.params.exp_name)

        if cfg.mlflow.autolog.enable:
            LOGGER.info("Enable mlflow autolog, params - %s.", cfg.mlflow.autolog.params)
            mlflow.pytorch.autolog(**cfg.mlflow.autolog.params)

        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.log_params(
                {
                    # TODO: add something?
                    "cfg": cfg,
                    "commit_hash": get_current_commit(),
                }
            )
            trainer.fit(model=model, datamodule=datamodule)

        # TODO: fix, save not as checkpoint + save best model (based on validation) and not last
        final_model_path = (
            pathlib.Path(cfg.callbacks.model_checkpoint.params.dirpath) / f"{cfg.params.model_filename}.ckpt"
        )
        trainer.save_checkpoint(final_model_path)
        LOGGER.info("Model saved to %s.", final_model_path)
