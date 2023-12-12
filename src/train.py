"""Module with model's train."""
import logging
import logging.config

from hydra import compose, initialize
from omegaconf import OmegaConf

from .utils import prepare_overrides

# TODO: create yaml config for logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def train(
    config_name: str,
    config_path: str | None = None,
    **config_override_params,
):
    """Train model.

    Args:
        config_name: name of config to use
        config_path: path to config (default None)
        **config_override_params: params for override config
    """
    config_path = config_path or "../configs"
    overrides = prepare_overrides(config_override_params)
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)
        LOGGER.info("Current_config:\n%s", OmegaConf.to_yaml(cfg))

        # TODO: implement train
