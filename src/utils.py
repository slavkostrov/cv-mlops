"""Module with common utils functionality."""
from typing import Any


def prepare_overrides(config_override_params: dict[str, Any]) -> list[str]:
    """Parse function's kwargs to hydra's overrides compatible format (list of strings)."""
    overrides = list(map(lambda key, value: f"{key}={value}", config_override_params.items()))
    return overrides
