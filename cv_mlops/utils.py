"""Module with common utils functionality."""
import subprocess
from importlib import import_module
from typing import Any


def prepare_overrides(config_override_params: dict[str, Any]) -> list[str]:
    """Parse function's kwargs to hydra's overrides compatible format (list of strings)."""
    overrides = list(map(lambda item: f"{item[0]}={item[1]}", config_override_params.items()))
    return overrides


def get_current_commit() -> str:
    """Get current commit hash in short format."""
    # TODO: check if in git repo now.
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()


def load_object_from_path(path: str, return_cls: bool = False, *args, **kwargs):
    """Load module from specified path (for example, `torch.optim.Adam`).

    Create object (make call) if return_cls is False with provided args and kwargs.
    Return without call if return_cls is true.
    """
    module_name, class_name = path.rsplit(".", maxsplit=1)
    target_class = getattr(import_module(module_name), class_name)
    if return_cls:
        return target_class
    obj = target_class(*args, **kwargs)
    return obj
