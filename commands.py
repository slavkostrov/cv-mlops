"""CLI runner for train and infer proccesses."""
import fire

from cv_mlops.infer import infer  # noqa: F401
from cv_mlops.train import train  # noqa: F401

# TODO: rewrite to pure argparse, it seems this might be more flexible and safe
if __name__ == "__main__":
    fire.Fire()
