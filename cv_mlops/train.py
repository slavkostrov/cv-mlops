"""Module with model's train."""
import logging
import logging.config
import os
import pathlib
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torchvision
from hydra import compose, initialize
from omegaconf import OmegaConf
from tqdm import tqdm

from .metrics import calculate_metrics
from .utils import get_current_commit, load_object_from_path, prepare_overrides

# TODO: create yaml config for logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@torch.no_grad()
def test_model(
    model, batch_generator, loss_function, subset_name="test", print_log=True, plot_scores=False, device="cpu"
):
    """Evaluate the model using data from batch_generator and metrics defined above."""
    model.train(False)

    score_list = []
    label_list = []
    loss_list = []

    for X_batch, y_batch in batch_generator:
        logits = model(X_batch.to(device))
        scores = torch.softmax(logits, -1)[:, 1]
        labels = y_batch.numpy().tolist()

        loss = loss_function(logits, y_batch.to(device))

        loss_list.append(loss.detach().cpu().numpy().tolist())
        score_list.extend(scores)
        label_list.extend(labels)

    if print_log:
        print("Results on {} set | ".format(subset_name), end="")

    metric_results = calculate_metrics(score_list, label_list, print_log)
    metric_results["scores"] = score_list
    metric_results["labels"] = label_list
    metric_results["loss"] = loss_list

    return metric_results


def compute_loss(model, data_batch, loss_function):
    """Compute the loss using loss_function for the batch of data and return mean loss value for this batch."""
    img_batch = data_batch["img"]
    label_batch = data_batch["label"]
    logits = model(img_batch)
    loss = loss_function(logits, label_batch)
    return loss, model


def get_score_distributions(epoch_result_dict):
    """Return per-class score arrays."""
    scores = epoch_result_dict["scores"]
    labels = epoch_result_dict["labels"]

    # save per-class scores
    for class_id in [0, 1]:
        epoch_result_dict["scores_" + str(class_id)] = np.array(scores)[np.array(labels) == class_id]

    return epoch_result_dict


def train_model(  # noqa: D103
    model: torch.nn.Module,
    train_batch_generator: torch.utils.data.DataLoader,
    val_batch_generator: torch.utils.data.DataLoader,
    opt: torch.optim.Optimizer,
    epoch_num: int,
    loss_function: torch.nn.Module,
    device: str | torch.device | None = None,
    save_path: str | pathlib.Path = None,
    log_mlflow: bool = False,
):
    if device is None:
        LOGGER.info("Device isn't set, use cpu.")
        device = torch.device("cpu")
    else:
        LOGGER.info("Device is provided, will use %s.", device)

    train_loss, val_loss = [], [1]
    val_loss_idx = [0]
    top_val_accuracy = 0

    model.to(device)
    for epoch in range(epoch_num):
        start_time = time.time()

        # Train phase
        model.train(True)  # enable dropout / batch_norm training behavior
        for X_batch, y_batch in tqdm(train_batch_generator, desc="Training", leave=False):
            # move data to target device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            data_batch = {"img": X_batch, "label": y_batch}

            # train on batch: compute loss, calc grads, perform optimizer step and zero the grads
            loss, model = compute_loss(model, data_batch, loss_function=loss_function)

            # compute backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            # log train loss
            train_loss.append(loss.detach().cpu().numpy())

        # Evaluation phase
        # TODO: maybe add score distribution
        # TODO: add mlflow logging & tensorboard
        validation_metric_results = test_model(
            model, val_batch_generator, subset_name="val", loss_function=loss_function, device=device
        )
        validation_metric_results = get_score_distributions(validation_metric_results)
        train_metric_results = test_model(
            model, train_batch_generator, subset_name="train", loss_function=loss_function, device=device
        )  # TODO: optimize

        # Logging
        val_loss_value = np.mean(validation_metric_results["loss"])
        train_loss_value = np.mean(train_metric_results["loss"])
        if log_mlflow:
            mlflow.log_metric("validation_loss", val_loss_value, step=epoch)
            mlflow.log_metric("train_loss", train_loss_value, step=epoch)

        val_loss_idx.append(len(train_loss))
        val_loss.append(val_loss_value)
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, epoch_num, time.time() - start_time))

        val_accuracy_value = validation_metric_results["accuracy"]
        if val_accuracy_value > top_val_accuracy and save_path is not None:
            top_val_accuracy = val_accuracy_value

            with open(f"{save_path}.ckpt", "wb") as f:
                torch.save(model, f)

    if log_mlflow:
        figure = plt.figure(figsize=(15, 5))
        ax1 = figure.add_subplot(121)
        ax2 = figure.add_subplot(122)

        ax1.plot(train_loss, color="b", label="train")
        ax1.plot(val_loss_idx, val_loss, color="c", label="val")
        ax1.legend()
        ax1.set_title("Train/val loss.")

        ax2.hist(validation_metric_results["scores_0"], bins=50, range=[0, 1.01], color="r", alpha=0.7, label="cats")
        ax2.hist(validation_metric_results["scores_1"], bins=50, range=[0, 1.01], color="g", alpha=0.7, label="dogs")
        ax2.legend()
        ax2.set_title("Validation set score distribution.")

        mlflow.log_figure(figure, "score_distribution.png")
        plt.close("all")

    return model, opt


def get_dataloader(path, batch_size, shuffle, transformer=None, num_workers=1):  # noqa: D103
    train_dataset = torchvision.datasets.ImageFolder(path, transform=transformer)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=num_workers,
    )
    return dataloader


def train(
    config_name: str = "train",
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
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        LOGGER.info("Current_config:\n%s", OmegaConf.to_yaml(cfg))

        model = load_object_from_path(path=cfg.model.name, **cfg.model.params)
        dataloader_params = {
            "transformer": model.get_transformer(),
            **cfg.data.loader.params,
        }

        train_data_path = pathlib.Path(cfg.data.path) / cfg.data.folders.train
        validation_data_path = pathlib.Path(cfg.data.path) / cfg.data.folders.validation

        train_dataloader = get_dataloader(path=train_data_path, shuffle=True, **dataloader_params)
        validation_dataloader = get_dataloader(path=validation_data_path, shuffle=False, **dataloader_params)

        opt = load_object_from_path(path=cfg.optimizer.name, params=model.parameters(), **cfg.optimizer.params)

        # TODO: fix uri
        mlflow.set_tracking_uri(uri=cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.params.exp_name)
        with mlflow.start_run(run_name="TESTRUN"):
            # TODO: add logging of hydra configs
            mlflow.log_params(
                {
                    "params": cfg.params,
                    "commit_hash": get_current_commit(),
                }
            )

            loss_function = torch.nn.CrossEntropyLoss()

            os.makedirs(cfg.params.models_folder, exist_ok=True)
            model_save_path = pathlib.Path(cfg.params.models_folder) / cfg.params.model_filename
            model, opt = train_model(
                model=model,
                train_batch_generator=train_dataloader,
                val_batch_generator=validation_dataloader,
                opt=opt,
                loss_function=loss_function,
                epoch_num=cfg.params.epoch_num,
                device=cfg.params.device,
                save_path=model_save_path,
                log_mlflow=True,
            )
