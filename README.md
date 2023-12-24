# cv-mlops
Project using MLops technologies using the example of CV classification

##  Description

By default, the problem of binary classification of pictures into cats and dogs is solved. Moreover, the capabilities of the library are not limited to this task; parts can be adapted to any task (at least CV classification).

## Quick start

Install dependencies with poetry:

```bash
poetry install --without dev,research
```

Repository contains python package with some models & `PyTorch Lightning` usage. Also there are script for train and infer model from command line and Hydra usage:

* `cv_mlops` - package root;
* `cv_mlops.train` - script for train models;
* `cv_mlops.infer` - script for infer models;
* `cv_mlops.models` - package with models implementations (can be extended);

In root of repository you can found `commands.py`, it is CLI interface for train/infer steps, so you can simply use it with `python commands.py train` or `python commands.py infer`.

Package can work Hydra configs, example and available paramse can be found in configs directory. By default, train and infer steps use train and infer configs respectivly, but you can simply use your own config:

```bash
python commands.py train --config_name=my_config --config_path=base_path_to_my_config
```
__warn__: it is better to use absolute path (hydra use relative path from runner (ex. train.py) by default in Copmpose API)

Also can override some params:

```bash
python commands.py train --trainer.params.max_epochs=5
```
All parameters can be found in [configs](configs/).

Repository use DVC for data versions control and model saving, in example we use simple GDrive remote, but it also cat be extended.

For logging package use `lightning` logger (such as `TensorBoardLogger` and `MLFlowLogger`) and `MLflow` utils for logging experiments (se. [configs/mlflow](configs/mlflow)).

## Development

For develop install use:

```bash
poetry install --without research
```

Setup pre-commit with:
```bash
pre-commit install
```
