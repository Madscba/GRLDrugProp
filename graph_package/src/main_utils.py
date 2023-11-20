from graph_package.configs.directories import Directories
import torch
from typing import List, Tuple, Dict
from graph_package.configs.definitions import model_dict, dataset_dict
from graph_package.src.etl.dataloaders import KnowledgeGraphDataset
from graph_package.configs.directories import Directories
from graph_package.src.pl_modules import BasePL
from torch.utils.data import Subset
from torch.utils.data import random_split, Subset
from torchdrug.data import DataLoader
import os
from pytorch_lightning import Trainer
import sys
import shutil
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def pretrain_single_model(config, data_loaders, k):
    model_name = config.model.pretrain_model
    check_point_path = Directories.CHECKPOINT_PATH / model_name

    if os.path.isdir(check_point_path):
        shutil.rmtree(check_point_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=get_checkpoint_path(model_name, k), **config.checkpoint_callback
    )

    model = init_model(model=model_name, model_kwargs=config.model[model_name])

    trainer = Trainer(
        logger=[],
        callbacks=[checkpoint_callback],
        **config.trainer,
    )

    trainer.fit(
        model,
        train_dataloaders=data_loaders["train"],
        val_dataloaders=data_loaders["val"],
    )

    return checkpoint_callback.best_model_path


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def load_data(dataset: str = "oneil"):
    """Fetch formatted data depending on modelling task"""
    dataset_path = dataset_dict[dataset.lower()]
    data_loader = KnowledgeGraphDataset(dataset_path)
    return data_loader


def init_model(model_name: str = "deepdds", model_kwargs: dict = {}):
    """Load model from registry"""
    model = model_dict[model_name.lower()](**model_kwargs)
    pl_module = BasePL(model, model_name)
    return pl_module


def get_model_name(config: dict, sys_args: List[str]):
    for arg in sys_args:
        if arg.startswith("model="):
            return arg.split("=")[1]
    else:
        return "deepdds"


def update_rescal_args(dataset):
    update_dict = {
        "ent_tot": dataset.num_nodes,
        "rel_tot": int(dataset.num_relations),
    }
    return update_dict


def update_deepdds_args(config):
    return {"dataset_path": dataset_dict[config.dataset]}


def update_model_kwargs(config: dict, model_name: str, dataset):
    if model_name == "deepdds":
        config.model.update(update_deepdds_args(config))
    elif model_name == "rescal":
        config.model.update(update_rescal_args(dataset))
    elif model_name == "hybridmodel":
        config.model.deepdds.update(update_deepdds_args(config))
        config.model.rescal.update(update_rescal_args(dataset))


def get_checkpoint_path(model_name: str, k: int):
    checkpoint_path = Directories.CHECKPOINT_PATH / model_name / f"fold_{k}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_path)


def get_dataloaders(datasets: List[DataLoader], batch_sizes: Dict[str, int]):
    dataloaders = {}
    for dataset, (key, val) in zip(datasets, batch_sizes.items()):
        dataloaders[key] = DataLoader(dataset, batch_size=val, num_workers=0)
    return dataloaders


def split_dataset(
    dataset,
    split_method: str = "custom",
    split_idx: Tuple[List[int], List[int]] = None,
):
    if split_method == "random":
        split_fracs = [0.8, 0.1, 0.1]
        n_datapoints = len(dataset)
        split_lengths = [int(frac * len(dataset)) for frac in split_fracs]
        train_set, valid_set, test_set = random_split(
            dataset, split_lengths, generator=torch.Generator().manual_seed(42)
        )
    elif split_method == "custom":
        train_set = Subset(dataset, split_idx[0])
        val_set = Subset(dataset, split_idx[1])

    return train_set, val_set
