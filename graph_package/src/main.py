"""main module."""

import torch
from typing import List, Tuple, Union
from torchdrug import core
from models import RESCAL
from graph_package.configs.definitions import model_dict, dataset_dict, task_dict
from graph_package.configs.directories import Directories
from torch.utils.data import random_split, Subset
from models import DeepDDS, RESCAL
import hydra
from dotenv import load_dotenv
from graph_package.utils.meter import WandbMeter

import os
import sys
import wandb 


def load_data(model: str = "deepdds", dataset: str = "oneil"):
    """Fetch formatted data depending on modelling task"""
    dataset_key =dataset.lower()+ "_" + model.lower() 
    dataset = dataset_dict[dataset_key]()
    return dataset


def wrap_model_in_task(model: Union[DeepDDS, RESCAL], model_name: str = "deepdds"):
    """Wrap model in task"""
    task = task_dict[model_name](model=model)
    return task


def load_model(model: str = "deepdds", model_kwargs: dict = {}):
    """Load model from registry"""
    model = model_dict[model.lower()](**model_kwargs)
    return model

def get_model_name(config: dict, sys_args: List[str]):
    for arg in sys_args:
        if arg.startswith("model="):
            return arg.split("=")[1]
    return "deepdds"


def split_dataset(
    dataset,
    split_method: str = "random",
    split_idx: Tuple[List[int], List[int], List[int]] = None,
):
    if split_method == "random":
        split_fracs = [0.8, 0.1, 0.1]
        train_set, valid_set, test_set = random_split(dataset, split_fracs)
    else:
        train_set = Subset(dataset, split_idx[0])
        valid_set = Subset(dataset, split_idx[1])
        test_set = Subset(dataset, split_idx[2])
    return train_set, valid_set, test_set


@hydra.main(config_path=str(Directories.CONFIG_PATH / "hydra_configs"),config_name='config.yaml')
def main(config):
    if config.wandb:
        api_key = os.getenv("WANDB_API_KEY")
        project = os.getenv("WANDB_PROJECT")
        entity = os.getenv("WANDB_ENTITY")
        wandb.login(key=api_key)
        wandb.init(config=config,project=project,entity=entity)


    model_name = get_model_name(config, sys_args = sys.argv)
    dataset = load_data(model=model_name, dataset=config.dataset)
    train_set, valid_set, test_set = split_dataset(dataset, split_method="random")
    model = load_model(model=model_name, model_kwargs=config.model)
    task = wrap_model_in_task(model)

    optimizer = torch.optim.Adam(task.parameters(), config.lr)

    solver = core.Engine(
        task, train_set, valid_set, test_set, optimizer, **config.engine
    )

    if config.wandb:
        solver.meter = WandbMeter()

    solver.train(**config.trainer)
    solver.evaluate(**config.evaluate)


if __name__ == "__main__":
    load_dotenv(".env")
    main()
