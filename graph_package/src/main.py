"""main module."""

import torch
from typing import List, Tuple, Union
import json
import pandas as pd
from torchdrug import data, core, datasets, tasks, models
from torchdrug.core import Registry as R
from .models import RESCAL
from .etl.dataloaders import ONEIL_RESCAL, ONIEL_DeepDDS
from configs.definitions import Directories, model_dict, dataset_dict, task_dict
from torch.utils.data import random_split, Subset
from models import DeepDDS, RESCAL


# https://torchdrug.ai/docs/quick_start.html


def load_data(model: str = "deepdds", dataset: str = "oneil"):
    """Fetch formatted data depending on modelling task"""
    dataset_key = model.lower() + "_" + dataset.lower()
    dataset = dataset_dict[dataset_key]()
    return dataset

def wrap_model_in_task(model: Union[DeepDDS,RESCAL]):
    """Wrap model in task"""
    task = task_dict[model.__name__.lower()](model=model)
    return task

def load_model(model: str = "deepdds", model_kwargs: dict = {}):
    """Load model from registry"""
    model = model_dict[model.lower()](**model_kwargs)
    return model


def split_dataset(dataset, split_method: str = "random", split_idx: Tuple[List[int], List[int], List[int]] = None
):
    if split_method == "random":
        split_fracs = [0.8, 0.1, 0.1]
        train_set, valid_set, test_set = random_split(
            dataset,
            split_fracs)
    else:
        train_set = Subset(dataset, split_idx[0])
        valid_set = Subset(dataset, split_idx[1])
        test_set = Subset(dataset, split_idx[2])
    return train_set, valid_set, test_set


def main(model_name: str = 'deepdds', dataset_name: str = 'oneil', kwargs: dict = {}):
    
    dataset = load_data(model=model_name, dataset=dataset_name)
    train_set, valid_set, test_set = split_dataset(dataset, split_method="random")
    model = load_model(model=model_name, model_kwargs=kwargs["model"])
    task = wrap_model_in_task(model)

    optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
    
    solver = core.Engine(
            task,
            train_set,
            valid_set,
            test_set,
            optimizer,
            **kwargs["engine"]
        )
    
    solver.train(**kwargs["train"])
    solver.evaluate(**kwargs["evaluate"])



if __name__ == "__main__":
    main()