from graph_package.configs.directories import Directories
import torch
from typing import List, Tuple, Dict, Optional
from graph_package.configs.definitions import model_dict, dataset_dict
from graph_package.src.etl.dataloaders import KnowledgeGraphDataset
from graph_package.configs.directories import Directories
from graph_package.src.pl_modules import BasePL
from torch.utils.data import random_split, Subset
from torchdrug.data import DataLoader
import os
from pytorch_lightning import Trainer
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import shutil
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def get_drug_split(dataset, config, n_drugs_per_fold=3):
    splits = []
    df = dataset.data_df
    for i in range(0, dataset.num_nodes, n_drugs_per_fold):
        drug_ids = list(range(i, min(i + n_drugs_per_fold, dataset.num_nodes)))
        drug_1_idx = df[df["drug_1_id"].isin(drug_ids)].index
        drug_2_idx = df[df["drug_2_id"].isin(drug_ids)].index
        test_idx = list(set(drug_1_idx).union(set(drug_2_idx)))
        train_idx = list(set(dataset.data_df.index).difference(test_idx))
        splits.append((train_idx, test_idx))
    return splits


def get_cv_splits(dataset, config):
    if config.group_val == "drug":
        splits = get_drug_split(dataset, config)
        return splits
    else:
        if config.group_val == "drug_combination":
            group = dataset.data_df.groupby(["drug_1_id", "drug_2_id"]).ngroup()
        elif config.group_val == "cell_line":
            group = dataset.data_df.groupby(["context_id"]).ngroup()
        else:
            group = np.arange(len(dataset))
        kfold = StratifiedGroupKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.seed
        )
        return kfold.split(dataset, dataset.get_labels(dataset.indices), group)


def pretrain_single_model(config, data_loaders, k):
    model_name = config.model.pretrain_model
    check_point_path = Directories.CHECKPOINT_PATH / model_name
    if os.path.isdir(check_point_path):
        shutil.rmtree(check_point_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=get_checkpoint_path(model_name, k), **config.checkpoint_callback
    )

    model = init_model(
        model=model_name,
        task=config.task,
        model_kwargs=config.model[model_name],
        logger_enabled=False,
        target=config.dataset.target,
    )

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


def load_data(dataset_config: dict, task="reg"):
    """Fetch formatted data depending on modelling task"""
    dataset_path = dataset_dict[dataset_config.name.lower()]
    data_loader = KnowledgeGraphDataset(
        dataset_path, task=task, target=dataset_config.target,
        use_node_features=dataset_config.use_node_features,
        use_edge_features=dataset_config.use_edge_features
    )
    return data_loader


def init_model(
    model: str = "deepdds",
    task: str = "clf",
    target: str = "zip_mean",
    model_kwargs: dict = {},
    graph: Optional[KnowledgeGraphDataset] = None,
    logger_enabled: bool = True,
):
    """Load model from registry"""
    if model == "rgcn":
        model = model_dict[model.lower()](graph,**model_kwargs)
    else:
        model = model_dict[model.lower()](**model_kwargs)
    pl_module = BasePL(model, task=task, logger_enabled=logger_enabled, target=target)
    return pl_module


def get_model_name(config: dict, sys_args: List[str]):
    for arg in sys_args:
        if arg.startswith("model="):
            return arg.split("=")[1]
    else:
        return "deepdds"


def update_shallow_embedding_args(dataset):
    update_dict = {
        "ent_tot": dataset.graph.num_node.tolist(),
        "rel_tot": dataset.graph.num_relation.tolist(),
    }
    return update_dict

def update_deepdds_args(config):
    return {"dataset_path": dataset_dict[config.dataset.name]}

def update_rgcn_args(config):
    return {"dataset_path": dataset_dict[config.dataset.name]}

def update_model_kwargs(config: dict, model_name: str, dataset):
    if model_name.startswith("deepdds"):
        config.model.update(update_deepdds_args(config))
    elif model_name == "hybridmodel":
        config.model.deepdds.update(update_deepdds_args(config))
        config.model.rescal.update(update_shallow_embedding_args(dataset))
    elif model_name =="rgcn":
        pass
        #config.model.update(update_rgcn_args(config))
    else:
        config.model.update(update_shallow_embedding_args(dataset))


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
