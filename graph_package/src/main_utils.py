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
import random
from pytorch_lightning import Trainer
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import shutil
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd

random.seed(4)  # set seed for reproducibility of shuffle get_drug_few_shot_split


def generate_histogram_of_node_degrees(dataset, dataset_name):
    """For group_val = drug_few_shot, generate histogram of node degrees to properly design limited connectivity exp."""
    degrees = (
        (dataset.graph.data.degree_in + dataset.graph.data.degree_out).cpu().numpy()
    )
    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 6)
    )
    ax1.hist(degrees, bins=40, color="skyblue", edgecolor="black")
    ax1.set_title("Degree Distribution Histogram")
    ax1.set_xlabel("Degrees")
    ax1.set_ylabel("Frequency")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    df_degrees = pd.DataFrame(degrees)
    ax2.boxplot(degrees, vert=False)
    ax2.set_title("Summary Statistics of Degrees")
    ax2.set_xlabel("Degrees")
    ax2.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    output_path = Directories.OUTPUT_PATH
    plt.savefig(output_path / f"degree_dist_histogram_{dataset_name}.png")
    plt.show()


def get_drug_few_shot_split(
    dataset, config, n_drugs_per_fold=8, print_split_stats=True
):
    """Split into 5 folds"""
    generate_histogram_of_node_degrees(dataset, dataset_name=config.dataset.name)
    all_drugs_ids = list(range(0, dataset.graph.num_node))
    random.shuffle(all_drugs_ids)
    splits = []
    df = dataset.data_df
    for i in range(0, dataset.graph.num_node, n_drugs_per_fold):
        drug_ids = all_drugs_ids[
            i : min(i + n_drugs_per_fold, dataset.graph.num_node.cpu().numpy())
        ]
        # put x triplets from test drugs in train
        test_idx = []
        for drug_id in drug_ids:
            drug_1_idx = df[df["drug_1_id"] == drug_id].index
            drug_2_idx = df[df["drug_2_id"] == drug_id].index
            drug_test_idx = list(set(drug_1_idx).union(set(drug_2_idx)))
            random.shuffle(drug_test_idx)
            n_triplets_to_include_in_train = min(
                len(drug_test_idx), config.max_train_triplets
            )
            # take n_triplets_to_include_in_train and put in test, use remainder for test
            test_idx = test_idx + drug_test_idx[n_triplets_to_include_in_train:]

        test_idx = list(set(test_idx))
        train_idx = list(set(dataset.data_df.index).difference(test_idx))
        splits.append((train_idx, test_idx))
    train_ratio = [len(split[0]) / (len(split[0]) + len(split[1])) for split in splits]
    if print_split_stats:
        print(
            f"train ratio mean, min, max: {np.mean(train_ratio),np.min(train_ratio), np.max(train_ratio)}"
        )
        print(f"amount of splits: {len(splits)}")
    return splits


def get_drug_split(dataset, config, n_drugs_per_fold=3):
    splits = []
    df = dataset.data_df
    for i in range(0, dataset.graph.num_node, n_drugs_per_fold):
        drug_ids = list(range(i, min(i + n_drugs_per_fold, dataset.graph.num_node)))
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
    elif config.group_val == "drug_few_shot":
        splits = get_drug_few_shot_split(dataset, config)
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


def pretrain_single_model(model_name, config, data_loaders, k):
    check_point_path = Directories.CHECKPOINT_PATH / model_name
    if os.path.isdir(check_point_path):
        shutil.rmtree(check_point_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=get_checkpoint_path(config.model.pretrain_model, k),
        **config.checkpoint_callback,
    )

    model = init_model(
        model=model_name,
        config=config,
        pretrain=True,
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


def init_model(
    model: str = "deepdds",
    config: dict = None,
    graph: Optional[KnowledgeGraphDataset] = None,
    logger_enabled: bool = True,
    pretrain: bool = False,
):
    """Load model from registry"""

    if model == "gnn":
        model = model_dict[model.lower()](
            graph=graph, dataset=config.dataset.name, **config.model
        )
    elif pretrain:
        pretrain_model = config.model.pretrain_model
        model = model_dict[pretrain_model](**config.model[pretrain_model])
    else:
        model = model_dict[model.lower()](**config.model)

    pl_module = BasePL(
        model,
        lr=config.lr,
        task=config.task,
        logger_enabled=logger_enabled,
        target=config.dataset.target,
        l2_reg=config.l2_reg,
        model_config=config.model,
    )
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
    elif model_name == "gnn":
        pass
        # config.model.update(update_rgcn_args(config))
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
