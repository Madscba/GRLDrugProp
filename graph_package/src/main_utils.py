from graph_package.configs.directories import Directories
import torch
import json
from typing import List, Tuple, Dict, Optional
from graph_package.configs.definitions import model_dict, dataset_dict
from graph_package.src.etl.dataloaders import KnowledgeGraphDataset
from graph_package.configs.directories import Directories
from graph_package.src.pl_modules import BasePL
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split as train_val_split
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


def get_drug_few_shot_split(dataset, config, print_split_stats=True):
    """Split into 5 folds"""
    n_drugs_per_fold = config.n_drugs_per_fold
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


def get_cell_line_few_shot_split(dataset, config, print_split_stats=True):
    """Split into 5 folds"""
    n_cell_lines_per_fold = config.n_cell_lines_per_fold
    num_cell_lines = dataset.graph.num_relation.cpu().numpy()
    all_cell_line_ids = list(range(0, num_cell_lines))
    splits = []
    df = dataset.data_df
    for i in range(0, num_cell_lines, n_cell_lines_per_fold):
        cell_line_ids = all_cell_line_ids[
            i : min(i + n_cell_lines_per_fold, num_cell_lines)
        ]
        # put x triplets from test drugs in train
        test_idx = []
        for cell_line_id in cell_line_ids:
            cell_line_idx = list(df[df["context_id"] == cell_line_id].index)
            random.shuffle(list(cell_line_idx))
            n_triplets_to_include_in_train = min(
                len(cell_line_idx), config.max_train_triplets
            )
            # take n_triplets_to_include_in_train and put in test, use remainder for test
            test_idx = test_idx + cell_line_idx[n_triplets_to_include_in_train:]

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
    elif config.group_val == "cell_line_few_shot":
        splits = get_cell_line_few_shot_split(dataset, config)
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
    fold: int = 0,
    config: dict = None,
    graph: Optional[KnowledgeGraphDataset] = None,
    logger_enabled: bool = True,
    pretrain: bool = False,
):
    """Load model from registry"""

    if model == "gnn":
        if config.dataset.drug_representation in ["distmult", "deepdds"]:
            load_pretrained_drug_embeddings_into_graph(
                graph=graph,
                model=config.dataset.drug_representation,
                dataset_str=config.dataset.name,
                fold=fold,
            )
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
        config.model.distmult.update(update_shallow_embedding_args(dataset))
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


def split_train_val_test(dataset, train_idx, test_idx, config):
    train_set, test_set = split_dataset(
        dataset, split_method="custom", split_idx=(train_idx, test_idx)
    )

    if config.group_val == "drug_few_shot":
        test_idx, val_idx = train_val_split(
            test_set.indices,
            test_size=0.4,
            random_state=config.seed,
            stratify=dataset.get_labels(test_set.indices),
        )

        test_set, val_set = split_dataset(
            dataset, split_method="custom", split_idx=(list(test_idx), list(val_idx))
        )

    else:
        train_idx, val_idx = train_val_split(
            train_set.indices,
            test_size=0.1,
            random_state=config.seed,
            stratify=dataset.get_labels(train_set.indices),
        )

        train_set, val_set = split_dataset(
            dataset, split_method="custom", split_idx=(list(train_idx), list(val_idx))
        )

    return train_set, val_set, test_set


def save_pretrained_drug_embeddings(model, fold):
    model_name = model.model._get_name()
    if model_name.lower() == "deepdds":
        drug_ids = torch.arange(len(model.model.entity_vocab), device=model.device)
        molecules = model.model._get_drug_molecules(drug_ids)
        features = model.model.drug_conv(
            molecules, molecules.data_dict["atom_feature"].float()
        )["node_feature"]
        features = model.model.drug_readout(molecules, features).tolist()
    else:
        drug_ids = torch.arange(model.model.num_entity, device=model.device)
        features = model.model.entity[drug_ids].tolist()
    dataset_path = dataset_dict["oneil_almanac"]
    with open(dataset_path.parent / "entity_vocab.json") as f:
        drug_vocab = json.load(f)
    reverse_vocab = {i: drug for drug, i in drug_vocab.items()}
    drug_feature_dict = {reverse_vocab[i]: features[i] for i in drug_ids.tolist()}
    file_name = (
        f"drug_embedding_{model_name.lower()}_f{fold}_d{np.shape(features)[1]}.json"
    )
    save_path = Directories.DATA_PATH / "features" / "pretrained_features"
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / file_name, "w") as json_file:
        json.dump(drug_feature_dict, json_file)
    return


def load_pretrained_drug_embeddings_into_graph(graph, model, dataset_str, fold, dim=83):
    dataset_path = dataset_dict[dataset_str.lower()]
    with open(dataset_path.parent / "entity_vocab.json") as f:
        drug_vocab = json.load(f)
    file_name = f"drug_embedding_{model}_f{fold}_d{dim}.json"
    feature_path = Directories.DATA_PATH / "features" / "pretrained_features"
    with open(feature_path / file_name) as f:
        node_feature_dict = json.load(f)
    # Convert to a list in correct order determined by graph node ID
    node_features = [
        node_feature_dict[name]
        for name in drug_vocab.keys()
        if name in node_feature_dict.keys()
    ]
    # Convert to float arraylike
    node_features = np.array(node_features).astype(np.float32)
    graph.node_feature = torch.as_tensor(node_features, device=graph.device)
    return
