from graph_package.configs.directories import Directories
import torch
import json
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
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.data import HeteroData



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
        dirpath=get_checkpoint_path(config.model.pretrain_model, k), **config.checkpoint_callback
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
                fold=fold 
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
        graph=graph,
        lr=config.lr,
        task=config.task,
        logger_enabled=logger_enabled,
        target=config.dataset.target,
        l2_reg=config.l2_reg,
        model_config=config.model,
    )
    return pl_module

def transform_hetero_data(graph: KnowledgeGraphDataset):
    edge_list = graph.edge_list
    drug_ids = torch.unique(edge_list[:, [0, 1]])
    hetero_data = HeteroData({
        'drug': {'x': graph.node_feature[[drug_ids]]}
    })
    # Create edge_index for the 'drug', 'interacts_with', 'drug' edge type
    edge_index = torch.stack([edge_list[:, 0], edge_list[:, 1]])
    edge_attr = edge_list[:, 2]  # Assign unique attributes to edges

    # Add edge_index information to the HeteroData object
    hetero_data['drug', 'interacts_with', 'drug'].edge_index = edge_index
    hetero_data['drug', 'interacts_with', 'drug'].edge_attr = edge_attr
    hetero_data.edge_list = edge_list

    return hetero_data

def init_explainer(
    model: torch.nn.Module,
    explainer_algorithm: str,
    explainer_args: dict = None
    ):
    default_explainer_args = {
        'explanation_type': 'model',
        'model_config': {
            'mode': 'regression',
            'task_level': 'edge',
            'return_type': 'raw',
        },
        'node_mask_type': 'attributes',
        'edge_mask_type': 'object',
        'threshold_config': {
            'threshold_type': 'topk',
            'value': 200,
        },
    }
    # Merge the default arguments with the optional provided arguments
    explainer_args = explainer_args or {}
    explainer_args = {**default_explainer_args, **explainer_args}
    if explainer_algorithm == 'IG':
        algorithm = CaptumExplainer(
            'IntegratedGradients',
        )
    else:
        raise ValueError("Passed explainer algorithm is not supported")
    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        **explainer_args
    )
    return explainer 

def get_explaination(explainer: Explainer, hetero_data: HeteroData):
    """ Silly example with the first 10 triplets (2 drugs, 10 cell lines)"""

    index = torch.tensor([2, 5]) # Explain edge labels with index 2 and 10.

    explanation = explainer(
        x=hetero_data.x_dict,
        edge_index=hetero_data.edge_index_dict,
        index=index,
        edge_label_index=hetero_data['drug','drug'].edge_attr
    )
    print(f'Generated explanations in {explanation.available_explanations}')

    path = 'feature_importance.png'
    explanation.visualize_feature_importance(path, top_k=10)
    print(f"Feature importance plot has been saved to '{path}'")
    return


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

def save_pretrained_drug_embeddings(model, fold):
    model_name = model.model._get_name()
    if model_name.lower() == 'deepdds':
        drug_ids = torch.arange(len(model.model.entity_vocab), device=model.device)
        molecules = model.model._get_drug_molecules(drug_ids)
        features = model.model.drug_conv(
                molecules, molecules.data_dict["atom_feature"].float()
            )["node_feature"]
        features = model.model.drug_readout(molecules, features).tolist()
    else:
        drug_ids = torch.arange(model.model.num_entity, device=model.device)
        features = model.model.entity[drug_ids].tolist()
    dataset_path = dataset_dict['oneil_almanac']
    with open(dataset_path.parent / "entity_vocab.json") as f:
        drug_vocab = json.load(f)
    reverse_vocab = {i: drug for drug, i in drug_vocab.items()}
    drug_feature_dict = {reverse_vocab[i]: features[i] for i in drug_ids.tolist()}
    file_name = f"drug_embedding_{model_name.lower()}_f{fold}_d{np.shape(features)[1]}.json"
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
        node_feature_dict[name] for name in drug_vocab.keys() 
        if name in node_feature_dict.keys()
    ]
    # Convert to float arraylike 
    node_features = np.array(node_features).astype(np.float32)
    graph.node_feature = torch.as_tensor(node_features, device=graph.device)
    return 