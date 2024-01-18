import os
import sys
import torch
import hydra
from graph_package.src.models.hybridmodel import remove_prefix_from_keys
from graph_package.src.explainability.utils import explain_attention
from graph_package.configs.directories import Directories
from graph_package.src.main_utils import (
    init_model, 
    get_model_name, 
    get_cv_splits, 
    split_dataset, 
)
from sklearn.model_selection import train_test_split as train_val_split
from dotenv import load_dotenv
from graph_package.src.etl.dataloaders import KnowledgeGraphDataset


@hydra.main(
    config_path=str(Directories.CONFIG_PATH / "hydra_configs"),
    config_name="config.yaml",
    version_base="1.1",
)
def explain(config):
    model_name = get_model_name(config=config, sys_args=sys.argv)
    if (model_name == "gnn") & (config.dataset.drug_representation not in ["distmult", "deepdds"]):
        config.dataset.update({"use_node_features": True})
    dataset = KnowledgeGraphDataset(**config.dataset)
    splits = get_cv_splits(dataset, config)
    for k, (train_idx, test_idx) in enumerate(splits):
        train_set = get_train_data_split(train_idx, test_idx, dataset, config.seed)
        checkpoint_path = str(Directories.CHECKPOINT_PATH / "gnn" / f"fold_{k}")
        all_items = os.listdir(checkpoint_path)
        files = [item for item in all_items if os.path.isfile(os.path.join(checkpoint_path, item))]
        best_model_path = os.path.join(checkpoint_path,files[-1])
        model = init_model(
                    model=model_name,
                    config=config,
                    graph=train_set.dataset.graph.edge_mask(train_set.indices),
                ).model
        state_dict = remove_prefix_from_keys(
            torch.load(best_model_path)["state_dict"], "model."
        )
        model.load_state_dict(state_dict)
        model.eval()
        explain_attention(df=train_set.dataset.data_df, graph=dataset.graph, model=model, topk=50)
        #explain_attention(df=train_set.dataset.data_df, graph=dataset.graph, model=model, topk=200)

def get_train_data_split(train_idx, test_idx, dataset, seed):
    train_set, _ = split_dataset(
        dataset, split_method="custom", split_idx=(train_idx, test_idx)
    )
    train_idx, val_idx = train_val_split(
        train_set.indices,
        test_size=0.1,
        random_state=seed,
        stratify=dataset.get_labels(train_set.indices),
    )
    train_set, _ = split_dataset(
        dataset, split_method="custom", split_idx=(list(train_idx), list(val_idx))
    )
    return train_set

if __name__ == "__main__":
    load_dotenv(".env")
    explain()