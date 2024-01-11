import os
import sys
import torch
import hydra
from graph_package.src.models.hybridmodel import remove_prefix_from_keys
from graph_package.src.explainability.utils import explain_attention
from graph_package.configs.directories import Directories
from graph_package.src.main_utils import init_model, get_model_name
from dotenv import load_dotenv
from graph_package.src.etl.dataloaders import KnowledgeGraphDataset


@hydra.main(
    config_path=str(Directories.CONFIG_PATH / "hydra_configs"),
    config_name="config.yaml",
    version_base="1.1",
)
def explain(config, fold=0):
    model_name = get_model_name(config=config, sys_args=sys.argv)
    if (model_name == "gnn") & (config.dataset.drug_representation not in ["distmult", "deepdds"]):
        config.dataset.update({"use_node_features": True})
    dataset = KnowledgeGraphDataset(**config.dataset)
    checkpoint_path = str(Directories.CHECKPOINT_PATH / "gnn" / f"fold_{fold}")
    all_items = os.listdir(checkpoint_path)
    files = [item for item in all_items if os.path.isfile(os.path.join(checkpoint_path, item))]
    best_model_path = os.path.join(checkpoint_path,files[-1])
    model = init_model(
                model=model_name,
                config=config,
                graph=dataset.graph,
            ).model
    state_dict = remove_prefix_from_keys(
        torch.load(best_model_path)["state_dict"], "model."
    )
    model.load_state_dict(state_dict)
    model.eval()
    explain_attention(df=dataset.data_df, graph=dataset.graph, model=model)


if __name__ == "__main__":
    load_dotenv(".env")
    explain()