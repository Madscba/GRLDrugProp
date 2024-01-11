from graph_package.src.error_analysis.utils import (
    save_performance_plots,
    save_model_pred,
)
from graph_package.src.main_utils import (
    init_model,
    save_pretrained_drug_embeddings
)
from graph_package.src.models.hybridmodel import remove_prefix_from_keys
from graph_package.src.explainability.utils import explain_attention
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig
from pathlib import Path
from graph_package.src.etl.dataloaders import KnowledgeGraphDataset
import torch
import pandas as pd

class TestDiagnosticCallback(Callback):
    def __init__(self, model_name, config: DictConfig, graph: KnowledgeGraphDataset, fold: int, triplets: pd.DataFrame) -> None:
        self.model_name = model_name
        self.config = config
        self.graph = graph
        self.fold = fold
        self.triplets = triplets

    def on_test_end(self, trainer, pl_module):
        # save and perform err_diag
        (
            df_cm,
            metrics,
            preds,
            target,
            batch,
            batch_idx,
        ) = pl_module.test_step_outputs.values()
        print("conf_matrix:\n", df_cm)

        #save_performance_plots(
        #    df_cm, metrics, preds, target, self.config, self.model_name, save_path=Path("")
        #)
        #save_model_pred(
        #    batch_idx, batch, preds, target, self.config, self.model_name, save_path=Path("")
        #)
        if (self.model_name=="gnn") & (self.config.model.layer in ['rgat', 'rgac', 'gat', 'gac']):
            model = init_model(
                model=self.model_name,
                config=self.config,
                graph=self.graph,
            ).model
            state_dict = remove_prefix_from_keys(
                torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"], "model."
            )
            model.load_state_dict(state_dict)
            model.eval()
            explain_attention(df=self.triplets, graph=self.graph, model=model)
        # save pretrained drug embeddings
        if self.model_name in ["deepdds", "distmult"]:
            save_pretrained_drug_embeddings(model=pl_module,fold=self.fold)
        pl_module.test_step_outputs.clear()