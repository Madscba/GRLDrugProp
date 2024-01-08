from graph_package.src.error_analysis.utils import (
    save_performance_plots,
    save_model_pred,
)
from graph_package.src.main_utils import (
    init_model
)
from graph_package.src.models.hybridmodel import remove_prefix_from_keys
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig
from pathlib import Path
from graph_package.src.etl.dataloaders import KnowledgeGraphDataset
import torch

class TestDiagnosticCallback(Callback):
    def __init__(self, model_name, config: DictConfig, graph: KnowledgeGraphDataset) -> None:
        self.model_name = model_name
        self.config = config
        self.graph = graph
        super().__init__()

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

        save_performance_plots(
            df_cm, metrics, preds, target, self.config, self.model_name, save_path=Path("")
        )
        save_model_pred(
            batch_idx, batch, preds, target, self.config, self.model_name, save_path=Path("")
        )
        if self.model_name=="gnn":
            model = init_model(
                model=self.model_name,
                config=self.config,
                graph=self.graph,
            ).model
            state_dict = remove_prefix_from_keys(
                torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"], "model."
            )
            model.load_state_dict(state_dict)
        pl_module.test_step_outputs.clear()

    def _explain_attention(self,batch,model):

        return
