from pytorch_lightning import LightningModule, Trainer
from graph_package.src.error_analysis.utils import (
    save_performance_plots,
    save_model_pred,
)
from graph_package.src.main_utils import (
    save_pretrained_drug_embeddings
)
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.core import LightningModule
class TestDiagnosticCallback(Callback):
    def __init__(self, model_name, config: DictConfig, fold: int) -> None:
        self.model_name = model_name
        self.config = config
        self.fold = fold
        super().__init__()

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
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
        # save pretrained drug embeddings
        if self.model_name in ["deepdds", "distmult"]:
            save_pretrained_drug_embeddings(model=pl_module,fold=self.fold)
        pl_module.test_step_outputs.clear()

class LossFnCallback(Callback):
    def __init__(self, epochs_wo_var) -> None:
        self.epochs_wo_var = epochs_wo_var
        super().__init__()  

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.current_epoch == self.epochs_wo_var:
            pl_module.loss_func.set_var_grad()
