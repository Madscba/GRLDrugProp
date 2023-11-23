from graph_package.src.error_analysis.utils import (
    save_performance_plots,
    save_model_pred,
)
from pytorch_lightning.callbacks import Callback


class TestDiagnosticCallback(Callback):
    def __init__(self,model_name: str) -> None:
        self.model_name = model_name
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
            df_cm, metrics, preds, target, self.model_name, save_path=""
        )
        save_model_pred(
            batch_idx, batch, preds, target, self.model_name, save_path=False
        )
        pl_module.test_step_outputs.clear()
