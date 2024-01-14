from pytorch_lightning import LightningModule
from graph_package.src.pl_modules.metrics import RegMetrics, ClfMetrics
from torchmetrics import MeanSquaredError
from torch.optim import Adam
from torch.nn import ModuleDict, BCEWithLogitsLoss, MSELoss
from torchmetrics import AUROC
import torch

loss_func_dict = {"clf": BCEWithLogitsLoss(), "reg": MSELoss()}


class BasePL(LightningModule):
    def __init__(
        self,
        model,
        lr: float = 0.001,
        task: str = "clf",
        logger_enabled: bool = True,
        target: str = "zip_mean",
        l2_reg: bool = False,
        model_config: dict = {},
    ):
        super().__init__()
        self.lr = lr
        self.task = task
        self.loss_func = loss_func_dict[task]
        metric = ClfMetrics if task == "clf" else RegMetrics
        self.val_metrics = metric("val", target)
        self.test_metrics = metric("test", target)
        self.model = model
        self.logger_enabled = logger_enabled
        self.l2_reg = l2_reg
        self.model_config = model_config

    def forward(self, inputs):
        return self.model(inputs)

    def _step(self, batch):
        inputs = batch[0]
        target = batch[1]
        preds = self(inputs)
        preds = preds.view(-1)
        loss = self.loss_func(preds, target)
        return loss, target, preds

    def training_step(self, batch, batch_idx):
        loss, target, preds = self._step(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=self.logger_enabled,
        )
        if str(self.model) == "hybridmodel":
            self.log(
                "deepdds_weight",
                self.model.deepdds_weight.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=self.logger_enabled,
            )
            self.log(
                "distmult_weight",
                self.model.distmult_weight.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=self.logger_enabled,
            )

        if self.l2_reg:
            self.weight_decay = 1e-5
            l2_regularization = 0.0
            for param in self.parameters():
                l2_regularization += torch.norm(param, p=2)
            loss += 0.5 * self.weight_decay * l2_regularization

        return loss

    def validation_step(self, batch, batch_idx):
        loss, target, preds = self._step(batch)
        metrics = self.val_metrics(preds, target)
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=self.logger_enabled
        )
        for key, val in metrics.items():
            self.log(
                f"{key}",
                val,
                on_epoch=True,
                prog_bar=True,
                logger=self.logger_enabled,
                batch_size=len(batch),
            )
        return

    def test_step(self, batch, batch_idx):
        loss, target, preds = self._step(batch)
        metrics = self.test_metrics(preds, target)
        for key, val in metrics.items():
            if "CM" not in key:
                self.log(
                    f"{key}",
                    val,
                    on_epoch=True,
                    prog_bar=True,
                    logger=self.logger_enabled,
                    batch_size=len(batch),
                )
            else:
                df_cm = val.cpu()

        metrics.pop("CM", None)

        self.test_step_outputs = {
            "df_cm": df_cm.cpu(),
            "metrics": {key: val.cpu() for key, val in metrics.items()},
            "preds": preds.cpu(),
            "target": (target.cpu() >= 10).to(torch.int64),
            "batch": [t.cpu() for t in batch],
            "batch_idx": batch_idx,
        }

        return

    def configure_optimizers(self):
        if "ph_kwargs" in self.model_config:
            if self.model.prediction_head.__class__.__name__ == "MLP":
                pk_kwargs = self.model_config.ph_kwargs
                self.custom_lr_setup = pk_kwargs.custom_lr_setup.value
                if self.custom_lr_setup:
                    lr_setup = pk_kwargs.custom_lr_setup
                    return Adam(
                        [
                            {
                                "params": self.model.gnn_layers.parameters(),
                                "lr": self.lr,
                            },
                            {
                                "params": self.model.prediction_head.cell_line_mlp.parameters(),
                                "lr": lr_setup.lr_ph_cell_mlp,
                            },
                            {
                                "params": self.model.prediction_head.global_mlp.parameters(),
                                "lr": lr_setup.lr_ph_global_mlp,
                            },
                        ]
                    )
            return Adam(self.model.parameters(), lr=self.lr)

        else:
            return Adam(self.model.parameters(), lr=self.lr)
