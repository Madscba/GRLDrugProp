from pytorch_lightning import LightningModule
from graph_package.src.pl_modules.metrics import RegMetrics, ClfMetrics
from torchmetrics import MeanSquaredError
from torch.optim import Adam
from torch.nn import ModuleDict, BCEWithLogitsLoss, MSELoss
import torch

loss_func_dict = {"clf": BCEWithLogitsLoss(), "reg": MSELoss()}


class BasePL(LightningModule):
    def __init__(self, model, lr: float = 0.001, task: str = "clf"):
        super().__init__()
        self.lr = lr
        self.task = task
        self.loss_func = loss_func_dict[task]
        metric  = ClfMetrics if task == "clf" else RegMetrics
        self.val_metrics = metric("val")
        self.test_metrics = metric("test")
        self.model =  model

    
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
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, target, preds = self._step(batch)
        metrics = self.val_metrics(preds, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        for key, val in metrics.items():
            self.log(
                f"{key}",
                val,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=len(batch),
            )
        return

    def test_step(self, batch, batch_idx):
        loss, target, preds = self._step(batch)
        metrics = self.test_metrics(preds, target)
        for key, val in metrics.items():
                self.log(
                    f"{key}",
                    val.item(),
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=len(batch),
                )
        return

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)

