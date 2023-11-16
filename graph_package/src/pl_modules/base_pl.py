from pytorch_lightning import LightningModule
from graph_package.src.models import DeepDDS, DeepDDS_HPC
from torch.optim import Adam
from torch.nn import ModuleDict, BCEWithLogitsLoss
from torchmetrics import AUROC, AveragePrecision
import torch


class BasePL(LightningModule):
    def __init__(self, model, lr: float = 0.001):
        super().__init__()
        self.lr = lr
        self.loss_func = BCEWithLogitsLoss()
        self.val_metrics = self.build_metrics(type="val")
        self.test_metrics = self.build_metrics(type="test")
        self.model =  model

    
    def forward(self, inputs):
        return self.model(inputs)
    
    def _step(self, batch):
        inputs = batch[0]
        target = batch[1]
        preds = self(inputs)
        preds = preds.view(-1)
        loss = self.loss_func(preds, target)
        return loss, target.to(torch.int), preds


    def training_step(self, batch, batch_idx):
        loss, target, preds = self._step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, target, preds = self._step(batch)
        metrics = {key: val(preds, target) for key, val in self.val_metrics.items()}
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
        metrics = {key: val(preds, target) for key, val in self.test_metrics.items()}
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

    def build_metrics(self, type):
        kwargs = {"task": "binary"}
        module_dict = ModuleDict(
            {
                f"{type}_auroc": AUROC(**kwargs),
                f"{type}_auprc": AveragePrecision(**kwargs),
            }
        )
        return module_dict