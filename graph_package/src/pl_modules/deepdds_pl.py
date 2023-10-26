from pytorch_lightning import LightningModule
from graph_package.src.models import DeepDDS, DeepDDS_HPC
from torch.optim import Adam
from torch.nn import BCELoss, ModuleDict
from torchmetrics import AUROC, AveragePrecision
import torch


class DeepDDS_PL(LightningModule):
    def __init__(self, lr: float = 0.001, hpc: bool = False):
        super().__init__()
        self.lr = lr
        self.model = DeepDDS_HPC() if hpc else DeepDDS()
        self.loss_func = BCELoss()
        self.val_metrics = self.build_metrics(type="val")
        self.test_metrics = self.build_metrics(type="test")

    def forward(self, inputs):
        return self.model(*inputs)

    def _step(self, batch):
        inputs = (
            batch["context_features"],
            batch["drug_molecules_left"],
            batch["drug_molecules_right"],
        )
        target = batch["label"]
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
