from pytorch_lightning import LightningModule
from graph_package.src.models import RESCAL
from torch.optim import Adam
from torch.nn import BCELoss, BCEWithLogitsLoss, ModuleDict
from torchmetrics import AUROC, AveragePrecision
import torch


class Rescal_PL(LightningModule):
    def __init__(self, ent_tot, rel_tot, dim=100, lr: float = 1e-4):
        super().__init__()
        self.lr = lr
        self.model = RESCAL(ent_tot, rel_tot, dim)
        self.loss_func = BCEWithLogitsLoss()  # BCELoss()
        self.val_metrics = self.build_metrics(type="val")
        self.test_metrics = self.build_metrics(type="test")

    def forward(self, inputs):
        h_index, t_index, r_index = (
            inputs[:, 0].reshape(1, -1),
            inputs[:, 1].reshape(1, -1),
            inputs[:, 2].reshape(1, -1),
        )
        return self.model(h_index, t_index, r_index)

    def _step(self, batch):
        inputs = batch[:, :3]
        target = batch[:, 3].to(dtype=torch.float32)  # needs to be float 32
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
                val,
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
                f"{type}_auprc": AveragePrecision(**kwargs),
                f"{type}_auroc": AUROC(**kwargs),
            }
        )
        return module_dict
