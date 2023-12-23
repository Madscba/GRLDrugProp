from pytorch_lightning import LightningModule
from graph_package.src.pl_modules.metrics import RegMetrics, ClfMetrics
from graph_package.src.pl_modules.explainer import Explainer
from typing import List, Tuple, Dict, Optional, Union
from graph_package.src.etl.dataloaders import KnowledgeGraphDataset
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
        graph,
        lr: float = 0.001,
        task: str = "clf",
        logger_enabled: bool = True,
        target: str = "zip_mean"
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
        self.graph = graph

    def forward(self, inputs, *explainer_node_ids, **explainer_edge_ids):
        node_features = self.graph.node_feature
        if isinstance(inputs, dict):
            # Construct the explainer inputs to fit the model
            drug_1_ids = explainer_node_ids[0]['drug', 'interacts_with', 'drug'][0]
            drug_2_ids = explainer_node_ids[0]['drug', 'interacts_with', 'drug'][1]
            node_features = inputs['drug']
            if not explainer_edge_ids:
                context_ids = explainer_node_ids[1]
            else:
                context_ids = explainer_edge_ids['edge_label_index']
            inputs = torch.stack([drug_1_ids, drug_2_ids, context_ids], dim=1)
        return self.model(inputs, node_features)

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
                "rescal_weight",
                self.model.rescal_weight.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=self.logger_enabled,
            )
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
        return Adam(self.model.parameters(), lr=self.lr)
