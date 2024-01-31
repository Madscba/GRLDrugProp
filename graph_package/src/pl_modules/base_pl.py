from pytorch_lightning import LightningModule
from graph_package.src.pl_modules.metrics import RegMetrics, ClfMetrics
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from graph_package.src.pl_modules.loss_functions import MSECellLineVar
import torch

loss_func_dict = {"clf": BCEWithLogitsLoss, "reg": MSECellLineVar}


class BasePL(LightningModule):
    def __init__(
        self,
        model,
        num_relation,
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
        self.loss_func = (
            loss_func_dict[task](num_relation)
            if task == "reg"
            else loss_func_dict[task]()
        )
        metric = ClfMetrics if task == "clf" else RegMetrics
        self.val_metrics = metric("val", target)
        self.test_metrics = metric("test", target)
        self.model = model
        self.graph = graph
        self.logger_enabled = logger_enabled
        self.l2_reg = l2_reg
        self.model_config = model_config

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
        return self.model(inputs)

    def _step(self, batch):
        inputs = batch[0]
        target = batch[1]
        preds = self(inputs)
        preds = preds.view(-1)
        if self.task == "reg":
            cell_line_ids = inputs[:, 2]
            loss = self.loss_func(preds, target, cell_line_ids)
        else:
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

        step_o = {
            "df_cm": df_cm.cpu(),
            "metrics": {key: val.cpu() for key, val in metrics.items()},
            "preds": preds.cpu(),
            "target": (target.cpu() >= 10).to(torch.int64),
            "batch": [t.cpu() for t in batch],
            "batch_idx": [batch_idx],
        }

        keys_to_concat = ["preds", "target"]

        if batch_idx == 0:
            self.test_outputs = step_o
        else:
            self.test_outputs["df_cm"] = self.test_outputs["df_cm"] + step_o["df_cm"]
            self.test_outputs["batch"] = [torch.vstack([ self.test_outputs["batch"][0], step_o["batch"][0]]),
                                          torch.hstack([ self.test_outputs["batch"][1], step_o["batch"][1]])]
            self.test_outputs["batch_idx"] += [step_o["batch_idx"]]

            for key in keys_to_concat:
                self.test_outputs[key] = torch.cat([self.test_outputs[key], step_o[key]], dim=0)

            for key, val in metrics.items():
                if key in ("test_auprc", "test_auroc", "test_F1", "test_mse"):
                    #incremental estimate of the mean of the metric
                    # mu_k = mu_{k-1} + (x_k - mu_{k-1})/k
                    self.test_outputs["metrics"][key] += + (self.test_outputs["metrics"][key] - step_o["metrics"][key]) / (batch_idx + 1)

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
                            {"params": self.loss_func.parameters(), "lr": self.lr},
                        ]
                    )
            return Adam(
                [
                    {"params": self.model.parameters()},
                    {"params": self.loss_func.parameters()},
                ],
                lr=self.lr,
            )

        else:
            return Adam(
                [
                    {"params": self.model.parameters()},
                    {"params": self.loss_func.parameters()},
                ],
                lr=self.lr,
            )
