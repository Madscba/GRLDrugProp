from torchmetrics import AUROC, AveragePrecision, MeanSquaredError, Accuracy,CalibrationError, ConfusionMatrix, F1Score
from torch.nn import ModuleDict
import torch

metric_dict = {"auprc":AveragePrecision,
               "auroc":AUROC,
                "acc":Accuracy,
                "calib_err":CalibrationError,
                "CM":ConfusionMatrix,
                "F1":F1Score}

test = ["auprc","auroc","acc","calib_err","CM","F1"]
train_val = ["auprc","auroc"]

class RegMetrics(torch.nn.Module):
    def __init__(self, type):
        super().__init__()
        self.set = type
        self.metric_types = test if type == "test" else train_val
        self.clf_metrics = ModuleDict({f"{type}_{metric}":metric_dict[metric](task="binary") for metric in self.metric_types})
        self.reg_metrics = ModuleDict({f"{type}_mse":MeanSquaredError()})

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        target_clf = target.apply_(lambda x: 1 if x >= 10 else 0).to(torch.int64)
        clf_metrics = {key:metric(preds,target_clf) for key,metric in self.clf_metrics.items()}
        reg_metrics = {key:metric(preds,target) for key, metric in self.reg_metrics.items()}
        return clf_metrics.update(reg_metrics)

class ClfMetrics(torch.nn.Module):
    def __init__(self, type):
        super().__init__()
        self.set = type
        self.metric_types = test if type == "test" else train_val
        self.clf_metrics = ModuleDict({f"{type}_{metric}":metric_dict[metric](task="binary") for metric in self.metric_types})

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        target = target.to(torch.int64)
        clf_metrics = {key:metric(preds,target) for key,metric in self.clf_metrics.items()}
        return clf_metrics