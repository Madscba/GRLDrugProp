from torchmetrics import AUROC, AveragePrecision, MeanSquaredError, Accuracy,CalibrationError, ConfusionMatrix, F1Score
from torch.nn import ModuleDict
import torch

metric_dict = {"auprc":AveragePrecision,
               "auroc":AUROC,
                "CM":ConfusionMatrix,
                "F1":F1Score}

test = ["auprc","auroc","CM","F1"]
train_val = ["auprc","auroc"]

class RegMetrics(torch.nn.Module):
    def __init__(self, type, target):
        super().__init__()
        self.set = type
        self.metric_types = test if type == "test" else train_val
        self.clf_metrics = ModuleDict({f"{type}_{metric}":metric_dict[metric](task="binary") for metric in self.metric_types})
        self.reg_metrics = ModuleDict({f"{type}_mse":MeanSquaredError()})
        self.thres = 5 if target == "zip_mean" else 10

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        target_clf = (target >= self.thres).to(torch.int64)
        clf_metrics = {key:metric(preds,target_clf) for key,metric in self.clf_metrics.items()}
        reg_metrics = {key:metric(preds,target) for key, metric in self.reg_metrics.items()}
        clf_metrics.update(reg_metrics)
        return clf_metrics

class ClfMetrics(torch.nn.Module):
    def __init__(self, type, target):
        super().__init__()
        self.set = type
        self.metric_types = test if type == "test" else train_val
        self.clf_metrics = ModuleDict({f"{type}_{metric}":metric_dict[metric](task="binary") for metric in self.metric_types})

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        target = target.to(torch.int64)
        clf_metrics = {key:metric(preds,target) for key,metric in self.clf_metrics.items()}
        return clf_metrics