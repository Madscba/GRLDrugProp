from torchmetrics import AUROC, AveragePrecision, MeanSquaredError
import torch


class RegMetrics(torch.nn.Module):
    def __init__(self, type):
        super().__init__()
        self.set = type
        self.auroc = AUROC(task="binary")
        self.auprc = AveragePrecision(task="binary")
        self.mse = MeanSquaredError()

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        target_clf = target.apply_(lambda x: 1 if x >= 10 else 0).to(torch.int64)
        return {
            f"{self.set}_auroc": self.auroc(preds, target_clf),
            f"{self.set}_auprc": self.auprc(preds, target_clf),
            f"{self.set}_mse": self.mse(preds, target),
        }


class ClfMetrics(torch.nn.Module):
    def __init__(self, type):
        super().__init__()
        self.set = type
        self.auroc = AUROC(task="binary")
        self.auprc = AveragePrecision(task="binary")

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        target = target.to(torch.int64)
        return {
            f"{self.set}_auroc": self.auroc(preds, target),
            f"{self.set}_auprc": self.auprc(preds, target),
        }
