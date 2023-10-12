import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, tasks, metrics
from torchdrug.core import Registry as R
from torchdrug.layers import functional

class SynergyPrediction(tasks.Task, core.Configurable):
    """
    Graph / molecule property prediction task.

    This class is also compatible with semi-supervised learning.

    Parameters:
        model (nn.Module): graph representation model
        task (str, list or dict, optional): training task(s).
            For dict, the keys are tasks and the values are the corresponding weights.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse`` and ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = set(["task", "criterion", "metric"])

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), verbose=0):
        super(SynergyPrediction, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.verbose = verbose
        self.weight = 1
        self.loss_func = torch.nn.BCELoss()


    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        pred = self.predict(batch, all_loss, metric)
        target = batch['label'].unsqueeze(1)
        loss = self.loss_func(pred, target)
        name = tasks._get_criterion_name("bce")
        metric[name] = loss
        return loss, metric 

    def predict(self, batch, all_loss=None, metric=None):
        batch = (batch['context_features'], batch["drug_molecules_left"], batch["drug_molecules_right"])
        output = self.model(*batch)
        return output

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)

        metric = {}
        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred * self.std + self.mean, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred * self.std + self.mean, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            elif _metric == "auroc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.area_under_roc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "auprc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.area_under_prc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "r2":
                score = []
                new_pred = pred * self.std + self.mean
                for _pred, _target, _labeled in zip(new_pred.t(), target.t(), labeled.t()):
                    _score = metrics.r2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return metric
