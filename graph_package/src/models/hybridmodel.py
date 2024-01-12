from torch import nn
import torch
from graph_package.src.models import DeepDDS
from graph_package.src.models.distmult import DistMult
from collections import OrderedDict


class HybridModel(nn.Module):
    def __init__(
        self,
        distmult: dict,
        deepdds: dict,
        ckpt_path=None,
        pretrain_model="distmult",
        comb_weight = 0.95,
        comb_weight_req_grad=True,
    ) -> None:
        super().__init__()

        self.deepdds = self.load_model(
            DeepDDS, deepdds, ckpt_path, freeze=pretrain_model == "deepdds"
        )
        self.distmult = self.load_model(
            DistMult, distmult, ckpt_path, freeze=pretrain_model == "distmult"
        )

        if not pretrain_model:
            deepdds_init_weight=distmult_init_weight=0.5
        else:
            deepdds_init_weight = comb_weight if pretrain_model=='deepdds' else 1-comb_weight
            distmult_init_weight = 1-deepdds_init_weight

        self.deepdds_weight = nn.Parameter(
            torch.tensor([deepdds_init_weight]), requires_grad=comb_weight_req_grad
        )
        self.distmult_weight = nn.Parameter(
            torch.tensor([distmult_init_weight]), requires_grad=comb_weight_req_grad
        )

    def load_model(self, model_construct, model_kwargs, ckpt_path, freeze=False):
        model = model_construct(**model_kwargs)
        if freeze:
            state_dict = remove_prefix_from_keys(
                torch.load(ckpt_path)["state_dict"], "model."
            )
            model.load_state_dict(state_dict)
            self.freeze_model(model)
        return model

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, input):
        deepdds_out = self.deepdds(input)
        distmult_out = self.distmult(input)
        return self.deepdds_weight * deepdds_out + self.distmult_weight * distmult_out

    def __str__(self) -> str:
        return "hybridmodel"


def remove_prefix_from_keys(d, prefix):
    """
    Recursively removes a prefix from the keys of an ordered dictionary and all its sub-dictionaries.

    Args:
        d (OrderedDict): The ordered dictionary to modify.
        prefix (str): The prefix to remove from the keys.

    Returns:
        OrderedDict: The modified ordered dictionary.
    """
    new_dict = OrderedDict()
    for key, value in d.items():
        new_key = key[len(prefix) :] if key.startswith(prefix) else key
        new_dict[new_key] = value
    return new_dict
