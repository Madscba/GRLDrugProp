from torch import nn
import torch
from graph_package.src.models import RESCAL, DeepDDS
from collections import OrderedDict



class HybridModel(nn.Module):
    def __init__(
        self,
        hpc,
        rescal: dict,
        deepdds: dict,
        ckpt_path=None,
        pretrain_model="rescal",
    ) -> None:
        super().__init__()

        self.deepdds = self.load_model(
            DeepDDS, deepdds, ckpt_path, freeze=pretrain_model == "deepdds"
        )
        self.rescal = self.load_model(
            RESCAL, rescal, ckpt_path, freeze=pretrain_model == "rescal"
        )

        self.deepdds_weight = nn.Parameter(torch.tensor([0.5]))
        self.rescal_weight = nn.Parameter(torch.tensor([0.5]))

    def load_model(self, model_construct, model_kwargs,  ckpt_path, freeze=False):
        model = model_construct(**model_kwargs)
        if freeze:
            state_dict = remove_prefix_from_keys(torch.load(ckpt_path)["state_dict"],'model.')
            model.load_state_dict(state_dict)
            self.freeze_model(model)
        return model

    def freeze_model(self,model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, input):
        deepdds_out = self.deepdds(input)
        rescal_out = self.rescal(input)
        return self.deepdds_weight * deepdds_out + self.rescal_weight * rescal_out

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
        new_key = key[len(prefix):] if key.startswith(prefix) else key
        new_dict[new_key] = value
    return new_dict
