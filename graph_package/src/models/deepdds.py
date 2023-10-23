from typing import List, Optional
from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.models import DeepDDS as DeepDDS_cx

context_channels = 288
context_hidden_dims = (2048, 512)
drug_gcn_hidden_dims = [1024, 512, 156]
drug_mlp_hidden_dims = None
context_output_size = 156
fc_hidden_dims = [1024, 512, 128]
dropout = 0.2


class DeepDDS(DeepDDS_cx):
    def __init__(
        self,
        context_channels: int = 288,  # cx only have 288 instead 954 as in paper, might have to use other implementation
        context_hidden_dims: List[int] = (2048, 512),  # same as paper
        drug_channels: int = TORCHDRUG_NODE_FEATURES,  # don't know about paper
        drug_gcn_hidden_dims: List[int] = [1024, 512, 156],  # same as paper
        drug_mlp_hidden_dims: List[int] = None,  # not in paper, only for cx
        context_output_size: int = 156,  # same as paper, based on figure 1 of paper
        fc_hidden_dims: List[int] = [1024, 512, 128],  # s ame as paper
        dropout: float = 0.2,  # same as paper
    ):
        super().__init__(
            context_channels=context_channels,
            context_hidden_dims=context_hidden_dims,
            drug_channels=drug_channels,
            drug_gcn_hidden_dims=drug_gcn_hidden_dims,
            drug_mlp_hidden_dims=drug_mlp_hidden_dims,
            context_output_size=context_output_size,
            fc_hidden_dims=fc_hidden_dims,
            dropout=dropout,
        )

    def __name__(self) -> str:
        return "DeepDDS"
