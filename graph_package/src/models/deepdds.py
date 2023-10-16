from typing import List, Optional
from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.models import DeepDDS as DeepDDS_cx


class DeepDDS(DeepDDS_cx):
    def __init__(
        self,
        *,
        context_channels: int = 954,
        context_hidden_dims: List[int] = None,
        drug_channels: int = 1024,
        drug_gcn_hidden_dims: List[int] = [1024,512,156],
        drug_mlp_hidden_dims: List[int] = [2048, 512],
        context_output_size: int = 32,
        fc_hidden_dims: List[int] = [1024,512,128],
        dropout: float = 0.2
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
