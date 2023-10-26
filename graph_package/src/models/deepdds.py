from typing import List, Optional
from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.models import DeepDDS as DeepDDS_cx
import torch
from torchdrug.data import PackedGraph


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
    
    def _forward_molecules(self, molecules: PackedGraph) -> torch.FloatTensor:

        features = self.drug_conv(molecules, molecules.data_dict["atom_feature"].float())["node_feature"]
        features = self.drug_readout(molecules, features)
        return self.drug_mlp(features)
    

    def __name__(self) -> str:
        return "DeepDDS"
