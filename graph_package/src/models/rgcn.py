import torch
import torch.nn as nn

from torch import nn
from torch.nn.functional import normalize
from typing import List, Optional, Iterable

from torchdrug.layers import MLP
from torchdrug.models import RelationalGraphConvolutionalNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RGCN(nn.Module):
    def __init__(
        self,
        graph, 
        hidden_dims: List[int] = [512],
        batch_norm=False,
        activation="relu",
        last_dim_size: int = 128,
        fc_hidden_dims: List[int] = [128, 64], 
        dropout: float = 0.2, 
    ):
        super().__init__()
        self.graph = graph
        self.node_feature_dim = self.graph.node_feature.shape[1]
        self.edge_feature_dim = self.graph.edge_feature.shape[1]
        self.drug_conv = RelationalGraphConvolutionalNetwork(
            input_dim=self.node_feature_dim,
            hidden_dims=[*hidden_dims, last_dim_size],
            num_relation=self.graph.num_relation,
            edge_input_dim=self.edge_feature_dim,
            batch_norm=batch_norm,
            activation=activation
        )
        # MLP for cancer cell lines gene expression profiles
        self.ccle_mlp = MLP(
                input_dim=self.edge_feature_dim,
                hidden_dims=[256, last_dim_size],
                activation="relu"
        )
        # Final prediction head that takes node embddings of d
        self.final = nn.Sequential(
            MLP(
                input_dim=last_dim_size*3,
                hidden_dims=[*fc_hidden_dims, 1],
                dropout=dropout,
            )
        )

    def forward(
        self, inputs
    ) -> torch.FloatTensor:
        """Run a forward pass of the R-GCN model.

        :returns: A vector of predicted synergy scores
        """
        drug_1_ids, drug_2_ids, context_ids = map(lambda x: x.squeeze(),inputs.split(1, dim=1))
        x = self.drug_conv(self.graph,self.graph.node_feature)['node_feature']
        x1 = x[drug_1_ids]
        x2 = x[drug_2_ids]
        x3 = self.ccle_mlp(self.graph.edge_feature[context_ids])
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.final(x)
        return x

    def __name__(self) -> str:
        return "GCN"