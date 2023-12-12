import ssl
import json
import torch
import urllib.request
import numpy as np
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
        dataset_path,
        hidden_dims: List[int] = [512],
        batch_norm=False,
        activation="relu",
        context_channels: int = 288,
        context_hidden_dims: List[int] = (2048, 512),
        last_dim_size: int = 128,
        fc_hidden_dims: List[int] = [128, 64], 
        dropout: float = 0.2, 
    ):
        super().__init__()
        self.graph = graph
        #self.context_vocab = json.load(open(dataset_path.parent / "relation_vocab.json","r"))
        #self.context_features = self.load_context_features()
        #self.graph.edge_feature = torch.as_tensor(self.context_features, device=self.device)
        self.node_feature_dim = self.graph.node_feature.shape[1]
        #self.edge_feature_dim = self.context_features[0].shape[-1]
        self.drug_conv = RelationalGraphConvolutionalNetwork(
            input_dim=self.node_feature_dim,
            hidden_dims=[*hidden_dims, last_dim_size],
            num_relation=self.graph.num_relation,
            #edge_input_dim=self.edge_feature_dim,
            batch_norm=batch_norm,
            activation=activation
        )
        self.final = nn.Sequential(
            MLP(
                input_dim=last_dim_size*2,
                hidden_dims=[*fc_hidden_dims, 1],
                dropout=dropout,
            )
        )

    def load_context_features(self) -> dict:
        """Get the context feature set."""
       # Create an SSL context that does not verify the SSL certificate
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        path = "https://raw.githubusercontent.com/AstraZeneca/chemicalx/main/dataset/drugcomb/context_set.json"
        with urllib.request.urlopen(path, context=ssl_context) as url:
            raw_data = json.loads(url.read().decode())
        #raw_data = {v: torch.FloatTensor(np.array(raw_data[k]).reshape(1, -1)).to(device) for k, v in self.context_vocab.items()}
        context_features = [np.array(raw_data[k]).reshape(-1) for k, v in self.context_vocab.items()]
        return context_features
    
    def _get_context_features(self, context_identifiers: Iterable[int]) -> Optional[torch.FloatTensor]:
        return torch.cat([self.context_features[context.item()] for context in context_identifiers])

    def forward(
        self, inputs
    ) -> torch.FloatTensor:
        """Run a forward pass of the GCN model.

        :returns: A vector of predicted synergy scores
        """
        drug_1_ids, drug_2_ids, context_ids = map(lambda x: x.squeeze(),inputs.split(1, dim=1))
        x = self.drug_conv(self.graph,self.graph.node_feature)['node_feature']
        x1 = x[drug_1_ids]
        x2 = x[drug_2_ids]
        #x3 = self._get_context_features(context_ids)
        #x3 = self.cell_mlp(x3)
        x = torch.cat([x1, x2], dim=1)
        x = self.final(x)
        return x

    def __name__(self) -> str:
        return "GCN"