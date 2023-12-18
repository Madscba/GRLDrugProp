import torch
import torch.nn as nn

from torch import nn
from torch.nn.functional import normalize
from typing import List, Optional, Iterable

from torchdrug.layers import MLP


from collections.abc import Sequence

import torch
from torch import nn

from torchdrug import core, layers
from .layers import RelationalGraphConv
from torchdrug.core import Registry as R


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RelationalGraphConvolutionalNetwork(nn.Module, core.Configurable):
    """
    Relational Graph Convolutional Network proposed in `Modeling Relational Data with Graph Convolutional Networks?`_.

    .. _Modeling Relational Data with Graph Convolutional Networks?:
        https://arxiv.org/pdf/1703.06103.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
        super(RelationalGraphConvolutionalNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(RelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation, edge_input_dim,
                                                          batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Require the graph(s) to have the same number of relations as this module.

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }



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

        self.graph
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