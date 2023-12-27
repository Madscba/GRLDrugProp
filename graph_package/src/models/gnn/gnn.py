import torch
import torch.nn as nn

from torch import nn
from torch.nn.functional import normalize
from torchdrug.data import Graph
import torch
from torch import nn
from torchdrug import core
from graph_package.src.models.gnn.gnn_layers import (
    RelationalGraphConv,
    GraphConv,
    DummyLayer,
)
from torchdrug.core import Registry as R
from graph_package.src.models.gnn.prediction_head import MLP, DistMult


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer_dict = {"rgc": RelationalGraphConv, "gc": GraphConv, "dummy": DummyLayer}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prediction_head_dict = {"mlp": MLP, "distmult": DistMult}


class GNN(nn.Module, core.Configurable):
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

    def __init__(
        self,
        graph: Graph,
        hidden_dims: list,
        layer: str,
        prediction_head: str,
        dataset: str,
        concat_hidden: bool = False,
        short_cut: bool = False,
        batch_norm: bool = False,
        feature_dropout: float = 0.0,
        enc_kwargs: dict = {},
        ph_kwargs: dict = {},
    ):
        super(GNN, self).__init__()
        self.graph = graph
        input_dim_gnn = [graph.node_feature.shape[1]]
        self.enc_kwargs = enc_kwargs
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.enc_kwargs.update(
            {"batch_norm": batch_norm, "feature_dropout": feature_dropout}
        )

        if layer == "gc":
            self.enc_kwargs.update({"dataset": dataset})

        if prediction_head == "distmult":
            ph_kwargs.update({"rel_tot": graph.num_relation.item()})

        elif prediction_head == "mlp":
            ph_kwargs.update({"dataset": dataset})
            ph_kwargs.update(
                {"batch_norm": batch_norm, "feature_dropout": feature_dropout}
            )

        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.gnn_layers = self._init_gnn_layers(
            layer, input_dim_gnn, hidden_dims, enc_kwargs
        )

        self.prediction_head = prediction_head_dict[prediction_head](
            dim=self.output_dim, **ph_kwargs
        )
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

    def _init_gnn_layers(
        self, layer: str, input_dim: list, hidden_dims: list, enc_kwargs: dict
    ):
        layers = nn.ModuleList()
        dims = input_dim + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(
                layer_dict[layer](
                    dims[i], dims[i + 1], self.graph.num_relation, **enc_kwargs
                )
            )
        return layers

    def encode(self, graph, input, all_loss=None, metric=None):
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

        for layer in self.gnn_layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        return node_feature

    def forward(self, inputs) -> torch.FloatTensor:
        """Run a forward pass of the R-GCN model.

        :returns: A vector of predicted synergy scores
        """
        drug_1_ids, drug_2_ids, context_ids = map(
            lambda x: x.squeeze(), inputs.split(1, dim=1)
        )
        drug_embeddings = self.encode(self.graph, self.graph.node_feature)
        d1 = drug_embeddings[drug_1_ids]
        d2 = drug_embeddings[drug_2_ids]
        out = self.prediction_head(d1, d2, context_ids, drug_1_ids, drug_2_ids)
        return out
