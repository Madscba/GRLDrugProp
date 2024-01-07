import torch
import torch.nn as nn

from torch import nn
from torch.nn.functional import normalize
from torchdrug.data import Graph
import torch
from torch import nn
from torchdrug import core
from graph_package.src.models.gnn.att_layers import (
    GraphAttentionLayer,
    GraphAttentionConv,
    RelationalGraphAttentionLayer,
    RelationalGraphAttentionConv,
    GraphAttentionLayerPerCellLine,
)

from graph_package.src.models.gnn.conv_layers import (
    GraphConv,
    RelationalGraphConv,
    DummyLayer,
)

from torchdrug.core import Registry as R
from graph_package.src.models.gnn.prediction_head import MLP, DistMult, DotProduct


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


layer_dict = {
    "rgc": RelationalGraphConv,
    "gc": GraphConv,
    "dummy": DummyLayer,
    "gac": GraphAttentionConv,
    "gat": GraphAttentionLayer,
    "gac": GraphAttentionConv,
    "rgat": RelationalGraphAttentionLayer,
    "rgac": RelationalGraphAttentionConv,
    "gat_pr_rel": GraphAttentionLayerPerCellLine,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prediction_head_dict = {"mlp": MLP, "distmult": DistMult, "dotproduct": DotProduct}


class GNN(nn.Module, core.Configurable):
    """
    Relational Graph Convolutional Network proposed in `Modeling Relational Data with Graph Convolutional Networks?`_.

    .. _Modeling Relational Data with Graph Convolutional Networks?:
        https://arxiv.org/pdf/1703.06103.pdf

    Parameters:
        - graph (torchdrug.Graph): use for messsage passing
        - hidden_dims (list of int): hidden dimensions
        - layer (str): key in layer_dict for fetching layer class
        - prediction_head (str): key in prediction_head_dict for fetching prediction head class
        - dataset (str): name of dataset
        - concat_hidden (bool, optional): concat hidden representations from all layers as output
        - short_cut (bool, optional): use short cut or not
        - enc_kwargs (dict, optional): additional arguments for layer class
        - ph_kwargs (dict, optional): additional arguments for prediction head class

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
        enc_kwargs: dict = {},
        ph_kwargs: dict = {},
    ):
        super(GNN, self).__init__()
        self.graph = graph
        self.enc_kwargs = enc_kwargs
        self.layer_name = layer
        self.ph_kwargs = ph_kwargs
        self.update_kwargs(layer, prediction_head, dataset)

        input_dim_gnn = [graph.node_feature.shape[1]]

        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.gnn_layers = self._init_gnn_layers(
            layer, input_dim_gnn, hidden_dims, enc_kwargs
        )

        dim = self.output_dim if not layer == "dummy" else graph.num_node
        self.prediction_head = prediction_head_dict[prediction_head](
            dim=dim, **ph_kwargs
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
                if self.layer_name == "gat_pr_rel":
                    layer_input = layer_input.expand(
                        self.gnn_layers[0].num_relations + 1,
                        layer_input.shape[0],
                        layer_input.shape[1],
                    )
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        return node_feature

    def forward(self, inputs) -> torch.FloatTensor:
        """Run a forward pass of the GNN model.

        :returns: A vector of predicted synergy scores
        """
        drug_1_ids, drug_2_ids, context_ids = map(
            lambda x: x.squeeze(), inputs.split(1, dim=1)
        )
        drug_embeddings = self.encode(self.graph, self.graph.node_feature)
        drug_1_emb_ids, drug_2_emb_ids = self.get_drug_ids(
            drug_1_ids, drug_2_ids, context_ids
        )
        d1 = drug_embeddings[drug_1_emb_ids]
        d2 = drug_embeddings[drug_2_emb_ids]
        out = self.prediction_head(d1, d2, context_ids, drug_1_ids, drug_2_ids)
        return out

    def get_drug_ids(self, drug_1_ids, drug_2_ids, context_ids):
        if self.layer_name == "gat_pr_rel":
            # If we are using drug embeddings per cell line, we need to fetch the correct embeddings using the context ids
            drug_1_ids = [context_ids, drug_1_ids]
            drug_2_ids = [context_ids, drug_2_ids]
        return drug_1_ids, drug_2_ids

    def update_kwargs(self, layer, prediction_head, dataset):
        if layer in ["gc", "gac"]:
            self.enc_kwargs.update({"dataset": dataset})

        if prediction_head == "distmult":
            self.ph_kwargs.update({"rel_tot": self.graph.num_relation.item()})

        elif prediction_head == "mlp":
            self.ph_kwargs.update({"dataset": dataset})
