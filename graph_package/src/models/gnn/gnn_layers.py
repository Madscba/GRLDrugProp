import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_add, scatter_max

from torchdrug import data, layers, utils
from graph_package.configs.directories import Directories
from torchdrug.layers import functional
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MessagePassingBase(nn.Module):
    """
    Base module for message passing.

    Any custom message passing module should be derived from this class.
    """

    gradient_checkpoint = False

    def message(self, graph, input):
        """
        Compute edge messages for the graph.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: edge messages of shape :math:`(|E|, ...)`
        """
        raise NotImplementedError

    def aggregate(self, graph, message):
        """
        Aggregate edge messages to nodes.

        Parameters:
            graph (Graph): graph(s)
            message (Tensor): edge messages of shape :math:`(|E|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def message_and_aggregate(self, graph, input):
        """
        Fused computation of message and aggregation over the graph.
        This may provide better time or memory complexity than separate calls of
        :meth:`message <MessagePassingBase.message>` and :meth:`aggregate <MessagePassingBase.aggregate>`.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        message = self.message(graph, input)
        update = self.aggregate(graph, message)
        return update

    def _message_and_aggregate(self, *tensors):
        graph = data.Graph.from_tensors(tensors[:-1])
        input = tensors[-1]
        update = self.message_and_aggregate(graph, input)
        return update

    def combine(self, input, update):
        """
        Combine node input and node update.

        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`
            update (Tensor): node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def forward(self, graph, input):
        """
        Perform message passing over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(
                self._message_and_aggregate, *graph.to_tensors(), input
            )
        else:
            update = self.message_and_aggregate(graph, input)
        output = self.combine(input, update)
        return output


class RelationalGraphConv(MessagePassingBase):
    """
    Relational graph convolution operator from `Modeling Relational Data with Graph Convolutional Networks`_.

    .. _Modeling Relational Data with Graph Convolutional Networks:
        https://arxiv.org/pdf/1703.06103.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    eps = 1e-10

    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        batch_norm=False,
        feature_dropout: float = 0.0,
    ):
        super(RelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None

        self.activation = F.relu
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        self.feature_dropout = nn.Dropout(feature_dropout)

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation
        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(
            message * edge_weight,
            node_out,
            dim=0,
            dim_size=graph.num_node * self.num_relation,
        ) / (
            scatter_add(
                edge_weight,
                node_out,
                dim=0,
                dim_size=graph.num_node * self.num_relation,
            )
            + self.eps
        )
        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation
        node_in, node_out, relation = graph.edge_list.t()
        # make an index of size num_edges*num_relations
        node_out = node_out * self.num_relation + relation
        # sum the edge weights for each node_out
        degree_out = scatter_add(
            graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation
        )
        # normalize weights for each node_out by dividing by the sum of the weights
        edge_weight = graph.edge_weight / (10e-10 + degree_out[node_out])

        # make sparse adjacency matrix of size (graph.num_node, graph.num_node * graph.num_relation)
        adjacency = utils.sparse_coo_tensor(
            torch.stack([node_in, node_out]),
            edge_weight,
            (graph.num_node, graph.num_node * graph.num_relation),
        )
        # multiply the adjacency matrix with the input tensor, corresponds to the aggregation step
        update = torch.sparse.mm(adjacency.t(), input)
        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def combine(self, input, update):
        # combine the input tensor with the update tensor, corresponds to the update step
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        output = self.activation(output)
        if self.feature_dropout:
            output = self.feature_dropout(output)
        return output


class DummyLayer(MessagePassingBase):
    def __init__(self, input_dim, output_dim, num_relation, dataset, batch_norm=False):
        "For testing MLP predictionhead without any GNN aggregation"
        super(DummyLayer, self).__init__()

    def message_and_aggregate(self, graph, input):
        return input

    def combine(self, input, update):
        return input


class GraphConv(MessagePassingBase):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        dataset,
        batch_norm=False,
        feature_dropout=0.0,
    ):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.dataset = dataset

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        self.ccle = self._load_ccle()
        cell_feature_size = self.ccle.shape[1]
        self.activation = F.relu
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(input_dim + cell_feature_size, output_dim)
        self.feature_dropout = nn.Dropout(feature_dropout)

    def _load_ccle(self):
        feature_path = (
            Directories.DATA_PATH
            / "features"
            / "cell_line_features"
            / "CCLE_954_gene_express_pca.json"
        )
        with open(feature_path) as f:
            all_edge_features = json.load(f)
        vocab_path = (
            Directories.DATA_PATH / "gold" / self.dataset / "relation_vocab.json"
        )
        with open(vocab_path) as f:
            relation_vocab = json.load(f)
        vocab_reverse = {v: k for k, v in relation_vocab.items()}
        ids = sorted(list(vocab_reverse.keys()))
        ccle = torch.tensor(
            [all_edge_features[vocab_reverse[id]] for id in ids], device=device
        )
        return ccle

    def transform_input(self, input: torch.Tensor):
        """
        Combine the input tensor with the CCLE tensor,
        by making a tensor of shape (num_relations*num_nodes, input_dim + ccle_dim)
        """
        ccle = self.ccle
        input_reshaped = input.unsqueeze(1).expand(-1, self.ccle.shape[0], -1)
        ccle_reshaped = ccle.unsqueeze(0).expand(input.shape[0], -1, -1)
        combined = torch.cat((input_reshaped, ccle_reshaped), dim=2)

        # Reshape the combined tensor to the desired shape
        combined = combined.reshape(-1, input.shape[1] + ccle.shape[1])
        return combined

    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation
        node_in, node_out, relation = graph.edge_list.t()
        node_out = node_out * self.num_relation + relation
        degree_out = scatter_add(
            graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation
        )
        # add small value to demoninator to avoid division by zero when using discrete edge weights
        edge_weight = graph.edge_weight / (10e-10 + degree_out[node_out])

        adjacency = utils.sparse_coo_tensor(
            torch.stack([node_in, node_out]),
            edge_weight,
            (graph.num_node, graph.num_node * graph.num_relation),
        )
        transform_input = self.transform_input(input)
        update = torch.sparse.mm(adjacency, transform_input)
        return update.view(graph.num_node, self.input_dim + self.ccle.shape[1])

    def combine(self, input, update):
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        output = self.activation(output)
        if self.feature_dropout:
            output = self.feature_dropout(output)

        return output
