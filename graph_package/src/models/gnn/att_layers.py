import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_scatter import (
    scatter_mean,
    scatter_add,
    scatter_max,
    scatter_softmax,
)


from torchdrug import data, layers, utils
from graph_package.configs.directories import Directories
from torchdrug.layers import functional
from graph_package.src.models.gnn.base_layer import MessagePassingBase
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RelationalGraphAttentionConv(MessagePassingBase):
    """
    Graph attentional convolution operator from `Graph Attention Networks`_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        num_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    eps = 1e-10

    def __init__(
        self,
        input_dim,
        output_dim,
        num_relations,
        feature_dropout=False,
        w_per_relation=False,
        num_head=1,
        negative_slope=0.2,
        concat=True,
        batch_norm=False,
    ):
        super(RelationalGraphAttentionConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_head = num_head
        self.concat = concat
        self.num_relations = num_relations
        self.w_per_relation = w_per_relation
        # call relu bu with a slightly negative slop for negative values
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=negative_slope)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None

        self.activation = F.relu

        if output_dim % num_head != 0:
            raise ValueError(
                "Expect output_dim to be a multiplier of num_head, but found `%d` and `%d`"
                % (output_dim, num_head)
            )
        if w_per_relation:
            self.W_tau = nn.Parameter(
                torch.zeros(self.num_relations + 1, output_dim, input_dim)
            )
        else:
            self.W_tau = nn.Parameter(torch.zeros(1, output_dim, input_dim))
        # what they do in torch-geometric
        nn.init.xavier_uniform(self.W_tau)
        # the idea is that different heads are each allocated to a slice of the hidden vector
        self.query = nn.Parameter(
            torch.zeros(self.num_relations + 1, num_head, output_dim * 2 // num_head)
        )
        nn.init.kaiming_uniform_(self.query, negative_slope, mode="fan_in")
        self.feature_dropout = nn.Dropout(feature_dropout)

    def message(self, graph, input: torch.Tensor):
        # torch.arange(graph.num_node, device=graph.device) is added to make self-loop
        node_in = torch.cat(
            [graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)]
        )
        node_out = torch.cat(
            [graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)]
        )
        relation = torch.cat(
            [
                graph.edge_list[:, 2],
                (self.num_relations) * torch.ones(graph.num_node, device=graph.device),
            ]
        ).to(torch.int32)
        edge_weight = torch.cat(
            [graph.edge_weight, torch.ones(graph.num_node, device=graph.device)]
        )
        edge_weight = edge_weight.unsqueeze(-1)
        input_per_relation = input.expand(self.num_relations + 1, *input.shape)
        # tedious way to make [0,0,+ for num_cell_lines,...num_nodes,num_nodes,num_nodes for num_cell_lines]

        if self.w_per_relation:
            weight_index = (
                torch.arange(0, self.num_relations + 1)
                .expand(graph.num_node, self.num_relations + 1)
                .T.reshape(-1)
            )
        else:
            weight_index = torch.zeros(
                graph.num_node * (self.num_relations + 1), dtype=torch.int32
            )

        input_per_relation.reshape(-1, input.shape[-1], 1)

        hidden = torch.bmm(
            self.W_tau[weight_index], input_per_relation.reshape(-1, input.shape[-1], 1)
        )

        hidden = hidden.reshape(self.num_relations + 1, graph.num_node, -1)

        key = torch.stack(
            [hidden[relation, node_in], hidden[relation, node_out]], dim=-1
        )

        # shape is (n_triplets+n_nodes, num_heads, output_dim * 2 // num_head)
        key = key.view(key.shape[0], self.num_head, -1)

        # Calculate the dot product between the self.query tensor and the key tensor
        # using Einstein summation notation. The resulting tensor represents the
        # similarity between the query and key vectors for each sample and head.
        # numerator of 5.20
        weight = torch.einsum("nhd, nhd -> nh", self.query[relation], key)
        weight = self.leaky_relu(weight)

        # the maximum attention for each node, denominator in [Hamilton] eq 5.20, but uses max instead of sum
        # used to force the largest value to be 1 after taking exp
        max_attention_per_node = scatter_max(
            weight, node_out, dim=0, dim_size=graph.num_node
        )[0][node_out]

        # see [Hamilton] eq 5.20
        attention = (weight - max_attention_per_node).exp()
        attention = attention * edge_weight
        # Comment from source: why mean? because with mean we have normalized message scale across different node degrees

        normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[
            node_out
        ]

        attention = attention / (normalizer + self.eps)
        # see eq. 5.19 [Hamilton], this is 'h'

        value = hidden[relation, node_in].view(
            -1, self.num_head, self.query.shape[-1] // 2
        )
        # Copies a_{v,u} for each dimension of output
        attention = attention.unsqueeze(-1).expand_as(value)
        message = (attention * value).flatten(1)
        return message

    def aggregate(self, graph, message):
        # add self loop
        node_out = torch.cat(
            [graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)]
        )
        update = scatter_mean(message, node_out, dim=0, dim_size=graph.num_node)
        return update

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        if self.feature_dropout:
            output = self.feature_dropout(output)
        return output



class GraphAttentionConv(MessagePassingBase):
    """
    This is a PyTorch implementation of the GAT operator from the paper 'Graph Attention Networks?'_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): dimension of edge features
        dataset (str): name of dataset
        n_heads (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        dropout (float, optional): Whether to apply dropout
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        dataset,
        n_heads: int = 1,
        negative_slope: int = 0.2,
        dropout: int = 0.0,
        concat_hidden=True,
        batch_norm=False,
    ):
        super(GraphAttentionConv, self).__init__()
        self.input_dim = input_dim
        self.num_relation = num_relation
        self.dataset = dataset
        self.dropout = dropout
        self.concat_hidden = concat_hidden
        self.negative_slope = negative_slope
        self.n_heads = n_heads
        self.ccle = self._load_ccle()
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None

        self.n_hidden = output_dim

        dim = self.input_dim + self.ccle.shape[1]

        self.output_dim = self.n_hidden
        self.W = nn.Parameter(torch.empty(size=(output_dim, dim)))
        self.self_loop = nn.Parameter(torch.empty(size=(output_dim, input_dim)))
        nn.init.xavier_uniform_(self.W.data)
        self.attention = nn.Parameter(torch.zeros(n_heads, output_dim * 2 // n_heads))
        nn.init.xavier_uniform_(self.attention.data)
        self.activation = nn.ELU()

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
        by making a tensor of shape (num_relations, num_nodes, input_dim + ccle_dim)
        """
        ccle = self.ccle
        input_reshaped = input.unsqueeze(1).expand(-1, ccle.shape[0], -1)
        ccle_reshaped = ccle.unsqueeze(0).expand(input.shape[0], -1, -1)
        combined = torch.cat((input_reshaped, ccle_reshaped), dim=2)
        # Reshape the combined tensor to the desired shape
        combined = combined.reshape(
            ccle.shape[0], input.shape[0], input.shape[1] + ccle.shape[1]
        )
        return combined

    def message_and_aggregate(self, graph, input):
        # Extract duplets from the graph
        node_in, node_out, relation = graph.edge_list.t()

        # add cell line embeddings for each node
        combined = self.transform_input(input)

        # Initial linear transformation to obtain Wh [n_nodes, n_heads, n_hidden]
        loop_states = torch.mm(self.self_loop, input.T).T

        Wh_in = torch.cat(
            [torch.mm(self.W, combined[relation, node_in].T).T, loop_states], dim=0
        )
        Wh_out = torch.cat(
            [torch.mm(self.W, combined[relation, node_out].T).T, loop_states], dim=0
        )

        # add self_loop
        node_in = torch.cat(
            [node_in, torch.arange(graph.num_node, device=graph.device)]
        )
        node_out = torch.cat(
            [node_out, torch.arange(graph.num_node, device=graph.device)]
        )

        # Prepare the input for the attention mechanism
        Wh_concat = torch.concat([Wh_in, Wh_out], dim=-1).reshape(
            -1, self.n_heads, 2 * self.n_hidden // self.n_heads
        )

        # Calculate the attention score e_ij with shape [n_nodes, n_nodes, n_heads]
        e = torch.einsum("nhj, hj -> nh", Wh_concat, self.attention)

        a = F.leaky_relu(e, negative_slope=self.negative_slope)

        a = scatter_softmax(a, node_out, dim=0)

        a = F.dropout(a, self.dropout, training=self.training)

        value = Wh_in[node_in].view(-1, self.n_heads, self.attention.shape[-1] // 2)

        a = a.unsqueeze(-1).expand_as(value)

        message = a * value

        h_prime = scatter_mean(message, node_out, dim=0)

        # Calculate output for each attention head following eq. 4 [Velickovic] (without non-linearity)

        # Whether to mean across the multiple attention heads or not
        if self.concat_hidden:
            return h_prime.flatten(1)
        else:
            return h_prime.mean(dim=1)

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        output = self.activation(output)
        return output

class GraphAttentionLayer(MessagePassingBase):
    """
    This is a PyTorch implementation of the GAT operator from the paper 'Graph Attention Networks?'_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): dimension of edge features
        dataset (str): name of dataset
        n_heads (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        dropout (float, optional): Whether to apply dropout
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        dataset,
        n_heads: int = 4,
        negative_slope: int = 0.2,
        dropout: int = 0.6,
        concat_hidden=True,
        batch_norm=False,
    ):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.num_relation = num_relation
        self.dataset = dataset
        self.dropout = dropout
        self.concat_hidden = concat_hidden
        self.negative_slope = negative_slope
        self.n_heads = n_heads

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None

        if self.concat_hidden:
            assert output_dim % n_heads == 0
            self.n_hidden = output_dim // n_heads
        else:
            self.n_hidden = output_dim

        self.output_dim = self.n_hidden * n_heads
        self.W = nn.Parameter(
            torch.empty(size=(self.input_dim, self.n_hidden * n_heads))
        )
        nn.init.xavier_uniform_(self.W.data)
        self.attention = nn.Parameter(torch.empty(size=(2 * self.n_hidden, 1)))
        nn.init.xavier_uniform_(self.attention.data)
        self.activation = nn.ELU()

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        Prepares the input for the attention mechanism.
        """
        n_nodes = Wh.shape[0]
        # Duplicate the transformed input to allow for pairwise interactions
        Wh_repeat = Wh.repeat(n_nodes, 1, 1)
        # Repeat and interleave the transformed input for pairwise interactions
        Wh_repeat_interleave = Wh.repeat_interleave(n_nodes, dim=0)
        # Concatenate the repeated and interleaved inputs along the last dimension
        Wh_concat = torch.cat([Wh_repeat_interleave, Wh_repeat], dim=-1)
        # Reshape the concatenated tensor to include the number of heads and hidden dimensions
        Wh_concat = Wh_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        return Wh_concat

    def message_and_aggregate(self, graph, input):
        # Extract duplets from the graph
        node_in, node_out, _ = graph.edge_list.t()
        n_nodes = input.shape[0]

        # Create an uni-relational adjacency matrix
        adjacency = torch.zeros(
            (graph.num_node, graph.num_node, 1),
            dtype=torch.float32,
            device=graph.device,
        )
        adjacency[node_in, node_out] = 1

        # Add self-loops
        adjacency += torch.eye(
            graph.num_node, graph.num_node, device=graph.device
        ).unsqueeze(-1)

        # Initial linear transformation to obtain Wh [n_nodes, n_heads, n_hidden]
        Wh = torch.mm(input, self.W).view(n_nodes, self.n_heads, self.n_hidden)

        # Prepare the input for the attention mechanism
        Wh_concat = self._prepare_attentional_mechanism_input(Wh)

        # Calculate the attention score e_ij with shape [n_nodes, n_nodes, n_heads]
        e = F.leaky_relu(
            torch.matmul(Wh_concat, self.attention), negative_slope=self.negative_slope
        ).squeeze(-1)

        # Mask based on adjacency matrix
        e = e.masked_fill(adjacency == 0, float("-inf"))

        # Normalizer following eq. 3 [Velickovic]
        a = F.softmax(e, dim=1)
        a = F.dropout(a, self.dropout, training=self.training)

        # Calculate output for each attention head following eq. 4 [Velickovic] (without non-linearity)
        h_prime = torch.einsum("ijh,jhf->ihf", a, Wh)

        # Whether to mean across the multiple attention heads or not
        if self.concat_hidden:
            return h_prime.reshape(input.shape[0], self.n_heads * self.n_hidden)
        else:
            return h_prime.mean(dim=1)

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        output = self.activation(output)
        return output


class RelationalGraphAttentionLayer(MessagePassingBase):
    """
    This is a PyTorch implementation of the Relational GAT operator from the paper 'Graph Attention Networks?'_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): dimension of edge features
        dataset (str): name of dataset
        n_heads (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        dropout (float, optional): Whether to apply dropout
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        dataset,
        n_heads: int = 1,
        negative_slope: int = 0.2,
        dropout: int = 0.6,
        concat_hidden=True,
        batch_norm=False,
    ):
        super(RelationalGraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.num_relation = num_relation
        self.dataset = dataset
        self.dropout = dropout
        self.concat_hidden = concat_hidden
        self.negative_slope = negative_slope
        self.n_heads = n_heads

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None

        if self.concat_hidden:
            assert output_dim % n_heads == 0
            self.n_hidden = output_dim // n_heads
        else:
            self.n_hidden = output_dim

        self.output_dim = output_dim
        # Weight matrix for linear transformation
        self.W = nn.Linear(self.input_dim,  self.n_hidden * n_heads * (self.num_relation+1))
        # Attention mechanism
        self.attention = nn.Parameter(torch.empty(size=((self.num_relation+1), n_heads , output_dim * 2 // n_heads)))
        nn.init.xavier_uniform_(self.attention.data)
        self.activation = nn.ELU() if self.concat_hidden else None

    def _add_relation_specific_self_loops(self, n_nodes, device):
        # Creating self-loop edges for each node and each relation
        self_loop_edges = (
            torch.stack(
                [
                    torch.arange(n_nodes, device=device)
                    for _ in range(self.num_relation)
                ],
                dim=1,
            )
            .view(-1, 1)
            .repeat(1, 2)
        )
        relation_indices = torch.repeat_interleave(
            torch.arange(self.num_relation, device=device), n_nodes
        ).view(-1, 1)
        self_loop_edge_list = torch.cat([self_loop_edges, relation_indices], dim=1)
        return self_loop_edge_list

    def message_and_aggregate(self, graph, input):
        # Extract triplets from the graph and create self-loop edges
        n_nodes = input.shape[0]
        self_loop_edges = torch.stack([torch.arange(n_nodes, device=graph.device)], dim=1).view(-1, 1).repeat(1, 2)
        self_loop_edge_list = torch.cat([self_loop_edges, torch.ones(n_nodes,1,dtype=torch.int,device=graph.device)*self.num_relation], dim=1)
        edge_list = torch.cat([graph.edge_list, self_loop_edge_list], dim=0)
        
        node_in, node_out, relation = edge_list.t()

        # Handling edge weights, assigning small positive weights to self-edges
        self_loop_weights = torch.ones(n_nodes, device=graph.device)
        edge_weight = torch.cat([graph.edge_weight, self_loop_weights])

        # Linear transformation with reshaping for relations
        Wh = self.W(input).view(-1, self.n_heads, self.n_hidden, self.num_relation+1)

        # Calculate attention coefficients for all attention heads and relations
        Wh_i = Wh[node_in, :, :, relation]  # Source node features
        Wh_j = Wh[node_out, :, :, relation]  # Target node features

        Wh_concat = torch.cat([Wh_i, Wh_j], dim=2) # Concatenate source and target features

        # Calculate attention coefficients e_ij for each node pair following eq. 3 [Velickovic] with leaky rely
        e = torch.einsum("nhd, nhd -> nh", self.attention[relation], Wh_concat)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # Multiply synergy scores to the attention coefficients including self-loop
        e *= edge_weight.unsqueeze(1)

        # Normalize attention coefficients and apply dropout
        a = scatter_softmax(e, node_out, dim=0)
        a = F.dropout(a, p=self.dropout, training=self.training)

        # Expand 'a' to have the same number of dimensions as 'Wh_j'
        a = a.unsqueeze(-1).expand_as(Wh_j)

        # Aggregate the attention-weighted node features for each source node and compute final output features
        # following eq. 4 [Velickovic]
        h_prime = scatter_add(a * Wh_j, node_in, dim=0, dim_size=input.size(0))

        if self.concat_hidden:
            # eq. 5 [Velickovic]
            return h_prime.view(input.shape[0], self.n_heads * self.n_hidden)
        else:
            return h_prime.mean(dim=1)

    def message_and_aggregate_old(self, graph, input):
        # Extract triplets from the graph and create self-loop edges
        n_nodes = input.shape[0]
        self_loop_edge_list = self._add_relation_specific_self_loops(n_nodes, graph.device)
        edge_list = torch.cat([graph.edge_list, self_loop_edge_list], dim=0)
        node_in, node_out, relation = edge_list.t()

        # Handling edge weights, assigning small positive weights to self-loops
        num_self_loops = n_nodes * self.num_relation
        self_loop_weights = torch.ones(num_self_loops, device=graph.device) * 0.1
        edge_weight = torch.cat([graph.edge_weight, self_loop_weights])

        # Linear transformation
        Wh = self.W(input)
        Wh = Wh.view(-1, self.n_heads, self.n_hidden)

        # Initialize output
        h_prime = torch.zeros(
            (input.size(0), self.n_heads, self.n_hidden, self.num_relation),
            device=graph.device,
        )

        # Apply attention mechanism for each relation and each head
        for rel in range(self.num_relation):
            rel_mask = relation == rel
            edges = torch.stack([node_in[rel_mask], node_out[rel_mask]])
            weights = edge_weight[rel_mask].unsqueeze(1)

            for head in range(self.n_heads):
                Wh_i = Wh[edges[0], head, :]  # Features of source nodes
                Wh_j = Wh[edges[1], head, :]  # Features of target nodes

                # Calculate attention coefficients e_ij for each node pair following eq. 3 [Velickovic] with leaky rely
                Wh_concat = torch.cat([Wh_i, Wh_j], dim=-1)
                e = self.attentions[rel][head](Wh_concat)
                e = F.leaky_relu(e, negative_slope=self.negative_slope)

                # Multiply synergy scores to the attention coefficients including self-loop
                e *= weights

                # Now normalize following eq. 3 [Velickovic]
                a = scatter_softmax(e, edges[1], dim=0)
                a = F.dropout(a, p=self.dropout, training=self.training)

                # Compute final output features for every node for the attention head following eq. 4 [Velickovic]
                h_prime[:, head, :, rel] = scatter_add(
                    a * Wh_j, edges[1], dim=0, dim_size=input.size(0)
                )

        # Mean across relations
        h_prime = h_prime.mean(dim=-1)

        # Whether to mean across the multiple attention heads or not
        if self.concat_hidden:
            # eq. 5 [Velickovic]
            return h_prime.reshape(input.shape[0], self.n_heads * self.n_hidden)
        else:
            return h_prime.mean(dim=1)

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

class GraphAttentionLayerPerCellLine(RelationalGraphAttentionConv):
    def __init__(self, *args, **kwargs):
        super(GraphAttentionLayerPerCellLine, self).__init__(*args, **kwargs)

    def message(self, graph, input: torch.Tensor):
        # torch.arange(graph.num_node, device=graph.device) is added to make self-loop
        node_in = torch.cat(
            [graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)]
        ).to(torch.int64)
        node_out = torch.cat(
            [graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)]
        ).to(torch.int64)
        relation = torch.cat(
            [
                graph.edge_list[:, 2],
                (self.num_relations) * torch.ones(graph.num_node, device=graph.device),
            ]
        ).to(torch.int64)
        edge_weight = torch.cat(
            [graph.edge_weight, torch.ones(graph.num_node, device=graph.device)]
        )
        edge_weight = edge_weight.unsqueeze(-1)
        input_per_relation = input.expand(self.num_relations + 1, *input.shape)
        # tedious way to make [0,0,+ for num_cell_lines,...num_nodes,num_nodes,num_nodes for num_cell_lines]

        weight_index = torch.zeros(
            graph.num_node * (self.num_relations + 1), dtype=torch.int64
        )
        input_per_relation.reshape(-1, input.shape[-1], 1)

        hidden = torch.bmm(
            self.W_tau[weight_index], input_per_relation.reshape(-1, input.shape[-1], 1)
        )

        hidden = hidden.reshape(self.num_relations + 1, graph.num_node, -1)

        key = torch.stack(
            [hidden[relation, node_in], hidden[relation, node_out]], dim=-1
        )

        # shape is (n_triplets+n_nodes, num_heads, output_dim * 2 // num_head)
        key = key.view(key.shape[0], self.num_head, -1)

        # Calculate the dot product between the self.query tensor and the key tensor
        # using Einstein summation notation. The resulting tensor represents the
        # similarity between the query and key vectors for each sample and head.
        # numerator of 5.20
        weight = torch.einsum("nhd, nhd -> nh", self.query[relation], key)
        weight = self.leaky_relu(weight)

        # the maximum attention for each node, denominator in [Hamilton] eq 5.20, but uses max instead of sum
        # used to force the largest value to be 1 after taking exp
        node_out_rel = node_out * self.num_relations + relation
        n_rel_nodes = graph.num_node * (self.num_relations + 1)
        # now find the max attention for each node_out_rel using the node_out_rel as index and expand the max attention for each relation
        max_attention_per_node = scatter_max(
            weight, node_out_rel, dim=0, dim_size=n_rel_nodes
        )[0][node_out_rel]

        # see [Hamilton] eq 5.20
        attention = (weight - max_attention_per_node).exp()
        attention = attention * edge_weight
        # Comment from source: why mean? because with mean we have normalized message scale across different node degrees

        normalizer = scatter_mean(attention, node_out_rel, dim=0, dim_size=n_rel_nodes)[
            node_out_rel
        ]

        attention = attention / (normalizer + self.eps)
        # see eq. 5.19 [Hamilton], this is 'h'

        value = hidden[relation, node_in].view(
            -1, self.num_head, self.query.shape[-1] // 2
        )
        # Copies a_{v,u} for each dimension of output
        attention = attention.unsqueeze(-1).expand_as(value)
        message = (attention * value).flatten(1)
        return message

    def aggregate(self, graph, message):
        # add self loop
        node_out = torch.cat(
            [graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)]
        )
        relation = torch.cat(
            [
                graph.edge_list[:, 2],
                (self.num_relations) * torch.ones(graph.num_node, device=graph.device),
            ]
        ).to(torch.int64)
        node_out_rel = node_out * self.num_relations + relation
        n_rel_nodes = graph.num_node * (self.num_relations + 1)
        # create an update for each cell line specific drug embedding ((num_relations+1), num_nodes),1)
        update = scatter_mean(
            message, node_out_rel, dim=0, dim_size=n_rel_nodes
        ).reshape((self.num_relations + 1), graph.num_node, -1)
        return update

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
