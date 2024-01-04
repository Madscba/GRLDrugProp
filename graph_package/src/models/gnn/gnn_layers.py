import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_sum, scatter_add, scatter_max, scatter_softmax

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

    def __init__(self, input_dim, output_dim, num_relation, batch_norm=False):
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
    def __init__(self, input_dim, output_dim, num_relation, dataset, batch_norm=False):
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
            batch_norm=False
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
        self.W = nn.Parameter(torch.empty(size=(self.input_dim,  self.n_hidden * n_heads)))
        nn.init.xavier_uniform_(self.W.data)
        self.attention = nn.Parameter(torch.empty(size=(2*self.n_hidden, 1)))
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
        adjacency = torch.zeros((graph.num_node, graph.num_node, 1), dtype=torch.float32, device=graph.device)
        adjacency[node_in, node_out] = 1

        # Add self-loops
        adjacency += torch.eye(graph.num_node, graph.num_node, device=graph.device).unsqueeze(-1)

        # Initial linear transformation to obtain Wh [n_nodes, n_heads, n_hidden]
        Wh = torch.mm(input, self.W).view(n_nodes, self.n_heads, self.n_hidden)

        # Prepare the input for the attention mechanism
        Wh_concat = self._prepare_attentional_mechanism_input(Wh)

        # Calculate the attention score e_ij with shape [n_nodes, n_nodes, n_heads]
        e = F.leaky_relu(torch.matmul(Wh_concat,self.attention),negative_slope=self.negative_slope).squeeze(-1)

        # Mask based on adjacency matrix
        e = e.masked_fill(adjacency == 0, float('-inf'))

        # Normalizer following eq. 3 [Velickovic]
        a = F.softmax(e, dim=1)
        a = F.dropout(a, self.dropout, training=self.training)

        # Calculate output for each attention head following eq. 4 [Velickovic] (without non-linearity)
        h_prime = torch.einsum('ijh,jhf->ihf', a, Wh)

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
            batch_norm=False
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
        self.W = nn.Linear(self.input_dim,  self.n_hidden * n_heads)

        # Define parameters for each relation type and each head
        self.attentions = nn.ModuleList()
        for _ in range(num_relation):
            head_attentions = nn.ModuleList([nn.Linear(2 * self.n_hidden, 1) for _ in range(n_heads)])
            self.attentions.append(head_attentions)

        self.activation = nn.ELU() if self.concat_hidden else None

    def _add_relation_specific_self_loops(self, n_nodes, device):
        # Creating self-loop edges for each node and each relation
        self_loop_edges = torch.stack([torch.arange(n_nodes, device=device) for _ in range(self.num_relation)], dim=1).view(-1, 1).repeat(1, 2)
        relation_indices = torch.repeat_interleave(torch.arange(self.num_relation, device=device), n_nodes).view(-1, 1)
        self_loop_edge_list = torch.cat([self_loop_edges, relation_indices], dim=1)
        return self_loop_edge_list


    def message_and_aggregate(self, graph, input):
        # Extract triplets from the graph and create self-loop edges
        n_nodes = input.shape[0]
        self_loop_edge_list = self._add_relation_specific_self_loops(n_nodes, graph.device)
        edge_list = torch.cat([graph.edge_list, self_loop_edge_list], dim=0)
        node_in, node_out, relation = edge_list.t()

        # Handling edge weights, assigning small positive weights to self-loops
        num_self_loops = n_nodes * self.num_relation
        self_loop_weights = torch.ones(num_self_loops, device=graph.device)*0.1
        edge_weight = torch.cat([graph.edge_weight, self_loop_weights])

        # Linear transformation
        Wh = self.W(input)
        Wh = Wh.view(-1, self.n_heads, self.n_hidden)

        # Initialize output
        h_prime = torch.zeros((input.size(0), self.n_heads, self.n_hidden, self.num_relation), device=graph.device)

        # Apply attention mechanism for each relation and each head
        for rel in range(self.num_relation):
            rel_mask = relation == rel
            edges = torch.stack([node_in[rel_mask], node_out[rel_mask]])
            weights = edge_weight[rel_mask].unsqueeze(1) 

            for head in range(self.n_heads):

                Wh_i = Wh[edges[0], head, :]   # Features of source nodes
                Wh_j = Wh[edges[1], head, :]   # Features of target nodes

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
                h_prime[:, head, :, rel] = scatter_add(a * Wh_j, edges[1], dim=0, dim_size=input.size(0))
        
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