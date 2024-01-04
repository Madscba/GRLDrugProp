import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_add, scatter_mean, scatter_max

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
        
        # Normalising edge weights does not make sense in our setting
        # sum the edge weights for each node_out 
        # degree_out = scatter_add(
        #     graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation
        # )
        # normalize weights for each node_out by dividing by the sum of the weights
        #edge_weight = graph.edge_weight / (10e-10 + degree_out[node_out])
        
        # make sparse adjacency matrix of size (graph.num_node, graph.num_node * graph.num_relation)
        edge_weight = graph.edge_weight
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

        # Normalising edge weights does not make sense in our setting  
        # degree_out = scatter_add(
        #     graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation
        # )
        # # add small value to demoninator to avoid division by zero when using discrete edge weights
        # edge_weight = graph.edge_weight / (10e-10 + degree_out[node_out])
        
        edge_weight = graph.edge_weight
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

    def __init__(self, input_dim, output_dim, num_relations, edge_input_dim=None, num_head=1, negative_slope=0.2, concat=True,
                 batch_norm=False):
        super(RelationalGraphAttentionConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_head = num_head
        self.concat = concat
        self.num_relations = num_relations
        # call relu bu with a slightly negative slop for negative values
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=negative_slope)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None

        self.activation = F.relu

        if output_dim % num_head != 0:
            raise ValueError("Expect output_dim to be a multiplier of num_head, but found `%d` and `%d`"
                             % (output_dim, num_head))

        self.W_tau = nn.Parameter(torch.zeros(self.num_relations+1, output_dim,input_dim))
        # what they do in torch-geometric 
        nn.init.xavier_uniform(self.W_tau)
        # the idea is that different heads are each allocated to a slice of the hidden vector 
        self.query = nn.Parameter(torch.zeros(self.num_relations+1, num_head, output_dim * 2 // num_head))
        nn.init.kaiming_uniform_(self.query, negative_slope, mode="fan_in")

    def message(self, graph, input: torch.Tensor):
        # torch.arange(graph.num_node, device=graph.device) is added to make self-loop
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        relation =  torch.cat([graph.edge_list[:, 2], (self.num_relations)*torch.ones(graph.num_node, device=graph.device)]).to(torch.int32)
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        input_per_relation = input.expand(self.num_relations+1,*input.shape)
        # tedious way to make [1,1,1 for num_cell_lines,...num_nodes,num_nodes,num_nodes for num_cell_lines]
        weight_index = torch.arange(0,self.num_relations+1).expand(graph.num_node,self.num_relations+1).T.reshape(-1)

        hidden = torch.bmm(self.W_tau[weight_index],input_per_relation.reshape(-1,input.shape[-1],1))
        hidden = hidden.reshape(self.num_relations+1,graph.num_node,-1)

        key = torch.stack([hidden[relation,node_in], hidden[relation,node_out]], dim=-1)

        # shape is (n_triplets+n_nodes, num_heads, output_dim * 2 // num_head)
        key = key.view(key.shape[0], self.num_head, -1)
        
        # Calculate the dot product between the self.query tensor and the key tensor
        # using Einstein summation notation. The resulting tensor represents the
        # similarity between the query and key vectors for each sample and head.
        weight = torch.einsum("nhd, nhd -> nh", self.query[relation], key)
        weight = self.leaky_relu(weight)

        # the maximum attention for each node, denominator in [Hamilton] eq 5.20, but uses max instead of sum 
        # used to force the largest value to be 1 after taking exp
        max_attention_per_node=scatter_max(weight, node_out, dim=0, dim_size=graph.num_node)[0][node_out]

        # see [Hamilton] eq 5.20
        attention = (weight - max_attention_per_node).exp()
        attention = attention * edge_weight
        # Comment from source: why mean? because with mean we have normalized message scale across different node degrees 

        normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[node_out]

        attention = attention / (normalizer + self.eps)
        # see eq. 5.19 [Hamilton], this is 'h'

        value = hidden[relation,node_in].view(-1, self.num_head, self.query.shape[-1] // 2)
        # Copies a_{v,u} for each dimension of output
        attention = attention.unsqueeze(-1).expand_as(value)
        message = (attention * value).flatten(1)
        return message

    def aggregate(self, graph, message):
        # add self loop
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        update = scatter_mean(message, node_out, dim=0, dim_size=graph.num_node)
        return update

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output



class GraphAttentionConv(MessagePassingBase):
    """
    ORIGINAL TORCHDRUG IMPLEMENTATION WITH COMMENTS. THIS IS JUST FOR REFERENCE.

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

    def __init__(self, input_dim, output_dim, edge_input_dim=None, num_head=1, negative_slope=0.2, concat=True,
                 batch_norm=False):
        super(GraphAttentionConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.num_head = num_head
        self.concat = concat
        # call relu bu with a slightly negative slop for negative values
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=negative_slope)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None

        self.activation = F.relu

        if output_dim % num_head != 0:
            raise ValueError("Expect output_dim to be a multiplier of num_head, but found `%d` and `%d`"
                             % (output_dim, num_head))

        self.linear = nn.Linear(input_dim, output_dim)
        # the idea is that different heads are each allocated to a slice of the hidden vector 
        self.query = nn.Parameter(torch.zeros(num_head, output_dim * 2 // num_head))
        nn.init.kaiming_uniform_(self.query, negative_slope, mode="fan_in")

    def message(self, graph, input):
        # torch.arange(graph.num_node, device=graph.device) is added to make self-loop
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        hidden = self.linear(input)

        key = torch.stack([hidden[node_in], hidden[node_out]], dim=-1)

        # shape is (n_triplets+n_nodes, num_heads, output_dim * 2 // num_head)
        key = key.view(-1, *self.query.shape)
        
        # Calculate the dot product between the self.query tensor and the key tensor
        # using Einstein summation notation. The resulting tensor represents the
        # similarity between the query for each cell line and key vectors for each sample and head.
        weight = torch.einsum("nhd, nhd -> nh", self.query, key)
        weight = self.leaky_relu(weight)

        # the maximum attention for each node, denominator in [Hamilton] eq 5.20, but uses max instead of sum 
        # used to force the largest value to be 1 after taking exp
        max_attention_per_node=scatter_max(weight, node_out, dim=0, dim_size=graph.num_node)[0][node_out]

        # see [Hamilton] eq 5.20
        attention = (weight - max_attention_per_node).exp()
        attention = attention * edge_weight
        # Comment from source: why mean? because with mean we have normalized message scale across different node degrees
        # I don't know if we want some cell-line specific normalizartion? 
        normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[node_out]

        attention = attention / (normalizer + self.eps)
        # see eq. 5.19 [Hamilton], this is 'h'
        value = hidden[node_in].view(-1, self.num_head, self.query.shape[-1] // 2)
        # Copies a_{v,u} for each dimension of output
        attention = attention.unsqueeze(-1).expand_as(value)
        message = (attention * value).flatten(1)
        return message

    def aggregate(self, graph, message):
        # add self loop
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        update = scatter_mean(message, node_out, dim=0, dim_size=graph.num_node)
        return update

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


