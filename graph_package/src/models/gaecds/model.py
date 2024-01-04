import torch
import torch.nn as nn
from torchdrug.models import GraphConvolutionalNetwork
import torch.nn.functional as F
from torchdrug.data import Graph
from torchdrug import data
from graph_package.configs.directories import Directories
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model
class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(GCN, self).__init__()
        self.drug_conv = GraphConvolutionalNetwork(
            input_dim=in_feats, hidden_dims=[hid_feats, out_feats], activation="relu"
        )
        self.drug_conv.to(device)

    def forward(self, g_adj, feature):
        h = self.drug_conv(g_adj, feature)
        h = h["node_feature"]
        return h


class CNN(nn.Module):
    def __init__(self, in_feats, hid1_feats, hid2_feats, out_feats):
        super().__init__()
        self.hid2_feats = hid2_feats
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_feats,
                out_channels=hid1_feats,
                kernel_size=(2, 2),
                padding="same",
            ),
            nn.BatchNorm2d(hid1_feats),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=hid1_feats,
                out_channels=hid2_feats,
                kernel_size=(2, 2),
                padding="same",
            ),
            nn.BatchNorm2d(hid2_feats),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(hid2_feats, out_feats),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class InnerProductDecoder(nn.Module):
    def forward(self, inputs):
        x = inputs.T
        x = torch.mm(inputs, x)
        x = torch.reshape(x, [-1])
        outputs = torch.sigmoid(x)
        return outputs


class GAE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.encoder = GCN(in_features, hidden_features, out_features)
        self.decoder = InnerProductDecoder()

    def forward(self, adj, feature):
        h = self.encoder(adj, feature)
        h_noise = self.decoder(h)

        return h_noise, h


class MLP(nn.Module):
    def __init__(self, in_feats, out_feats, hid1_feats=512, hid2_feats=256):
        super().__init__()
        self.mlp1 = nn.Linear(in_feats, hid1_feats)
        self.mlp2 = nn.Linear(hid1_feats, hid2_feats)
        self.mlp3 = nn.Linear(hid2_feats, out_feats)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.in_feats = in_feats

    def forward(self, x, method):
        x = x.view(-1, self.in_feats)
        out = self.mlp1(x)
        out = self.dropout(out)
        out = self.sigmoid(out)

        out = self.mlp2(out)
        out = self.dropout(out)
        out = self.sigmoid(out)
        out = self.mlp3(out)
        if method == "relu":
            out = self.relu(out)
        elif method == "sigmoid":
            out = self.sigmoid(out)

        return out


class GAECDS(nn.Module):
    def __init__(
        self,
        graph: Graph,
        hidden_dims: list,
        dataset: str,
        concat_hidden: bool = False,
        short_cut: bool = False,
        enc_kwargs: dict = {},
        ph_kwargs: dict = {},
    ):
        super().__init__()
        self.graph = graph

        self.gae = GAE(300, 256, 128)
        self.cnn = CNN(128, 64, 32, 1)
        self.mlp_cell = MLP(in_feats=128, out_feats=128)
        self.dataset = dataset
        self.ccle = self._load_ccle()

        _, gcn_feature = self.gae(self.graph, self.graph.node_feature)
        self.gcn_feature = gcn_feature

        self.optim_gae = torch.optim.Adam(
            [{"params": self.gae.parameters(), "lr": 0.00001}]
        )

    def forward(self, inputs):
        drug_1_ids, drug_2_ids, context_ids = map(
            lambda x: x.squeeze(), inputs.split(1, dim=1)
        )

        # c = self.cell_line_mlp(self.ccle[context_ids])
        cell_feature = self.ccle
        cell_out = self.mlp_cell(cell_feature, "sigmoid")
        cell_out = cell_out.cpu().detach().numpy()
        x_matrix = self.gcn_feature
        x_matrix = x_matrix.cpu().detach().numpy()
        # todo debug code below and extract relevant node_drugs
        # todo setup function new_matrix_cell from gaecds model
        drug1 = np.array(node_drug["g_id1"])
        drug2 = np.array(node_drug["g_id2"])
        x_new_matrix = new_matrix_with_cell(drug1, drug2, cell_out, x_matrix)

        pred_cnn = self.cnn(batch_x)
        pred_new = pred_cnn
        batch_y_new = batch_y
        pred_cnn = pred_cnn.to(torch.float32)
        batch_y = batch_y.to(torch.float32)
        loss_net = torch.mean(F.binary_cross_entropy(pred_cnn, batch_y))

        acc_train = acc(pred_new, batch_y_new)

        loss_net.requires_grad_(True)
        opt_cnn.zero_grad()
        loss_net.backward()
        opt_cnn.step()

    def train_gae(self):
        """
        Train the graph autoencoder model.
        """
        node_feature = self.graph.node_feature
        tensors = self.graph
        graph = data.Graph.from_tensors(tensors[:-1])
        adj = graph.adjacency
        norm = (
            adj.shape[0]
            * adj.shape[0]
            / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        )
        adj = torch.tensor(adj)
        labels = torch.reshape(adj, [-1])
        pred_gcn, gcn_feature = self.gae(adj, node_feature)
        self.gcn_feature = gcn_feature
        pred_1 = torch.mul(labels, pred_gcn)
        pos_weight = torch.tensor((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        loss_model = norm * torch.mean(
            F.binary_cross_entropy_with_logits(
                input=pred_1, target=labels.float(), pos_weight=pos_weight
            )
        )
        self.optim_gae.zero_grad()
        loss_model.requires_grad_(True)
        loss_model.backward()
        self.optim_gae.step()
        print("loss:", loss_model.item())

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
