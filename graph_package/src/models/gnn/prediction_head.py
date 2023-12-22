from torch import nn
import torch
import json
from graph_package.configs.directories import Directories
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP_PredictionHead(nn.Module):
    def __init_(self, drug_input_dim, use_mono_response: bool = False):
        super(MLP_PredictionHead, self).__init__()
        self.use_mono_response = use_mono_response
        self.ccle = self._load_ccle()
        cell_line_input_dim = self.ccle.shape[1]
        self.cell_line_mlp = nn.Sequential(
            nn.Linear(cell_line_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        if self.use_mono_response:
            self.mono_r = self._load_mono_response()
            global_mlp_input_dim = 2 * drug_input_dim + 64 + 3 * 2
        else:
            global_mlp_input_dim = 2 * drug_input_dim + 64

        self.global_mlp = nn.Sequential(
            nn.Linear(global_mlp_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _load_mono_response(self):
        gold_oneil_alm = Directories.DATA_PATH / "gold" / "oneil_almanac"
        mono_response_path = path = gold_oneil_alm / "mono_response.csv"
        mono_response = pd.read_csv(mono_response_path)
        vocab_path = gold_oneil_alm / "entity_vocab.json"
        with open(vocab_path) as f:
            entity_vocab = json.load(f)
        vocab_reverse = {v: k for k, v in entity_vocab.items()}
        ids = sorted(list(vocab_reverse.keys()))
        ccle = torch.tensor(
            [all_edge_features[vocab_reverse[id]] for id in ids], device=device
        )
        return ccle

    def _load_ccle(self):
        feature_path = (
            Directories.DATA_PATH
            / "features"
            / "cell_line_features"
            / "CCLE_954_gene_express.json"
        )
        with open(feature_path) as f:
            all_edge_features = json.load(f)
        vocab_path = (
            Directories.DATA_PATH / "gold" / "oneil_almanac" / "relation_vocab.json"
        )
        with open(vocab_path) as f:
            relation_vocab = json.load(f)
        vocab_reverse = {v: k for k, v in relation_vocab.items()}
        ids = sorted(list(vocab_reverse.keys()))
        ccle = torch.tensor(
            [all_edge_features[vocab_reverse[id]] for id in ids], device=device
        )
        return ccle

    def forward(
        self, d1_embd: torch.Tensor, d2_embd: torch.Tensor, context_ids: torch.Tensor
    ) -> torch.FloatTensor:
        c = self.cell_line_mlp(self.ccle[context_ids])
        if self.use_mono_response:
            pass
        input = torch.concat([d1_embd, d2_embd, c], dim=1)
        output = self.global_mlp(input)
        return output


class DistMult:
    """
    DistMult embedding proposed in `Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_.

    .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases:
        https://arxiv.org/pdf/1412.6575.pdf

    Parameters:
        num_entity (int): number of entities
        num_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        l3_regularization (float, optional): weight for l3 regularization
    """

    def __init__(self, ent_tot, rel_tot, dim, max_score=30):
        super().__init__(num_entity=ent_tot, num_relation=rel_tot, embedding_dim=dim)
        self.num_entity = ent_tot
        self.num_relation = rel_tot

        self.entity = nn.Parameter(torch.empty(ent_tot, dim))
        self.relation = nn.Parameter(torch.empty(rel_tot, dim))

        self.max_score = max_score
        nn.init.xavier_uniform_(self.relation)

    def score_triplet(self, h, t, r_index):
        """
        DistMult score function from `Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_.

        .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases:
            https://arxiv.org/pdf/1412.6575.pdf

        Parameters:
            entity (Tensor): entity embeddings of shape :math:`(|V|, d)`
            relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
            h_index (LongTensor): index of head entities
            t_index (LongTensor): index of tail entities
            r_index (LongTensor): index of relations
        """
        r = self.relation[r_index]
        score = (h * r * t).sum(dim=-1)
        return score

    def forward(self, d1_emb, d2_emb, context_ids):
        score = self.score_triplet(context_ids, d1_emb, d2_emb)
        return score

    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.cpu().numpy()

    def __name__(self) -> str:
        return "DistMult"
