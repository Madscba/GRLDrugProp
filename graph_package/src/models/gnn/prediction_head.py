from torch import nn
import torch
import json
from graph_package.configs.directories import Directories
import pandas as pd
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        dataset: str,
        use_mono_response: bool = False,
        cell_line_features: str = "ccle",
        hidden_dims: List[int] = [256, 128, 64],
        custom_lr_setup: bool = False,
        batch_norm: bool = False,
        feature_dropout: float = 0.0,
    ):
        super(MLP, self).__init__()
        self.use_mono_response = use_mono_response
        self.dataset = dataset
        self.encoding = self._load_ccle(cell_line_features) if cell_line_features in ("ccle","ccle_full") else self.load_cell_line_onehot()
        self.feature_dropout = nn.Dropout(feature_dropout)
        cell_line_input_dim = self.encoding.shape[1]
        
        cell_layers = [
            nn.Linear(cell_line_input_dim, 128),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.ReLU(),
            self.feature_dropout,
            nn.Linear(128, 64),
            nn.BatchNorm1d(64) if batch_norm else nn.Identity(),
            nn.ReLU(),
            self.feature_dropout,
        ]
        self.cell_line_mlp = nn.Sequential(*cell_layers)

        if self.use_mono_response:
            self.mono_r_index, self.mono_r = self._load_mono_response(self.dataset)

            global_mlp_input_dim = 2 * dim + 64 + 3 * 2
        else:
            global_mlp_input_dim = 2 * dim + 64

        self.global_mlp = self.create_global_mlp(global_mlp_input_dim, hidden_dims, batch_norm, feature_dropout)


    def _load_ccle(self, cell_line_features):
        if cell_line_features == "ccle":
            f_name = "CCLE_954_gene_express_pca.json"
        elif cell_line_features == "ccle_full":
            f_name = "CCLE_954_gene_express.json"
        else:
            raise ValueError("Invalid cell feature arg (++model.ph_kwargs.cell_line_feature)")
        feature_path = (
            Directories.DATA_PATH
            / "features"
            / "cell_line_features"
            / f_name
        )
        with open(feature_path) as f:
            all_edge_features = json.load(f)
        vocab_path = (
            Directories.DATA_PATH / "gold" / self.dataset / "relation_vocab.json"
        )
        with open(vocab_path) as f:
            entity_vocab = json.load(f)
        vocab_reverse = {v: k for k, v in entity_vocab.items()}
        ids = sorted(list(vocab_reverse.keys()))
        ccle = torch.tensor(
            [all_edge_features[vocab_reverse[id]] for id in ids], device=device
        )
        return ccle
    
    
    def load_cell_line_onehot(self):
        # Define the path to the cell line features file
        vocab_path = (
            Directories.DATA_PATH / "gold" / self.dataset / "relation_vocab.json"
        )
        with open(vocab_path) as f:
            entity_vocab = json.load(f)

        return torch.eye(len(entity_vocab), device=device)

    def _load_mono_response(self, study_name):
        d_path = Directories.DATA_PATH / "gold" / study_name
        mono_response = pd.read_csv(d_path / "mono_response.csv")
        mono_response = self._enrich_mono_response_with_ids_and_drop_na_and_col(
            mono_response, d_path
        )
        mono_response_tensor = torch.tensor(mono_response.values, device=device)
        return mono_response.index, mono_response_tensor

    def _enrich_mono_response_with_ids_and_drop_na_and_col(self, mono_response, d_path):
        with open(d_path / "entity_vocab.json") as f:
            entity_vocab = json.load(f)
        mono_response["drug_id"] = mono_response["drug"].map(entity_vocab)

        with open(d_path / "relation_vocab.json") as f:
            relation_vocab = json.load(f)
        mono_response["context_id"] = mono_response["cell_line"].map(relation_vocab)

        mono_response = mono_response.dropna().drop(columns=["drug", "cell_line"])
        mono_response.set_index(["drug_id", "context_id"], inplace=True)
        if "Unnamed: 0" in mono_response.columns:
            mono_response.drop("Unnamed: 0",axis=1,inplace=True)
        return mono_response

    def _get_mono_response(self, drug_ids, context_ids):
        sample_ids = list(zip(drug_ids.cpu().numpy(), context_ids.cpu().numpy()))
        batch_mono_val = torch.vstack(
            [self.mono_r[self.mono_r_index.get_loc(ids)] for ids in sample_ids]
        )
        batch_mono_val = batch_mono_val.to(torch.float32)
        return batch_mono_val
    
    def create_global_mlp(self, global_mlp_input_dim, hidden_dims, batch_norm=True, feature_dropout=None):
        layers = []
        input_dim = global_mlp_input_dim
        
        # Iterate over the hidden dimensions and create linear layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.Identity())
            
            layers.append(nn.ReLU())
            
            if feature_dropout is not None:
                layers.append(nn.Dropout(feature_dropout))
            
            input_dim = hidden_dim
        
        # Add the final linear layer
        layers.append(nn.Linear(input_dim, 1))
        
        global_mlp = nn.Sequential(*layers)
    
        return global_mlp

    def forward(
        self,
        d1_embd: torch.Tensor,
        d2_embd: torch.Tensor,
        context_ids: torch.Tensor,
        drug_1_ids: torch.Tensor,
        drug_2_ids: torch.Tensor,
    ) -> torch.FloatTensor:
        c = self.cell_line_mlp(self.encoding[context_ids])
        if self.use_mono_response:
            d1_mono = self._get_mono_response(drug_1_ids, context_ids)
            d2_mono = self._get_mono_response(drug_2_ids, context_ids)
            input = torch.concat([d1_embd, d2_embd, c, d1_mono, d2_mono], dim=1)
        else:
            input = torch.concat([d1_embd, d2_embd, c], dim=1)
        output = self.global_mlp(input)
        return output


class DistMult(nn.Module):
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

    def __init__(self, rel_tot, dim, max_score=30):
        super(DistMult, self).__init__()
        self.num_relation = rel_tot
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

    def forward(
        self,
        d1_embd: torch.Tensor,
        d2_embd: torch.Tensor,
        context_ids: torch.Tensor,
        drug_1_ids: torch.Tensor,
        drug_2_ids: torch.Tensor,
    ) -> torch.FloatTensor:
        score = self.score_triplet(d1_embd, d2_embd, context_ids)
        return score

    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.cpu().numpy()

    def __name__(self) -> str:
        return "DistMult"


class DotProduct(nn.Module):
    """
    DotProduct prediction head
    """

    def __init__(self, dim, dataset: str = ""):
        super(DotProduct, self).__init__()

    def forward(
        self,
        d1_embd: torch.Tensor,
        d2_embd: torch.Tensor,
        context_ids: torch.Tensor,
        drug_1_ids: torch.Tensor,
        drug_2_ids: torch.Tensor,
    ) -> torch.FloatTensor:
        score = (d1_embd * d2_embd).sum(-1)
        return score

    def __name__(self) -> str:
        return "DotProduct"
