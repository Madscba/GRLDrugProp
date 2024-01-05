from typing import List, Optional, Iterable
from graph_package.configs.directories import Directories
import torch
from torchdrug.data import PackedGraph
import torch.nn as nn
from torchdrug.layers import MLP
from torch import nn
from torch.nn.functional import normalize
import urllib.request
from torchdrug.data import PackedGraph
from torchdrug.data.feature import atom_default
from torchdrug.layers import MLP, MaxReadout
from torchdrug.models import GraphConvolutionalNetwork
from torchdrug.data import Graph, Molecule
import json 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TORCHDRUG_NODE_FEATURES = len(atom_default(Molecule.dummy_mol.GetAtomWithIdx(0)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepDDS(nn.Module):
    def __init__(
        self,
        dataset_path,
        context_hidden_dims: List[int] = (2048, 512),  # same as paper
        drug_channels: int = TORCHDRUG_NODE_FEATURES,  # don't know about paper
        drug_gcn_hidden_dims: List[int] = [1024, 512, 156],  # same as paper
        drug_mlp_hidden_dims: List[int] = None,  # not in paper, only for cx
        context_output_size: int = 156,  # same as paper, based on figure 1 of paper
        fc_hidden_dims: List[int] = [1024, 512, 128],  # same as paper
        dropout: float = 0.2,  # same as paper
    ):

        super().__init__()
        self.entity_vocab = json.load(open(dataset_path.parent / "entity_vocab.json","r"))
        self.context_vocab = json.load(open(dataset_path.parent / "relation_vocab.json","r"))
        self.drug_features = self.load_drug_features()
        self.context_features = self.load_context_features()
        context_channels = list(self.context_features.values())[0].shape[1]
        # Check default parameters:
        # Defaults are different from the original implementation.
        if context_hidden_dims is None:
            context_hidden_dims = [32, 32]
        if drug_gcn_hidden_dims is None:
            drug_gcn_hidden_dims = [drug_channels, drug_channels * 2, drug_channels * 4]
        if drug_mlp_hidden_dims is None:
            drug_mlp_hidden_dims = [drug_channels * 2]
        if fc_hidden_dims is None:
            fc_hidden_dims = [32, 32]

        # Cell feature extraction with MLP
        self.cell_mlp = MLP(
            input_dim=context_channels,
            # Paper: [2048, 512, context_output_size]
            # Code: [512, 256, context_output_size]
            # Our code: [32, 32, context_output_size]
            hidden_dims=[*context_hidden_dims, context_output_size],
        )

        # GCN
        # Paper: GCN with three hidden layers + global max pool
        # Code: Same as paper + two FC layers. With different layer sizes.
        self.drug_conv = GraphConvolutionalNetwork(
            # Paper: [1024, 512, 156],
            # Code: [drug_channels, drug_channels * 2, drug_channels * 4]
            input_dim=drug_channels,
            hidden_dims=drug_gcn_hidden_dims,
            activation="relu",
        )
        self.drug_readout = MaxReadout()

        # Paper: no FC layers after GCN layers and global max pooling
        self.drug_mlp = MLP(
            input_dim=drug_gcn_hidden_dims[-1],
            hidden_dims=[*drug_mlp_hidden_dims, context_output_size],
            dropout=dropout,
            activation="relu",
        )

        self.final = nn.Sequential(
            MLP(
                input_dim=context_output_size * 3,
                hidden_dims=[*fc_hidden_dims, 1],
                dropout=dropout,
            ))

    def load_drug_features(self):
        path = Directories.DATA_PATH / "bronze" / "drugcomb" / "drug_dict.json"
        drug_dict = json.load(path.open())
        
        drug_features = {
            id: {
                "molecule": Molecule.from_smiles(drug_dict[drug]["smiles"].split(';')[0]).to(device),
            }
            for drug, id in self.entity_vocab.items()
        }

        return drug_features

    def load_context_features(self) -> dict:
        """Get the context feature set."""
        feature_path = (
            Directories.DATA_PATH
            / "features"
            / "cell_line_features"
            / "CCLE_954_gene_express_pca.json"
        )
        with open(feature_path) as f:
            all_edge_features = json.load(f)

        raw_data = {v: torch.FloatTensor(np.array(all_edge_features[k]).reshape(1, -1)).to(device) for k, v in self.context_vocab.items()}
        return raw_data

    def _get_drug_molecules(self, drug_identifiers: Iterable[int]) -> Optional[PackedGraph]:
        return Graph.pack([self.drug_features[drug.item()]["molecule"] for drug in drug_identifiers])

    
    def _get_context_features(self, context_identifiers: Iterable[int]) -> Optional[torch.FloatTensor]:
        return torch.cat([self.context_features[context.item()] for context in context_identifiers])


    def _forward_molecules(self, molecules: PackedGraph) -> torch.FloatTensor:
        features = self.drug_conv(
            molecules, molecules.data_dict["atom_feature"].float()
        )["node_feature"]
        features = self.drug_readout(molecules, features)
        return self.drug_mlp(features)
    

    def forward(
        self, inputs
    ) -> torch.FloatTensor:
        """Run a forward pass of the DeeDDS model.

        :param context_features: A matrix of cell line features
        :param molecules_left: A matrix of left drug features
        :param molecules_right: A matrix of right drug features
        :returns: A vector of predicted synergy scores
        """

        drug_1_ids, drug_2_ids, context_ids = map(lambda x: x.squeeze(),inputs.split(1, dim=1))
        molecules_left = self._get_drug_molecules(drug_1_ids)
        molecules_right = self._get_drug_molecules(drug_2_ids)
        context_features = self._get_context_features(context_ids).squeeze()

        # Run the MLP forward for the cell line features
        mlp_out = self.cell_mlp(normalize(context_features, p=2, dim=1))

        # Run the GCN forward for the drugs: GCN -> Global Max Pool -> MLP
        features_left = self._forward_molecules(molecules_left)
        features_right = self._forward_molecules(molecules_right)

        # Concatenate the output of the MLP and the GNN
        concat_in = torch.cat([mlp_out, features_left, features_right], dim=1)

        return self.final(concat_in).squeeze()

    def __name__(self) -> str:
        return "DeepDDS"
    
if __name__=="__main__":
    model = DeepDDS(Directories.DATA_PATH / "gold" / "oneil")
    print(model(torch.LongTensor([[0,1,2]])))