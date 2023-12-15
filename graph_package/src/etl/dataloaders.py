import json
import torch
import pandas as pd
from graph_package.configs.directories import Directories
from graph_package.src.etl.medallion.gold import (
    create_drug_id_vocabs, 
    create_cell_line_id_vocabs
)
from torch.utils.data import Dataset
from torchdrug.data import Graph
import numpy as np

target_dict = {
    "reg": {
        "zip_mean": "synergy_zip_mean",
        "zip_max": "synergy_zip_max",
        "css": "css",
    },
    "clf": {"zip_mean": "mean_label", "zip_max": "max_label", "loewe": "label"},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KnowledgeGraphDataset(Dataset):
    def __init__(
        self, 
        dataset_path, 
        target: str = "zip_mean", 
        task: str = "reg", 
        use_node_features: bool = False,
        use_edge_features: bool = False 
    ):
        """
        Initialize the Knowledge Graph.

        Parameters:
        - dataset_path (str): The path to the dataset.
        - target (str, optional): The target variable for the task.
        - task (str, optional): The type of task ("reg" for regression, "clf" for classification).
        - use_node_features (bool, optional): Whether to use node features and load them into the KG.
        - use_edge_features (bool, optional): Whether to use edge features and load them into the KG.
        """
        self.target = target
        self.task = task
        self.dataset_path = dataset_path
        self.device = device
        self.use_node_features = use_node_features
        self.use_edge_features = use_edge_features
        self.label = target_dict[task][target]
        self.data_df = pd.read_csv(
            self.dataset_path,
            dtype={
                "drug_1_id": int,
                "drug_2_id": int,
                "drug_1_name": str,
                "drug_2_name": str,
                "context": str,
                "context_id": int,
                self.label: float,
            },
        )
        triplets = self.data_df.loc[
            :, ["drug_1_id", "drug_2_id", "context_id"]
        ].to_numpy()
        self.graph = self._init_graph(triplets)
        self.indices = list(range(len(self.data_df)))

    def get_labels(self, indices=None):
        if indices is None:
            indices = self.indices
        if self.target == 'css': 
            labels = np.random.randint(0,1,len(indices))
        else: 
            labels = self.data_df.iloc[indices][target_dict['clf'][self.target]]
        return labels
    
    def _init_graph(self, triplets):
        node_features = self._get_node_features() if self.use_node_features else None
        edge_features = self._get_edge_features() if self.use_edge_features else None
        self.num_relations = len(set(self.data_df["context"]))
        self.num_nodes = len(
            set(self.data_df["drug_1_id"]).union(set(self.data_df["drug_2_id"]))
        )
        triplets = torch.as_tensor(triplets, dtype=torch.long, device=self.device)
        graph = Graph(
            triplets, 
            num_node=self.num_nodes, 
            num_relation=self.num_relations, 
            node_feature=node_features,
            edge_feature=edge_features
        )
        return graph

    def _update_dataset(self, df: pd.DataFrame):
        self.data_df = pd.concat([self.data_df, df], ignore_index=True)
        triplets = self.data_df.loc[
            :, ["drug_1_id", "drug_2_id", "context_id"]
        ].to_numpy()
        self.graph = self._init_graph(triplets)
    
    def _get_node_features(self):
        feature_path = Directories.DATA_PATH / "features" / "node_features" / f"{self.dataset_path.parts[-2]}_drug_features.json"
        with open(feature_path) as f:
            node_features = json.load(f)
        with open(self.dataset_path.parent / "entity_vocab.json") as f:
            drug_vocab = json.load(f)
        feature_dict = {}
        for node, feature in node_features.items():
            concatenated_features = []
            for value in feature.values():
                concatenated_features.extend(value)
            feature_dict[node] = concatenated_features
        node_features = [
            feature_dict[name.lower()] for name in drug_vocab.keys() 
            if name.lower() in feature_dict.keys()
        ]
        return node_features

    def _get_edge_features(self):
        feature_path = Directories.DATA_PATH / "features" / "cell_line_features" / "CCLE_954_gene_express.json"
        with open(feature_path) as f:
            all_edge_features = json.load(f)
        edge_df = self.data_df['context'].map(all_edge_features)
        edge_features = [edge_df[i] for i in edge_df.index]
        return edge_features

    def make_inv_triplets(self,indices):
        """Create inverse triplets so that if (h,r,t) then (t,r,h) is also in the graph"""
        df_subset = self.data_df.iloc[indices]
        df_inv = df_subset.copy()
        df_inv["drug_1_name"], df_inv["drug_2_name"] = df_subset["drug_2_name"], df_subset["drug_1_name"]
        df_inv["drug_1_id"], df_inv["drug_2_id"] = df_subset["drug_2_id"], df_subset["drug_1_id"]
        inv_idx_start = len(self.data_df)
        self._update_dataset(df_inv)
        sub_set_indices = list(range(inv_idx_start, len(self.data_df)))  
        return sub_set_indices 
    
    def del_inv_triplets(self):
        self.data_df = self.data_df.iloc[:len(self.indices)]
        self.graph = self.graph.edge_mask(self.indices)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.graph.edge_list[index], self.data_df.iloc[index][self.label]
