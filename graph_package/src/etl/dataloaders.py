import json
import pandas as pd
from graph_package.configs.directories import Directories
from graph_package.src.etl.gold import (
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


class KnowledgeGraphDataset(Dataset):
    def __init__(
        self, 
        dataset_path, 
        target: str = "zip_mean", 
        task: str = "reg", 
        use_node_features: bool = True, 
    ):
        """
        Initialize the Knowledge Graph.

        Parameters:
        - dataset_path (str): The path to the dataset.
        - target (str, optional): The target variable for the task.
        - task (str, optional): The type of task ("reg" for regression, "clf" for classification).
        - use_node_features (bool, optional): Whether to use node features and load them into the KG.
        """
        self.target = target
        self.task = task
        self.use_node_features = use_node_features
        self.label = target_dict[task][target]
        self.data_df = pd.read_csv(
            dataset_path,
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
        if self.use_node_features:
            feature_path = Directories.DATA_PATH / "node_features" / "ONEIL_drug_features.json"
            with open(feature_path) as f:
                node_features = json.load(f)
            self.data_df, drug_vocab = self._filter_drugs(node_features)
            self.node_features = [node_features[name.lower()] for name in drug_vocab.keys() if name.lower() in node_features.keys()]
        else:    
            self.node_features = None
        triplets = self.data_df.loc[
            :, ["drug_1_id", "drug_2_id", "context_id"]
        ].to_numpy()
        self.num_relations = len(set(self.data_df["context"]))
        self.num_nodes = len(
            set(self.data_df["drug_1_id"]).union(set(self.data_df["drug_2_id"]))
        )
        self.graph = Graph(
            triplets, 
            num_node=self.num_nodes, 
            num_relation=self.num_relations, 
            node_feature=self.node_features
        )
        self.indices = list(range(len(self.data_df)))

    def get_labels(self, indices=None):
        if indices is None:
            indices = self.indices
        if self.target == 'css': 
            labels = np.random.randint(0,1,len(indices))
        else: 
            labels = self.data_df.iloc[indices][target_dict['clf'][self.target]]
        return labels

    def _filter_drugs(self, node_features):
        het_drugs = [drug.lower() for drug in list(node_features.keys())]
        filtered_drugs = self.data_df[
            (self.data_df['drug_1_name'].str.lower().isin(het_drugs)) &
            (self.data_df['drug_2_name'].str.lower().isin(het_drugs))
        ]           
        filtered_drugs, drug_vocab = create_drug_id_vocabs(filtered_drugs)
        filtered_drugs, _ = create_cell_line_id_vocabs(filtered_drugs)
        return filtered_drugs, drug_vocab

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.graph.edge_list[index], self.data_df.iloc[index][self.label] #, self.graph.node_feature[index]

    def _create_inverse_triplets(self, df: pd.DataFrame):
        """Create inverse triplets so that if (h,r,t) then (t,r,h) is also in the graph"""
        df_inv = df.copy()
        df_inv["drug_1_name"], df_inv["drug_2_name"] = df["drug_2_name"], df["drug_1_name"]
        df_inv["drug_1_id"], df_inv["drug_2_id"] = df["drug_2_id"], df["drug_1_id"]
        df_combined = pd.concat([df, df_inv], ignore_index=True)
        return df_combined