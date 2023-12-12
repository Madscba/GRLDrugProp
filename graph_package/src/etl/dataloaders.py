from graph_package.configs.directories import Directories
import pandas as pd
from torchdrug.data import KnowledgeGraphDataset
from torch.utils.data import Dataset
from torchdrug.core import Registry as R
from torchdrug.core import Registry as R
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
    def __init__(self, dataset_path, target: str = "zip_mean", task: str = "reg"):
        self.target = target
        self.task = task
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
        triplets = self.data_df.loc[
            :, ["drug_1_id", "drug_2_id", "context_id"]
        ].to_numpy()
        self.num_relations = len(set(self.data_df["context"]))
        self.num_nodes = len(
            set(self.data_df["drug_1_id"]).union(set(self.data_df["drug_2_id"]))
        )
        self.graph = Graph(
            triplets, num_node=self.num_nodes, num_relation=self.num_relations
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
    
    def _update_dataset(self, df: pd.DataFrame):
        self.data_df = pd.concat([self.data_df, df], ignore_index=True)
        triplets = self.data_df.loc[
            :, ["drug_1_id", "drug_2_id", "context_id"]
        ].to_numpy()
        self.graph = Graph(triplets, num_node=self.num_nodes, num_relation=self.num_relations)
    
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
