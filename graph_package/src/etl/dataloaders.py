from graph_package.configs.directories import Directories
import pandas as pd
from chemicalx.data import dataset_resolver, DrugFeatureSet
from torchdrug.data import KnowledgeGraphDataset
from torch.utils.data import Dataset
from torchdrug.core import Registry as R
from chemicalx.data.datasetloader import RemoteDatasetLoader, LabeledTriples
from chemicalx.data import BatchGenerator
from torchdrug.core import Registry as R
import numpy as np
from torchdrug.data import Graph
import torch.utils.data
import json


class KnowledgeGraphDataset(Dataset):
    def __init__(self, dataset_path):
        self.data_df = pd.read_csv(
            dataset_path,
            dtype={
                "drug_1_id": int,
                "drug_2_id": int,
                "drug_1_name": str,
                "drug_2_name": str,
                "context": str,
                "context_id": int,
                "label": float,
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
        return self.data_df.iloc[indices]["label"]
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        return self.graph.edge_list[index], self.data_df.iloc[index]["label"]


    def _create_inverse_triplets(self, df: pd.DataFrame):
        """Create inverse triplets so that if (h,r,t) then (t,r,h) is also in the graph"""
        df_inv = df.copy()
        df_inv["drug_1"], df_inv["drug_2"] = df["drug_2"], df["drug_1"]
        df_inv["drug_1_id"], df_inv["drug_2_id"] = df["drug_2_id"], df["drug_1_id"]
        df_combined = pd.concat([df, df_inv], ignore_index=True)
        return df_combined


    