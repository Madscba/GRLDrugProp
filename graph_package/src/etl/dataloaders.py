import json
import pandas as pd
from graph_package.configs.directories import Directories
from torch.utils.data import Dataset
from torchdrug.data import Graph

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
        load_node_features: bool = False, 
        use_random_features: bool = False
    ):
        """
        Initialize the Knowledge Graph.

        Parameters:
        - dataset_path (str): The path to the dataset.
        - target (str, optional): The target variable for the task.
        - task (str, optional): The type of task ("reg" for regression, "clf" for classification).
        - load_node_features (bool, optional): Whether to load node features into the KG.
        - use_random_features (bool, optional): 
        Whether to use random features for the drugs not in Hetionet. 

            * If 'False' all drugs not in Hetionet are filtered from the KG
        """
        self.target = target
        self.task = task
        self.load_node_features = load_node_features
        self.use_random_features = use_random_features
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
        # TODO: Use random node features or filter on drugs not in hetionet
        assert not self.use_random_features or self.load_node_features, "The boolean variables are not the same."
        if self.load_node_features:
            feature_path = Directories.DATA_PATH / "ONEIL_drug_features.json"
            with open(feature_path) as f:
                node_features = json.load(f)
        else:    
            self.node_features = None
        if not self.use_random_features:
            self.data_df = self.data_df
        triplets = self.data_df.loc[
            :, ["drug_1_id", "drug_2_id", "context_id"]
        ].to_numpy()
        self.num_relations = len(set(self.data_df["context"]))
        self.num_nodes = len(
            set(self.data_df["drug_1_id"]).union(set(self.data_df["drug_2_id"]))
        )
        # TODO: Load node features (set as argument) into graph  
        #self.node_features = pd.read_csv("node_features") if self.load_node_features else None
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
        return self.data_df.iloc[indices][target_dict['clf'][self.target]]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.graph.edge_list[index], self.data_df.iloc[index][self.label] #, self.graph.node_feature[index]

    def _create_inverse_triplets(self, df: pd.DataFrame):
        """Create inverse triplets so that if (h,r,t) then (t,r,h) is also in the graph"""
        df_inv = df.copy()
        df_inv["drug_1"], df_inv["drug_2"] = df["drug_2"], df["drug_1"]
        df_inv["drug_1_id"], df_inv["drug_2_id"] = df["drug_2_id"], df["drug_1_id"]
        df_combined = pd.concat([df, df_inv], ignore_index=True)
        return df_combined