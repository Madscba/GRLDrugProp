import json
import torch
import pandas as pd
from graph_package.configs.directories import Directories
from graph_package.configs.definitions import dataset_dict
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
        name: str="oneil_almanac",
        discrete_edge_weights: bool = False,
        target: str = "zip_mean", 
        task: str = "reg", 
        use_node_features: bool = False,
        modalities: str = 'None',
        use_edge_features: bool = False 
    ):
        """
        Initialize the Knowledge Graph.

        Parameters:
        - name (str): The name of the dataset.
        - discrete_edge_weights (bool): Whether to discretize the edge weights to -1, 0 and 1.
        - dataset_path (str): The path to the dataset.
        - target (str, optional): The target variable for the task.
        - task (str, optional): The type of task ("reg" for regression, "clf" for classification).
        - use_node_features (bool, optional): Whether to use node features and load them into the KG.
        - modalities (list, optional): Which modalities in Hetionet to include as PCA node features:
            Options:
                - 'None' - only drug features are used as node features then (Default)
                - 'Gene'
                - 'Side Effect'
                - 'Disease'
                - 'Pharmacological Class'
                - 'All' for including all possible nearest modalities 
        - use_edge_features (bool, optional): Whether to use edge features and load them into the KG.
        """
        self.target = target
        self.discrete_edge_weights = discrete_edge_weights
        self.task = task
        self.dataset_path = dataset_dict[name.lower()]
        self.use_node_features = use_node_features
        self.modalities = modalities
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
        
        self.graph = self._init_graph(self.data_df)
            
        self.indices = list(range(len(self.data_df)))

    def get_labels(self, indices=None):
        if indices is None:
            indices = self.indices
        if self.target == 'css': 
            labels = np.random.randint(0,1,len(indices))
        else: 
            labels = self.data_df.iloc[indices][target_dict['clf'][self.target]]
        return labels
    
    def _init_graph(self, data_df: pd.DataFrame):
        triplets = self.data_df.loc[
            :, ["drug_1_id", "drug_2_id", "context_id"]
        ].to_numpy()
        targets = self.data_df[self.label].to_numpy() 
        if self.discrete_edge_weights:
            targets = np.where(targets > 5, 1, np.where(targets < -5, -1, 0))
        node_features = self._get_node_features() if self.use_node_features else None
        edge_features = self._get_edge_features() if self.use_edge_features else None
        num_relations = len(set(self.data_df["context"]))
        num_nodes = len(
            set(self.data_df["drug_1_id"]).union(set(self.data_df["drug_2_id"]))
        )
        triplets = torch.as_tensor(triplets, dtype=torch.long, device=device)
        graph = Graph(
            triplets, 
            num_node=num_nodes, 
            num_relation=num_relations, 
            node_feature=node_features,
            edge_feature=edge_features, 
            edge_weight=targets,
        )
        return graph
    
    def _get_node_features(self):
        # Load drug features and vocab with graph node IDs
        drug_feature_path = Directories.DATA_PATH / "features" / "drug_features" / "drug_ECFP_fp_2D.csv"
        drug_features = pd.read_csv(drug_feature_path,index_col=0)
        with open(self.dataset_path.parent / "entity_vocab.json") as f:
            drug_vocab = json.load(f)
        node_feature_dict = {}
        
        # In case only drug features are used
        if self.modalities == 'None':
            for drug in drug_features.index:
                node_feature_dict[drug] = drug_features.loc[drug].to_list()
                
        # Load PCA nearest neighbor features
        else: 
            pca_feature_path = Directories.DATA_PATH / "features" / "node_features" / "oneil_almanac_drug_features.json"
            with open(pca_feature_path) as f:
                pca_features = json.load(f)
            relation_dict = {
                'Gene': ['binds', 'downregulates','upregulates'],
                'Side Effect': ['causes'],
                'Disease': ['treats'],
                'Pharmacologic Class': ['includes']
            }
            if self.modalities[0] == 'All':
                self.modalities = ['Gene', 'Side Effect', 'Disease', 'Pharmacologic Class'] 
            relations_to_include = [
                item for entity, rel_list in relation_dict.items() 
                if entity in self.modalities for item in rel_list
            ]
            # Concat drug and PCA features 
            for node, feature in pca_features.items():
                concatenated_pca_features = []
                for relation, value in feature.items():
                    if relation in relations_to_include:
                        concatenated_pca_features.extend(value)
                node_feature_dict[node] = drug_features.loc[node].to_list() + concatenated_pca_features

        # Convert to a list in correct order determined by graph node ID
        node_features = [
            node_feature_dict[name] for name in drug_vocab.keys() 
            if name in node_feature_dict.keys()
        ]
        # Convert to float arraylike 
        node_features = np.array(node_features).astype(np.float32)
        return node_features

    def _get_edge_features(self):
        feature_path = Directories.DATA_PATH / "features" / "cell_line_features" / "CCLE_954_gene_express_pca.json"
        with open(feature_path) as f:
            all_edge_features = json.load(f)
        edge_df = self.data_df['context'].map(all_edge_features)
        edge_features = edge_df.tolist()
        return edge_features

    def make_inv_triplets(self,indices):
        """Create inverse triplets so that if (h,r,t) then (t,r,h) is also in the graph"""
        df_subset = self.data_df.iloc[indices]
        df_inv = df_subset.copy()
        df_inv["drug_1_name"], df_inv["drug_2_name"] = df_subset["drug_2_name"], df_subset["drug_1_name"]
        df_inv["drug_1_id"], df_inv["drug_2_id"] = df_subset["drug_2_id"], df_subset["drug_1_id"]
        inv_idx_start = len(self.data_df)
        self.data_df = pd.concat([self.data_df, df_inv], ignore_index=True)
        self.graph = self._init_graph(self.data_df)
        sub_set_indices = list(range(inv_idx_start, len(self.data_df)))  
        return sub_set_indices 
    
    def del_inv_triplets(self):
        self.data_df = self.data_df.iloc[:len(self.indices)]
        self.graph = self.graph.edge_mask(self.indices)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.graph.edge_list[index], self.data_df.iloc[index][self.label]
