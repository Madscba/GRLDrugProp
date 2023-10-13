from graph_package.configs.directories import Directories
import pandas as pd
from chemicalx.data import dataset_resolver
from torchdrug import data
from torch.utils import data as torch_data
from torch.utils.data import Dataset
from torchdrug.core import Registry as R
from chemicalx.data.datasetloader import RemoteDatasetLoader, LabeledTriples
from chemicalx.data import BatchGenerator
from torchdrug.core import Registry as R
import torch.utils.data



class ONEIL_DeepDDS_CX(RemoteDatasetLoader):
    data_path = Directories.DATA_PATH / "oneil" / "oneil.csv"

    def __init__(self) -> None:
        super().__init__(dataset_name="drugcomb")
    
    def get_labeled_triples(self) -> LabeledTriples:
        """Get the labeled triples file from the storage."""
        path = Directories.DATA_PATH / "gold" / "chemicalx" / "oneil" / "oneil.csv"
        dtype = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        df = pd.read_csv(path,dtype=dtype)
        return LabeledTriples(df)
    

class ONEIL_DeepDDS(RemoteDatasetLoader, BatchGenerator, Dataset):
    def __init__(self) -> None:
        RemoteDatasetLoader.__init__(self, dataset_name="drugcomb")

        BatchGenerator.__init__(
            self,
            batch_size=1,
            context_features=True,
            drug_features=True,
            drug_molecules=True,
            context_feature_set=self.get_context_features(),
            drug_feature_set=self.get_drug_features(),
            labeled_triples=self.get_labeled_triples(),
        )

        path = Directories.DATA_PATH / "gold" / "chemicalx" / "oneil" / "oneil.csv"
        dtype = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        self.df = pd.read_csv(path, dtype=dtype).reset_index(drop=True)
        self.batch_names = (
            "drug_features_left",
            "drug_molecules_left",
            "drug_features_right",
            "drug_molecules_right",
            "context_features",
            "label"
        )

    def get_labeled_triples(self) -> LabeledTriples:
        """Get the labeled triples file from the storage."""
        path = Directories.DATA_PATH / "gold" / "chemicalx" / "oneil" / "oneil.csv"
        dtype = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        df = pd.read_csv(path, dtype=dtype)
        return LabeledTriples(df)

    def get_item(self, index):
        return {name: value for name, value in zip(self.batch_names, data)}

    def __getitem__(self, index):
        row = self.df.loc[index]
        drug_features_left = self._get_drug_features([row["drug_1"]])
        drug_molecules_left = self._get_drug_molecules([row["drug_1"]])
        drug_features_right = self._get_drug_features([row["drug_2"]])
        drug_molecules_right = self._get_drug_molecules([row["drug_2"]])
        context_features = self._get_context_features([row["context"]]).squeeze()

        label = torch.tensor(self.df.loc[index, "label"],dtype=torch.float32)
        data = (
            drug_features_left,
            drug_molecules_left,
            drug_features_right,
            drug_molecules_right,
            context_features,
            label
        )

        return {name: value for name, value in zip(self.batch_names, data)}

    def __len__(self) -> int:
        return len(self.df)



class ONEIL_RESCAL(data.KnowledgeGraphDataset):
    def __init__(self, data):
        super().__init__()
        df = self._create_inverse_triplets(data)
        # Convert relevant columns to a NumPy array and load it into the dataset
        self.load_triplet(df.loc[:, ['drug_1_id', 'drug_2_id', 'context_id']].to_numpy())
        n_samples = self.num_triplet.tolist()
        self.num_samples = [int(n_samples*.8),int(n_samples*.1),int(n_samples*.1)]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

    def _create_inverse_triplets(self,df: pd.DataFrame):
        """ Create inverse triplets so that if (h,r,t) then (t,r,h) is also in the graph"""
        df_inv = df.copy()
        df_inv['drug_1'], df_inv['drug_2'] = df['drug_2'], df['drug_1']
        df_inv['drug_1_id'], df_inv['drug_2_id'] = df['drug_2_id'], df['drug_1_id']
        df_combined = pd.concat([df,df_inv], ignore_index=True)
        return df_combined