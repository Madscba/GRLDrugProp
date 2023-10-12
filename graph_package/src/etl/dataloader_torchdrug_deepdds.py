from chemicalx.data.datasetloader import RemoteDatasetLoader, LabeledTriples
from chemicalx.data import BatchGenerator
from graph_package.configs.directories import Directories
import pandas as pd
from chemicalx.data import dataset_resolver
from torchdrug import data
from torch.utils import data as torch_data
from torchdrug.core import Registry as R
import torch.utils.data


class OneilCX(RemoteDatasetLoader, BatchGenerator, torch_data.Dataset):
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

    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = range(start, stop, step)
        elif not isinstance(index, list):
            raise ValueError("Unknown index `%s`" % index)
        return index

    def get_item(self, index):
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

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)

        index = self._standarize_index(index, len(self))
        return [self.get_item(i) for i in index]

    def __len__(self) -> int:
        return len(self.df)


if __name__ == "__main__":
    dataset = OneilCX()
    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
