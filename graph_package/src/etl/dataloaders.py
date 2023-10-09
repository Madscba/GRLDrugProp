from chemicalx.data.datasetloader import RemoteDatasetLoader, LabeledTriples
from graph_package.configs.directories import Directories
import pandas as pd
from chemicalx.data import dataset_resolver
from torchdrug import data
from torch.utils import data as torch_data
from torchdrug.core import Registry as R

class OneilCX(RemoteDatasetLoader):
    data_path = Directories.DATA_PATH / "oneil" / "oneil.csv"

    def __init__(self) -> None:
        super().__init__(dataset_name="drugcomb")
    
    def get_labeled_triples(self) -> LabeledTriples:
        """Get the labeled triples file from the storage."""
        path = Directories.DATA_PATH / "gold" / "chemicalx" / "oneil" / "oneil.csv"
        dtype = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        df = pd.read_csv(path,dtype=dtype)
        return LabeledTriples(df)
    


@R.register("datasets.ONEIL")
class OneilTD(data.KnowledgeGraphDataset):
    def __init__(self, data):
        super().__init__()
        # Convert relevant columns to a NumPy array and load it into the dataset
        self.load_triplet(data.loc[:, ['drug1_id', 'drug2_id', 'context_id']].to_numpy())
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




if __name__ == "__main__":
    dataset = OneilCX()
    loader = dataset_resolver.make(dataset)
    train_generator, test_generator = loader.get_generators(
        batch_size=512,
        context_features=True,
        drug_features=True,
        drug_molecules=False,
        train_size=None,
        random_state=1,
    )
