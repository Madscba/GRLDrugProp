from torchdrug import data
from torch.utils import data as torch_data
from torchdrug.core import Registry as R

@R.register("datasets.ONEIL")
class ONEIL(data.KnowledgeGraphDataset):
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