from chemicalx.data.datasetloader import RemoteDatasetLoader, LabeledTriples
from graph_package.configs.directories import Directories
from chemicalx import pipeline
import pandas as pd
from chemicalx.data import dataset_resolver
import csv
import io
import json
import urllib.request
from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import chain
from pathlib import Path
from textwrap import dedent
from typing import ClassVar, Dict, Mapping, Optional, Sequence, Tuple, cast

class ONEIL(RemoteDatasetLoader):
    data_path = Directories.DATA_PATH / "oneil" / "oneil.csv"

    def __init__(self) -> None:
        super().__init__(dataset_name="drugcomb")
    
    def get_labeled_triples(self) -> LabeledTriples:
        """Get the labeled triples file from the storage."""
        path = Directories.DATA_PATH / "gold" / "chemicalx" / "oneil" / "oneil.csv"
        dtype = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        df = pd.read_csv(path,dtype=dtype)
        return LabeledTriples(df)


if __name__ == "__main__":
    dataset = ONEIL()
    loader = dataset_resolver.make(dataset)
    train_generator, test_generator = loader.get_generators(
        batch_size=512,
        context_features=True,
        drug_features=True,
        drug_molecules=False,
        train_size=None,
        random_state=1,
    )

