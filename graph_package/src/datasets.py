from chemicalx.data.datasetloader import DatasetLoader, RemoteDatasetLoader, LabeledTriples, DrugFeatureSet, ContextFeatureSet
from configs.directories import Directories
import pandas as pd

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

class ONEIL(DatasetLoader):
    data_path = Directories.DATA_PATH / "oneil" / "oneil.csv"

    def __init__(self) -> None:
        super().__init__()
    
    def get_generator(
        self,
        batch_size: int,
        context_features: bool,
        drug_features: bool,
        drug_molecules: bool,
        labeled_triples: Optional[LabeledTriples] = None,
    ) -> BatchGenerator:
        """Initialize a batch generator.

        :param batch_size: Number of drug pairs per batch.
        :param context_features: Indicator whether the batch should include biological context features.
        :param drug_features: Indicator whether the batch should include drug features.
        :param drug_molecules: Indicator whether the batch should include drug molecules
        :param labeled_triples:
            A labeled triples object used to generate batches. If none is given, will use
            all triples from the dataset.
        :returns: A batch generator
        """
        return BatchGenerator(
            batch_size=batch_size,
            context_features=context_features,
            drug_features=drug_features,
            drug_molecules=drug_molecules,
            context_feature_set=self.get_context_features() if context_features else None,
            drug_feature_set=self.get_drug_features() if drug_features else None,
            labeled_triples=self.get_labeled_triples() if labeled_triples is None else labeled_triples,
        )

    @abstractmethod
    def get_context_features(self) -> ContextFeatureSet:
        """Get the context feature set."""

    @property
    def num_contexts(self) -> int:
        """Get the number of contexts."""
        return len(self.get_context_features())

    @property
    def context_channels(self) -> int:
        """Get the number of features for each context."""
        return next(iter(self.get_context_features().values())).shape[1]

    @abstractmethod
    def get_drug_features(self):
        """Get the drug feature set."""

    @property
    def num_drugs(self) -> int:
        """Get the number of drugs."""
        return len(self.get_drug_features())

    @property
    def drug_channels(self) -> int:
        """Get the number of features for each drug."""
        return next(iter(self.get_drug_features().values()))["features"].shape[1]

    @abstractmethod
    def get_labeled_triples(self) -> LabeledTriples:
        """Get the labeled triples file from the storage."""
        df = pd.read_csv(self.data_path)
        return LabeledTriples(df)

    @property
    def num_labeled_triples(self) -> int:
        """Get the number of labeled triples."""
        return len(self.get_labeled_triples())

    def summarize(self) -> None:
        """Summarize the dataset."""
        print(
            dedent(
                f"""\
            Name: {self.__class__.__name__}
            Contexts: {self.num_contexts}
            Context Feature Size: {self.context_channels}
            Drugs: {self.num_drugs}
            Drug Feature Size: {self.drug_channels}
            Triples: {self.num_labeled_triples}
        """
            )
        )

