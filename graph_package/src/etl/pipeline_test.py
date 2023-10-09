"""Script to test the chemicalx pipeline for DeepDDS."""
from chemicalx import pipeline
from chemicalx.models import DeepSynergy, DeepDDS
from chemicalx.data import DrugCombDB
from .dataloaders import OneilCX

model = DeepDDS(context_channels=288)


dataset = OneilCX()


results = pipeline(dataset=dataset,
                   model=model,
                   batch_size=1024,
                   context_features=True,
                   drug_features=True,
                   drug_molecules=True,
                   epochs=20)

results.summarize()