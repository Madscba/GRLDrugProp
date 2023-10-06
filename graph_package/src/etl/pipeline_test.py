
from chemicalx import pipeline
from chemicalx.models import DeepSynergy, DeepDDS
from chemicalx.data import DrugCombDB
from graph_package.src.etl.datasets import ONEIL

model = DeepDDS(context_channels=112,
                drug_gcn_hidden_dims=[112, 112 * 2, 112 * 4],
                drug_channels=256)

# model = DeepSynergy(context_channels=112,
#                     drug_channels=256)







dataset = ONEIL()

drug_feature = dataset.get_drug_features()
drug_triplets = dataset.get_labeled_triples()   
drug_feature_keys = drug_feature.keys()
drugs = set(drug_triplets.data["drug_1"])-set(drug_feature_keys)
sum = drug_triplets.data.apply(lambda x: (x["drug_1"] in drugs) or (x["drug_2"] in drugs))
drug_triplets.data.columns
results = pipeline(dataset=dataset,
                   model=model,
                   batch_size=1024,
                   context_features=True,
                   drug_features=True,
                   drug_molecules=False,
                   epochs=100)


results.summarize()