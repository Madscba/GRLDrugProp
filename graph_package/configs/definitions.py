from graph_package.src.models import DeepDDS, RESCAL
from graph_package.src.pl_modules import DeepDDS_PL, Rescal_PL
from graph_package.src.pl_modules import KnowledgeGraphCompletion
from graph_package.src.etl.dataloaders import ONEIL_DeepDDS, ONEIL_RESCAL


model_dict = {"deepdds": DeepDDS_PL, "rescal": Rescal_PL}
dataset_dict = {"oneil_deepdds": ONEIL_DeepDDS, "oneil_rescal": ONEIL_RESCAL}
