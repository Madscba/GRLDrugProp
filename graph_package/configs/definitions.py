from graph_package.src.models import DeepDDS, RESCAL
from graph_package.src.pl_modules import DeepDDS_PL
from graph_package.src.pl_modules import  KnowledgeGraphCompletion
from graph_package.src.etl.dataloaders import ONEIL_DeepDDS, ONEIL_RESCAL




model_dict = {"deepdds": DeepDDS_PL, "rescal": RESCAL} 

dataset_dict = {"oneil_deepdds": ONEIL_DeepDDS, "oneil_rescal": ONEIL_RESCAL}
