from graph_package.src.models import DeepDDS, RESCAL
from graph_package.src.tasks import SynergyPrediction, KnowledgeGraphCompletion
from graph_package.src.etl.dataloaders import ONEIL_DeepDDS, ONEIL_RESCAL


model_dict = {"deepdds": DeepDDS, "rescal": RESCAL} 

dataset_dict = {"oneil_deepdds": ONEIL_DeepDDS, "oneil_rescal": ONEIL_RESCAL}

task_dict = {"deepdds": SynergyPrediction, "rescal": KnowledgeGraphCompletion}