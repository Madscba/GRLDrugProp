from dataclasses import dataclass
from pathlib import Path
from graph_package.src.models import DeepDDS, RESCAL
from graph_package.src.tasks import SynergyPrediction, KnowledgeGraphCompletion
from graph_package.src.etl.dataloaders import ONEIL_DeepDDS, ONEIL_RESCAL
import graph_package



@dataclass
class Directories:
    """Class with all paths used in the repository."""

    REPO_PATH = Path(graph_package.__file__).parent.parent
    MODULE_PATH = Path(graph_package.__file__).parent
    DATA_PATH = REPO_PATH / "data"
    TESTS = REPO_PATH / "tests"
    TESTS_DATA = TESTS / "data"
    LOGGING_FOLDER = REPO_PATH / "logs"
    


model_dict = {"deepdds": DeepDDS, "rescal": RESCAL} 

dataset_dict = {"oneil_deepdds": ONEIL_DeepDDS, "oneil_rescal": ONEIL_RESCAL}

task_dict = {"deepdds": SynergyPrediction, "rescal": KnowledgeGraphCompletion}