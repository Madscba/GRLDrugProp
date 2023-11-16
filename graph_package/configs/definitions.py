from graph_package.src.models import DeepDDS, RESCAL, DeepDDS_HPC
from graph_package.configs.directories import Directories  


model_dict = {"deepdds": DeepDDS, "rescal": RESCAL, "deepdds_hpc": DeepDDS_HPC}

dataset_dict = {"oneil": Directories.DATA_PATH / "gold" / "oneil" / "oneil.csv",
                "deepdds_original":  Directories.DATA_PATH / "gold" / "deepdds_original" / "deepdds_original.csv"}




