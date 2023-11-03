from graph_package.src.models import DeepDDS, RESCAL
from graph_package.src.pl_modules import DeepDDS_PL, Rescal_PL
from graph_package.src.etl.dataloaders import DeepDDS_DataSet, RESCAL_DataSet
from graph_package.configs.directories import Directories  

model_dict = {"deepdds": DeepDDS_PL, "rescal": Rescal_PL}

dataset_dict = {"oneil": Directories.DATA_PATH / "gold" / "oneil" / "oneil.csv",
                "deepdds_original":  Directories.DATA_PATH / "gold" / "oneil" / "deepdds_original.csv"}

dataloader_dict = {
    "deepdds": DeepDDS_DataSet,
    "rescal": RESCAL_DataSet,
}
