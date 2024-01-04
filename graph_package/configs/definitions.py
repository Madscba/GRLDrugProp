from graph_package.src.models import (
    DeepDDS,
    RESCAL,
    HybridModel,
    TransE,
    DistMult,
    ComplEx,
    RotatE,
    GNN,
)
from graph_package.src.models.gaecds.model import GAECDS
from graph_package.configs.directories import Directories


model_dict = {
    "deepdds": DeepDDS,
    "rescal": RESCAL,
    "hybridmodel": HybridModel,
    "transe": TransE,
    "distmult": DistMult,
    "complex": ComplEx,
    "rotate": RotatE,
    "gnn": GNN,
    "gaecds": GAECDS,
}

dataset_dict = {
    "oneil_legacy": Directories.DATA_PATH / "gold" / "oneil_legacy" / "oneil.csv",
    "oneil": Directories.DATA_PATH / "gold" / "oneil" / "oneil.csv",
    "deepdds_original": Directories.DATA_PATH
    / "gold"
    / "deepdds_original"
    / "deepdds_original.csv",
    "oneil_almanac": Directories.DATA_PATH
    / "gold"
    / "oneil_almanac"
    / "oneil_almanac.csv",
}
