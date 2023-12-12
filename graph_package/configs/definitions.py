from graph_package.src.models import DeepDDS, RESCAL, HybridModel, TransE, DistMult, ComplEx, RotatE
from graph_package.configs.directories import Directories  


model_dict = {"deepdds": DeepDDS, "rescal": RESCAL, "hybridmodel": HybridModel,
              "transe": TransE, "distmult": DistMult, "complex": ComplEx,
              "rotate": RotatE}

dataset_dict = {"oneil_legacy": Directories.DATA_PATH / "gold" / "oneil_legacy" / "oneil.csv",
                "oneil": Directories.DATA_PATH / "gold" / "oneil" / "oneil.csv",
                "deepdds_original":  Directories.DATA_PATH / "gold" / "deepdds_original" / "deepdds_original.csv"}




