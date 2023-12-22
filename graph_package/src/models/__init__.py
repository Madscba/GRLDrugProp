import torchvision
from graph_package.src.models.rescal import RESCAL
from graph_package.src.models.deepdds import DeepDDS
from graph_package.src.models.hybridmodel import HybridModel
from graph_package.src.models.transe import TransE
from graph_package.src.models.distmult import DistMult
from graph_package.src.models.complex import ComplEx
from graph_package.src.models.rotate import RotatE
from graph_package.src.models.gnn.gnn import GNN

__all__ = ["RESCAL", "DeepDDS", "HybridModel", "TransE", "DistMult", "ComplEx", "RotatE","GNN"]
