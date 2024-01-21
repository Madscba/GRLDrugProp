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
    "oneil_almanac_het": Directories.DATA_PATH
    / "gold"
    / "oneil_almanac_het"
    / "oneil_almanac_het.csv",
    "drugcomb": Directories.DATA_PATH
    / "gold"
    / "drugcomb"
    / "drugcomb.csv",
     "drugcomb_het": Directories.DATA_PATH
    / "gold"
    / "drugcomb_het"
    / "drugcomb_het.csv"
}


d_feature_path = Directories.DATA_PATH / "features" / "drug_features"
drug_representation_path_dict = {
    "morgan": d_feature_path / "drug_ECFP_fp_2D.csv",
    "morgan83_rad6": d_feature_path / "drug_ECFP_fp_2D_83.csv",
    "morgan83_rad3": d_feature_path / "drug_ECFP_fp_2D_83_rad3.csv",
    "e3fp": d_feature_path / "drug_E3FP_fp_3D.csv",
    "maccs": d_feature_path / "drug_MACCS_fp_2D_.csv",
    "minhash": d_feature_path / "drug_MINHASH_fp_2D.csv",
    "minhash83": d_feature_path / "drug_MINHASH_fp_2D_83.csv",
    "minhash83_rad3": d_feature_path / "drug_MINHASH_fp_2D_83_rad3.csv",
    "rdkit": d_feature_path / "drug_rdkit_descriptor_2D.csv",
}
