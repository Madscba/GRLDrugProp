from graph_package.configs.directories import Directories
from graph_package.src.etl.medallion.gold import get_drug_info
import pandas as pd
from graph_package.src.error_analysis.utils import get_drug_info as get_drug_info_full

from rdkit import Chem
from chemopy import Fingerprint
from rdkit.Chem import Descriptors
from pathlib import Path


def get_feature_path() -> Path:
    path_to_drug_feature_folder = Directories.DATA_PATH / "features" / "drug_features"
    path_to_drug_feature_folder.mkdir(parents=True, exist_ok=True)
    return path_to_drug_feature_folder


def make_drug_fingerprint_features(get_extended_repr: bool = False, dim=None):
    """Function takes drugs from oneil_almanack

    We could consider a more advanced fingerprint method:
    Smiles string as input
    https://github.com/HannesStark/3DInfomax
    Returns:

    """
    drug_SMILES, drug_names = get_drug_SMILES_repr()
    mols = get_molecules_from_SMILES(drug_SMILES)
    save_path = get_feature_path()

    # morgan_fingerprint:
    generate_and_save_morgan_fp(dim, drug_names, mols, save_path)

    # calculate_minhash_atompair_fp
    generate_and_save_minhash_fp(dim, drug_names, mols, save_path)

    generate_and_save_maccs_fp(drug_names, mols, save_path)

    generate_and_save_rdkit_descriptor(drug_names, mols, save_path)
    #
    # if get_extended_repr:
    #     # To obtain 11 2D molecular fingerprints with default folding size, one can use the following:
    #     drug_2d_fingerprint = []
    #     for mol in mols:
    #         drug_2d_fingerprint.append(Fingerprint.get_all_fps(mol))
    #     pd.DataFrame(drug_2d_fingerprint, index=drug_names).to_csv(
    #         save_path / "drug_all_fp_2D.csv"
    #     )


def generate_and_save_rdkit_descriptor(drug_names, mols, save_path):
    drug_2d_rdkit_descriptor = []
    for mol in mols:
        drug_2d_rdkit_descriptor.append(calculate_molecular_descriptors(mol))
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    drug_features_scaled = scaler.fit_transform(drug_2d_rdkit_descriptor)
    df = pd.DataFrame(drug_features_scaled, index=drug_names).dropna(axis=1, how="all")
    column_sums = df.sum(axis=0)
    zero_sum_columns = column_sums[column_sums == 0].index
    df = df.drop(columns=zero_sum_columns)
    df.to_csv(save_path / "drug_rdkit_descriptor_2D.csv")


def generate_and_save_maccs_fp(drug_names, mols, save_path):
    drug_2d_maccs_fingerprint = []
    for mol in mols:
        drug_2d_maccs_fingerprint.append(Fingerprint.calculate_maccs_fp(mol))
    pd.DataFrame(drug_2d_maccs_fingerprint, index=drug_names).to_csv(
        save_path / "drug_MACCS_fp_2D_.csv"
    )


def generate_and_save_minhash_fp(dim, drug_names, mols, save_path):
    drug_2d_minhash_fingerprint = []
    if dim is None:
        minhash_dim = 2048
        minhash_fname = "drug_MINHASH_fp_2D.csv"
    else:
        minhash_dim = dim
        minhash_fname = f"drug_MINHASH_fp_2D_{minhash_dim}.csv"
    for mol in mols:
        drug_2d_minhash_fingerprint.append(
            Fingerprint.calculate_minhash_atompair_fp(mol, radius=6, nbits=minhash_dim)
        )  # Settings from SOTA paper we are implementing
    pd.DataFrame(drug_2d_minhash_fingerprint, index=drug_names).to_csv(
        save_path / minhash_fname
    )


def generate_and_save_morgan_fp(dim, drug_names, mols, save_path):
    drug_2d_morgan_fingerprint = []
    if dim is None:
        morgan_dim = 300
        morgan_fpath = "drug_ECFP_fp_2D.csv"
    else:
        morgan_dim = dim
        morgan_fpath = f"drug_ECFP_fp_2D_{morgan_dim}.csv"
    for mol in mols:
        drug_2d_morgan_fingerprint.append(
            Fingerprint.calculate_morgan_fp(mol, radius=6, nbits=morgan_dim)
        )  # Settings from SOTA paper we are implementing
    pd.DataFrame(drug_2d_morgan_fingerprint, index=drug_names).to_csv(
        save_path / morgan_fpath
    )


def calculate_molecular_descriptors(mol, print_descriptors=False):
    """
    get mol descriptors for a molecule (rdkit-type)

    Args:
    - mol (RDKit Mol): RDKit molecule object.

    Returns:
    - concatenated_descriptor (list): List of calculated molecular descriptors.
    """

    mol_descriptors = Descriptors.CalcMolDescriptors(mol)
    concatenated_descriptor = list(mol_descriptors.values())
    # fill nan values with 0
    concatenated_descriptor = [0 if x is None else x for x in concatenated_descriptor]

    if print_descriptors:
        print(mol_descriptors.keys())

    return concatenated_descriptor


def get_drug_SMILES_repr():
    datasets_name = "oneil_almanac"
    save_path = Directories.DATA_PATH / "features" / "drug_features"
    save_path.mkdir(parents=True, exist_ok=True)
    # Get info dict with drug name, DrugBank ID and inchikey for each drug in dataset
    data_path = (
        Directories.DATA_PATH / "silver" / datasets_name / f"{datasets_name}.csv"
    )
    drugs = pd.read_csv(data_path)
    unique_drug_names = set(drugs["drug_row"]).union(set(drugs["drug_col"]))
    drug_info = get_drug_info(drugs, unique_drug_names, add_SMILES=True)
    df_drug_info = pd.DataFrame(drug_info).T.reset_index()
    df_drug_info = df_drug_info.rename(columns={"index": "drug_name"})
    drug_SMILES = [d.split(";")[0] for d in df_drug_info["SMILES"]]
    drug_drug_names = list(df_drug_info["drug_name"])
    return drug_SMILES, drug_drug_names


def get_molecules_from_SMILES(drug_SMILES):
    mols = []
    for idx, smiles in enumerate(drug_SMILES):
        smiles_ = smiles.split(";")
        if len(smiles_) == 1:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        else:
            mol = Chem.AddHs(
                Chem.MolFromSmiles(smiles_[0])
            )  # one drug appears to have two SMILES str
        mols.append(mol)
    return mols


if __name__ == "__main__":
    make_drug_fingerprint_features(dim=83)
