from graph_package.configs.directories import Directories
from graph_package.src.etl.medallion.silver import get_drug_info, filter_from_hetionet
import pandas as pd
from graph_package.src.error_analysis.utils import get_drug_info as get_drug_info_full
from rdkit.Chem import inchi

from rdkit import Chem
from chemopy import ChemoPy, Fingerprint, Fingerprint3D
from rdkit.Chem import AllChem
import os
import time as t
from tqdm import tqdm
import json
from pathlib import Path


def get_feature_path() -> Path:
    path_to_drug_feature_folder = Directories.DATA_PATH / "features" / "drug_features"
    path_to_drug_feature_folder.mkdir(parents=True, exist_ok=True)
    return path_to_drug_feature_folder


def make_drug_fingerprint_features(get_extended_repr: bool = False):
    """10.1093/bioinformatics/btt105:
    #requirements
    conda install openbabel mopac -c conda-forge
    pip install chemopy2


    We could consider a more advanced fingerprint method:
    Smiles string as input
    https://github.com/HannesStark/3DInfomax
    Returns:

    """
    drug_SMILES, drug_drug_names = get_drug_SMILES_repr()
    mols = get_molecules_from_SMILES(drug_SMILES)
    save_path = get_feature_path()

    # morgan_fingerprint:
    drug_2d_morgan_fingerprint = []
    for mol in mols:
        drug_2d_morgan_fingerprint.append(
            Fingerprint.calculate_morgan_fp(mol, radius=6, nbits=300)
        )  # Settings from SOTA paper we are implementing
    pd.DataFrame(drug_2d_morgan_fingerprint).to_csv(save_path / "drug_ECFP_fp_2D.csv")

    # calculate_minhash_atompair_fp
    drug_2d_minhash_fingerprint = []
    for mol in mols:
        drug_2d_minhash_fingerprint.append(
            Fingerprint.calculate_minhash_atompair_fp(mol)
        )  # Settings from SOTA paper we are implementing
    pd.DataFrame(drug_2d_minhash_fingerprint).to_csv(
        save_path / "drug_MINHASH_fp_2D.csv"
    )

    if get_extended_repr:
        # To obtain 11 2D molecular fingerprints with default folding size, one can use the following:
        drug_2d_fingerprint = []
        for mol in mols:
            drug_2d_fingerprint.append(Fingerprint.get_all_fps(mol))
        pd.DataFrame(drug_2d_fingerprint).to_csv(save_path / "drug_all_fp_2D.csv")

        ## 2D molecular descriptors and fingerprints
        cmp = ChemoPy(include_fps=True)
        drug_2d_desc_and_fingerprint = cmp.calculate(mols)

        # The script beneath calculates 632 two - dimensional(2D) molecular descriptors.
        drug_2d_desc_path = "2d_desc.csv"
        if os.path.exists(drug_2d_desc_path):
            drug_2d_desc = pd.read_csv(drug_2d_desc_path, index_col=0)
        else:
            drug_2d_desc = pd.DataFrame()

        cmp = ChemoPy()
        ## Obtain details for all descriptors and fingerprints
        desc_fing_details = cmp.get_details()

        for idx, mol in tqdm(enumerate(mols), desc="2D desc"):
            drug_n = drug_drug_names[idx]
            print("DRUG: ", drug_n)
            if drug_n not in drug_2d_desc.index:
                drug_desc = cmp.calculate([mol])
                if not drug_2d_desc.empty:
                    drug_2d_desc.loc[drug_n] = drug_desc.iloc[0]
                else:
                    drug_2d_desc = drug_desc
                    drug_2d_desc.index = [drug_n]
                drug_2d_desc.to_csv(drug_2d_desc_path)
                print("saved drug")
        # todo: for each molecule generate 2D & 3D molecular descriptors and save

        # The additional 552 three-dimensional (3D) molecular descriptors can be computed as follows:
        # Ensure molecules have 3D conformations
        for mol in mols:
            _ = AllChem.EmbedMolecule(mol)

        cmp = ChemoPy(ignore_3D=False)
        # print(cmp.calculate(mols))
        start = t.time()
        drug_2d_and_3d_desc = cmp.calculate(mols)
        print("time for 2d & 3d desc", t.time() - start)


def get_drug_SMILES_repr():
    datasets_name = "oneil_almanac"
    save_path = Directories.DATA_PATH / "features" / "drug_features"
    save_path.mkdir(parents=True, exist_ok=True)
    # Get info dict with drug name, DrugBank ID and inchikey for each drug in dataset
    data_path = (
        Directories.DATA_PATH / "silver" / datasets_name / f"{datasets_name}.csv"
    )
    drugs = pd.read_csv(data_path)
    if datasets_name == "oneil":
        drugs = filter_from_hetionet(drugs)
    drug_info = get_drug_info(drugs, add_SMILES=True)
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
    make_drug_fingerprint_features()
