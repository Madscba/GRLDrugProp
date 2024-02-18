from graph_package.configs.directories import Directories
from graph_package.src.etl.medallion.gold import get_drug_info
import pandas as pd
from sklearn.decomposition import PCA
from rdkit import Chem
from chemopy import Fingerprint
from rdkit.Chem import Descriptors
from pathlib import Path
from e3fp.conformer.util import smiles_to_dict
from e3fp.config.params import default_params
from e3fp.pipeline import params_to_dicts,fprints_from_smiles, fprints_from_mol, confs_from_smiles
from python_utilities.parallel import Parallelizer
from tqdm.notebook import tqdm
from itertools import islice
import os
from python_utilities.parallel import Parallelizer
import pickle
import json
from glob import glob
import numpy as np

def get_feature_path() -> Path:
    path_to_drug_feature_folder = Directories.DATA_PATH / "features" / "drug_features"
    path_to_drug_feature_folder.mkdir(parents=True, exist_ok=True)
    return path_to_drug_feature_folder


def make_drug_fingerprint_features(get_extended_repr: bool = False, dim=None):
    """Function takes drugs from oneil_almanack

    We could consider a more advanced fingerprint method:
    Smiles string as input
    Returns:

    """
    drug_SMILES, drug_names = get_drug_SMILES_repr()
    save_path = get_feature_path()

    mols = get_molecules_from_SMILES(drug_SMILES)

    generate_and_save_morgan_fp(dim, drug_names, mols, save_path)

    generate_and_save_minhash_fp(dim, drug_names, mols, save_path)

    generate_and_save_maccs_fp(drug_names, mols, save_path)

    generate_and_save_rdkit_descriptor(drug_names, mols, save_path)

    generate_and_save_e3fp_3d_fp(drug_SMILES, drug_names, save_path) #can take hours to run!
    #
    # if get_extended_repr:
    #     # To obtain 11 2D molecular fingerprints with default folding size, one can use the following:
    #     drug_2d_fingerprint = []
    #     for mol in mols:
    #         drug_2d_fingerprint.append(Fingerprint.get_all_fps(mol))
    #     pd.DataFrame(drug_2d_fingerprint, index=drug_names).to_csv(
    #         save_path / "drug_all_fp_2D.csv"
    #     )


def generate_and_save_e3fp_3d_fp(drug_SMILES, drug_names, save_path):
    confgen_params, fprint_params = params_to_dicts(default_params)
    del confgen_params['protonate']
    del confgen_params['standardise']
    fprint_params['include_disconnected'] = True
    fprint_params['stereo'] = False
    fprint_params['first'] = 1
    fprint_params['bits'] = 512
    confgen_params['first'] = 20

    drug_SMILES_ = [(sm.split(";")[0], d_name) for sm, d_name in zip(drug_SMILES, drug_names)]
    kwargs = {"confgen_params": confgen_params, "fprint_params": fprint_params}
    parallelizer = Parallelizer(parallel_mode="processes", num_proc=3)
    fprints_list = parallelizer.run(fprints_from_smiles, drug_SMILES_, kwargs=kwargs)
    representation = [np.zeros((fprint_params['bits'])) for i in range(len(fprints_list))]
    fprints_ = [(i, fprints_list[i][0][0].indices) if fprints_list[i][0] else (i, np.array([])) for i in
                range(len(fprints_list))]
    drugs_without_repr = 0
    for i, fprint in fprints_:
        if fprint.size > 0:
            representation[i][fprint] = 1
        else:
            try:
                mol = confs_from_smiles(drug_SMILES[i], drug_names[i], confgen_params=confgen_params)
                tmp_fprint = fprints_from_mol(mol, fprint_params=fprint_params)[0].indices
                representation[i][tmp_fprint] = 1
            except:
                drugs_without_repr += 1
                print("no 3D repr found for", drug_names[i])
                pass
    print("Drugs without E3FP repr", drugs_without_repr)


    pd.DataFrame(representation, index=drug_names).to_csv(save_path / "drug_E3FP_fp_3D.csv")


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
        save_path / morgan_fpath)



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
    datasets_names = ["oneil_almanac", "drugcomb"]
    save_path = Directories.DATA_PATH / "features" / "drug_features"
    save_path.mkdir(parents=True, exist_ok=True)
    # Get info dict with drug name, DrugBank ID and inchikey for each drug in dataset

    data_paths = [Directories.DATA_PATH / "silver" / datasets_names[i] / f"{datasets_names[i]}.csv" for i in range(len(datasets_names))]

    drugs = pd.concat([pd.read_csv(data_paths[i]) for i in range(len(data_paths))])
    unique_drug_names = set(drugs["drug_row"]).union(set(drugs["drug_col"]))
    drug_info = get_drug_info(drugs, unique_drug_names, add_SMILES=True)
    df_drug_info = pd.DataFrame(drug_info).T.reset_index()
    df_drug_info = df_drug_info.rename(columns={"index": "drug_name"})
    drug_SMILES = [d.split(";")[0] for d in df_drug_info["SMILES"]]
    drug_names = list(df_drug_info["drug_name"])
    #remove drugs without SMILES and drug names
    drug_names = [drug_names[i] for i in range(len(drug_names)) if drug_SMILES[i] not in ("NULL", 'Antibody (MEDI3622)')]
    drug_SMILES = [drug_SMILES[i] for i in range(len(drug_SMILES)) if drug_SMILES[i] not in ("NULL", 'Antibody (MEDI3622)')]
    return drug_SMILES, drug_names


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

def make_e3fp_3d_pca():
    file_path = Directories.DATA_PATH / "features" / "drug_features" / "drug_E3FP_fp_3D.csv"
    drug_comb_drugs = json.load(open('data/gold/drugcomb_filtered/entity_vocab.json','r'))
    drug_descriptors = pd.read_csv(file_path,index_col=0)
    drug_descriptors = drug_descriptors.loc[list(drug_comb_drugs.keys())]  
    components = len(drug_descriptors.index)
    # Generate PCA feature vectors

    drug_node_array = np.array(drug_descriptors)
    pca = PCA(n_components=min(components, drug_node_array.shape[1]))
    pca_feature_vectors = pca.fit_transform(drug_node_array)
    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    # Calculate the accumulated explained variance
    accumulated_explained_variance = np.cumsum(explained_variance_ratio)
    print(f"Acc explained variance for {components} cell line pca features: {accumulated_explained_variance[-1]}"
    )
    pd.DataFrame(pca_feature_vectors,index=drug_descriptors.index).to_csv(file_path.parent / f"drug_E3FP_fp_3D_PCA_{components}.csv")

if __name__ == "__main__":
    #make_drug_fingerprint_features()
    make_e3fp_3d_pca()
