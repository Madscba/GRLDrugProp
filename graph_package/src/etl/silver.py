from graph_package.configs.directories import Directories
from graph_package.utils.helpers import init_logger
import pandas as pd
import asyncio
import shutil
import jsonlines
import json
import os
from graph_package.src.etl.bronze import download_response_info_drugcomb, load_jsonl, load_block_ids
from graph_package.src.node_features.utils import filter_drugs_in_graph

logger = init_logger()

def load_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "summary_v_1_5.csv"
    return pd.read_csv(data_path)

def partial_name_match(filtered_drug_dict):
    # Check for different name conventions and edge cases
    filtered_drugs = [drug for drug in filtered_drug_dict.keys()]
    fd_lower = [d.lower() for d in filtered_drugs]
    fd_cap = [d.capitalize() for d in filtered_drugs]
    fd_title = [d.title() for d in filtered_drugs]
    final_drugs = filtered_drugs+fd_lower+fd_cap+fd_title+['5-Aminolevulinic acid hydrochloride']
    return final_drugs

def filter_from_hetionet(studies):
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "summary_v_1_5.csv"
    dict_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "drug_dict.json"

    # Load drug dict from DrugComb with metadata on drugs
    with open(dict_path) as f:
        drug_dict = json.load(f)

    # Load ONEIL-ALMANAC dataset 
    drugs = pd.read_csv(data_path)

    drugs = drugs[drugs['study_name'].isin(studies)]
    unique_drug_names = list(set(drugs['drug_row'].tolist() + drugs['drug_col'].tolist()))
    # Make info dict with drug name, DrugBank ID and inchikey for each drug
    drug_info = {}
    nans = []
    for drug in unique_drug_names:
        # Check for different name conventions and edge cases
        drug_name = drug.capitalize() if not drug[0].isupper() else drug
        drug_name = drug_name.title() if not drug_name[0].isalpha() else drug_name
        if drug == 'mitomycin C':
            drug_name = 'mitomycin C'
        drug_info[drug_name] = {}
        drug_info[drug_name]['synonyms'] = drug_dict[drug]['synonyms'].split(";")
        drug_info[drug_name]['inchikey'] = drug_dict[drug]['inchikey'] 
        drug_info[drug_name]['DB'] = drug_dict[drug]['drugbank_id']

    # Find drugs in Hetionet
    _, filtered_drug_dict, _ = filter_drugs_in_graph(drug_info)

    # Filter drugs from ONEIL-ALMANAC that exists in Hetionet
    potential_drug_names = partial_name_match(filtered_drug_dict)
    filtered_df = drugs[
        drugs['drug_row'].isin(potential_drug_names) 
        & drugs['drug_col'].isin(potential_drug_names)
    ]
    unique_final_names = list(set(filtered_df['drug_row'].tolist() + filtered_df['drug_col'].tolist()))
    print("Num drugs: ", len(unique_final_names))
    print("Num triplets:", filtered_df.shape[0])
    return filtered_df

def generate_study_dataset(studies=["ONEIL"]):
    """
    Generate the ONEIL and/or ALMANAC dataset from the DrugComb dataset.
    """
    study_names = "oneil_almanac" if len(studies)>1 else studies[0].lower()
    df = load_drugcomb()
    df_study = df[df["study_name"].isin(studies)]
    print("Num triplets:", df_study.shape[0])
    df_study_cleaned = df_study.dropna(subset=["drug_row", "drug_col", "synergy_zip"])
    df_study_cleaned = filter_from_hetionet(studies)
    print("Num cell lines:", df_study_cleaned.cell_line_name.nunique())
    unique_block_ids = df_study_cleaned["block_id"].unique().tolist()
    download_response_info(unique_block_ids,study_names=study_names,overwrite=False)
    study_path = Directories.DATA_PATH / "silver" / study_names
    study_path.mkdir(exist_ok=True,parents=True)
    df_study_cleaned = df_study_cleaned.loc[
        :, ~df_study_cleaned.columns.str.startswith("Unnamed")
    ]
    df_study_cleaned.to_csv(study_path / f"{study_names}.csv", index=False)

def generate_oneil_dataset():
    """
    Generate the Oniel dataset from the DrugComb dataset.
    """
    df = load_drugcomb()
    df_oneil = df[df["study_name"] == "ONEIL"]
    df_oneil_cleaned = df_oneil.dropna(subset=["drug_row", "drug_col", "synergy_zip"])   
    unique_drug_names = list(set(df_oneil_cleaned['drug_row'].tolist() + df_oneil_cleaned['drug_col'].tolist()))
    print("Num drugs: ", len(unique_drug_names))
    print("Num triplets:", df_oneil_cleaned.shape[0])
    print("Num cell lines:", df_oneil_cleaned.cell_line_name.nunique())
    unique_block_ids = df_oneil_cleaned["block_id"].unique().tolist()
    download_response_info(unique_block_ids,overwrite=False)
    oneil_path = Directories.DATA_PATH / "silver" / "oneil"
    oneil_path.mkdir(exist_ok=True,parents=True)
    df_oneil_cleaned = df_oneil_cleaned.loc[
        :, ~df_oneil_cleaned.columns.str.startswith("Unnamed")
    ]
    df_oneil_cleaned.to_csv(oneil_path / "oneil.csv", index=False)


def download_response_info(list_entities, study_names = "oneil", overwrite=False):
    """Download response information from DrugComb API. This is in silver
    since downloading it for all of DrugComb take up too much space."""
    data_path = Directories.DATA_PATH / "silver" / study_names
    data_path.mkdir(exist_ok=True,parents=True)
    if (not (data_path / "block_dict.json").exists()) | overwrite:
        if (data_path / "block_dict.json").exists():
              os.remove(data_path / "block_dict.json")
        while list_entities:
            logger.info("Downloading block info from DrugComb API.")
            asyncio.run(
                download_response_info_drugcomb(
                    path=data_path,
                    type="blocks",
                    base_url="https://api.drugcomb.org/response",
                    list_entities=list_entities,
                    file_name="block_dict.json"
                )
            )  
            # the server times out when making this many requests so we have to do in chunks
            entities_downloaded = set(load_block_ids(data_path / "block_dict.json"))
            list_entities = list(set(list_entities)- entities_downloaded)
            if list_entities:
                logger.info(f"Retrying for {len(list_entities)} entities.")



if __name__ == "__main__":
    #generate_study_dataset(studies=["ONEIL"])
    generate_study_dataset(studies=["ONEIL","ALMANAC"])
    generate_oneil_dataset()

    
