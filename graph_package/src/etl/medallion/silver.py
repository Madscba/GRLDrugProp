from graph_package.configs.directories import Directories
from graph_package.utils.helpers import logger
import pandas as pd
import asyncio
import json
import os
import requests
import bz2
from graph_package.src.etl.medallion.bronze import (
    download_response_info_drugcomb,
    load_block_ids,
)


def load_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "summary_v_1_5.csv"
    return pd.read_csv(data_path)


def partial_name_match(filtered_drug_dict):
    # Check for different name conventions and edge cases
    filtered_drugs = [drug for drug in filtered_drug_dict.keys()]
    fd_lower = [d.lower() for d in filtered_drugs]
    fd_cap = [d.capitalize() for d in filtered_drugs]
    fd_title = [d.title() for d in filtered_drugs]
    final_drugs = (
        filtered_drugs
        + fd_lower
        + fd_cap
        + fd_title
        + ["5-Aminolevulinic acid hydrochloride"]
    )
    return final_drugs


def download_hetionet(data_path):
    url = "https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/json/hetionet-v1.0.json.bz2?download=true"
    response = requests.get(url)
    if response.status_code == 200:
        logger.info("Downloading Hetionet json file from GitHub..")
        # Decompress the content
        decompressed_content = bz2.decompress(response.content)

        # Decode bytes to string
        decompressed_content_str = decompressed_content.decode("utf-8")

        # Save the decompressed content to a file
        with open(data_path / "hetionet-v1.0.json", "w") as file:
            file.write(decompressed_content_str)
    else:
        logger.info(
            f"Failed to download Hetionet json file. Status code: {response.status_code}"
        )


def filter_drugs_in_graph(drug_info):
    """
    Function for filtering drugs in DrugComb found in Hetionet
    """

    # Load Hetionet from json
    data_path = Directories.DATA_PATH / "hetionet"
    data_path.mkdir(exist_ok=True, parents=True)
    if not os.path.exists(data_path / "hetionet-v1.0.json"):
        download_hetionet(data_path)
    with open(data_path / "hetionet-v1.0.json") as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Select subset of drug IDs corresponding to drug IDs that exist in the graph
    drugs_in_graph = {}
    for node in nodes:
        if node["kind"] == "Compound":
            drugs_in_graph[node["name"]] = {}
            drugs_in_graph[node["name"]]["inchikey"] = node["data"]["inchikey"][9:]
            drugs_in_graph[node["name"]]["DB"] = node["identifier"]

    # Check each drug in drug_info and add it to filtered dict if found in drugs_in_graph
    filtered_drug_dict = {}
    logger.info("Filtering drugs in Hetionet..")
    for drug_name, identifiers in drug_info.items():
        for graph_drug_name, graph_identifiers in drugs_in_graph.items():
            # Match on drug names
            if (
                graph_drug_name.lower() in drug_name.lower()
                or graph_drug_name in drug_name.title()
            ):
                filtered_drug_dict[drug_name] = graph_identifiers
            else:
                for identifier_key, identifier_value in identifiers.items():
                    # Match on drug synonyms
                    if identifier_key == "synonyms":
                        if graph_drug_name in identifier_value:
                            filtered_drug_dict[drug_name] = graph_identifiers
                    # Match on drugbank ID or inchikey
                    elif graph_identifiers.get(identifier_key) == identifier_value:
                        filtered_drug_dict[drug_name] = graph_identifiers

    drug_ids = [filtered_drug_dict[drug]["DB"] for drug in filtered_drug_dict.keys()]
    logger.info(f"{len(drug_ids)} of {len(drug_info)} drugs found in Hetionet")
    drug_edges = [
        e
        for e in edges
        if (e["target_id"][0] == "Compound") or (e["source_id"][0] == "Compound")
    ]
    return drug_ids, filtered_drug_dict, drug_edges


def get_drug_info(drugs: pd.DataFrame, add_SMILES: bool = False):
    """
    Filter drugs from DrugComb to match drugs in Hetionet
    """
    dict_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "drug_dict.json"

    # Load drug dict from DrugComb with metadata on drugs
    with open(dict_path) as f:
        drug_dict = json.load(f)

    unique_drug_names = sorted(
        list(set(drugs["drug_row"].tolist() + drugs["drug_col"].tolist()))
    )
    # Make info dict with drug name, DrugBank ID and inchikey for each drug
    drug_info = {}
    for drug in unique_drug_names:
        if add_SMILES:
            drug_info[drug] = {}
            drug_info[drug]["SMILES"] = drug_dict[drug]["smiles"]
        else:
            # Check for different name conventions and edge cases
            drug_name_v1 = drug.capitalize() if not drug[0].isupper() else drug
            drug_name_v2 = drug_name_v1.title() if not drug_name_v1[0].isalpha() else drug
            drug_info[drug] = {}
            drug_info[drug]["synonyms"] = drug_dict[drug]["synonyms"].split(";") + [drug_name_v1, drug_name_v2]
            drug_info[drug]["inchikey"] = drug_dict[drug]["inchikey"]
            drug_info[drug]["DB"] = drug_dict[drug]["drugbank_id"]

    return drug_info


def filter_from_hetionet(drugs):
    # Get drug info
    drug_info = get_drug_info(drugs)

    # Find drugs in Hetionet
    _, filtered_drug_dict, _ = filter_drugs_in_graph(drug_info)

    # Filter drugs from ONEIL-ALMANAC that exists in Hetionet
    potential_drug_names = partial_name_match(filtered_drug_dict)
    filtered_df = drugs[
        drugs["drug_row"].isin(potential_drug_names)
        & drugs["drug_col"].isin(potential_drug_names)
    ]
    return filtered_df


def generate_oneil_almanac_dataset(studies=["oneil", "oneil_almanac"]):
    """
    Generate the ONEIL and ONEIL-ALMANAC dataset from the DrugComb dataset.
    """
    df = load_drugcomb()
    for study in studies:
        study_path = Directories.DATA_PATH / "silver" / study
        study_path.mkdir(exist_ok=True, parents=True)
        study_names = ["ONEIL", "ALMANAC"] if study == "oneil_almanac" else ["ONEIL"]
        df_study = df[df["study_name"].isin(study_names)]
        df_study_cleaned = df_study.dropna(
            subset=["drug_row", "drug_col", "synergy_zip"]
        )
        if study == "oneil_almanac":
            df_study_cleaned = filter_from_hetionet(df_study_cleaned)
        unique_block_ids = df_study_cleaned["block_id"].unique().tolist()
        download_response_info(unique_block_ids, study, overwrite=False)
        df_study_cleaned = df_study_cleaned.loc[
            :, ~df_study_cleaned.columns.str.startswith("Unnamed")
        ]
        df_study_cleaned.to_csv(study_path / f"{study}.csv", index=False)


def download_response_info(list_entities, study_names="oneil", overwrite=False):
    """Download response information from DrugComb API. This is in silver
    since downloading it for all of DrugComb take up too much space."""
    data_path = Directories.DATA_PATH / "silver" / study_names
    data_path.mkdir(exist_ok=True, parents=True)
    if (not (data_path / "block_dict.json").exists()) | overwrite:
        if (data_path / "block_dict.json").exists():
            os.remove(data_path / "block_dict.json")
        (data_path / "block_dict.json").touch()
        while list_entities:
            logger.info("Downloading block info from DrugComb API.")
            asyncio.run(
                download_response_info_drugcomb(
                    path=data_path,
                    type="blocks",
                    base_url="https://api.drugcomb.org/response",
                    list_entities=list_entities,
                    file_name="block_dict.json",
                )
            )
            # the server times out when making this many requests so we have to do in chunks
            entities_downloaded = set(load_block_ids(data_path / "block_dict.json"))
            list_entities = list(set(list_entities) - entities_downloaded)
            if list_entities:
                logger.info(f"Retrying for {len(list_entities)} entities.")


if __name__ == "__main__":
    generate_oneil_almanac_dataset(studies=["oneil_almanac"])
