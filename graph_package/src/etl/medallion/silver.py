from graph_package.configs.directories import Directories
from graph_package.utils.helpers import logger
import pandas as pd
import asyncio
import json
import os
from graph_package.src.etl.medallion.bronze import download_response_info_drugcomb
from graph_package.src.etl.medallion.load import load_block_ids, load_drugcomb


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
        unique_block_ids = df_study_cleaned["block_id"].unique().tolist()
        download_response_info(unique_block_ids, study, overwrite=True)
        df_study_cleaned = df_study_cleaned.loc[
            :, ~df_study_cleaned.columns.str.startswith("Unnamed")
        ]
        df_study_cleaned.to_csv(study_path / f"{study}.csv", index=False)

def generate_rest_of_drugcomb_dataset():
    """
    Generate the DrugComb dataset which does not include the ONEIL and ALMANAC studies.
    """
    df = load_drugcomb()
    # Remove mono-studies
    df = df.dropna(subset=["drug_col"])
    # Remove non-cancer studies
    non_cancer_studies = ["MOTT", "NCATS_SARS-COV-2DPI", "BOBROWSKI", "DYALL"]
    cancer_studies = set(df.study_name.unique()).difference(non_cancer_studies)
    df = df[df["study_name"].isin(cancer_studies)]
    df = df.dropna(
        subset=["drug_row", "drug_col", "synergy_zip"]
    )
    # Remove ONEIL & ALMANAC and generate block-dict for remaining 
    df = df[df["study_name"].isin(cancer_studies-set(["ONEIL", "ALMANAC"]))]
    study = "drugcomb"
    study_path = Directories.DATA_PATH / "silver" / study
    study_path.mkdir(exist_ok=True, parents=True)
    unique_block_ids = df["block_id"].unique().tolist()
    download_response_info(unique_block_ids, study, overwrite=True)
    df_study_cleaned = df.loc[
        :, ~df.columns.str.startswith("Unnamed")
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
