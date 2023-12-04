from graph_package.configs.directories import Directories
from graph_package.utils.helpers import init_logger
import pandas as pd
import asyncio
import shutil
import jsonlines
import json
import os
from graph_package.src.etl.bronze import download_response_info_drugcomb, load_jsonl, load_block_ids

logger = init_logger()

def load_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "summary_v_1_5.csv"
    return pd.read_csv(data_path)

def generate_oneil_dataset():
    """
    Generate the Oniel dataset from the DrugComb dataset.
    """
    df = load_drugcomb()
    df_oneil = df[df["study_name"] == "ONEIL"]
    df_oneil_cleaned = df_oneil.dropna(subset=["drug_row", "drug_col", "synergy_loewe"])   
    unique_block_ids = df_oneil_cleaned["block_id"].unique().tolist()
    download_response_info_oneil(unique_block_ids,overwrite=False)
    oneil_path = Directories.DATA_PATH / "silver" / "oneil"
    oneil_path.mkdir(exist_ok=True,parents=True)
    df_oneil_cleaned = df_oneil_cleaned.loc[
        :, ~df_oneil_cleaned.columns.str.startswith("Unnamed")
    ]
    df_oneil_cleaned.to_csv(oneil_path / "oneil.csv", index=False)


def download_response_info_oneil(list_entities, overwrite=False):
    """Download response information from DrugComb API. This is in silver
    since downloading it for all of DrugComb take up too much space."""
    data_path = Directories.DATA_PATH / "silver" / "oneil"
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
    generate_oneil_dataset()

    
