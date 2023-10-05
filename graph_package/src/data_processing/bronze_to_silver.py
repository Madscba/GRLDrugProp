from graph_package.configs.directories import Directories
import pandas as pd
import numpy as np
from graph_package.utils.helpers import init_logger
from collections import defaultdict
import requests
import asyncio
import aiohttp
from tqdm import tqdm
import json

logger = init_logger()

def download_drugcomb(data_path: str):
    url = "https://drugcomb.fimm.fi/jing/summary_v_1_5.csv"
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(data_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            progress_bar.update(len(chunk))
            f.write(chunk)
    progress_bar.close()

async def download_drug_info_drugcomb():
    drug_dict = {}
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(1, 8398):
            base_url = "https://api.drugcomb.org/drugs"
            url = f"{base_url}/{i}"
            task = asyncio.ensure_future(download_drug(session, url))
            tasks.append(task)
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                block_dict = await task
                del block_dict["synonyms"]
                drug_dict[block_dict["dname"]] = block_dict
            except:
                logger.error(f"Failed to download information for drug")
    path = Directories.DATA_PATH / "bronze" / "drugcomb" / "drug_dict.json"
    with open(path, "w") as f:
        json.dump(drug_dict, f)

async def download_drug(session, url):
    async with session.get(url) as response:
        return await response.json()


def load_drugcomb():
    data_path = Directories.DATA_PATH / "bronze"/  "drugcomb" / "summary_v_1_5.csv"
    if not data_path.exists():
        logger.info("Downloading DrugComb dataset.")
        download_drugcomb(data_path)
    return pd.read_csv(data_path)

def load_drug_info_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "drug_dict.json"
    if not data_path.exists():
        logger.info("Downloading drug info from DrugComb API.")
        asyncio.run(download_drug_info_drugcomb())
    with open(data_path) as f:
        drug_dict = json.load(f)
    return drug_dict

def get_CIDs(df: pd.DataFrame, dataset: str = "oneil"):
    """
    Get CIDs and smile strings for drugs.
    """
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "drug_dict.json"
    drug_dict = load_drug_info_drugcomb()
    df["drug_row_cid"] = df["drug_row"].apply(lambda x: drug_dict[x]['cid'])
    df["drug_col_cid"] = df["drug_col"].apply(lambda x: drug_dict[x]['cid'])

    return df


def generate_oneil_dataset():
    """
    Generate the Oniel dataset from the DrugComb dataset.
    """
    df = load_drugcomb()

    df_oneil = df[df["study_name"]== "ONEIL"]
    df_oneil_cleaned = df_oneil.dropna(subset=["drug_row", "drug_col", "synergy_loewe"])
    logger.info(f"Dropped {len(df_oneil)-len(df_oneil_cleaned)} NaN values.")
    df_oneil_cleaned = get_CIDs(df_oneil_cleaned)
    oneil_path = Directories.DATA_PATH / "silver" / "oneil"
    oneil_path.mkdir(exist_ok=True)
    df_oneil_cleaned = df_oneil_cleaned.loc[:, ~df_oneil_cleaned.columns.str.startswith('Unnamed')]
    df_oneil_cleaned.to_csv(oneil_path / "oneil.csv",index=False)    

    
if __name__ == "__main__":
    print("hej")
    generate_oneil_dataset() 


    