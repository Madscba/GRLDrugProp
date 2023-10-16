from graph_package.configs.definitions import Directories
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


def download_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "summary_v_1_5.csv"
    if not data_path.exists():
        logger.info("Downloading DrugComb dataset.")
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


async def download_info_drugcomb(
    type: str = "drugs",
    base_url: str = "https://api.drugcomb.org/drugs",
    n_entities: int = 8397,
    file_name: str = "drug_dict.json",
):
    drug_dict = {}
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(1, n_entities + 1):
            url = f"{base_url}/{i}"
            task = asyncio.ensure_future(download_info(session, url))
            tasks.append(task)
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                info_dict = await task
                del info_dict["synonyms"]
                if type == "drugs":
                    name = info_dict["dname"]
                else:
                    name = info_dict["name"]
                drug_dict[name] = info_dict
            except:
                logger.error(f"Failed to download information for drug")
    path = Directories.DATA_PATH / "bronze" / "drugcomb" / file_name
    with open(path, "w") as f:
        json.dump(drug_dict, f)


async def download_info(session, url):
    async with session.get(url) as response:
        return await response.json()


def download_cell_line_info_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb"
    if not (data_path / "cell_line_dict.json").exists():
        logger.info("Downloading cell-line info from DrugComb API.")
        asyncio.run(
            download_info_drugcomb(
                type="cell_lines",
                base_url="https://api.drugcomb.org/cell_lines",
                n_entities=2320,
                file_name="cell_line_dict.json",
            )
        )


def download_drug_info_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb"
    if not (data_path / "drug_dict.json").exists():
        logger.info("Downloading drug info from DrugComb API.")
        asyncio.run(
            download_info_drugcomb(
                type="drugs",
                base_url="https://api.drugcomb.org/drugs",
                n_entities=8397,
                file_name="drug_dict.json",
            )
        )



def get_drugcomb():
    download_drugcomb()
    download_drug_info_drugcomb()
    download_cell_line_info_drugcomb()

if __name__ == "__main__":
    get_drugcomb()
