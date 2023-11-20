from graph_package.configs.directories import Directories
import pandas as pd
import numpy as np
from graph_package.utils.helpers import init_logger
import requests
import asyncio
import aiohttp
from tqdm import tqdm
from pathlib import Path
import jsonlines
import json
import ssl

logger = init_logger()

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def write_jsonl(file_path, data):
    with jsonlines.open(file_path, mode='w') as writer:
        for item in data:
            writer.write(item)


def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for item in reader:
            data.append(item)
    return data

def load_block_ids(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for item in reader:
            data.append(item['block_id'])
    return data

def download_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "summary_v_1_5.csv"
    if not data_path.exists():
        Path(Directories.DATA_PATH / "bronze" / "drugcomb").mkdir(
            exist_ok=True, parents=True
        )
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
    path: Path = Directories.DATA_PATH / "bronze" / "drugcomb",
    type: str = "drugs",
    base_url: str = "https://api.drugcomb.org/drugs",
    list_entities: int = 8397,
    file_name: str = "drug_dict.json",
):
    path = path / file_name
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        tasks = []
        drug_dict = {}
        failure_counts = 0
        for i in list_entities:
            url = f"{base_url}/{i}"
            task = asyncio.ensure_future(download_info(session, url))
            tasks.append(task)
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                info_dict = await task
                if type == "drugs":
                    name = info_dict["dname"]
                elif type == "cell_lines":
                    name = info_dict["name"]
                drug_dict[name] = info_dict
            except:
                failure_counts +=1 
                logger.error(f"Failed to download information for {type} with id {i}")
        with open(path, "w") as f:
            json.dump(drug_dict, f)


async def download_response_info_drugcomb(
    path: Path = Directories.DATA_PATH / "bronze" / "drugcomb",
    type: str = "drugs",
    base_url: str = "https://api.drugcomb.org/drugs",
    list_entities: int = 8397,
    file_name: str = "drug_dict.json",
):
    path = path / file_name
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        tasks = []
        failure_counts = 0
        for i in list_entities:
            url = f"{base_url}/{i}"
            task = asyncio.ensure_future(download_info(session, url))
            tasks.append(task)
        with jsonlines.open(path, mode='a') as writer:
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                try:
                    response_list = await task
                    for val in response_list:
                        writer.write(val)
                except:
                    failure_counts +=1 
                    logger.error(f"Failed to download information for {type} with id {i}")
                if failure_counts > 10:
                    logger.error(f"Failed to download information for {type} with id {i} more than 10 times. Stopping.")
                    break

async def download_info(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await asyncio.wait_for(response.json(), timeout=1000000)


def download_cell_line_info_drugcomb(overwrite=False):
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb"
    if (not (data_path / "cell_line_dict.json").exists()) | overwrite:
        logger.info("Downloading cell-line info from DrugComb API.")
        asyncio.run(
            download_info_drugcomb(
                type="cell_lines",
                base_url="https://api.drugcomb.org/cell_lines",
                list_entities=range(1,2321),
                file_name="cell_line_dict.json",
            )
        )



def download_drug_info_drugcomb(overwrite=True):
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb"
    if (not (data_path / "drug_dict.json").exists()) | overwrite:
        logger.info("Downloading drug info from DrugComb API.")
        asyncio.run(
            download_info_drugcomb(
                type="drugs",
                base_url="https://api.drugcomb.org/drugs",
                list_entities=range(1,8398),
                file_name="drug_dict.json",
            )
        )



def get_drugcomb(over):
    download_drugcomb()
    download_drug_info_drugcomb(overwrite=False)
    download_cell_line_info_drugcomb(overwrite=False)



if __name__ == "__main__":
    get_drugcomb()
