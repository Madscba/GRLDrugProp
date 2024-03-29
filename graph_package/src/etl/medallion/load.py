from graph_package.configs.directories import Directories
import pandas as pd
import numpy as np
import os
from graph_package.utils.helpers import init_logger
from graph_package.src.etl.feature_engineering.cell_line_features import (
    make_cell_line_features,
)
import json
import requests
import re
from tqdm import tqdm
import jsonlines

logger = init_logger()

def load_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "summary_v_1_5.csv"
    return pd.read_csv(data_path)


def load_silver_csv(study):
    data_path = Directories.DATA_PATH / "silver" / study/ f"{study}.csv"
    return pd.read_csv(data_path)


def load_drug_info_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "drug_dict.json"
    with open(data_path) as f:
        drug_dict = json.load(f)
    return drug_dict


def load_cell_info_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "cell_line_dict.json"
    with open(data_path) as f:
        cell_line_dict = json.load(f)
    return cell_line_dict


def load_block_as_df(study_name: str):
    data_path = Directories.DATA_PATH / "silver" / study_name / "block_dict.json"
    block_dict_json = load_jsonl(data_path)
    df_block = pd.DataFrame(block_dict_json)
    df_block = df_block.loc[:, ["conc_r", "conc_c", "inhibition", "block_id"]]
    return df_block


def load_mono_response(study_name: str):
    data_path = Directories.DATA_PATH / "gold" / study_name / "mono_response.csv"
    df_mono_response = pd.read_csv(data_path, index_col=[0, 1])
    return df_mono_response


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
            data.append(item["block_id"])
    return data
