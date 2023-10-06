from graph_package.configs.directories import Directories
import pandas as pd
import numpy as np
from graph_package.utils.helpers import init_logger
from collections import defaultdict
import requests
import json


def load_oneil():
    data_path = Directories.DATA_PATH / "silver" / "oneil" / "oneil.csv"
    return pd.read_csv(data_path)

def load_drug_info_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "drug_dict.json"
    with open(data_path) as f:
        drug_dict = json.load(f)
    return drug_dict

def get_CIDs(df: pd.DataFrame, dataset: str = "oneil"):
    """
    Get CIDs and smile strings for drugs.
    """
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "drug_dict.json"
    drug_dict = load_drug_info_drugcomb()
    df["drug_row_cid"] = df["drug_row"].apply(lambda x: drug_dict[x]["cid"])
    df["drug_col_cid"] = df["drug_col"].apply(lambda x: drug_dict[x]["cid"])
    return df


def make_triplets_oneil_chemicalx():
    save_path = Directories.DATA_PATH / "gold" / "chemicalx" / "oneil" / "oneil.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_oneil()
    df = get_CIDs(df)
    columns_to_keep = ["drug_row_cid", "drug_col_cid", "cell_line_name", "synergy_loewe"]
    df = df[columns_to_keep]
    df = df.groupby(["drug_row_cid", "drug_col_cid", "cell_line_name"]).mean().reset_index()
    df["synergy_loewe"] = df["synergy_loewe"].apply(
        lambda x: 0 if x < 0 else (1 if x > 10 else pd.NA)
    )
    df.dropna(subset=["synergy_loewe"], inplace=True)
    rename_dict = {
        "drug_row_cid": "drug_1",
        "drug_col_cid": "drug_2",
        "synergy_loewe": "label",
        "cell_line_name": "context",
    }
    df.rename(columns=rename_dict, inplace=True)
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    make_triplets_oneil_chemicalx()