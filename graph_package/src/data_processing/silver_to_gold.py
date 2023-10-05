from graph_package.configs.directories import Directories
import pandas as pd
import numpy as np
from graph_package.utils.helpers import init_logger
from collections import defaultdict
import requests


def load_oneil():
    data_path = Directories.DATA_PATH / "silver" / "oneil" / "oneil.csv"
    return pd.read_csv(data_path)


def make_triplets_oneil():
    save_path = Directories.DATA_PATH / "gold" / "oneil" / "oneil.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_oneil()
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
    print("hello")
    make_triplets_oneil()