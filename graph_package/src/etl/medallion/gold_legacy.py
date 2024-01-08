from graph_package.configs.directories import Directories
import pandas as pd
import numpy as np
import os
from graph_package.utils.helpers import init_logger
import json
from tqdm import tqdm
import requests
from .load import (
    load_oneil,
    load_drug_info_drugcomb,
    load_cell_info_drugcomb
)
from .gold import create_drug_id_vocabs, create_cell_line_id_vocabs

logger = init_logger()

def download_drug_dict_deepdds_original():
    url = "https://raw.githubusercontent.com/Sinwang404/DeepDDs/master/data/drugs_0_10.csv"
    save_path = Directories.DATA_PATH / "gold" / "oneil"
    save_path.mkdir(parents=True, exist_ok=True)
    response = requests.get(url)
    with open(save_path / "drug_dict.json", "wb") as f:
        f.write(response.content)
    df = pd.read_csv(save_path / "drug_dict.json")
    return df


def get_drug_dict_deepdds_original():
    drug_name = pd.read_csv(
        "https://raw.githubusercontent.com/Sinwang404/DeepDDs/master/data/drugs_0_10.csv",
        header=None,
    )
    drug_names_dict = {val[0]: val[1] for val in drug_name.values}
    drug_names_list = drug_names_dict.keys()
    drug_dict = load_drug_info_drugcomb()
    deepdds_drug_dict = {}
    for val in drug_dict.values():
        synonyms = val["synonyms"].split("; ")
        for synonym in synonyms:
            for drug in drug_names_list:
                synonym = synonym.lower()
                if (drug.lower() == synonym) or (drug.lower() in synonym.split(" ")):
                    assert drug not in deepdds_drug_dict
                    deepdds_drug_dict[drug_names_dict[drug]] = val["dname"]
                    del drug_names_dict[drug]
                    break
    assert len(drug_names_list) == 0
    return deepdds_drug_dict


def get_cell_line_dict_deepdds_original(df: pd.DataFrame):
    cell_line_dict = load_cell_info_drugcomb()
    cell_line_names = set(df["cell"])
    deepdds_cell_dict = {}
    for val in cell_line_dict.values():
        synonyms = val["synonyms"].split("; ")
        for synonym in synonyms:
            for cell_line in cell_line_names:
                synonym = synonym.lower()
                if (cell_line.lower() == synonym) or (
                    cell_line.lower() in synonym.split(" ")
                ):
                    assert cell_line not in deepdds_cell_dict
                    deepdds_cell_dict[cell_line] = val["name"]
                    cell_line_names.remove(cell_line)
                    break
    assert len(cell_line_names) == 0
    return deepdds_cell_dict


def make_original_deepdds_dataset():
    logger.info("Making original DeepDDS dataset.")
    # Download triplets from DeepDDS repo
    save_path = Directories.DATA_PATH / "gold" / "deepdds_original"
    save_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        "https://raw.githubusercontent.com/Sinwang404/DeepDDs/master/data/new_labels_0_10.csv"
    )
    drug_dict_deepdds = get_drug_dict_deepdds_original()
    cell_line_dict_deepdds = get_cell_line_dict_deepdds_original(df)

    df["drug_1_name"] = df["drug1"].map(drug_dict_deepdds)
    df["drug_2_name"] = df["drug2"].map(drug_dict_deepdds)
    df["context"] = df["cell"].map(cell_line_dict_deepdds)

    df, drug_vocab = create_drug_id_vocabs(df)
    df, cell_line_vocab = create_cell_line_id_vocabs(df)
    for vocab, name in zip(
        (drug_vocab, cell_line_vocab), ["entity_vocab.json", "relation_vocab.json"]
    ):
        with open(save_path / name, "w") as json_file:
            json.dump(vocab, json_file)

    df = df[
        [
            "drug_1_name",
            "drug_2_name",
            "label",
            "context",
            "drug_1_id",
            "drug_2_id",
            "context_id",
        ]
    ]
    assert df.isna().sum().sum() == 0
    df.to_csv(
        Directories.DATA_PATH / "gold" / "deepdds_original" / "deepdds_original.csv",
        index=False,
    )


def make_oneil_legacy_dataset():
    logger.info("Making Oneil legacy dataset.")
    save_path = Directories.DATA_PATH / "gold" / "oneil_legacy"
    save_path.mkdir(parents=True, exist_ok=True)
    df = load_oneil()
    rename_dict = {
        "drug_row": "drug_1_name",
        "drug_col": "drug_2_name",
        "synergy_loewe": "label",
        "cell_line_name": "context",
    }
    df = df[rename_dict.keys()]
    df.rename(columns=rename_dict, inplace=True)

    df, drug_vocab = create_drug_id_vocabs(df)
    df, cell_line_vocab = create_cell_line_id_vocabs(df)

    for vocab, name in zip(
        (drug_vocab, cell_line_vocab), ["entity_vocab.json", "relation_vocab.json"]
    ):
        with open(save_path / name, "w") as json_file:
            json.dump(vocab, json_file)

    df.to_csv(save_path / "oneil.csv", index=False)