#import sys
#sys.path.insert(0,'/Users/johannesreiche/Library/Mobile Documents/com~apple~CloudDocs/DTU/MMC/Thesis/Code/GRLDrugProp')
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

def create_vocab(df: pd.DataFrame, subset: list, save_path: str = ''):
    """
    Create json file with CID / cell-line name to entity / relation ID 
    """
    sub_df = df.drop_duplicates(subset=subset)
    values = sub_df.loc[:,subset[0]].to_list()
    ids = sub_df.loc[:,subset[1]].to_list()

    vocab = {index: value for index, value in zip(ids,values)}
    
    # Save the vocab to a JSON file
    with open(save_path, 'w') as json_file:
        json.dump(vocab, json_file)


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

def make_triplets_oneil_torchdrug():
    save_path = Directories.DATA_PATH / "gold" / "torchdrug" / "oneil" / "oneil.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    entity_vocab_path = Directories.DATA_PATH / "gold" / "torchdrug" / "oneil" / "entity_vocab.json"
    relation_vocab_path = Directories.DATA_PATH / "gold" / "torchdrug" / "oneil" / "relation_vocab.json"
    load_path = Directories.DATA_PATH / "gold" / "chemicalx" / "oneil" / "oneil.csv"
    df = pd.read_csv(load_path)

    # Create unique cell-line ID's based on context and label
    df['context_id'] = df.groupby(['context', 'label']).ngroup()
        
    # Factorize drug_1, and drug_2 columns to ID's
    df['drug_1_id'] = df['drug_1'].factorize()[0]
    df['drug_2_id'] = df['drug_2'].factorize()[0]

    # Create vocab json files
    create_vocab(df=df, subset=['drug_1','drug_1_id'], save_path=entity_vocab_path)
    create_vocab(df=df, subset=['context','context_id'], save_path=relation_vocab_path)

    # Filter dataframe
    columns_to_keep = ["drug_1_id", "drug_2_id", "context_id"]
    df = df[columns_to_keep]
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    make_triplets_oneil_chemicalx()
    make_triplets_oneil_torchdrug()