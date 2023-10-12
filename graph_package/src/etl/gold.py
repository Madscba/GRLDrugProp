# import sys
# sys.path.insert(0,'/Users/johannesreiche/Library/Mobile Documents/com~apple~CloudDocs/DTU/MMC/Thesis/Code/GRLDrugProp')
from graph_package.configs.directories import Directories
import pandas as pd
import numpy as np
from graph_package.utils.helpers import init_logger
from collections import defaultdict
import json
import urllib.request


logger = init_logger()


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
    df["drug_row_cid"] = df["drug_row"].apply(lambda x: drug_dict[x]["cid"]).astype(str)
    df["drug_col_cid"] = df["drug_col"].apply(lambda x: drug_dict[x]["cid"]).astype(str)
    return df


def agg_loewe_and_make_binary(df: pd.DataFrame):
    df = (
        df.groupby(["drug_row_cid", "drug_col_cid", "cell_line_name"])
        .mean()
        .reset_index()
    )
    df["synergy_loewe"] = df["synergy_loewe"].apply(
        lambda x: 0 if x < 0 else (1 if x > 10 else pd.NA)
    )
    df.dropna(subset=["synergy_loewe"], inplace=True)

    return df


def remove_drugs_not_in_cx(df: pd.DataFrame):
    path = "https://raw.githubusercontent.com/AstraZeneca/chemicalx/main/dataset/drugcomb/drug_set.json"
    with urllib.request.urlopen(path) as url:
        raw_data = json.loads(url.read().decode())

    drugs_not_in_cx = set(df["drug_row_cid"]).union(set(df["drug_col_cid"])) - set(
        raw_data.keys()
    )
    drugcomb_not_in_cx = df["drug_row_cid"].apply(lambda x: x in drugs_not_in_cx) | df[
        "drug_col_cid"
    ].apply(lambda x: x in drugs_not_in_cx)
    drugcombs_in_oneil = df.shape[0]
    df = df[~drugcomb_not_in_cx]

    logger.info(
        f"Removed {drugcomb_not_in_cx.sum()} of {drugcombs_in_oneil} drug pairs not in chemicalx"
    )
    return df


def create_vocab(df: pd.DataFrame, subset: list, save_path: str = ""):
    """
    Create json file with CID / cell-line name to entity / relation ID
    """
    sub_df = df.drop_duplicates(subset=subset, keep="first")
    keys = sub_df.loc[:, subset[0]].to_list()
    ids = sub_df.loc[:, subset[1]].to_list()

    vocab = {key: id for key, id in zip(keys, ids)}

    # Save the vocab to a JSON file
    with open(save_path, "w") as json_file:
        json.dump(vocab, json_file)

def create_inverse_triplets(df: pd.DataFrame):
    """ Create inverse triplets so that if (h,r,t) then (t,r,h) is also in the graph"""
    df_inv = df.copy()
    df_inv['drug_1'], df_inv['drug_2'] = df['drug_2'], df['drug_1']
    df_inv['drug_1_id'], df_inv['drug_2_id'] = df['drug_2_id'], df['drug_1_id']
    df_combined = pd.concat([df,df_inv], ignore_index=True)
    return df_combined

def make_triplets_oneil_chemicalx():
    save_path = Directories.DATA_PATH / "gold" / "chemicalx" / "oneil" / "oneil.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_oneil()
    df = get_CIDs(df)
    df = remove_drugs_not_in_cx(df)
    df = agg_loewe_and_make_binary(df)

    rename_dict = {
        "drug_row_cid": "drug_1",
        "drug_col_cid": "drug_2",
        "synergy_loewe": "label",
        "cell_line_name": "context",
    }
    df.rename(columns=rename_dict, inplace=True)
    df = df[rename_dict.values()]

    df.to_csv(save_path, index=False)


def make_triplets_oneil_torchdrug():
    save_path = Directories.DATA_PATH / "gold" / "torchdrug" / "oneil"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    entity_vocab_path = save_path / "entity_vocab.json"
    relation_vocab_path = save_path / "relation_vocab.json"
    
    load_path = save_path /  "oneil.csv"
    df = pd.read_csv(load_path)

    # Create unique drug ID's
    unique_drugs = set(df["drug_1"]).union(set(df["drug_2"]))
    drug_dict = load_drug_info_drugcomb()
    drug_id_mapping = {drug: idx for idx, drug in enumerate(unique_drugs)}
    df["drug_1_id"] = df["drug_1"].map(drug_id_mapping)
    df["drug_2_id"] = df["drug_2"].map(drug_id_mapping)

    # Create vocab to map drug name identifier to graph drug ID
    drug_cid_mapping = {
        name: drug_dict[name]["cid"]
        for name in drug_dict
        if drug_dict[name]["cid"] in list(unique_drugs)
    }
    drug_name_mapping = {
        name: drug_id_mapping[cid] for name, cid in drug_cid_mapping.items()
    }

    # Save the vocab to a JSON file
    with open(entity_vocab_path, "w") as json_file:
        json.dump(drug_name_mapping, json_file)

    # Create unique cell-line ID's based on context and label
    df = df[df["label"]==1]
    df['context_id'] = df.groupby(['context', 'label']).ngroup()
    sub_df = df.drop_duplicates(subset=['context','context_id'],keep='first')
    # Create vocab to map cell line name to graph cell line ID
    cell_line_mapping = {name: idx for name, idx in sub_df.loc[:,['context', 'context_id']].values}
    

    # Save the vocab to a JSON file
    with open(relation_vocab_path, "w") as json_file:
        json.dump(cell_line_mapping, json_file)


    # Create inverse triplets
    df = create_inverse_triplets(df)

    columns_to_keep = ["drug_1_id", "drug_2_id", "context_id"]
    df = df[columns_to_keep]
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    make_triplets_oneil_chemicalx()
    make_triplets_oneil_torchdrug()
