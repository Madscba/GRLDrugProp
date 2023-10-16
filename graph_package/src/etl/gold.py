from graph_package.configs.directories import Directories
import pandas as pd
import numpy as np
from graph_package.utils.helpers import init_logger
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


def get_CIDs(df: pd.DataFrame):
    """
    Get CIDs and smile strings for drugs.
    """
    drug_dict = load_drug_info_drugcomb()
    df["drug_row_cid"] = df["drug_row"].apply(lambda x: drug_dict[x]["cid"]).astype(str)
    df["drug_col_cid"] = df["drug_col"].apply(lambda x: drug_dict[x]["cid"]).astype(str)
    return df


def agg_loewe_and_make_binary(df: pd.DataFrame):
    sub_df = (
        df.groupby(["drug_row_cid", "drug_col_cid", "cell_line_name"])
        .mean()
        .reset_index()
    )
    sub_df["synergy_loewe"] = sub_df["synergy_loewe"].apply(
        lambda x: 0 if x < 0 else (1 if x > 10 else pd.NA)
    )
    #sub_df["drug_row"], sub_df["drug_col"]  = df["drug_row"].reset_index(), df["drug_col"].reset_index()
    sub_df.dropna(subset=["synergy_loewe"], inplace=True)
    return sub_df


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

def create_drug_id_vocabs(df: pd.DataFrame):
    # Create unique drug ID's
    unique_drugs = set(df["drug_row_cid"]).union(set(df["drug_col_cid"]))
    drug_dict = load_drug_info_drugcomb()
    drug_id_mapping = {drug: idx for idx, drug in enumerate(unique_drugs)}
    df["drug_row_id"] = df["drug_row_cid"].map(drug_id_mapping)
    df["drug_col_id"] = df["drug_col_cid"].map(drug_id_mapping)

    # Create vocab to map drug name identifier to graph drug ID
    drug_cid_mapping = {
        name: drug_dict[name]["cid"]
        for name in drug_dict
        if str(drug_dict[name]["cid"]) in list(unique_drugs)
    }
    drug_name_mapping = {
        name: drug_id_mapping[str(cid)] for name, cid in drug_cid_mapping.items()
    }
    return df, drug_name_mapping

def create_cell_line_id_vocabs(df: pd.DataFrame):
    # Create unique cell-line ID's based on context and label
    sub_df = df.loc[df["label"]==1]
    sub_df["context_id"] = sub_df.groupby(['context', 'label']).ngroup()
    sub_df.drop_duplicates(subset=['context','context_id'],keep='first',inplace=True)

    # Create vocab to map cell line name to graph cell line ID
    cell_line_mapping = {name: idx for name, idx in sub_df.loc[:,['context', 'context_id']].values}
    cell_line_name_mapping = {idx: name for name, idx in sub_df.loc[:,['context', 'context_id']].values}
    df["context_id"] = df["context"].map(cell_line_mapping)
    
    return df, cell_line_name_mapping

def make_triplets_oneil():
    save_path = Directories.DATA_PATH / "gold" / "oneil"
    save_path.mkdir(parents=True, exist_ok=True)

    df = load_oneil()
    df = get_CIDs(df)
    df = remove_drugs_not_in_cx(df)
    df, drug_vocab = create_drug_id_vocabs(df)
    df = agg_loewe_and_make_binary(df)

    rename_dict = {
        "drug_row_cid": "drug_1",
        "drug_col_cid": "drug_2",
        "synergy_loewe": "label",
        "cell_line_name": "context",
        "drug_row_id": "drug_1_id",
        "drug_col_id": "drug_2_id",
    }
    df.rename(columns=rename_dict, inplace=True)
    df = df[rename_dict.values()]

    df, cell_line_vocab = create_cell_line_id_vocabs(df)

    # Save the vocabs to a JSON file
    for vocab, name in zip((drug_vocab,cell_line_vocab),["entity_vocab.json","relation_vocab.json"]):
        with open(save_path / name, "w") as json_file:
            json.dump(vocab, json_file)

    df.to_csv(save_path / "oneil.csv", index=False)


if __name__ == "__main__":
    make_triplets_oneil()
    make_triplets_oneil_chemicalx()
    make_triplets_oneil_torchdrug()
