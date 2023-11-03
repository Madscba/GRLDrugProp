from graph_package.configs.directories import Directories
import pandas as pd
import numpy as np
from graph_package.utils.helpers import init_logger
import json
import requests
import json


logger = init_logger()


def load_oneil():
    data_path = Directories.DATA_PATH / "silver" / "oneil" / "oneil.csv"
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



def agg_loewe_and_make_binary(df: pd.DataFrame):
    sub_df = (
        df.groupby(["drug_row", "drug_col", "cell_line_name"])
        .mean()
        .reset_index()
    )
    sub_df["synergy_loewe"] = sub_df["synergy_loewe"].apply(
        lambda x: 0 if x < 0 else (1 if x > 10 else pd.NA)
    )
    # sub_df["drug_row"], sub_df["drug_col"]  = df["drug_row"].reset_index(), df["drug_col"].reset_index()
    sub_df.dropna(subset=["synergy_loewe"], inplace=True)
    return sub_df


def create_drug_id_vocabs(df: pd.DataFrame):
    # Create unique drug ID's
    unique_drugs = set(df["drug_row"]).union(set(df["drug_col"]))
    drug_id_mapping = {drug: idx for idx, drug in enumerate(unique_drugs)}
    df["drug_row_id"] = df["drug_row"].map(drug_id_mapping)
    df["drug_col_id"] = df["drug_col"].map(drug_id_mapping)

    return df, drug_id_mapping


def create_cell_line_id_vocabs(df: pd.DataFrame):
    # Create unique cell-line ID's based on context and label
    sub_df = df.loc[df["label"] == 1]
    sub_df.loc[:,["context_id"]] = sub_df.groupby(["context", "label"]).ngroup()
    sub_df.drop_duplicates(subset=["context", "context_id"], keep="first", inplace=True)

    # Create vocab to map cell line name to graph cell line ID
    cell_line_mapping = {
        name: idx for name, idx in sub_df.loc[:, ["context", "context_id"]].values
    }
    cell_line_name_mapping = {
        idx: name for name, idx in sub_df.loc[:, ["context", "context_id"]].values
    }
    
    df["context_id"] = df["context"].map(cell_line_mapping)
    return df, cell_line_name_mapping


def save_vocabs(drug_vocab, cell_line_vocab):
    save_path = Directories.DATA_PATH / "gold" / "oneil"
    save_path.mkdir(parents=True, exist_ok=True)
    for vocab, name in zip(
        (drug_vocab, cell_line_vocab), ["entity_vocab.json", "relation_vocab.json"]
    ):
        with open(save_path / name, "w") as json_file:
            json.dump(vocab, json_file)


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
    drug_name = pd.read_csv("https://raw.githubusercontent.com/Sinwang404/DeepDDs/master/data/drugs_0_10.csv",header=None)
    drug_names_dict = {val[0]:val[1] for val in drug_name.values}
    drug_names_list = drug_names_dict.keys()
    drug_dict = load_drug_info_drugcomb()
    deepdds_drug_dict = {}
    for val in drug_dict.values():
        synonyms = val["synonyms"].split("; ")
        for synonym in synonyms:
            for drug in drug_names_list:
                synonym  = synonym.lower()
                if  (drug.lower() == synonym) or (drug.lower() in synonym.split(" ")):
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
                synonym  = synonym.lower()
                if  (cell_line.lower() == synonym) or (cell_line.lower() in synonym.split(" ")):
                    assert cell_line not in deepdds_cell_dict
                    deepdds_cell_dict[cell_line] = val["name"]
                    cell_line_names.remove(cell_line)
                    break
    assert len(cell_line_names) == 0
    return deepdds_cell_dict



def get_drug_cell_line_ids(df: pd.DataFrame):    
    load_path = Directories.DATA_PATH / "gold" / "oneil"
    entity_vocab = json.load(open(load_path / "entity_vocab.json"))
    relation_vocab = json.load(open(load_path / "relation_vocab.json"))
    relation_vocab = {v:k for k,v in relation_vocab.items()}
    df["drug_1_id"] = df["drug_1_name"].map(entity_vocab)
    df["drug_2_id"] = df["drug_2_name"].map(entity_vocab)
    set(df["cell"]).difference(set(relation_vocab.keys()))
    df["context_id"] = df["context"].map(relation_vocab)
    return df


def make_original_deepdds_dataset():
    # Download triplets from DeepDDS repo 
    df = pd.read_csv("https://raw.githubusercontent.com/Sinwang404/DeepDDs/master/data/new_labels_0_10.csv")
    drug_dict_deepdds = get_drug_dict_deepdds_original()
    cell_line_dict_deepdds = get_cell_line_dict_deepdds_original(df)
    df["drug_1_name"] = df["drug1"].map(drug_dict_deepdds)
    df["drug_2_name"] = df["drug2"].map(drug_dict_deepdds)
    df["context"] = df["cell"].map(cell_line_dict_deepdds)
    df = get_drug_cell_line_ids(df)
    df.drop(["drug1","drug2","cell"],axis=1,inplace=True)
    df =  df[["drug_1_name","drug_2_name","label","context","drug_1_id","drug_2_id","context_id"]]
    df.to_csv(Directories.DATA_PATH / "gold" / "oneil" / "deepdds_original.csv", index=False)


def make_triplets_oneil():
    save_path = Directories.DATA_PATH / "gold" / "oneil"
    save_path.mkdir(parents=True, exist_ok=True)

    df = load_oneil()
    df, drug_vocab = create_drug_id_vocabs(df)
    df = agg_loewe_and_make_binary(df)

    rename_dict = {
        "drug_row": "drug_1_name",
        "drug_col": "drug_2_name",
        "synergy_loewe": "label",
        "cell_line_name": "context",
        "drug_row_id": "drug_1_id",
        "drug_col_id": "drug_2_id",
    }
    df.rename(columns=rename_dict, inplace=True)
    df = df[rename_dict.values()]

    df, cell_line_vocab = create_cell_line_id_vocabs(df)

    save_vocabs(drug_vocab, cell_line_vocab)
    df.to_csv(save_path / "oneil.csv", index=False)


if __name__ == "__main__":
    #make_triplets_oneil()
    make_original_deepdds_dataset()
