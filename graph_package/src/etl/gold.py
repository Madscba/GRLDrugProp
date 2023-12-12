from graph_package.configs.directories import Directories
from graph_package.src.etl.bronze import load_jsonl
import pandas as pd
import numpy as np
from graph_package.utils.helpers import init_logger
import json
import requests



logger = init_logger()


def load_oneil():
    data_path = Directories.DATA_PATH / "silver" / "oneil" / "oneil.csv"
    return pd.read_csv(data_path)

def load_oneil_almanac():
    data_path = Directories.DATA_PATH / "silver" / "oneil_almanac" / "oneil_almanac.csv"
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
        df.groupby(["drug_1_name", "drug_2_name", "context"])
        .mean()
        .reset_index()
    )
    sub_df["label"] = sub_df["label"].apply(
        lambda x: 0 if x < 0 else (1 if x > 10 else pd.NA)
    )
    # sub_df["drug_row"], sub_df["drug_col"]  = df["drug_row"].reset_index(), df["drug_col"].reset_index()
    sub_df.dropna(subset=["label"], inplace=True)
    return sub_df


def create_drug_id_vocabs(df: pd.DataFrame):
    # Create unique drug ID's
    unique_drugs = set(df["drug_1_name"]).union(set(df["drug_2_name"]))
    drug_id_mapping = {drug: idx for idx, drug in enumerate(unique_drugs)}
    df["drug_1_id"] = df["drug_1_name"].map(drug_id_mapping)
    df["drug_2_id"] = df["drug_2_name"].map(drug_id_mapping)

    return df, drug_id_mapping


def create_cell_line_id_vocabs(df: pd.DataFrame):
    # Create unique cell-line ID's based on context and label
    unique_contexts = set(df["context"])
    cell_line_mapping = {drug: idx for idx, drug in enumerate(unique_contexts)}
    df["context_id"] = df["context"].map(cell_line_mapping)
    return df, cell_line_mapping


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


def get_drug_cell_line_ids(df: pd.DataFrame):
    load_path = Directories.DATA_PATH / "gold" / "oneil"
    entity_vocab = json.load(open(load_path / "entity_vocab.json"))
    relation_vocab = json.load(open(load_path / "relation_vocab.json"))
    relation_vocab = {v: k for k, v in relation_vocab.items()}
    df["drug_1_id"] = df["drug_1_name"].map(entity_vocab)
    df["drug_2_id"] = df["drug_2_name"].map(entity_vocab)
    set(df["cell"]).difference(set(relation_vocab.keys()))
    df["context_id"] = df["context"].map(relation_vocab)
    return df


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


def get_max_zip_response(df: pd.DataFrame, study: str='oneil'):
    block_dict = load_jsonl(Directories.DATA_PATH / "silver" / study / "block_dict.json")
    block_df = pd.DataFrame(block_dict)
    block_df = block_df.groupby(['block_id','conc_c','conc_r']).agg({'synergy_zip': 'mean'}).reset_index()
    block_df = block_df.groupby(['block_id']).agg({'synergy_zip': ['max', 'mean']}).reset_index()
    block_df.columns = ['block_id', 'synergy_zip_max', 'synergy_zip_mean']  
    df = df.merge(block_df, on='block_id', how='left',validate='1:1')
    df = df.groupby(['drug_1_name', 'drug_2_name', 'context']).mean().reset_index()
    df["mean_label"] = df["synergy_zip_mean"].apply(lambda x: 1 if x >= 5 else 0)
    df["max_label"] = df["synergy_zip_max"].apply(lambda x: 1 if x >= 10 else 0)
    return df

def make_oneil_almanac_dataset(studies=["oneil","oneil_almanac"]):
    """
    Make ONEIL and ONEIL-ALMANAC datasets
    """
    for study in studies:
        logger.info(f"Making {study} dataset.")
        save_path = Directories.DATA_PATH / "gold" / study
        save_path.mkdir(parents=True, exist_ok=True)
        df = load_oneil() if study == 'oneil' else load_oneil_almanac()
        rename_dict = {
            "block_id": "block_id",
            "drug_row": "drug_1_name",
            "drug_col": "drug_2_name",
            "cell_line_name": "context",
        }
        df.rename(columns=rename_dict, inplace=True)
        columns_to_keep = list(rename_dict.values())+["css_col","css_row"]
        df = df[columns_to_keep]

        df = get_max_zip_response(df,study)
        df['css'] = (df['css_col'] + df['css_row'])/2
        df, drug_vocab = create_drug_id_vocabs(df)
        df, cell_line_vocab = create_cell_line_id_vocabs(df)
        for vocab, name in zip(
            (drug_vocab, cell_line_vocab),
            ["entity_vocab.json", "relation_vocab.json"]):
            with open(save_path / name, "w") as json_file:
                json.dump(vocab, json_file)
        df.to_csv(save_path / f"{study}.csv", index=False)

if __name__ == "__main__":
    make_oneil_almanac_dataset()