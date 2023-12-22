from graph_package.configs.directories import Directories
from graph_package.src.etl.medallion.bronze import load_jsonl
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


def load_block_as_df(study_name: str):
    data_path = Directories.DATA_PATH / "silver" / study_name / "block_dict.json"
    block_dict_json = load_jsonl(data_path)
    df_block = pd.DataFrame(block_dict_json)
    df_block = df_block.loc[:, ["conc_r", "conc_c", "inhibition", "block_id"]]
    return df_block


def load_mono_response(study_name: str):
    data_path = Directories.DATA_PATH / "gold" / study_name / "mono_response.json"
    with open(data_path) as f:
        mono_response_dict = json.load(f)
    return mono_response_dict


def agg_loewe_and_make_binary(df: pd.DataFrame):
    sub_df = df.groupby(["drug_1_name", "drug_2_name", "context"]).mean().reset_index()
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


def get_max_zip_response(df: pd.DataFrame, study: str = "oneil"):
    block_dict = load_jsonl(
        Directories.DATA_PATH / "silver" / study / "block_dict.json"
    )
    block_df = pd.DataFrame(block_dict)
    block_df = (
        block_df.groupby(["block_id", "conc_c", "conc_r"])
        .agg({"synergy_zip": "mean"})
        .reset_index()
    )
    block_df = (
        block_df.groupby(["block_id"])
        .agg({"synergy_zip": ["max", "mean"]})
        .reset_index()
    )
    block_df.columns = ["block_id", "synergy_zip_max", "synergy_zip_mean"]
    df = df.merge(block_df, on="block_id", how="left", validate="1:1")
    df = df.groupby(["drug_1_name", "drug_2_name", "context"]).mean().reset_index()
    df["mean_label"] = df["synergy_zip_mean"].apply(lambda x: 1 if x >= 5 else 0)
    df["max_label"] = df["synergy_zip_max"].apply(lambda x: 1 if x >= 10 else 0)
    return df


def filter_cell_lines(data_df):
    feature_path = (
        Directories.DATA_PATH
        / "features"
        / "cell_line_features"
        / "CCLE_954_gene_express.json"
    )
    if not os.path.exists(feature_path):
        make_cell_line_features()
    with open(feature_path) as f:
        all_edge_features = json.load(f)
    cell_lines = [cl for cl in list(all_edge_features.keys())]
    data_df = data_df[(data_df["context"].isin(cell_lines))]
    return data_df


def insert_inhibition_and_concentration_into_dict(
    df: pd.DataFrame, mono_response_dict: dict
):
    for index, row in df.iterrows():
        cell_line = row["cell_line_name"]
        drug = row["drug"]

        # Check if drug exists, if not, create an empty dictionary for it
        if drug not in mono_response_dict:
            mono_response_dict[drug] = {}

        # Check if cell line exists for the drug, if not, create an empty dictionary for it
        if cell_line not in mono_response_dict[drug]:
            mono_response_dict[drug][cell_line] = {}

        concentrations = [
            c for c in df.columns if re.match("[-+]?\d*\.\d+|\d+ ", str(c))
        ]
        for c in concentrations:
            if isinstance(row[c], float):
                mono_response_dict[drug][cell_line][c] = row[c]
            else:
                print("nan here:", row[c])
    return mono_response_dict


def generate_mono_responses(study_name: str = "oneil_almanac", overwrite: bool = False):
    """From the relevant block dict from data/silver/<study_name>/block_dict.json fetch mono responses
    and per drug and cell line and aggregate inhibition per concentration.
    Params:
        study_name (str): indicates which study that the mono responses should be generated from
        overwrite (bool): if true then the current (if it exists) will be removed before a new is created.

    Returns:
        None
    """
    data_path = Directories.DATA_PATH / "gold" / study_name
    data_path.mkdir(exist_ok=True, parents=True)
    m_file_name = "mono_response.json"
    if (not (data_path / m_file_name).exists()) | overwrite:
        if (data_path / m_file_name).exists():
            os.remove(data_path / m_file_name)
        else:
            df_block = load_block_as_df(study_name)
            df = load_oneil() if study_name == "oneil" else load_oneil_almanac()
            df_block = df_block.merge(
                df.loc[:, ["drug_row", "drug_col", "cell_line_name", "block_id"]],
                how="left",
                on=["block_id"],
            )
            del df
            filter_both = (df_block["conc_r"] > 0) & (df_block["conc_c"] > 0)
            filter_zero = (df_block["conc_r"] == 0) & (df_block["conc_c"] == 0)
            df_block = df_block[~(filter_both | filter_zero)]
            unique_drugs = list(
                set(
                    np.append(
                        df_block["drug_row"].unique(), df_block["drug_col"].unique()
                    )
                )
            )
            mono_response_dict = {}

            for drug in tqdm(unique_drugs, desc="creating mono response dict"):
                drug_included = df_block["drug_row"].isin([drug]) | df_block[
                    "drug_col"
                ].isin([drug])
                df_block_sub = df_block[drug_included]
                df_block_sub["conc"] = np.where(
                    df_block_sub["drug_row"] == drug,
                    df_block_sub["conc_r"],
                    np.where(
                        df_block_sub["drug_col"] == drug, df_block_sub["conc_c"], np.nan
                    ),
                )
                df_block_sub["drug"] = np.where(
                    df_block_sub["drug_row"] == drug,
                    df_block_sub["drug_row"],
                    np.where(
                        df_block_sub["drug_col"] == drug,
                        df_block_sub["drug_col"],
                        np.nan,
                    ),
                )

                df_block_sub = df_block_sub[df_block_sub["conc"] > 0]
                df_block_sub_grouped = (
                    df_block_sub.groupby(["drug", "cell_line_name", "conc"])[
                        "inhibition"
                    ]
                    .mean()
                    .reset_index()
                )
                cell_line_inhibition_per_conc = df_block_sub_grouped.pivot_table(
                    index=["cell_line_name", "drug"],
                    values="inhibition",
                    columns="conc",
                ).reset_index()
                mono_response_dict = insert_inhibition_and_concentration_into_dict(
                    cell_line_inhibition_per_conc, mono_response_dict
                )

            with open(data_path / m_file_name, "w") as json_file:
                json.dump(mono_response_dict, json_file)


def make_oneil_almanac_dataset(studies=["oneil", "oneil_almanac"]):
    """
    Make ONEIL and ONEIL-ALMANAC datasets
    """
    for study in studies:
        logger.info(f"Making {study} dataset.")
        save_path = Directories.DATA_PATH / "gold" / study
        save_path.mkdir(parents=True, exist_ok=True)
        df = load_oneil() if study == "oneil" else load_oneil_almanac()
        rename_dict = {
            "block_id": "block_id",
            "drug_row": "drug_1_name",
            "drug_col": "drug_2_name",
            "cell_line_name": "context",
        }
        df.rename(columns=rename_dict, inplace=True)
        columns_to_keep = list(rename_dict.values()) + ["css_col", "css_row"]
        df = df[columns_to_keep]

        df = get_max_zip_response(df, study)
        df["css"] = (df["css_col"] + df["css_row"]) / 2
        if study == "oneil_almanac":
            df = filter_cell_lines(df)
        df, drug_vocab = create_drug_id_vocabs(df)
        df, cell_line_vocab = create_cell_line_id_vocabs(df)
        for vocab, name in zip(
            (drug_vocab, cell_line_vocab), ["entity_vocab.json", "relation_vocab.json"]
        ):
            with open(save_path / name, "w") as json_file:
                json.dump(vocab, json_file)
        df.to_csv(save_path / f"{study}.csv", index=False)
        generate_mono_responses(study_name=study)


if __name__ == "__main__":
    make_oneil_almanac_dataset()
