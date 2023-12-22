from graph_package.configs.directories import Directories
from graph_package.src.etl.medallion.load import load_jsonl
import pandas as pd
import numpy as np
import os
from graph_package.utils.helpers import init_logger
from graph_package.src.etl.feature_engineering.cell_line_features import (
    make_cell_line_features,
)
import json
from itertools import product
from tqdm import tqdm
from graph_package.src.etl.medallion.load import (
    load_oneil,
    load_oneil_almanac,
    load_block_as_df,
    load_mono_response,
)

logger = init_logger()


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
    m_file_name = "mono_response.csv"
    if (not (data_path / m_file_name).exists()) | overwrite:
        if (data_path / m_file_name).exists():
            os.remove(data_path / m_file_name)

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

        unique_drugs = set(df_block["drug_row"]).union(set(df_block["drug_col"]))
        unique_cell_lines = set(df_block["cell_line_name"])
        mono_response_dict = {}
        multi_index = pd.MultiIndex.from_tuples(
            product(unique_drugs, unique_cell_lines), names=("drug", "cell_line")
        )
        df_mono = pd.DataFrame(0, index=multi_index, columns=["min", ",median", "max"])
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
                df_block_sub.groupby(["drug", "cell_line_name", "conc"])["inhibition"]
                .mean()
                .reset_index()
            )
            cl_inhi_pr_conc = df_block_sub_grouped.pivot_table(
                index=["cell_line_name", "drug"],
                values="inhibition",
                columns="conc",
            ).reset_index()
            conc_cols = cl_inhi_pr_conc.columns[2:]
            stat_array = np.array(
                cl_inhi_pr_conc[conc_cols].agg(["min", "median", "max"], axis=1)
            )
            idx = list(
                zip(
                    cl_inhi_pr_conc["drug"].values,
                    cl_inhi_pr_conc["cell_line_name"].values,
                )
            )
            df_mono.loc[idx] = stat_array
        df_mono.to_csv(data_path / m_file_name, index=True)


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

        generate_mono_responses(study_name=study)

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


if __name__ == "__main__":
    make_oneil_almanac_dataset()
