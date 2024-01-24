from graph_package.configs.directories import Directories
import pandas as pd
import numpy as np
import os
import requests
import bz2
from graph_package.utils.helpers import init_logger
from graph_package.src.etl.feature_engineering.cell_line_features import (
    make_cell_line_features,
)
import json
from itertools import product
from tqdm import tqdm
from graph_package.src.etl.medallion.load import (
    load_silver_csv,
    load_block_as_df,
    load_mono_response,
    load_jsonl
)
import matplotlib.pyplot as plt 

logger = init_logger()

def partial_name_match(filtered_drug_dict):
    # Check for different name conventions and edge cases
    filtered_drugs = [drug for drug in filtered_drug_dict.keys()]
    fd_lower = [d.lower() for d in filtered_drugs]
    fd_cap = [d.capitalize() for d in filtered_drugs]
    fd_title = [d.title() for d in filtered_drugs]
    final_drugs = (
        filtered_drugs
        + fd_lower
        + fd_cap
        + fd_title
        + ["5-Aminolevulinic acid hydrochloride"]
    )
    return final_drugs


def download_hetionet(data_path):
    url = "https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/json/hetionet-v1.0.json.bz2?download=true"
    response = requests.get(url)
    if response.status_code == 200:
        logger.info("Downloading Hetionet json file from GitHub..")
        # Decompress the content
        decompressed_content = bz2.decompress(response.content)

        # Decode bytes to string
        decompressed_content_str = decompressed_content.decode("utf-8")

        # Save the decompressed content to a file
        with open(data_path / "hetionet-v1.0.json", "w") as file:
            file.write(decompressed_content_str)
    else:
        logger.info(
            f"Failed to download Hetionet json file. Status code: {response.status_code}"
        )


def filter_drugs_in_graph(drug_info):
    """
    Function for filtering drugs in DrugComb found in Hetionet
    """

    # Load Hetionet from json
    data_path = Directories.DATA_PATH / "hetionet"
    data_path.mkdir(exist_ok=True, parents=True)
    if not os.path.exists(data_path / "hetionet-v1.0.json"):
        download_hetionet(data_path)
    with open(data_path / "hetionet-v1.0.json") as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Select subset of drug IDs corresponding to drug IDs that exist in the graph
    drugs_in_graph = {}
    for node in nodes:
        if node["kind"] == "Compound":
            drugs_in_graph[node["name"]] = {}
            drugs_in_graph[node["name"]]["inchikey"] = node["data"]["inchikey"][9:]
            drugs_in_graph[node["name"]]["DB"] = node["identifier"]

    # Check each drug in drug_info and add it to filtered dict if found in drugs_in_graph
    filtered_drug_dict = {}
    logger.info("Filtering drugs in Hetionet..")
    for drug_name, identifiers in drug_info.items():
        for graph_drug_name, graph_identifiers in drugs_in_graph.items():
            # Match on drug names
            if (
                graph_drug_name.lower() in drug_name.lower()
                or graph_drug_name in drug_name.title()
            ):
                filtered_drug_dict[drug_name] = graph_identifiers
            else:
                for identifier_key, identifier_value in identifiers.items():
                    # Match on drug synonyms
                    if identifier_key == "synonyms":
                        if graph_drug_name in identifier_value:
                            filtered_drug_dict[drug_name] = graph_identifiers
                    # Match on drugbank ID or inchikey
                    elif graph_identifiers.get(identifier_key) == identifier_value:
                        filtered_drug_dict[drug_name] = graph_identifiers

    drug_ids = [filtered_drug_dict[drug]["DB"] for drug in filtered_drug_dict.keys()]
    logger.info(f"{len(drug_ids)} of {len(drug_info)} drugs found in Hetionet")
    drug_edges = [
        e
        for e in edges
        if (e["target_id"][0] == "Compound") or (e["source_id"][0] == "Compound")
    ]
    return drug_ids, filtered_drug_dict, drug_edges


def get_drug_info(drugs: pd.DataFrame, unique_drug_names: list, add_SMILES: bool = False):
    """
    Filter drugs from DrugComb to match drugs in Hetionet
    """
    dict_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "drug_dict.json"

    # Load drug dict from DrugComb with metadata on drugs
    with open(dict_path) as f:
        drug_dict = json.load(f)

    # Make info dict with drug name, DrugBank ID and inchikey for each drug
    drug_info = {}
    for drug in unique_drug_names:
        if add_SMILES:
            drug_info[drug] = {}
            drug_info[drug]["SMILES"] = drug_dict[drug]["smiles"]
        else:
            # Check for different name conventions and edge cases
            drug_name_v1 = drug.capitalize() if not drug[0].isupper() else drug
            drug_name_v2 = drug_name_v1.title() if not drug_name_v1[0].isalpha() else drug
            drug_info[drug] = {}
            drug_info[drug]["synonyms"] = drug_dict[drug]["synonyms"].split(";") + [drug_name_v1, drug_name_v2]
            drug_info[drug]["inchikey"] = drug_dict[drug]["inchikey"]
            drug_info[drug]["DB"] = drug_dict[drug]["drugbank_id"]

    return drug_info


def filter_from_hetionet(drugs, drug_identifiers = ["drug_row", "drug_col"]):

    unique_drug_names = set(drugs[drug_identifiers[0]]).union(set(drugs[drug_identifiers[1]]))

    # Get drug info
    drug_info = get_drug_info(drugs, unique_drug_names)

    # Find drugs in Hetionet
    _, filtered_drug_dict, _ = filter_drugs_in_graph(drug_info)

    # Filter drugs from ONEIL-ALMANAC that exists in Hetionet
    potential_drug_names = partial_name_match(filtered_drug_dict)
    filtered_df = drugs[
        drugs[drug_identifiers[0]].isin(potential_drug_names)
        & drugs[drug_identifiers[1]].isin(potential_drug_names)
    ]
    return filtered_df

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
        .agg({"synergy_zip": "mean", "synergy_loewe": "mean"})
        .reset_index()
    )
    block_df = (
        block_df.groupby(["block_id"])
        .agg({"synergy_zip": ["max", "mean"], "synergy_loewe": "mean"})
        .reset_index()
    )
    block_df.columns = ["block_id", "synergy_zip_max", "synergy_zip_mean", "synergy_loewe"]
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


def concat_oneil_almanac(df: pd.DataFrame, het: bool = False):
    oneil_almanac = "oneil_almanac_het" if het else "oneil_almanac"
    oneil_almanac_path = Directories.DATA_PATH / "gold" / oneil_almanac / f"{oneil_almanac}.csv"
    oneil_almanac_df = pd.read_csv(oneil_almanac_path)
    oneil_almanac_df = oneil_almanac_df.loc[:, df.columns.to_list()]
    df_combined = pd.concat([df,oneil_almanac_df],ignore_index=True)
    return df_combined

def get_combined_drugs_and_cell_lines(df: pd.DataFrame):
    unique_drugs_rest = set(df["drug_row"]).union(set(df["drug_col"]))
    oneil_almanac_path = Directories.DATA_PATH / "gold" / "oneil_almanac_het" / "oneil_almanac_het.csv"
    oneil_almanac_df = pd.read_csv(oneil_almanac_path)
    unique_drugs_almanac = set(oneil_almanac_df["drug_1_name"]).union(set(oneil_almanac_df["drug_2_name"]))
    unique_drugs = unique_drugs_rest.union(unique_drugs_almanac)
    unique_cell_lines = set(df.cell_line_name).union(set(oneil_almanac_df.context))
    return unique_drugs, unique_cell_lines


def generate_mono_responses(df: pd.DataFrame, study_name: str = "oneil_almanac", overwrite: bool = False):
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
    # Create a bool to check if dataset is filtered on het
    het = True if study_name in ["oneil_het", "oneil_almanac_het", "drugcomb_het"] else False
    if het:
        # Get subset of mono-responses since het is a subset of the corresponding dataset
        study = study_name[:-4] if het else study_name
        df_mono_study = pd.read_csv(Directories.DATA_PATH / "gold" / study / "mono_response.csv")
        if study == "drugcomb":
            unique_drugs, unique_cell_lines = get_combined_drugs_and_cell_lines(df)
        else:
            unique_drugs = set(df["drug_row"]).union(set(df["drug_col"]))
            unique_cell_lines = set(df.cell_line_name)
        # Filter drugs and cell lines not found in _het dataset
        df_mono_study = df_mono_study[df_mono_study["drug"].isin(unique_drugs)]
        df_mono_study = df_mono_study[df_mono_study["cell_line"].isin(unique_cell_lines)]
        df_mono_study.reset_index(drop=True,inplace=True)
        save_path = Directories.DATA_PATH / "gold" / study_name
        save_path.mkdir(exist_ok=True, parents=True)
        df_mono_study.to_csv(save_path / m_file_name,index=False)
        return
    
    file_path = data_path / m_file_name
    if (not (file_path).exists()) | overwrite:
        if (file_path).exists():
            os.remove(file_path)
        df_block = load_block_as_df(study_name)
        df_block = df_block.merge(
            df.loc[:, ["drug_row", "drug_col", "cell_line_name", "block_id"]],
            how="left",
            on=["block_id"],
        )

        df_block = df_block.dropna(subset=["drug_row", "drug_col", "cell_line_name"])
        filter_both = (df_block["conc_r"] > 0) & (df_block["conc_c"] > 0)
        filter_zero = (df_block["conc_r"] == 0) & (df_block["conc_c"] == 0)

        df_block = df_block[~(filter_both | filter_zero)]

        unique_drugs = set(df_block["drug_row"]).union(set(df_block["drug_col"]))
        unique_cell_lines = set(df_block["cell_line_name"])
        mono_response_dict = {}
        multi_index = pd.MultiIndex.from_tuples(
            product(unique_drugs, unique_cell_lines), names=("drug", "cell_line")
        )
        dummy_max = 0.0
        df_mono = pd.DataFrame(dummy_max, index=multi_index, columns=["min", ",median", "max"])
        
        for drug in tqdm(unique_drugs, desc="creating mono response dict"):
            drug_included = ((df_block["drug_row"]==drug) | (df_block[
                "drug_col"]==drug))
        
            df_block_sub = df_block[drug_included]
            df_block_sub["conc"] = np.where(
                df_block_sub["drug_row"] == drug,
                df_block_sub["conc_r"],
                np.where(
                    df_block_sub["drug_col"] == drug, df_block_sub["conc_c"], np.nan
                ),
            )
            df_block_sub["drug"] = drug

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
        
        save_path = Directories.DATA_PATH / "gold" / study_name
        save_path.mkdir(exist_ok=True, parents=True)
        # All drug-cell line pairs are created, but we only need to save the ones with mono response data
        if study_name == "drugcomb":
            # Concat ONEIL-ALMANAC to rest of DrugComb to create full DrugComb mono-responses
            df_mono_oneil_almanac = pd.read_csv(Directories.DATA_PATH / "gold" / "oneil_almanac" / "mono_response.csv")

            df_mono = pd.concat([df_mono, df_mono_oneil_almanac], ignore_index=True)
            df_mono = df_mono.groupby(["drug","cell_line"]).mean().reset_index()
            df_mono.to_csv(save_path / m_file_name, index=False)
        else:
            df_mono.to_csv(save_path / m_file_name, index=True)


def filter_drugs_from_smiles(df: pd.DataFrame):
    unique_drugs = set(df["drug_row"]).union(set(df["drug_col"]))
    drug_info = get_drug_info(df, unique_drugs, add_SMILES=True)
    drugs = [drug for drug in unique_drugs if drug_info[drug]["SMILES"] not in ("NULL", 'Antibody (MEDI3622)')]
    df_filtered = df[df["drug_row"].isin(drugs) & df["drug_col"].isin(drugs)]
    return df_filtered


def make_oneil_almanac_dataset(studies=["oneil","oneil_almanac","drugcomb"]):

    """
    Make ONEIL, ONEIL-ALMANAC and DrugComb datasets with and without filtering on Hetionet
    
    NOTE: The ONEIL-ALMANAC dataset should be created before the DrugComb dataset
    """

    for study in studies:
        for het in [False, True]:
            df = load_silver_csv(study)
            study_name = study + "_het" if het else study
            logger.info(f"Making {study_name} dataset.")
            save_path = Directories.DATA_PATH / "gold" / study_name
            save_path.mkdir(parents=True, exist_ok=True)
            if het:
                df = filter_from_hetionet(df)
            generate_mono_responses(df=df, study_name=study_name)
            df = filter_drugs_from_smiles(df)
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
    
            df = filter_cell_lines(df)

            # Concat ONEIL-ALMANAC to remaining of DrugComb
            if study=="drugcomb":
                df = concat_oneil_almanac(df, het)

            df, drug_vocab = create_drug_id_vocabs(df)
            df, cell_line_vocab = create_cell_line_id_vocabs(df)
            for vocab, name in zip(
                (drug_vocab, cell_line_vocab), ["entity_vocab.json", "relation_vocab.json"]
            ):
                with open(save_path / name, "w") as json_file:
                    json.dump(vocab, json_file)
            df.to_csv(save_path / f"{study_name}.csv", index=False)

def make_filtered_drugcomb_dataset():
    """ 
    Second gold layer. 
    Filters drugcomb dataset on triplets for which drugs and cell lines that are present
    in less than 50 triplets are removed.
    """

    save_path = Directories.DATA_PATH / "gold"
    study = "drugcomb"
    df = pd.read_csv(save_path / study / f"{study}.csv")
    df_mono = pd.read_csv(save_path / study / "mono_response.csv")
    # take only cell_lines with more than 50 triplets
    df_context_id=df.value_counts(subset=['context_id'])
    context_ids = df_context_id[df_context_id > 50].index.get_level_values(0)
    df_inv = df.copy()
    df_inv["drug_1_id"], df_inv["drug_2_id"] = (
        df["drug_2_id"],
        df["drug_1_id"],
    )
    df_undirected = pd.concat([df, df_inv], ignore_index=True)
    df_undirected = df_undirected.value_counts(["drug_1_id"]).sort_values(ascending=False)
    drug_index = df_undirected[df_undirected>50].index.get_level_values(0)
    # filter away drugs with less than 50 triplets
    drug_filter = df["drug_1_id"].isin(drug_index) & df["drug_2_id"].isin(drug_index)
    drug_context_filter = drug_filter & df["context_id"].isin(context_ids) 
    df_filter = df[drug_context_filter]
    # filter away outliers 
    df_filter = df_filter[df_filter["synergy_zip_max"] < 50]

    drug_filter = df_mono["drug"].isin(set(df_filter["drug_1_name"]).union(set(df_filter["drug_2_name"])))
    drug_cell_line_filter = drug_filter & df_mono["cell_line"].isin(df_filter["context"].unique())
    df_mono_filtered = df_mono[drug_cell_line_filter] 
    drug_names = df_mono_filtered["drug"].unique()
    cell_lines = df_mono_filtered["cell_line"].unique()  
    # make vocabs   
    entity_vocab = {name: i for i, name in enumerate(drug_names)}
    cell_line_vocab = {name: i for i, name in enumerate(cell_lines)}
    df_filter.loc[:,"drug_1_id"] = df_filter["drug_1_name"].map(entity_vocab)
    df_filter.loc[:,"drug_2_id"] = df_filter["drug_2_name"].map(entity_vocab)
    df_filter.loc[:,"context_id"] = df_filter["context"].map(cell_line_vocab)
    
    new_study_name = study + "_filtered"
    save_path_new = save_path / (new_study_name)
    with open(save_path_new / 'entity_vocab.json', "w") as json_file:
                    json.dump(entity_vocab, json_file)
    with open(save_path_new / 'relation_vocab.json', "w") as json_file:
                json.dump(cell_line_vocab, json_file)
    
    save_path_new.mkdir(parents=True, exist_ok=True)    
    df_filter.to_csv(save_path_new / f"{new_study_name}.csv", index=False)
    df_mono.to_csv(save_path_new / "mono_response.csv", index=False)

if __name__ == "__main__":
    make_oneil_almanac_dataset(studies=["drugcomb"])
    make_filtered_drugcomb_dataset()

