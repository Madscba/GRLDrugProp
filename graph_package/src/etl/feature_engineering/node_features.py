import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from graph_package.utils.helpers import logger
from graph_package.src.etl.medallion.silver import (
    download_hetionet,
    get_drug_info,
    filter_from_hetionet,
)
from graph_package.configs.directories import Directories
from sklearn.decomposition import PCA


def add_connected_node(edge, r, nodes_connected_to_drugs, filtered_edges, target_id):
    # Check if r is a key in nodes_connected_to_drugs
    if r not in nodes_connected_to_drugs:
        nodes_connected_to_drugs[r] = {}
        filtered_edges[r] = []
    # Check if a drug is already connected to this node
    if target_id in nodes_connected_to_drugs[r]:
        nodes_connected_to_drugs[r][target_id] += 1
    else:
        nodes_connected_to_drugs[r][target_id] = 1
    filtered_edges[r].append(edge)


def get_connected_nodes_ids(nodes_connected_to_drugs, min_degree):
    nodes_connected_to_at_least_one_drugs = [
        [node_id for node_id, _ in nodes_connected_to_drugs[rel].items()]
        for rel in nodes_connected_to_drugs
    ]
    node_ids = [
        [
            node_id
            for node_id, degree in nodes_connected_to_drugs[rel].items()
            if degree >= min_degree
        ]
        for rel in nodes_connected_to_drugs
    ]
    avg_node_degree = [
        sum(nodes_connected_to_drugs[r].values()) / len(nodes_connected_to_drugs[r])
        for r in nodes_connected_to_drugs
    ]
    return nodes_connected_to_at_least_one_drugs, node_ids, avg_node_degree


def filter_drug_node_graph(drug_ids, edges, node="Gene", min_degree=2):
    """Function for retrieving neigboring node ID's in Hetionet for drugs in DrugComb"""

    # Filter on nodes related to drugs
    nodes_connected_to_drugs = {}
    filtered_edges = {}
    logger.info(f"Filtering {node} in Hetionet..")
    for edge in tqdm(edges):
        source_type, source_id = edge["source_id"]
        target_type, target_id = edge["target_id"]
        r = edge["kind"]
        # Ignore palliates relation
        if r != "palliates":
            # Case where a drug is connected to the specified node through the relation r as source/head
            if (
                source_type == "Compound"
                and target_type == node
                and source_id in drug_ids
            ):
                add_connected_node(
                    edge, r, nodes_connected_to_drugs, filtered_edges, target_id
                )
            # Case where a drug is connected to the specified node through the relation r as target/tail
            elif (
                source_type == node
                and target_type == "Compound"
                and target_id in drug_ids
            ):
                add_connected_node(
                    edge, r, nodes_connected_to_drugs, filtered_edges, source_id
                )

    # Filter nodes based on degree
    (
        nodes_connected_to_at_least_one_drugs,
        node_ids,
        avg_node_degree,
    ) = get_connected_nodes_ids(nodes_connected_to_drugs, min_degree)

    # Basic descriptive statistics on drug-node graph (either per relation or across)
    for r, nodes_connected, node_id, avg_degree in zip(
        nodes_connected_to_drugs,
        nodes_connected_to_at_least_one_drugs,
        node_ids,
        avg_node_degree,
    ):
        logger.info(
            f"Number of {node} - {r} connected to at least one drug: {len(nodes_connected)}"
        )
        logger.info(
            f"Number of {node} - {r} connected to at least {min_degree} drugs: {len(node_id)}"
        )
        logger.info(f"Avg degree of {node} - {r} nodes to drugs: {round(avg_degree,3)}")
    return node_ids, filtered_edges


def build_adjacency_matrix(drug_ids, node_ids, edges):
    drug_indices = {drug: index for index, drug in enumerate(drug_ids)}
    node_indices = {node: index for index, node in enumerate(node_ids)}

    adjacency_matrix = torch.zeros((len(drug_ids), len(node_ids)), dtype=torch.float32)

    logger.info("Building adjacency matrix..")
    for edge in tqdm(edges):
        _, source_id = edge["source_id"]
        _, target_id = edge["target_id"]

        if source_id in drug_ids and target_id in node_ids:
            drug_index = drug_indices[source_id]
            node_index = node_indices[target_id]
            adjacency_matrix[drug_index, node_index] = 1.0

        elif target_id in drug_ids and source_id in node_ids:
            drug_index = drug_indices[target_id]
            node_index = node_indices[source_id]
            adjacency_matrix[drug_index, node_index] = 1.0

    return adjacency_matrix


def fit_pca_node_feature(adjacency_matrix, components, node_type, relation):
    # Generate PCA feature vectors
    drug_node_array = np.array(adjacency_matrix)
    pca = PCA(n_components=min(components, drug_node_array.shape[1]))
    pca_feature_vectors = pca.fit_transform(drug_node_array)

    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Calculate the accumulated explained variance
    accumulated_explained_variance = np.cumsum(explained_variance_ratio)
    logger.info(
        f"Acc explained variance for {node_type} - {relation} pca features: {accumulated_explained_variance[-1]}"
    )

    return pca_feature_vectors


def make_node_features(datasets=["ONEIL", "ALMANAC"], components=10, node_types="all"):
    """Function for generating PCA feature vectors based on Hetionet"""
    # Defining paths
    datasets_name = "oneil_almanac" if datasets == ["ONEIL", "ALMANAC"] else "oneil"
    save_path = Directories.DATA_PATH / "features" / "node_features"
    save_path.mkdir(parents=True, exist_ok=True)

    # Get info dict with drug name, DrugBank ID and inchikey for each drug in dataset
    data_path = (
        Directories.DATA_PATH / "silver" / datasets_name / f"{datasets_name}.csv"
    )
    drugs = pd.read_csv(data_path)
    if datasets_name == "oneil":
        drugs = filter_from_hetionet(drugs)
    drug_info = get_drug_info(drugs)
    drug_ids = [drug_info[drug]["DB"] for drug in drug_info.keys()]

    # Load Hetionet from json and filter edges
    het_path = Directories.DATA_PATH / "hetionet"
    if not os.path.exists(het_path / "hetionet-v1.0.json"):
        download_hetionet(het_path)
    with open(het_path / "hetionet-v1.0.json") as f:
        graph = json.load(f)
    edges = graph["edges"]
    edges = [
        e
        for e in edges
        if (e["target_id"][0] == "Compound") or (e["source_id"][0] == "Compound")
    ]

    # For each node_type and relation, build adjacency matrix and get pca feature vectors
    node_types_list = (
        ["Gene", "Disease", "Side Effect", "Pharmacologic Class"]
        if node_types == "all"
        else node_types
    )
    feature_vectors = []
    relations = []
    for i, node_type in enumerate(node_types_list):
        # Creates a list of node id's and filtered edges for each relation type
        node_ids, filtered_edges = filter_drug_node_graph(
            drug_ids=drug_ids, edges=edges, node=node_type
        )
        # Loop through relation types and build adjacency matrix per relation
        for node_indicies, relation in zip(node_ids, filtered_edges):
            filtered_edge = filtered_edges[relation]
            adjacency_matrix = build_adjacency_matrix(
                drug_ids, node_indicies, filtered_edge
            )
            degree = [
                adjacency_matrix[i, :].sum() for i in range(adjacency_matrix.shape[0])
            ]
            logger.info(
                f"Avg degree of drug nodes to {node_type} - {relation}: {sum(degree)/len(degree)} (max is {adjacency_matrix.shape[1]})"
            )

            # Fit PCA feature vectors
            pca_feature_vectors = fit_pca_node_feature(
                adjacency_matrix, components, node_type, relation
            )
            feature_vectors.append(pca_feature_vectors.tolist())
            relations.append(relation)

    # Save pca features to data/node_features as json/csv (name,relation,feature)
    drug_features = {drug.lower(): {} for drug in drug_info.keys()}
    for feature_vector, relation in zip(feature_vectors, relations):
        for i, drug in enumerate(drug_info.keys()):
            drug_features[drug.lower()][relation] = feature_vector[i]
    with open(save_path / f"{datasets_name}_drug_features.json", "w") as json_file:
        json.dump(drug_features, json_file)
    logger.info(f"Saved {len(drug_features)} PCA feature vectors to {save_path}")


if __name__ == "__main__":
    make_node_features()
