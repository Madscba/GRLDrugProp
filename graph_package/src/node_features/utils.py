import json
import torch
import pandas as pd
from tqdm import tqdm
from graph_package.utils.helpers import init_logger
from graph_package.configs.directories import Directories

logger = init_logger()

def filter_drugs_in_graph(drug_info):
    """
    Function for filtering drugs in DrugComb found in Hetionet
    """

    # Load Hetionet from json
    data_path = Directories.DATA_PATH / "node_features"
    logger.info("Loading Hetionet..")
    with open(data_path / "hetionet-v1.0.json") as f:
        graph = json.load(f)
    nodes = graph['nodes']
    edges = graph['edges']

    # Step 1: Select subset of drug IDs corresponding to drug IDs that exist in the graph
    drugs_in_graph = {}
    for node in nodes:
        if node['kind'] == 'Compound':
            drugs_in_graph[node['name']] = {}
            drugs_in_graph[node['name']]['inchikey'] = node['data']['inchikey'][9:]
            drugs_in_graph[node['name']]['DB'] = node['identifier']

    # Create filtered drug dict
    filtered_drug_dict = {}

    logger.info("Filtering drugs in Hetionet..")
    # Check each drug in drug_info and add it to filtered dict if found in drugs_in_graph
    for drug_name, identifiers in drug_info.items():
        for identifier_key, identifier_value in identifiers.items():
            for graph_drug_name, graph_identifiers in drugs_in_graph.items():
                if drug_name == graph_drug_name:
                    filtered_drug_dict[drug_name] = graph_identifiers
                elif graph_identifiers.get(identifier_key) == identifier_value:
                    filtered_drug_dict[drug_name] = graph_identifiers

    drug_ids = [filtered_drug_dict[drug]['DB'] for drug in filtered_drug_dict.keys()]
    logger.info(f"{len(drug_ids)} of {len(drug_info)} drugs found in Hetionet")
    return drug_ids, filtered_drug_dict, edges


def filter_drug_node_graph(drug_ids, edges, node='Gene', min_degree=2):
    """
    Function for retrieving neigboring node ID's in Hetionet for drugs in DrugComb
    """

    # Step 2: Filter on nodes related to drugs in the subset
    nodes_connected_to_drugs = {}
    filtered_edges = []
    logger.info(f"Filtering {node} in Hetionet..")
    for edge in tqdm(edges):
        source_type, source_id = edge['source_id']
        target_type, target_id = edge['target_id']

        if source_type == 'Compound' and target_type == node and source_id in drug_ids:
            filtered_edges.append(edge)
            if target_id in nodes_connected_to_drugs:
                nodes_connected_to_drugs[target_id] += 1 
            else:
                nodes_connected_to_drugs[target_id] = 1
        elif source_type == node and target_type == 'Compound'  and target_id in drug_ids:
            filtered_edges.append(edge)
            if source_id in nodes_connected_to_drugs:
                nodes_connected_to_drugs[source_id] += 1 
            else:
                nodes_connected_to_drugs[source_id] = 1
    
    # Step 3: Filter nodes based on degree
    nodes_connected_to_at_least_one_drugs = [
        node_id for node_id, _ in nodes_connected_to_drugs.items()
    ]
    node_ids = [
        node_id for node_id, degree in nodes_connected_to_drugs.items() if degree >= min_degree
    ]
    avg_node_degree = sum(nodes_connected_to_drugs.values())/len(nodes_connected_to_drugs)

    # Basic descriptive statistics on drug-node graph
    logger.info(f"Number of {node} connected to at least one drug: {len(nodes_connected_to_at_least_one_drugs)}")
    logger.info(f"Number of {node} connected to at least {min_degree} drugs: {len(node_ids)}")
    logger.info(f"Avg degree of {node} nodes to drugs: {round(avg_node_degree,3)}")

    return node_ids, filtered_edges

def build_adjacency_matrix(drug_ids, node_ids, edges):

    drug_indices = {drug: index for index, drug in enumerate(drug_ids)}
    node_indices = {node: index for index, node in enumerate(node_ids)}

    adjacency_matrix = torch.zeros((len(drug_ids), len(node_ids)), dtype=torch.float32)
    
    logger.info("Building adjacency matrix..")
    for edge in tqdm(edges):
        _, source_id = edge['source_id']
        _, target_id = edge['target_id']

        if source_id in drug_ids and target_id in node_ids:
            drug_index = drug_indices[source_id]
            node_index = node_indices[target_id]
            adjacency_matrix[drug_index, node_index] = 1.0

        elif target_id in drug_ids and source_id in node_ids:
            drug_index = drug_indices[target_id]
            node_index = node_indices[source_id]
            adjacency_matrix[drug_index, node_index] = 1.0

    return adjacency_matrix

def create_inverse_triplets(df: pd.DataFrame):
    """Create inverse triplets so that if (h,r,t) then (t,r,h) is also in the graph"""
    df_inv = df.copy()
    df_inv["drug_row"], df_inv["drug_col"] = df["drug_col"], df["drug_row"]
    df_combined = pd.concat([df, df_inv], ignore_index=True)
    return df_combined