import json
import torch
import pandas as pd
from tqdm import tqdm
from graph_package.utils.helpers import init_logger
from graph_package.configs.directories import Directories

logger = init_logger()

def load_drugs(datasets=["ONEIL","ALMANAC"]):
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "summary_v_1_5.csv"
    dict_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "drug_dict.json"

    # Load drug dict from DrugComb with metadata on drugs
    with open(dict_path) as f:
        drug_dict = json.load(f)

    # Load ONEIL dataset (or bigger dataset)
    drugs = pd.read_csv(data_path)
    drugs = drugs[drugs['study_name'].isin(datasets)]
    drugs = create_inverse_triplets(drugs)
    names = [drug_dict[drug]['dname'] for drug in drugs.drug_row.unique()]

    # Make info dict with drug name, DrugBank ID and inchikey for each drug
    drug_info = {}
    for drug in names:
        drug_name = drug.capitalize() if drug[0].isalpha() else drug
        drug_info[drug_name] = {}
        drug_info[drug_name]['inchikey'] = drug_dict[drug]['inchikey'] 
        drug_info[drug_name]['DB'] = drug_dict[drug]['drugbank_id']

    return drug_info


def filter_drugs_in_graph(drug_info):
    """
    Function for filtering drugs in DrugComb found in Hetionet
    """

    # Load Hetionet from json
    data_path = Directories.DATA_PATH / "gold" / "hetionet"
    logger.info("Loading Hetionet..")
    with open(data_path / "hetionet-v1.0.json") as f:
        graph = json.load(f)
    nodes = graph['nodes']
    edges = graph['edges']

    # Select subset of drug IDs corresponding to drug IDs that exist in the graph
    drugs_in_graph = {}
    for node in nodes:
        if node['kind'] == 'Compound':
            drugs_in_graph[node['name']] = {}
            drugs_in_graph[node['name']]['inchikey'] = node['data']['inchikey'][9:]
            drugs_in_graph[node['name']]['DB'] = node['identifier']

    # Check each drug in drug_info and add it to filtered dict if found in drugs_in_graph
    filtered_drug_dict = {}
    logger.info("Filtering drugs in Hetionet..")
    for drug_name, identifiers in drug_info.items():
        for identifier_key, identifier_value in identifiers.items():
            for graph_drug_name, graph_identifiers in drugs_in_graph.items():
                if drug_name == graph_drug_name:
                    filtered_drug_dict[drug_name] = graph_identifiers
                elif graph_identifiers.get(identifier_key) == identifier_value:
                    filtered_drug_dict[drug_name] = graph_identifiers

    drug_ids = [filtered_drug_dict[drug]['DB'] for drug in filtered_drug_dict.keys()]
    logger.info(f"{len(drug_ids)} of {len(drug_info)} drugs found in Hetionet")
    drug_edges = [e for e in edges if (e['target_id'][0] =='Compound') or (e['source_id'][0] =='Compound')]
    return drug_ids, filtered_drug_dict, drug_edges
    
def add_connected_node(edge, r, nodes_connected_to_drugs,filtered_edges, target_id):
    if not r in nodes_connected_to_drugs:
        nodes_connected_to_drugs[r] = {}
        filtered_edges[r] = []
    filtered_edges[r].append(edge)
    # Check if a drug is already connected to this node
    if target_id in nodes_connected_to_drugs[r]:
        nodes_connected_to_drugs[r][target_id] += 1
    else:
        nodes_connected_to_drugs[r][target_id] = 1
    return nodes_connected_to_drugs, filtered_edges

def get_connected_nodes_ids(nodes_connected_to_drugs, min_degree):
    nodes_connected_to_at_least_one_drugs = [
        [node_id for node_id, _ in nodes_connected_to_drugs[rel].items()] 
        for rel in nodes_connected_to_drugs
    ]
    node_ids = [
        [node_id for node_id, degree in nodes_connected_to_drugs[rel].items() if degree >= min_degree]
        for rel in nodes_connected_to_drugs
    ]
    avg_node_degree = [sum(nodes_connected_to_drugs[r].values())/len(nodes_connected_to_drugs[r]) for r in nodes_connected_to_drugs]
    return nodes_connected_to_at_least_one_drugs, node_ids, avg_node_degree

def filter_drug_node_graph(drug_ids, edges, node='Gene', min_degree=2, across_relation=True):
    """
    Function for retrieving neigboring node ID's in Hetionet for drugs in DrugComb
    """

    # Step 2: Filter on nodes related to drugs in the subset
    nodes_connected_to_drugs = {}
    filtered_edges = {}
    logger.info(f"Filtering {node} in Hetionet..")
    for edge in tqdm(edges):
        source_type, source_id = edge['source_id']
        target_type, target_id = edge['target_id']
        r = edge['kind'] # Type of relation
        if node == 'Compound':
            # Case where a drug is connected to another drug through the 'resembles' relation
            if r=='resembles':
                nodes_connected_to_drugs, filtered_edges = add_connected_node(
                    edge,
                    r,
                    nodes_connected_to_drugs,
                    filtered_edges,
                    target_id
                )
        else:
            # Case where a drug is connected to the specified node through the relation r as source/head
            if (source_type == 'Compound' and target_type == node and source_id in drug_ids):
                nodes_connected_to_drugs, filtered_edges = add_connected_node(
                    edge,
                    r,
                    nodes_connected_to_drugs,
                    filtered_edges,
                    target_id
                )
            # Case where a drug is connected to the specified node through the relation r as target/tail
            elif source_type == node and target_type == 'Compound'  and target_id in drug_ids:
                nodes_connected_to_drugs, filtered_edges = add_connected_node(
                    edge,
                    r,
                    nodes_connected_to_drugs,
                    filtered_edges,
                    target_id
                )
    
    # Step 3: Filter nodes based on degree
    nodes_connected_to_at_least_one_drugs, node_ids, avg_node_degree = get_connected_nodes_ids(
        nodes_connected_to_drugs,
        min_degree
    )
    if across_relation:
        nodes_connected_to_at_least_one_drugs_all = [
            item for sublist in nodes_connected_to_at_least_one_drugs for item in sublist
        ]
        filtered_edges_all = [filtered_edges[r] for r in filtered_edges]
        filtered_edges_all = [item for sublist in filtered_edges_all for item in sublist]
        node_ids_all = [item for sublist in node_ids for item in sublist]
        avg_node_degree_all = sum(avg_node_degree)/len(avg_node_degree)

    # Basic descriptive statistics on drug-node graph (either per relation or across)
    nodes_connected_to_at_least_one_drugs.append(nodes_connected_to_at_least_one_drugs_all),
    node_ids.append(node_ids_all),
    avg_node_degree.append(avg_node_degree_all)
    filtered_edges['all'] = filtered_edges_all
    for r, nodes_connected, node_id, avg_degree in zip(
        nodes_connected_to_drugs,
        nodes_connected_to_at_least_one_drugs,
        node_ids,
        avg_node_degree
    ):
        logger.info(f"Number of {node} - {r} connected to at least one drug: {len(nodes_connected)}")
        logger.info(f"Number of {node} - {r} connected to at least {min_degree} drugs: {len(node_id)}")
        logger.info(f"Avg degree of {node} - {r} nodes to drugs: {round(avg_degree,3)}")
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