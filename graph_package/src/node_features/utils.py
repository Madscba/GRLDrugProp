import json
import torch
from graph_package.utils.helpers import init_logger
from graph_package.configs.directories import Directories

logger = init_logger()

def filter_drug_gene_graph(drug_info, gene_degree=2):
    """
    Function for retrieving neigboring gene ID's in Hetionet for drugs in DrugComb
    """

    # Load Hetionet from json
    data_path = Directories.DATA_PATH / "node_features"
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

    logger.info("Filtering drugs in Hetionet")
    # Check each drug in drug_info and add it to filtered dict if found in drugs_in_graph
    for drug_name, identifiers in drug_info.items():
        for identifier_key, identifier_value in identifiers.items():
            for graph_drug_name, graph_identifiers in drugs_in_graph.items():
                if drug_name == graph_drug_name:
                    filtered_drug_dict[graph_drug_name] = graph_identifiers
                elif graph_identifiers.get(identifier_key) == identifier_value:
                    filtered_drug_dict[graph_drug_name] = graph_identifiers

    drug_ids = [filtered_drug_dict[drug]['DB'] for drug in filtered_drug_dict.keys()]
    logger.info(f"{len(drug_ids)} of {len(drug_info)} drugs found in Hetionet")

    # Step 2: Filter on genes related to drugs in the subset
    genes_connected_to_drugs = {}
    filtered_edges = []
    logger.info("Filtering genes in Hetionet")
    for edge in edges:
        source_type, source_id = edge['source_id']
        target_type, target_id = edge['target_id']

        if source_type == 'Compound' and target_type == 'Gene' and source_id in drug_ids:
            filtered_edges.append(edge)
            if target_id in genes_connected_to_drugs:
                genes_connected_to_drugs[target_id] += 1 
            else:
                genes_connected_to_drugs[target_id] = 1
    
    # Step 3: Filter genes based on degree
    genes_connected_to_at_least_one_drugs = [
        gene_id for gene_id, _ in genes_connected_to_drugs.items()
    ]
    gene_ids = [
        gene_id for gene_id, degree in genes_connected_to_drugs.items() if degree >= gene_degree
    ]
    avg_gene_degree = sum(genes_connected_to_drugs.values())/len(genes_connected_to_drugs)

    # Basic descriptive statistics on drug-gene graph
    logger.info(f"Number of genes connected to at least one drug: {len(genes_connected_to_at_least_one_drugs)}")
    logger.info(f"Number of genes connected to at least {gene_degree} drugs: {len(gene_ids)}")
    logger.info(f"Average degree of genes: {round(avg_gene_degree,3)}")

    return drug_ids, gene_ids, filtered_edges

def build_adjacency_matrix(drug_ids, gene_ids, edges):

    drug_indices = {drug: index for index, drug in enumerate(drug_ids)}
    gene_indices = {gene: index for index, gene in enumerate(gene_ids)}

    adjacency_matrix = torch.zeros((len(drug_ids), len(gene_ids)), dtype=torch.float32)
    
    logger.info("Building adjacency matrix")
    for edge in edges:
        _, source_id = edge['source_id']
        _, target_id = edge['target_id']

        if source_id in drug_ids and target_id in gene_ids:
            drug_index = drug_indices[source_id]
            gene_index = gene_indices[target_id]
            adjacency_matrix[drug_index, gene_index] = 1.0

    return adjacency_matrix
