import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from utils import (
    filter_drugs_in_graph,
    filter_drug_node_graph, 
    build_adjacency_matrix, 
    create_inverse_triplets
)
from graph_package.utils.helpers import init_logger
from graph_package.configs.directories import Directories

from sklearn.decomposition import PCA

logger = init_logger()

def generate_pca_feature_vectors(datasets=["ONEIL"], components=20, node_types="all"):
    """
    Function for generating PCA feature vectors based on Drug-Gene graph
    """
    # Defining paths
    save_path = Directories.DATA_PATH / "node_features"
    save_path.mkdir(parents=True, exist_ok=True)

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

    # Filter drugs and nodes in Hetionet and build adjacency matrix
    drug_ids, filtered_drug_dict, edges = filter_drugs_in_graph(drug_info)
    db_ids = [filtered_drug_dict[drug]['DB'] for drug in filtered_drug_dict.keys()]
    node_types_list = ["Gene", "Disease", "Side Effect", "Pharmacologic Class"] if node_types == 'all' else node_types
    feature_vectors = np.zeros((len(db_ids),len(node_types_list),components))
    for i, node_type in enumerate(node_types_list):
        node_ids, filtered_edges = filter_drug_node_graph(drug_ids=drug_ids,edges=edges,node=node_type)
        adjacency_matrix = build_adjacency_matrix(db_ids,node_ids,filtered_edges)
        degree = [adjacency_matrix[i,:].sum() for i in range(adjacency_matrix.shape[0])]
        logger.info(f"Avg degree of drug nodes to {node_type}: {sum(degree)/len(degree)} (max is {adjacency_matrix.shape[1]})")
        
        # Generate PCA feature vectors
        drug_node_array = np.array(adjacency_matrix)
        pca = PCA(n_components=min(components,drug_node_array.shape[1]))
        pca_feature_vectors = pca.fit_transform(drug_node_array)
        feature_vectors[:,i,:] = pca_feature_vectors

        # Get the explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_

        # Calculate the accumulated explained variance
        accumulated_explained_variance = np.cumsum(explained_variance_ratio)

        # Print the accumulated explained variance
        logger.info(f"Acc explained variance for {node_type} pca features: {accumulated_explained_variance[-1]}")

    # Reshape feature vectors 
    feature_vectors = feature_vectors.reshape(len(db_ids),-1)

    # Save pca features to data/node_features as json/csv (drugDB,feature)
    drug_features = {
        drug.lower(): feature 
        for drug, feature in zip(filtered_drug_dict.keys(), feature_vectors.tolist())
    }
    with open(save_path / f"{datasets}_drug_features.json", "w") as json_file:
            json.dump(drug_features, json_file)
    logger.info(f"Saved PCA feature vectors to {save_path}!")

if __name__ == "__main__":
    generate_pca_feature_vectors(["ONEIL","ALMANAC"],components=10, node_types=[ "Disease", "Side Effect", "Gene"])