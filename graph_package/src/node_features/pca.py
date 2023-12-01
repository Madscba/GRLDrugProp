import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from utils import (filter_drug_gene_graph, build_adjacency_matrix, create_inverse_triplets)
from graph_package.utils.helpers import init_logger
from graph_package.configs.directories import Directories

from sklearn.decomposition import PCA

logger = init_logger()

def generate_pca_feature_vectors(dataset="ONEIL", components=20):
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
    drugs = drugs[drugs['study_name']==dataset]
    drugs = create_inverse_triplets(drugs)
    names = [drug_dict[drug]['dname'] for drug in drugs.drug_row.unique()]

    # Make info dict with drug name, DrugBank ID and inchikey for each drug
    drug_info = {}
    for drug in names:
        drug_name = drug.capitalize() if drug[0].isalpha() else drug
        drug_info[drug_name] = {}
        drug_info[drug_name]['inchikey'] = drug_dict[drug]['inchikey'] 
        drug_info[drug_name]['DB'] = drug_dict[drug]['drugbank_id']

    # Filter drugs and genes in Hetionet and build adjacency matrix
    filtered_drug_dict, gene_ids, edges = filter_drug_gene_graph(drug_info,gene_degree=4)
    drug_ids = [filtered_drug_dict[drug]['DB'] for drug in filtered_drug_dict.keys()]
    adjacency_matrix = build_adjacency_matrix(drug_ids,gene_ids,edges)
    degree = [adjacency_matrix[i,:].sum() for i in range(adjacency_matrix.shape[0])]
    logger.info(f"Average drug degree: {sum(degree)/len(degree)} (max is {adjacency_matrix.shape[1]})")
    
    # Generate PCA feature vectors
    drug_gene_array = np.array(adjacency_matrix)
    pca = PCA(n_components=components)
    pca_feature_vectors = pca.fit_transform(drug_gene_array)

    # Plot the first two principal components
    plt.scatter(pca_feature_vectors[:, 0], pca_feature_vectors[:, 1])
    plt.title('PCA: First Two Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Calculate the accumulated explained variance
    accumulated_explained_variance = np.cumsum(explained_variance_ratio)

    # Print or use the explained variance ratio as needed
    print("Explained Variance Ratio:", explained_variance_ratio)
    print("Accumulated Explained Variance:", accumulated_explained_variance)

    # Save pca features to data/node_features as json/csv (drugDB,feature)
    drug_features = {
        drug.lower(): feature 
        for drug, feature in zip(filtered_drug_dict.keys(), pca_feature_vectors.tolist())
    }
    with open(save_path / f"{dataset}_drug_features.json", "w") as json_file:
            json.dump(drug_features, json_file)
    logger.info(f"Saved PCA feature vectors to {save_path}")

if __name__ == "__main__":
    generate_pca_feature_vectors("ONEIL",20)