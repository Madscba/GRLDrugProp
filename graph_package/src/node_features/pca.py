import numpy as np
from utils import (
    filter_drugs_in_graph,
    filter_drug_node_graph, 
    build_adjacency_matrix, 
    load_drugs
)
from graph_package.utils.helpers import init_logger
from graph_package.configs.directories import Directories

from sklearn.decomposition import PCA

logger = init_logger()

def fit_pca_node_feature(adjacency_matrix,components,node_type,relation):
    # Generate PCA feature vectors
    drug_node_array = np.array(adjacency_matrix)
    pca = PCA(n_components=min(components,drug_node_array.shape[1]))
    pca_feature_vectors = pca.fit_transform(drug_node_array)

    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Calculate the accumulated explained variance
    accumulated_explained_variance = np.cumsum(explained_variance_ratio)

    # Print the accumulated explained variance
    logger.info(f"Acc explained variance for {node_type} - {relation} pca features: {accumulated_explained_variance[-1]}")

    return pca_feature_vectors

def generate_pca_feature_vectors(datasets=["ONEIL"], components=20, node_types="all", 
                                 across_relation=True):
    """
    Function for generating PCA feature vectors based on Drug-Gene graph
    """
    # Defining paths
    datasets_name = "oneil_almanac" if datasets==["ONEIL","ALMANAC"] else 'oneil'
    save_path = Directories.DATA_PATH / "gold" / datasets_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Get info dict with drug name, DrugBank ID and inchikey for each drug in dataset
    drug_info = load_drugs(datasets)

    # Filter drugs and nodes in Hetionet and build adjacency matrix
    drug_ids, filtered_drug_dict, edges = filter_drugs_in_graph(drug_info)
    db_ids = [filtered_drug_dict[drug]['DB'] for drug in filtered_drug_dict.keys()]
    node_types_list = ["Gene", "Disease", "Side Effect", "Pharmacologic Class"] if node_types == 'all' else node_types
    feature_vectors = np.zeros((len(db_ids),len(node_types_list),components))

    # For each node_type and relation, build adjacency matrix and get pca feature vectors
    for i, node_type in enumerate(node_types_list):
        node_ids, filtered_edges = filter_drug_node_graph(drug_ids=drug_ids,edges=edges,node=node_type,across_relation=across_relation)
        for node_id, relation in zip(node_ids,filtered_edges):
            filtered_edge = filtered_edges[relation]
            adjacency_matrix = build_adjacency_matrix(db_ids,node_id,filtered_edge)
            degree = [adjacency_matrix[i,:].sum() for i in range(adjacency_matrix.shape[0])]
            logger.info(f"Avg degree of drug nodes to {node_type}: {sum(degree)/len(degree)} (max is {adjacency_matrix.shape[1]})")
            
            # Fit PCA feature vectors
            pca_feature_vectors = fit_pca_node_feature(adjacency_matrix,components,node_type,relation)
            #feature_vectors[:,i,:] = pca_feature_vectors

    # Reshape feature vectors 
    feature_vectors = feature_vectors.reshape(len(db_ids),-1)

    # Save pca features to data/node_features as json/csv (drugDB,feature)
    drug_features = {
        drug.lower(): feature 
        for drug, feature in zip(filtered_drug_dict.keys(), feature_vectors.tolist())
    }
    #with open(save_path / f"{datasets_name}_drug_features.json", "w") as json_file:
    #        json.dump(drug_features, json_file)
    logger.info(f"Saved PCA feature vectors to {save_path}")

if __name__ == "__main__":
    generate_pca_feature_vectors(["ONEIL","ALMANAC"], components=10, node_types=[ "Gene", "Disease", "Side Effect", "Gene"])