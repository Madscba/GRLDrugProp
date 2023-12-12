import argparse
from graph_package.src.node_features.pca import generate_pca_feature_vectors

def parse_list_or_all(value):
    if value.lower() == 'all':
        return 'all'
    else:
        return value.split(',')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="pca")
    parser.add_argument("--dataset", type=str, default="ONEIL")
    parser.add_argument("--node_types", type=parse_list_or_all, 
                        help='Specify either "all" or a comma-separated list of node types',
                        default='all')
    parser.add_argument("--n_components", type=int, default=16)
    args = parser.parse_args()
    if (args.method == "pca"):
        generate_pca_feature_vectors(
            dataset=args.dataset,
            components=args.n_components,
            node_types=args.node_types
        ) 

if __name__ == "__main__":
    main()