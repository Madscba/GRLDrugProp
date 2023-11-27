import argparse
from graph_package.src.node_features.pca import generate_pca_feature_vectors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="pca")
    parser.add_argument("--dataset", type=str, default="ONEIL")
    parser.add_argument("--n_components", type=int, default=16)
    args = parser.parse_args()
    if (args.method == "pca"):
        generate_pca_feature_vectors(
            dataset=args.dataset,
            components=args.n_components
        ) 

if __name__ == "__main__":
    main()