from graph_package.src.etl.medallion.bronze import get_drugcomb
from graph_package.src.etl.medallion.silver import generate_oneil_almanac_dataset
import argparse
from graph_package.src.etl.medallion.gold import make_oneil_almanac_dataset
from graph_package.src.etl.feature_engineering.node_features import make_node_features
from graph_package.src.etl.feature_engineering.drug_features import (
    make_drug_fingerprint_features,
)
from graph_package.src.etl.feature_engineering.cell_line_features import (
    make_cell_line_features,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=str, default="full")
    args = parser.parse_args()
    if (args.update == "bronze") | (args.update == "full"):
        get_drugcomb()
    if (args.update == "silver") | (args.update == "full"):
        generate_oneil_almanac_dataset()
    if (args.update == "feature") | (args.update == "full"):
        make_cell_line_features()
        make_node_features()
        make_drug_fingerprint_features()

    if (args.update == "gold") | (args.update == "full"):
        make_oneil_almanac_dataset()


if __name__ == "__main__":
    main()
