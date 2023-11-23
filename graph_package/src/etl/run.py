from graph_package.src.etl.bronze import get_drugcomb
from graph_package.src.etl.silver import generate_oneil_dataset
import argparse
from graph_package.src.etl.gold import make_oneil_dataset, make_oneil_legacy_dataset,make_original_deepdds_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=str, default="full")
    args = parser.parse_args()
    if (args.update == "bronze") | (args.update == "full"):
        get_drugcomb()
    if (args.update == "silver") | (args.update == "full"):
        generate_oneil_dataset()
    if (args.update == "gold") | (args.update == "full"):
        make_oneil_dataset()
        make_oneil_legacy_dataset()
        make_original_deepdds_dataset()  

if __name__ == "__main__":
    main() 

