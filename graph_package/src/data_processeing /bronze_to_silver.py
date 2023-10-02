from graph_package.configs.directories import Directories
import pandas as pd
import numpy as np
from graph_package.utils.helpers import init_logger

logger = init_logger()

def load_drugcomb(dataset: str = "drugcomb"):
    assert dataset in ["drugcomb"]
    data_path = Directories.DATA_PATH / dataset / "summary_v_1_5.csv"
    return pd.read_csv(data_path)

def generate_oneil_dataset():
    """
    Generate the Oniel dataset from the DrugComb dataset.
    """
    df = load_drugcomb()
    df_oneil = df[df["study_name"]== "ONEIL"]
    df_oneil_cleaned = df_oneil.dropna()
    logger.log(1, f"Dropped {len(df_oneil)-len(df_oneil_cleaned)} NaN values.")
    oneil_path = Directories.DATA_PATH / "oneil"
    oneil_path.mkdir(exist_ok=True)
    df_oneil_cleaned.to_csv(oneil_path / "oneil.csv")    

    


if __name__ == "__main__":
    generate_oneil_dataset()


    