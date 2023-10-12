from graph_package.configs.definitions import Directories
from graph_package.utils.helpers import init_logger
import pandas as pd

logger = init_logger()

def load_drugcomb():
    data_path = Directories.DATA_PATH / "bronze" / "drugcomb" / "summary_v_1_5.csv"
    return pd.read_csv(data_path)

def generate_oneil_dataset():
    """
    Generate the Oniel dataset from the DrugComb dataset.
    """
    df = load_drugcomb()

    df_oneil = df[df["study_name"] == "ONEIL"]
    df_oneil_cleaned = df_oneil.dropna(subset=["drug_row", "drug_col", "synergy_loewe"])
    logger.info(f"Dropped {len(df_oneil)-len(df_oneil_cleaned)} NaN values.")
    oneil_path = Directories.DATA_PATH / "silver" / "oneil"
    oneil_path.mkdir(exist_ok=True)
    df_oneil_cleaned = df_oneil_cleaned.loc[
        :, ~df_oneil_cleaned.columns.str.startswith("Unnamed")
    ]
    df_oneil_cleaned.to_csv(oneil_path / "oneil.csv", index=False)


if __name__ == "__main__":
    generate_oneil_dataset()