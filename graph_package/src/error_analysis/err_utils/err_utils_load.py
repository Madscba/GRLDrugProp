import errno
import json
import os
import pickle
import typing as t
from datetime import date
from pathlib import Path

import pandas as pd

from graph_package.configs.directories import Directories
from graph_package.src.error_analysis.err_utils.err_utils_data_manipulation import (
    format_and_return_as_dataframes,
)


def get_saved_pred(e_conf: t.Dict):
    """Err diagnostics function to get saved predictions given model names and path"""
    p_pre = Directories.OUTPUT_PATH / "model_predictions"
    path_to_prediction_folder_and_pred_file_name_tuple = [
        (
            p_pre / c["day_of_prediction"] / "_".join([c["task"], c["target"]]),
            c["prediction_file_name"],
        )
        for _, c in e_conf.items()
    ]

    # pred_file_names = [err_config[i]["prediction_file_name"] for model in model_names]
    pred_dfs = [
        get_prediction_dataframe(pred_file, save_path=path_to_prediction_folder)
        for path_to_prediction_folder, pred_file in path_to_prediction_folder_and_pred_file_name_tuple
    ]
    return pred_dfs


def load_json(filepath):
    with open(filepath) as f:
        file = json.load(f)
    return file


def get_prediction_dataframe(file_name, save_path: Path):
    """
    Load prediction dictionary and turn into dataframe
    Parameters:
        file_name (str):
        save_path (Path):

    Returns:
        df (pd.DataFrame): dataframe with saved model predictions
    """
    pred_dict = get_model_pred_dict(file_name, save_path)
    df = format_and_return_as_dataframes(pred_dict)
    return df


def get_model_pred_dict(
    file_name="rescal_model_pred_dict.pkl", save_path: Path = Path("")
):
    """
    Save model predictions, alongside batch_idx, batch triplets

    Parameters:
        file_name (str): name of pickle file with model predictions
        save_path (Path): path to file folder

    Returns:
        pred_dict (dict): prediction dictionary (batch_idx, batch, predictions, targets)
    """
    save_path = get_model_pred_path(save_path)
    file_path = save_path / file_name
    try:
        with open(file_path, "rb") as file:
            pred_dict = pickle.load(file)
    except:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    return pred_dict


def get_rel_vocab():
    """
    Get relation vocabulary from data/gold and return as dataframe

    Parameters:
    Returns:
        df_rel_vocab (pd.DataFrame):
    """
    gold_root_path = Directories.DATA_PATH / "gold" / "oneil"
    df_rel_vocab = pd.DataFrame(
        load_json(gold_root_path / "relation_vocab.json").items(),
        columns=["rel_name", "rel_id"],
    )
    return df_rel_vocab


def get_ent_vocab():
    """
    Get entity vocabulary from data/gold and return as dataframe

    Parameters:
    Returns:
        df_ent_vocab (pd.DataFrame):
    """
    gold_root_path = Directories.DATA_PATH / "gold" / "oneil"
    df_ent_vocab = pd.DataFrame(
        load_json(gold_root_path / "entity_vocab.json").items(),
        columns=["drug_name", "drug_id"],
    )
    return df_ent_vocab


def get_cell_line_info():
    bronze_root_path = Directories.DATA_PATH / "bronze" / "drugcomb"
    return pd.DataFrame(load_json(bronze_root_path / "cell_line_dict.json")).T


def get_drug_info():
    bronze_root_path = Directories.DATA_PATH / "bronze" / "drugcomb"
    return pd.DataFrame(load_json(bronze_root_path / "drug_dict.json")).T


def get_model_pred_path(save_path: Path, config={}):
    """
    Dummy function to retrieve default save path if none is supplied

    Parameters:
        save_path (str): path to save object
        config (DictConfig): hydra config
    Returns:
        None (displays the plot).
    """
    today = date.today()
    today_str = today.strftime("%d_%m_%Y")
    if save_path == Path(""):
        task_target = "_".join([config.task, config.dataset.target])
        save_path = (
            Directories.OUTPUT_PATH / "model_predictions" / today_str / task_target
        )
    return save_path


def get_err_analysis_path(save_path, config):
    """
    Dummy function to retrieve default save path if none is supplied

    Parameters:
        save_path (str): path to save object
    Returns:
        None (displays the plot).
    """
    today = date.today()
    today_str = today.strftime("%d_%m_%Y")
    if not save_path:
        task_target = "_".join([config.task, config.dataset.target])
        save_path = Directories.OUTPUT_PATH / "err_analysis" / today_str / task_target
    return save_path
