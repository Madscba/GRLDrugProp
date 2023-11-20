import hashlib
import json
import os

from matplotlib import pyplot as plt
from scipy.special import expit

from graph_package.configs.directories import Directories

import matplotlib.pyplot as plt
import pickle
import pandas as pd, numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import seaborn as sns

from graph_package.src.main_utils import load_data


def get_performance_curve(
    y_true, y_scores, model_name, curve_type="roc", save_output=False, save_path=None
):
    """
    Plot ROC or precision-recall curve for binary classification.

    Parameters:
        y_true (array-like): Ground truth binary labels.
        y_scores (array-like): Predicted probabilities for the positive class.
        curve_type (str): Type of curve to plot ('roc' for AUC-ROC, 'pr' for AUC-PR).

    Returns:
        None (displays the plot).
    """
    if curve_type not in ["roc", "pr"]:
        raise ValueError(
            "Invalid curve_type. Use 'roc' for AUC-ROC curve or 'pr' for AUC-PR curve."
        )

    plt.figure(figsize=(8, 6))
    if curve_type == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        x, y = fpr, tpr
        fig_label = f"AUC-ROC curve (AUC = {roc_auc:.2f})"
        fig_title = "Receiver Operating Characteristic (ROC) Curve"
        xlabel, ylabel = "False Positive Rate", "True Positive Rate"
        legend_loc = "lower right"
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    elif curve_type == "pr":
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        x, y = recall, precision
        fig_label = f"AUC-PR curve (AUC = {pr_auc:.2f})"
        fig_title = "Precision-Recall Curve"
        xlabel, ylabel = "Recall", "Precision"
        legend_loc = "upper right"

    plt.plot(x, y, color="darkorange", lw=2, label=fig_label)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(fig_title)
    plt.legend(loc=legend_loc)

    if save_output:
        save_path = get_err_analysis_path(save_path)

        plt.savefig(
            save_path / f"{model_name}_{curve_type}_curve.png", bbox_inches="tight"
        )
    plt.show()


def convert_metrics_to_summary_table(
    callback_metrics, model_name, save_output=False, save_path=False
):
    """
    Convert metric dictionary to summary table

    Parameters:
        callback_metrics (dict): dict with metrics
        save_output (bool): Save summary table
        save_path (str): path to save summary table

    Returns:
        None (displays the plot).

    callback_metrics retrieved from trainer object trainer.callback_metrics
    alternatively, metrics from the test_step can be used as first argument
    """
    filtered_metrics = {
        key: val for key, val in callback_metrics.items() if "confusion" not in key
    }

    df = pd.DataFrame(list(filtered_metrics.items()), columns=["Metric", "Value"])
    df.loc[:, "Value"] = df.Value.apply(lambda x: np.round(float(x.numpy()), 3))
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")

    if save_output:
        save_path = get_err_analysis_path(save_path)
        plt.savefig(
            save_path / f"{model_name}_summary_metric_table.png", bbox_inches="tight"
        )


def get_err_analysis_path(save_path):
    """
    Dummy function to retrieve default save path if none is supplied

    Parameters:
        save_path (str): path to save object
    Returns:
        None (displays the plot).
    """
    if not save_path:
        save_path = Directories.OUTPUT_PATH / "err_analysis"
    return save_path


def get_model_pred_path(save_path):
    """
    Dummy function to retrieve default save path if none is supplied

    Parameters:
        save_path (str): path to save object
    Returns:
        None (displays the plot).
    """
    if not save_path:
        save_path = Directories.OUTPUT_PATH / "model_predictions"
    return save_path


def get_confusion_matrix_heatmap(values, save_output, model_name, save_path=False):
    """
    Turn confusion matrix into heatmap w. values

    Parameters:
        values (2x2 array): confusion matrix values
        save_output (str): save heatmap
        save_path (str): path to save object

    Returns:
        save_path (str)
    """
    heatmap = sns.heatmap(values, annot=True, cmap="Spectral", fmt="d").get_figure()
    fig = heatmap.get_figure()
    ax = fig.gca()
    ax.set_xlabel("predicted values")
    ax.set_ylabel("actual values")
    if save_output:
        save_path = get_err_analysis_path(save_path)

        fig.savefig(save_path / f"{model_name}_conf_matrix.png")
    return fig


def save_model_pred(batch_idx, batch, preds, target, model_name, save_path=False):
    """
    Save model predictions, alongside batch_idx, batch triplets

    Parameters:
        batch_idx (int): Ground truth binary labels.
        batch (List[List[int]]): List of all relations (each a list of 4 integers) included in batch
        preds (str): model predictions
        target (str): ground truth
        save_path (str): path to save object

    Returns:
        None: Function saves dictionary with all of the above information as a pickle file
    """
    save_path = get_model_pred_path(save_path)
    if not save_path.exists():
        save_path.mkdir(exist_ok=True, parents=True)
    output_dict = {
        "batch_id": batch_idx,
        "batch": batch,
        "predictions": preds,
        "targets": target,
    }

    with open(save_path / f"{model_name}_model_pred_dict.pkl", "wb") as f:
        pickle.dump(output_dict, f)


def get_model_pred_dict(file_name="rescal_model_pred_dict.pkl", save_path=""):
    """
    Save model predictions, alongside batch_idx, batch triplets

    Parameters:
        file_name (str): name of pickle file with model predictions
        save_path (str): path to file folder

    Returns:
        pred_dict (dict): prediction dictionary (batch_idx, batch, predictions, targets)
    """
    save_path = get_model_pred_path(save_path)
    with open(save_path / file_name, "rb") as file:
        pred_dict = pickle.load(file)
    return pred_dict


def save_performance_plots(df_cm, metrics, preds, target, model_name, save_path=""):
    """
    Create and save confusion-matrix heatmap, roc- and pr curves and summary table as figures

    Parameters:
        df_cm (int): confusion matrix values
        metrics (dict): dict with metrics
        preds (str): model predictions
        target (str): ground truth
        save_path (str): path to save object

    Returns:
        None: Function saves dictionary with all of the above information as a pickle file
    """
    save_path = get_err_analysis_path(save_path)
    if not save_path.exists():
        save_path.mkdir(exist_ok=True, parents=True)
    get_confusion_matrix_heatmap(
        values=df_cm, save_output=True, save_path=save_path, model_name=model_name
    )
    get_performance_curve(
        target,
        preds,
        curve_type="roc",
        save_output=True,
        save_path=save_path,
        model_name=model_name,
    )
    get_performance_curve(
        target,
        preds,
        curve_type="pr",
        save_output=True,
        save_path=save_path,
        model_name=model_name,
    )
    convert_metrics_to_summary_table(
        metrics, save_output=True, save_path=save_path, model_name=model_name
    )


def find_best_model_ckpt(ckpt_folder):
    """
    Iterate over model checkpoints to find the best checkpoint and return path to it.

    Parameters:
        ckpt_folder (str): starting point for search (os.walk) is performed to check all ckpts

    Returns:
        max_checkpoint (str): path to best model checkpoint
    """
    checkpoints, scores = [], []
    for path, subdirs, files in os.walk(ckpt_folder):
        for name in files:
            scores.append(name[-11:-5])
            checkpoints.append(name)
    best_idx = np.where(np.array(scores) == max(scores))[0][0]
    max_checkpoint = ckpt_folder / f"fold_{best_idx}" / checkpoints[best_idx]
    return max_checkpoint


def barplot_mean_correct_prediction_grouped_by_entity(
    pred_dfs, model_names, group_by_columns, title
):
    """
    Generate and save barplot

    Parameters:
        pred_dfs (List[pd.DataFrame): prediction dataframe with side info
        model_names (List[str]): List of models

    Returns:
        None
    """
    plt_colors = ["red", "blue", "orange"]
    save_path = Directories.OUTPUT_PATH / "err_diagnostics"
    if not save_path.exists():
        save_path.mkdir(exist_ok=True, parents=True)

    for i, df in enumerate(pred_dfs):
        grouped_df = df.groupby(group_by_columns).agg({"correct_pred": "mean"})
        sorted_df = sort_df_by_mean_correct_pred(grouped_df)
        top20_df = sorted_df.tail(20)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        sorted_df.plot(kind="bar", ax=plt.gca(), color=plt_colors[i])
        plt.xticks(rotation=90)
        plt.title(f"{title}\nmean correct prediction")
        plt.legend([model_names[i]])

        plt.subplot(1, 2, 2)
        top20_df.plot(kind="bar", ax=plt.gca(), color=plt_colors[i])
        plt.xticks(rotation=90)
        plt.title(f"{title}\ntop 20 w. lowest mean correct prediction")
        plt.legend([model_names[i]])

        plt.savefig(save_path / f"{title}_bar_{model_names[i]}")
        plt.show()
        plt.clf()


def sort_df_by_mean_correct_pred(df):
    return df.sort_values(by="correct_pred", ascending=False)


def map_to_index(row, cols):
    """
    Function to generate unique hash index for a combination of rows. Meant to be applied to each row of a dataframe.

    Parameters:
        row (pd.DataFrame row):
        cols (List[str]): Columns to map to id

    Returns:
        index (int): unique id
    """
    sorted_values = tuple(sorted([str(row[col]) for col in cols]))
    value_string = ",".join(sorted_values)
    index = int(hashlib.md5(value_string.encode("utf-8")).hexdigest(), 16)
    return index


def load_json(filepath):
    with open(filepath) as f:
        file = json.load(f)
    return file


def get_prediction_dataframe(file_name, save_path=""):
    """
    Load prediction dictionary and turn into dataframe
    Parameters:
        file_name (str):
        save_path (str):

    Returns:
        df (pd.DataFrame): dataframe with saved model predictions
    """
    pred_dict = get_model_pred_dict(file_name, save_path)
    dataframes = [
        pd.DataFrame(ent)
        for ent in [
            pred_dict["batch"][0].numpy(),
            pred_dict["predictions"],
            pred_dict["targets"],
        ]
    ]
    df = pd.concat(dataframes, axis=1)
    df.columns = [
        "drug_molecules_left_id",
        "drug_molecules_right_id",
        "context_features_id",
        "predictions",
        "targets",
    ]
    return df


def get_bce_loss(pred_dfs):
    for df in pred_dfs:
        df["bce_loss"] = -(
            df["targets"] * np.log(df["pred_prob"])
            + (1 - df["targets"]) * np.log(1 - df["pred_prob"])
        )


def generate_node_degree(file_name, save_path):
    """
    Return node degree of each drugs.

    Generate pickle file containing node degrees for each node if it does not exist.

    Parameters:
        file_name (str):
        save_path (str):

    Returns:
        node_degree (np.array()): list of node degrees.
    """
    dataset = load_data(dataset="oneil")
    node_degree = (dataset.graph.degree_in + dataset.graph.degree_out).numpy()
    if not save_path.exists():
        save_path.mkdir(exist_ok=True, parents=True)
    with open(save_path / file_name, "wb") as f:
        pickle.dump(node_degree, f)
    return node_degree


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


def merge_cell_line_and_drug_info(df, cell_line_meta_data, drug_meta_data):
    """
    Merge the meta data from two the two dataframes (cell_line_meta_data, drug_meta_data) with the prediction dataframe df and return it

    Parameters:
        df (pd.DataFrame): Dataframe containing model prediction and targets
        cell_line_meta_data (pd.DataFrame): - cancer cell line meta data
        drug_meta_data (pd.DataFrame): - drug meta data
    Returns:
        df (pd.DataFrame):
    """
    df = pd.merge(
        df,
        drug_meta_data,
        left_on="drug_molecules_left_id",
        right_on="id",
        suffixes=("", "_left"),
    )
    df.drop("id", inplace=True, axis=1)
    df = pd.merge(
        df,
        drug_meta_data,
        left_on="drug_molecules_right_id",
        right_on="id",
        suffixes=("", "_right"),
    )
    df.drop("id", inplace=True, axis=1)
    df = pd.merge(df, cell_line_meta_data, left_on="context_features_id", right_on="id")
    return df


def merge_vocabs_with_predictions(df, df_ent_vocab, df_rel_vocab):
    """
    Merge the vocabulary info from two the two dataframes (df_ent_vocab, df_rel_vocab) with the prediction dataframe df and return it

    Parameters:
        df (pd.DataFrame): Dataframe containing model prediction and targets
        df_ent_vocab (pd.DataFrame): - entity/drug index and name
        df_rel_vocab (pd.DataFrame): - relation/cancer cell index and name
    Returns:
        df (pd.DataFrame):
    """
    df = df.merge(
        df_ent_vocab, left_on="drug_molecules_left_id", right_on="drug_id"
    ).rename(columns={"drug_name": "drug_name_left"})
    df.drop(["drug_id"], inplace=True, axis=1)
    df = df.merge(
        df_ent_vocab, left_on="drug_molecules_right_id", right_on="drug_id"
    ).rename(columns={"drug_name": "drug_name_right"})
    df.drop(["drug_id"], inplace=True, axis=1)
    df = df.merge(df_rel_vocab, left_on="context_features_id", right_on="rel_id")
    df.drop("rel_id", inplace=True, axis=1)
    return df


def get_cell_line_info():
    bronze_root_path = Directories.DATA_PATH / "bronze" / "drugcomb"
    return pd.DataFrame(load_json(bronze_root_path / "cell_line_dict.json")).T


def get_drug_info():
    bronze_root_path = Directories.DATA_PATH / "bronze" / "drugcomb"
    return pd.DataFrame(load_json(bronze_root_path / "drug_dict.json")).T


def get_node_degree():
    """
    Retrieve node degree of each drug.

    Returns:
        node_degree (np.array()): degree (in and out) of each drug.
    """
    save_path = Directories.DATA_PATH / "gold" / "node_attributes"
    file_name = "node_degree.pickle"
    if not os.path.exists(save_path / file_name):
        generate_node_degree(file_name, save_path)
    with open(save_path / file_name, "rb") as file:
        node_degree = pickle.load(file)
    return node_degree


def enrich_model_predictions(model_names, pred_dfs):
    """
    Enrich prediction dataframe with vocabulary and meta information on both drug and cancer cell line level.
    Parameters:
        model_names:
        pred_dfs:

    Returns:
        combined_df (pd.DataFrame): Dataframe with predictions from all models
        pred_dfs (List[pd.DataFrame]) List of dataframes each dataframe holding the prediction from one model.
    """
    # enrich df
    df_ent_vocab = get_ent_vocab()
    df_rel_vocab = get_rel_vocab()
    cell_line_meta_data = get_cell_line_info()
    drug_meta_data = get_drug_info()
    new_pred_dfs = []
    for idx, df in enumerate(pred_dfs):
        df["pred_prob"] = df["predictions"].apply(expit)
        df["pred_thresholded"] = df["pred_prob"].apply(lambda x: 1 if x > 0.5 else 0)
        df["correct_pred"] = np.isclose(df["pred_thresholded"], df["targets"]).astype(
            int
        )
        df["drug_pair_idx"] = df.apply(
            map_to_index,
            axis=1,
            cols=["drug_molecules_left_id", "drug_molecules_right_id"],
        )
        df["triplet_idx"] = df.apply(
            map_to_index,
            axis=1,
            cols=[
                "drug_molecules_left_id",
                "drug_molecules_right_id",
                "context_features_id",
            ],
        )
        df["model_name"] = model_names[idx]

        df = merge_vocabs_with_predictions(df, df_ent_vocab, df_rel_vocab)
        df = merge_cell_line_and_drug_info(df, cell_line_meta_data, drug_meta_data)
        new_pred_dfs.append(df)
    get_bce_loss(pred_dfs)
    pred_dfs = new_pred_dfs
    del new_pred_dfs
    combined_df = pd.concat(pred_dfs)
    return combined_df, pred_dfs
