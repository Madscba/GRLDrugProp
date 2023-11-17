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


def get_performance_curve(
    y_true, y_scores, curve_type="roc", save_output=False, save_path=None
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

        plt.savefig(save_path / f"{curve_type}_curve.png", bbox_inches="tight")
    plt.show()


def convert_metrics_to_summary_table(
    callback_metrics, save_output=False, save_path=False
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
        plt.savefig(save_path / "summary_metric_table.png", bbox_inches="tight")


def get_err_analysis_path(save_path):
    """
    Dummy function to retrieve default save path if none is supplied

    Parameters:
        save_path (str): path to save object
    Returns:
        None (displays the plot).
    """
    if not save_path:
        save_path = Directories.ERROR_ANALYSIS_PATH
    return save_path


def get_confusion_matrix_heatmap(values, save_output, save_path=False):
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

        fig.savefig(save_path / "conf_matrix.png")
    return fig


def save_model_pred(batch_idx, batch, preds, target, save_path=False):
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
    save_path = get_err_analysis_path(save_path)
    if not save_path.exists():
        save_path.mkdir(exist_ok=True, parents=True)
    output_dict = {
        "batch_id": batch_idx,
        "batch": batch.numpy(),
        "predictions": preds,
        "targets": target,
    }

    with open(save_path / "model_pred_dict.pkl", "wb") as f:
        pickle.dump(output_dict, f)


def get_model_pred(file_name="model_pred_dict.pkl", save_path=""):
    save_path = get_err_analysis_path(save_path)
    with open(save_path / file_name, "rb") as file:
        pred_dict = pickle.load(file)
    return pred_dict


def save_performance_plots(df_cm, metrics, preds, target, save_path=""):
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
    get_confusion_matrix_heatmap(values=df_cm, save_output=True, save_path=save_path)
    get_performance_curve(
        target, preds, curve_type="roc", save_output=True, save_path=save_path
    )
    get_performance_curve(
        target, preds, curve_type="pr", save_output=True, save_path=save_path
    )
    convert_metrics_to_summary_table(metrics, save_output=True, save_path=save_path)
