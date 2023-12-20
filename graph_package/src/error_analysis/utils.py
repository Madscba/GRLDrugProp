import json
import errno
import os

from scipy.special import expit

from graph_package.configs.directories import Directories

import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
import seaborn as sns
from datetime import date
from graph_package.src.main_utils import load_data
import numpy as np
import torch
from pathlib import Path
import typing as t


def get_performance_curve(
    y_true,
    y_scores,
    model_name,
    config,
    curve_type="roc",
    save_output=False,
    save_path=None,
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
        save_path = get_err_analysis_path(save_path, config)

        plt.savefig(
            save_path / f"{model_name}_{curve_type}_curve.png", bbox_inches="tight"
        )
    plt.show()


def convert_metrics_to_summary_table(
    callback_metrics, config, model_name, save_output=False, save_path=False
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
        key: val for key, val in callback_metrics.items() if "CM" not in key
    }

    df = pd.DataFrame(list(filtered_metrics.items()), columns=["Metric", "Value"])
    df.loc[:, "Value"] = df.Value.apply(lambda x: np.round(float(x.numpy()), 3))
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")

    if save_output:
        save_path = get_err_analysis_path(save_path, config)
        plt.savefig(
            save_path / f"{model_name}_summary_metric_table.png", bbox_inches="tight"
        )


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


def get_roc_visualization(y_true, y_pred_probability):
    from sklearn.metrics import RocCurveDisplay

    RocCurveDisplay.from_predictions(
        y_true,
        y_pred_probability,
        name=f"test vs the rest",
        color="darkorange",
        plot_chance_level=True,
    )
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
    plt.legend()
    plt.savefig("roc_curve")
    plt.show()


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


def get_confusion_matrix_heatmap(
    values, save_output, config, model_name, save_path=False
):
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
        save_path = get_err_analysis_path(save_path, config)

        fig.savefig(save_path / f"{model_name}_conf_matrix.png")
    return fig


def save_model_pred(batch_idx, batch, preds, target, config, model_name, save_path=""):
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
    output_dict = {
        "batch_id": [batch_idx],
        "batch": batch,
        "predictions": [preds],
        "targets": [target],
    }

    save_path = get_model_pred_path(save_path, config)
    if not save_path.exists():
        print("making a space to save preds: ", save_path)
        save_path.mkdir(exist_ok=True, parents=True),

    pred_path = save_path / f"{model_name}_model_pred_dict.pkl"
    if pred_path.exists():
        with open(save_path / f"{model_name}_model_pred_dict.pkl", "rb") as f:
            old_output_dict = pickle.load(f)

        # append old predictions with new
        for key, val in old_output_dict.items():
            concatenated_input = old_output_dict[key] + output_dict[key]
            output_dict.update({key: concatenated_input})

    with open(save_path / f"{model_name}_model_pred_dict.pkl", "wb") as f:
        print("saving at: ", save_path / f"{model_name}_model_pred_dict.pkl")
        pickle.dump(output_dict, f)


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


def save_performance_plots(
    df_cm, metrics, preds, target, config, model_name, save_path=""
):
    """
    Create and save confusion-matrix heatmap, roc- and pr curves and summary table as figures

    Parameters:
        df_cm (int): confusion matrix values
        metrics (dict): dict with metrics
        preds (str): model predictions
        target (str): ground truth
        config (DictConfig): hydra config
        model_name (str): model name
        save_path (str): path to save object

    Returns:
        None: Function saves dictionary with all of the above information as a pickle file
    """
    save_path = get_err_analysis_path(save_path, config)
    if not save_path.exists():
        save_path.mkdir(exist_ok=True, parents=True)
    get_confusion_matrix_heatmap(
        values=df_cm,
        save_output=True,
        save_path=save_path,
        config=config,
        model_name=model_name,
    )
    get_performance_curve(
        target,
        preds,
        curve_type="roc",
        save_output=True,
        save_path=save_path,
        config=config,
        model_name=model_name,
    )
    get_performance_curve(
        target,
        preds,
        curve_type="pr",
        save_output=True,
        save_path=save_path,
        config=config,
        model_name=model_name,
    )
    convert_metrics_to_summary_table(
        metrics,
        save_output=True,
        save_path=save_path,
        config=config,
        model_name=model_name,
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


def barplot_aucroc_grouped_by_entity(
    pred_dfs,
    model_names,
    group_by_columns,
    title,
    xlabel_col_name,
    add_bar_info: bool = True,
    run_name: str = "",
    metric_name: str = "MSE",
    task: str = "reg",
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

    grouped_dfs = []
    for i, df in enumerate(pred_dfs):
        if xlabel_col_name == "triplet_name":
            grouped_df = df.groupby(group_by_columns).agg(
                {metric_name: "mean", xlabel_col_name: "max"}
            )
            original_size, filtered_size = 1, 1
        else:
            # Goal is to say something about the difficulty of specific entities relative to others of the same kind
            grouped_df = df.groupby(group_by_columns)

            if metric_name == "AUC_ROC":
                # Ensure that we both have positive cases and negative cases for each group, and remove groups without
                original_size = df.shape[0]
                df["has_pos_and_neg_target"] = (
                    grouped_df["targets"].transform(lambda x: x.eq(0).any())
                ) & (grouped_df["targets"].transform(lambda x: x.eq(1).any()))
                df = df[df["has_pos_and_neg_target"] == 1]
                filtered_size = df.shape[0]
                check_accepted_sample_ratio(
                    original_size, filtered_size, group_by_columns
                )

            x_labels = df.groupby(group_by_columns)[xlabel_col_name].max().reset_index()
            metric_scores, n_exp, mean_target = {}, {}, {}
            unique_groups = df.loc[:, group_by_columns[0]].unique()
            for group in unique_groups:
                group_df = df[df[group_by_columns[0]] == group]
                metric_scores[group] = get_metric_from_pred_and_target(
                    group_df, metric_name
                )
                n_exp[group] = group_df.shape[0]
                mean_target[group] = (group_df["targets"]).mean()

            grouped_df = pd.concat(
                [
                    pd.DataFrame(dict_.items())
                    for dict_ in [metric_scores, n_exp, mean_target]
                ],
                axis=1,
            )
            grouped_df.columns = [
                group_by_columns[0],
                metric_name,
                group_by_columns[0] + "1",
                "n_exp",
                group_by_columns[0] + "2",
                "mean_target",
            ]
            grouped_df.drop(
                columns=[group_by_columns[0] + "1", group_by_columns[0] + "2"],
                inplace=True,
            )

            grouped_df = grouped_df.merge(x_labels, on=group_by_columns[0])
            grouped_dfs.append(grouped_df.copy())

        generate_bar_plot(
            grouped_df,
            i,
            metric_name,
            model_names,
            plt_colors,
            save_path,
            title,
            xlabel_col_name,
            add_bar_info,
            run_name,
            task,
        )

    if len(pred_dfs) == 2 and xlabel_col_name != "triplet_name":
        df_diff = generate_difference_df(
            group_by_columns, grouped_dfs, metric_name, model_names, x_labels
        )
        generate_bar_plot(
            df_diff,
            0,
            metric_name,
            ["absolute negative difference"],
            plt_colors,
            save_path,
            f"{xlabel_col_name}_diff",
            xlabel_col_name,
            add_bar_info,
            run_name,
            task=task,
        )


def get_metric_from_pred_and_target(df, metric_name):
    """Error diagnostics function to find the metric of interest for a subset of the data"""
    if metric_name == "AUC_ROC":
        metric = roc_auc_score(df["targets"], df["pred_prob"])
    elif metric_name == "MSE":
        metric = ((df["targets"] - df["predictions"]) ** 2).values.mean()
    return metric


def generate_difference_df(
    group_by_columns, grouped_dfs, metric_name, model_names, x_labels
):
    df_diff = grouped_dfs[0].merge(
        grouped_dfs[1],
        how="inner",
        suffixes=(f"_{model_names[0]}", f"_{model_names[1]}"),
        on=group_by_columns,
    )
    metrics_columns = [
        f"{metric_name}_{model_names[i]}" for i in range(len(model_names))
    ]
    mean_target_columns = [
        f"mean_target_{model_names[i]}" for i in range(len(model_names))
    ]
    n_exp_columns = [f"n_exp_{model_names[i]}" for i in range(len(model_names))]
    df_diff[metric_name] = -abs(
        df_diff[metrics_columns[0]] - df_diff[metrics_columns[1]]
    )
    df_diff["mean_target"] = (
        df_diff[mean_target_columns[0]].values + df_diff[mean_target_columns[1]].values
    ) / 2
    df_diff["n_exp"] = (
        df_diff[n_exp_columns[0]].values + df_diff[n_exp_columns[1]].values
    ) / 2
    df_diff = df_diff.merge(x_labels, on=group_by_columns[0])
    return df_diff


def generate_bar_plot(
    grouped_df,
    i,
    metric_name,
    model_names,
    plt_colors,
    save_path,
    title,
    xlabel_col_name,
    add_bar_info,
    run_name,
    task,
):
    sorted_df, top10_df = sort_df_by_metric(grouped_df, metric_name, task, model_names)

    top10_df.reset_index(inplace=True)
    plt.figure(figsize=(16, 10))
    plt.subplot(1, 2, 1)
    # sorted_df.plot(kind="bar", ax=plt.gca(), color=plt_colors[i])
    sorted_df[metric_name].plot(kind="bar", ax=plt.gca(), color=plt_colors[i])
    plt.gca().set_xticklabels([])
    plt.gca().set_ylabel(metric_name)
    # plt.xticks(rotation=90)
    # plt.gca().set_xticks(range(len(sorted_df)))
    # plt.gca().set_xticklabels(sorted_df[xlabel_col_name])
    avg_exp_and_mean_target = [
        np.round(np.mean(sorted_df[exp_data].values), 2)
        for exp_data in ["n_exp", "mean_target"]
    ]
    df_corr = get_err_correlations(sorted_df, metric_name, avg_exp_and_mean_target)
    corr_str = f"corr: MSE/n_exp {df_corr.iloc[0,1]}\ncorr: MSE/mt {df_corr.iloc[0,2]}\ncorr: MSE/abs_dev_mt {df_corr.iloc[0,3]}\ncorr: mt/n_exp {df_corr.iloc[2,1]}\ncorr: mt/abs_dev_mt {df_corr.iloc[2,3]}"
    avg_exp_and_mean_target_str = f"avg n_exp: {avg_exp_and_mean_target[0]}\navg mt: {avg_exp_and_mean_target[1]:.2f}"
    plt.text(
        sorted_df.shape[0] * 0.5,
        sorted_df[metric_name].max() * 0.92,
        corr_str,
        ha="center",
        va="bottom",
    )
    plt.text(
        sorted_df.shape[0] * 0.8,
        sorted_df[metric_name].max() * 0.92,
        avg_exp_and_mean_target_str,
        ha="center",
        va="bottom",
    )
    plt.title(f"{title}\n {metric_name}")
    plt.legend([model_names[i]])
    plt.subplot(1, 2, 2)
    top10_df[metric_name].plot(kind="bar", ax=plt.gca(), color=plt_colors[i])
    # plt.gca().set_ylim(0, 1)
    # plt.xticks(rotation=45)
    plt.gca().set_xticks(range(len(top10_df)))
    plt.gca().set_xticklabels(top10_df[xlabel_col_name])
    plt.title(f"{title}\ntop 10 w. lowest {metric_name}")
    if add_bar_info:
        for index, value in enumerate(top10_df[metric_name]):
            mt = np.round(top10_df.loc[index, ["mean_target"]].values[0], 2)
            bar_text = f"n:\n{top10_df.loc[index, ['n_exp']].values[0]}\nmt:\n{mt:.2f}"
            plt.text(index, value, bar_text, ha="center", va="bottom")
    plt.legend([model_names[i]])
    plt.tight_layout()
    plt.savefig(save_path / f"{run_name}_{title}_bar_{model_names[i]}")
    df_corr.to_excel(save_path / f"corr_{run_name}_{title}_bar_{model_names[i]}.xlsx")
    sorted_df.to_excel(save_path / f"df_{run_name}_{title}_bar_{model_names[i]}.xlsx")
    pd.DataFrame(avg_exp_and_mean_target).to_excel(
        save_path / f"avg_mt_n_exp_{run_name}_{title}_bar_{model_names[i]}.xlsx"
    )
    plt.show()
    plt.clf()

    plot_individual(
        add_bar_info,
        i,
        metric_name,
        model_names,
        plt_colors,
        run_name,
        save_path,
        title,
        sorted_df,
        top10_df,
        xlabel_col_name,
        avg_exp_and_mean_target,
    )


def plot_individual(
    add_bar_info,
    i,
    metric_name,
    model_names,
    plt_colors,
    run_name,
    save_path,
    title,
    sorted_df,
    top10_df,
    xlabel_col_name,
    avg_exp_and_mean_target,
):
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 1, 1)
    # sorted_df.plot(kind="bar", ax=plt.gca(), color=plt_colors[i])
    sorted_df[metric_name].plot(kind="bar", ax=plt.gca(), color=plt_colors[i])
    # plt.gca().set_ylim(0, 1)
    # plt.xticks(rotation=45)
    plt.gca().set_ylabel(metric_name)
    # plt.gca().set_xticks(range(len(sorted_df)))
    # plt.gca().set_xticklabels(sorted_df[xlabel_col_name])
    plt.title(f"{title}\n {metric_name}")
    # plt.ylim(-7, 0)
    df_corr = get_err_correlations(sorted_df, metric_name, avg_exp_and_mean_target)
    corr_str = f"corr: MSE/n_exp {df_corr.iloc[0, 1]}\ncorr: MSE/mt {df_corr.iloc[0, 2]}\ncorr: MSE/abs_dev_mt {df_corr.iloc[0, 3]}\ncorr: mt/n_exp {df_corr.iloc[2, 1]}\ncorr: mt/abs_dev_mt {df_corr.iloc[2, 3]}"
    avg_exp_and_mean_target_str = f"avg n_exp: {avg_exp_and_mean_target[0]}\navg mt: {avg_exp_and_mean_target[1]:.2f}"
    # plt.text(
    #     sorted_df.shape[0] * 0.5,
    #     sorted_df[metric_name].max() * 0.92,
    #     corr_str,
    #     ha="center",
    #     va="bottom",
    # )
    # plt.text(
    #     sorted_df.shape[0] * 0.8,
    #     sorted_df[metric_name].max() * 0.92,
    #     avg_exp_and_mean_target_str,
    #     ha="center",
    #     va="bottom",
    # )
    # if add_bar_info:
    #     for index, value in enumerate(top10_df[metric_name]):
    #         mt = np.round(top10_df.loc[index, ["mean_target"]].values[0], 2)
    #         bar_text = f"n:\n{top10_df.loc[index, ['n_exp']].values[0]}\nmt:\n{mt:.2f}"
    #         plt.text(index, value, bar_text, ha="center", va="bottom")
    plt.legend([model_names[i]])
    plt.tight_layout()
    plt.savefig(save_path / f"{run_name}_{title}_bar_{model_names[i]}_full")

    plt.figure(figsize=(8, 5))
    plt.subplot(1, 1, 1)
    # sorted_df.plot(kind="bar", ax=plt.gca(), color=plt_colors[i])
    top10_df[metric_name].plot(kind="bar", ax=plt.gca(), color=plt_colors[i])
    # plt.gca().set_ylim(0, 1)
    # plt.xticks(rotation=45)
    plt.gca().set_xticks(range(len(top10_df)))
    plt.gca().set_ylabel(metric_name)
    plt.gca().set_xticklabels(top10_df[xlabel_col_name])
    plt.title(f"{title}\ntop 10 w. worst {metric_name}")
    # plt.ylim(-7, 0)
    # if add_bar_info:
    #     for index, value in enumerate(top10_df[metric_name]):
    #         mt = np.round(top10_df.loc[index, ["mean_target"]].values[0], 2)
    #         bar_text = f"n:\n{top10_df.loc[index, ['n_exp']].values[0]}\nmt:\n{mt:.2f}"
    #         plt.text(index, value, bar_text, ha="center", va="bottom")
    plt.legend([model_names[i]])
    plt.tight_layout()
    plt.savefig(save_path / f"{run_name}_{title}_bar_{model_names[i]}_top10")
    plt.clf()


def check_accepted_sample_ratio(original_size, filtered_size, group_by_columns):
    ratio = round(filtered_size / original_size, 2)
    if ratio < 0.6:
        print(
            f"{group_by_columns}:\nAccepted percentage of experiments: {ratio}.\nA too low % suggest that the groupings are on a too granular level"
        )


def sort_df_by_metric(df, metric_name, task, model_names):
    """
    Sort values by the metric given. Return sorted values with the tail containing performance on the worst entities.
    Args:
        df (pd.DataFrame):
        metric_name (str):
        task (str):

    Returns:
        df (pd.DataFrame): sorted dataframe
    """
    if task == "clf":
        asc = False
    else:
        asc = True
    df = df.sort_values(by=metric_name, ascending=asc)
    # Get entitities with top 10 worst performance
    if "dif" in "".join(model_names):
        top10_df = df.head(10)
    else:
        top10_df = df.tail(10)
    return df, top10_df


def get_saved_pred(model_names: t.List[str], path_to_prediction_folder: Path("")):
    """Err diagnostics function to get saved predictions given model names and path"""
    pred_file_names = [f"{model}_model_pred_dict.pkl" for model in model_names]
    pred_dfs = [
        get_prediction_dataframe(pred_file, save_path=path_to_prediction_folder)
        for pred_file in pred_file_names
    ]
    return pred_dfs


def get_err_correlations(df, metric_name, avg_exp_and_mean_target) -> pd.DataFrame:
    metric_val = df[metric_name]
    n_exp = df["n_exp"]
    mt = df["mean_target"]
    abs_mt_deviation_from_avg_mt = abs(df["mean_target"] - avg_exp_and_mean_target[1])
    df_corr = pd.DataFrame(
        [metric_val, n_exp, mt, abs_mt_deviation_from_avg_mt]
    ).T.corr()
    return round(df_corr, 2)


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


def format_and_return_as_dataframes(pred_dict):
    """Error diagnostics function that unpacks the prediction dictionary that holds batch triplets, preductions and targets and was saved under trainer.Test() and returns it as a dataframe"""
    if len(pred_dict["batch"]) != 2:
        fold_triplets = torch.vstack(pred_dict["batch"][::2]).numpy()
        fold_predictions = torch.hstack(pred_dict["predictions"]).numpy()
        fold_target = torch.hstack(pred_dict["batch"][1::2]).numpy()
        dataframes = [
            pd.DataFrame(ent)
            for ent in [
                fold_triplets,
                fold_predictions,
                fold_target,
            ]
        ]
    else:
        dataframes = [
            pd.DataFrame(ent)
            for ent in [
                pred_dict["batch"][0],
                pred_dict["predictions"],
                pred_dict["batch"][1],
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


def get_task_loss(df, task):
    """Error diagnostics function to retrieve the task relevant loss and calculate for each (target-pred) pair"""
    if task == "clf":
        df["bce_loss"] = -(
            df["targets"] * np.log(df["pred_prob"])
            + (1 - df["targets"]) * np.log(1 - df["pred_prob"])
        )
    else:
        df["MSE"] = (
            df["targets"] - df["predictions"]
        ) ** 2  # technically it is just squared error, but for avoid naming confusion MSE will be used here.


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
    df = df.merge(
        drug_meta_data,
        how="left",
        left_on="drug_name_left",
        right_on="dname",
        suffixes=("", "_left"),
    )
    df = df.drop(columns=["dname", "id"], inplace=False, axis=1)

    df = df.merge(
        drug_meta_data,
        how="left",
        left_on="drug_name_right",
        right_on="dname",
        suffixes=("", "_right"),
    )
    df = df.drop(columns=["dname", "id"], inplace=False, axis=1)
    df = df.merge(cell_line_meta_data, how="left", left_on="rel_name", right_on="name")
    df = df.rename(columns={"name": "tissue_name", "rel_name": "cancer_cell_name"})
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
    df = df.drop(["drug_id"], inplace=False, axis=1)
    df = df.merge(
        df_ent_vocab, left_on="drug_molecules_right_id", right_on="drug_id"
    ).rename(columns={"drug_name": "drug_name_right"})
    df.drop(["drug_id"], inplace=True, axis=1)
    df = df.merge(df_rel_vocab, left_on="context_features_id", right_on="rel_id")
    df = df.drop("rel_id", inplace=False, axis=1)
    return df


def get_cell_line_info():
    bronze_root_path = Directories.DATA_PATH / "bronze" / "drugcomb"
    return pd.DataFrame(load_json(bronze_root_path / "cell_line_dict.json")).T


def get_drug_info():
    bronze_root_path = Directories.DATA_PATH / "bronze" / "drugcomb"
    return pd.DataFrame(load_json(bronze_root_path / "drug_dict.json")).T


def get_evaluation_metric_name(task):
    """Based on the specified task return the metrics that the error diagnostic plots should focus on"""
    if task == "clf":
        metric_name = "AUC_ROC"
        triplet_metric_name = "correct_pred"  # There is only a single observation, so AUCROC cannot be used
    else:
        metric_name, triplet_metric_name = "MSE", "MSE"
    return metric_name, triplet_metric_name


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


def enrich_model_predictions(model_names, pred_dfs, task):
    """
    Enrich prediction dataframe with vocabulary and meta information on both drug and cancer cell line level.
    Parameters:
        model_names:
        pred_dfs:
        task: classification "clf" or regression "reg". Value determines what loss is added.
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
        if task == "clf":
            df["pred_prob"] = df["predictions"].apply(expit)
            df["pred_thresholded"] = df["pred_prob"].apply(
                lambda x: 1 if x > 0.5 else 0
            )
            df["correct_pred"] = np.isclose(
                df["pred_thresholded"], df["targets"]
            ).astype(int)
        get_task_loss(df, task)

        df = merge_vocabs_with_predictions(df, df_ent_vocab, df_rel_vocab)
        df = merge_cell_line_and_drug_info(df, cell_line_meta_data, drug_meta_data)
        df["model_name"] = model_names[idx]

        generate_group_indices_and_names(df)

        # add triplet name
        # add drug pair name
        # add
        new_pred_dfs.append(df)
    combined_df = [pd.concat(new_pred_dfs)]
    return combined_df, new_pred_dfs


def generate_group_indices_and_names(df: pd.DataFrame):
    """Generate a unique idx and name for each drug_pair, triplet, drug_target & disease

    Parameters:
        df (pd.DataFrame)
    Returns:
        None
    """
    df["drug_pair_idx"] = df.groupby(
        ["drug_molecules_left_id", "drug_molecules_right_id"]
    ).ngroup()
    df["triplet_idx"] = df.groupby(
        [
            "drug_molecules_left_id",
            "drug_molecules_right_id",
            "context_features_id",
        ]
    ).ngroup()
    df["drug_targets_idx"] = df.groupby(["target_type", "target_type_right"]).ngroup()
    df["disease_idx"] = df.groupby(["disease_id"]).ngroup()
    df["triplet_name"] = df.apply(
        lambda row: ",".join(
            [
                str(row[key])
                for key in ["drug_name_left", "drug_name_right", "cancer_cell_name"]
            ]
        ),
        axis=1,
    )
    df["drug_targets_name"] = df.apply(
        lambda row: ",".join(
            [str(row[key]) for key in ["target_type", "target_type_right"]]
        ),
        axis=1,
    )
    df["drug_pair_name"] = df.apply(
        lambda row: ",".join(
            [str(row[key]) for key in ["drug_name_left", "drug_name_right"]]
        ),
        axis=1,
    )
