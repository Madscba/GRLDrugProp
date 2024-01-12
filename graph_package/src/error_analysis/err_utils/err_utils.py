from scipy.special import expit

from graph_package.configs.directories import Directories

import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
)

from graph_package.src.error_analysis.err_utils.err_utils_load import (
    get_model_pred_path,
    get_ent_vocab,
    get_rel_vocab,
    get_cell_line_info,
    get_drug_info,
)
import numpy as np

PLOT_COLORS = ["red", "blue", "orange"]

ENTITY_ERR_DICT = {
    "drug_pair": {
        "group_by": ["drug_pair_idx"],
        "x_label_column_name": "drug_pair_name",
    },
    "drug": {
        "group_by": ["drug_molecules_left_id"],
        "x_label_column_name": "drug_name_left",
    },
    "disease": {
        "group_by": ["disease_idx"],
        "x_label_column_name": "disease_id",
    },
    "tissue": {
        "group_by": ["tissue_id"],
        "x_label_column_name": "tissue_name",
    },
    "cancer_cell": {
        "group_by": ["context_features_id"],
        "x_label_column_name": "cancer_cell_name",
    },
    "drug_target": {
        "group_by": ["drug_targets_idx"],
        "x_label_column_name": "drug_targets_name",
    },
}


def generate_error_plots_per_entity(
    df,
    task,
    comparison,
    entity,
    model_name
    # pred_dfs,
    # model_names,
    # group_by_columns,
    # title,
    # xlabel_col_name,
    # add_bar_info: bool = True,
    # run_name: str = "",
    # metric_name: str = "MSE",
    # task: str = "reg",
):
    """
    Generate and save barplot

    Parameters:
        pred_dfs (List[pd.DataFrame): prediction dataframe with side info
        model_names (List[str]): List of models

    Returns:
        None
    """
    metric_name = "AUC_ROC" if task == "clf" else "MSE"
    save_path = Directories.OUTPUT_PATH / "err_diagnostics"

    if not save_path.exists():
        save_path.mkdir(exist_ok=True, parents=True)

    if metric_name == "AUC_ROC":
        df = filter_away_groups_without_pos_and_neg_cases(df, entity)

    sorted_df, top10_df = sort_df_by_metric(df, metric_name, task, comparison)

    for idx, df in enumerate([sorted_df, top10_df]):
        # todo add correlation and avg mt and n_exp to plot and save
        plot_individual(
            df,
            entity,
            metric_name,
            task,
            comparison,
            model_name,
            save_path,
            ["full", "top10"][idx],
        )

    # plot_full_and_top10_together(
    #     sorted_df,
    #     top10_df,
    #     i,
    #     metric_name,
    #     model_names,
    #     plt_colors,
    #     save_path,
    #     title,
    #     x_label_col_name,
    #     add_bar_info,
    #     run_name,
    #     task,
    # )


def enrich_df_w_metric_nexp_meantarget_per_group(df, group_by_columns, metric_name):
    """
    Get metric, n_exp and mean_target for each group that exists for the entity currently under investigation
    Args:
        df: predictions dataframe with meta data
        group_by_columns: entity currently being investigated
        metric_name: performance measure

    Returns:
        mean_target: mean target value for each group
        metric_scores: performance measure for each group
        n_exp: number of experiments for each group
    """
    metric_scores, n_exp, mean_target = {}, {}, {}

    unique_groups = df.loc[:, group_by_columns].squeeze().unique()
    for group in unique_groups:
        group_df = df[df[group_by_columns].squeeze() == group]
        metric_scores[group] = get_metric_from_pred_and_target(group_df, metric_name)
        n_exp[group] = group_df.shape[0]
        mean_target[group] = (group_df["targets"]).mean()

    grouped_df = pd.concat(
        [pd.DataFrame(dict_.items()) for dict_ in [metric_scores, n_exp, mean_target]],
        axis=1,
    )

    grouped_df = grouped_df.iloc[:, [0, 1, 3, 5]]
    grouped_df.columns = [
        group_by_columns[0],
        metric_name,
        "n_exp",
        "mean_target",
    ]

    return grouped_df


def filter_away_groups_without_pos_and_neg_cases(df, group_by_columns):
    # Ensure that we both have positive cases and negative cases for each group, and remove groups without
    grouped_df = df.groupby(group_by_columns)
    original_size = df.shape[0]
    df["has_pos_and_neg_target"] = (
        grouped_df["targets"].transform(lambda x: x.eq(0).any())
    ) & (grouped_df["targets"].transform(lambda x: x.eq(1).any()))
    df = df[df["has_pos_and_neg_target"] == 1]
    filtered_size = df.shape[0]
    check_accepted_sample_ratio(original_size, filtered_size, group_by_columns)
    return df


def get_metric_from_pred_and_target(df, metric_name):
    """Error diagnostics function to find the metric of interest for a subset of the data"""
    if metric_name == "AUC_ROC":
        metric = roc_auc_score(df["targets"], df["pred_prob"])
    elif metric_name == "MSE":
        metric = ((df["targets"] - df["predictions"]) ** 2).values.mean()
    return metric


def generate_difference_df(
    group_by_columns, grouped_dfs, metric_name, model_names, x_label_col_name
):
    """
    Generate dataframe with the difference between two models

    Args:
        group_by_columns:
        grouped_dfs:
        metric_name:
        model_names:
        x_labels:

    Returns:

    """
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
    df_diff[metric_name] = abs(
        df_diff[metrics_columns[0]] - df_diff[metrics_columns[1]]
    )
    df_diff["mean_target"] = (
        df_diff[mean_target_columns[0]].values + df_diff[mean_target_columns[1]].values
    ) / 2
    df_diff["n_exp"] = (
        df_diff[n_exp_columns[0]].values + df_diff[n_exp_columns[1]].values
    ) / 2
    df_diff = df_diff.rename(
        columns={f"{x_label_col_name}_{model_names[0]}": x_label_col_name}
    )
    df_diff.drop(columns=[f"{x_label_col_name}_{model_names[1]}"], inplace=True)
    return df_diff


def plot_full_and_top10_together(
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
    # sorted_df, top10_df = sort_df_by_metric(grouped_df, metric_name, task, model_names)

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


def plot_individual(
    df,
    entity,
    metric_name,
    task,
    comparison,
    model_name,
    save_path,
    scope,
    # add_bar_info,
    # i,
    # metric_name,
    # model_names,
    # plt_colors,
    # run_name,
    # save_path,
    # title,
    # sorted_df,
    # top10_df,
    # xlabel_col_name,
    # avg_exp_and_mean_target,
):
    x_label_col_name = ENTITY_ERR_DICT[entity]["x_label_column_name"]

    plt.figure(figsize=(8, 5))
    plt.subplot(1, 1, 1)
    # sorted_df.plot(kind="bar", ax=plt.gca(), color=plt_colors[i])
    df[metric_name].plot(kind="bar", ax=plt.gca(), color=PLOT_COLORS[0])
    # plt.gca().set_ylim(0, 1)
    # plt.xticks(rotation=45)
    plt.gca().set_ylabel(metric_name)
    if scope != "full":
        plt.gca().set_xticks(range(len(df)))
        plt.gca().set_xticklabels(df[x_label_col_name])
        title = f"{entity}_{model_name}_{comparison}"
    else:
        plt.xticks([])
        title = f"{entity}_{model_name}_{comparison} top10 errors"
    plt.title(f"{title}\n {metric_name}")
    # plt.ylim(-7, 0)
    # avg_exp_and_mean_target = [
    #     np.round(np.mean(df[exp_data].values), 2)
    #     for exp_data in ["n_exp", "mean_target"]
    # ]
    # df_corr = get_err_correlations(df, metric_name, avg_exp_and_mean_target)
    # corr_str = f"corr: MSE/n_exp {df_corr.iloc[0, 1]}\ncorr: MSE/mt {df_corr.iloc[0, 2]}\ncorr: MSE/abs_dev_mt {df_corr.iloc[0, 3]}\ncorr: mt/n_exp {df_corr.iloc[2, 1]}\ncorr: mt/abs_dev_mt {df_corr.iloc[2, 3]}"
    # avg_exp_and_mean_target_str = f"avg n_exp: {avg_exp_and_mean_target[0]}\navg mt: {avg_exp_and_mean_target[1]:.2f}"
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
    plt.legend([model_name])
    plt.tight_layout()
    plt.savefig(save_path / f"{title}_{scope}_bar")


def check_accepted_sample_ratio(original_size, filtered_size, group_by_columns):
    ratio = round(filtered_size / original_size, 2)
    if ratio < 0.6:
        print(
            f"{group_by_columns}:\nAccepted percentage of experiments: {ratio}.\nA too low % suggest that the groupings are on a too granular level"
        )


def sort_df_by_metric(df, metric_name, task, comparison):
    """
    Sort values by the metric given. Return sorted values with the tail containing performance on the worst entities.
    Args:
        df (pd.DataFrame):
        metric_name (str):
        task (str):

    Returns:
        df (pd.DataFrame): sorted dataframe
    """
    ascending_order = task != "clf"
    df = df.sort_values(by=metric_name, ascending=ascending_order)
    # Get entitities with top 10 worst performance
    if "diff" in comparison and task == "clf":
        top10_df = df.head(10)
    else:
        top10_df = df.tail(10)
    return df, top10_df


def get_err_correlations(df, metric_name, avg_exp_and_mean_target) -> pd.DataFrame:
    metric_val = df[metric_name]
    n_exp = df["n_exp"]
    mt = df["mean_target"]
    abs_mt_deviation_from_avg_mt = abs(df["mean_target"] - avg_exp_and_mean_target[1])
    df_corr = pd.DataFrame(
        [metric_val, n_exp, mt, abs_mt_deviation_from_avg_mt]
    ).T.corr()
    return round(df_corr, 2)


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


def get_drug_level_df(df_list, task):
    if task == "clf":
        additional_cols = ["pred_prob", "correct_pred"]
    else:
        additional_cols = ["MSE"]
    df = df_list[0]
    df_sub = df.loc[
        :,
        [
            "drug_molecules_left_id",
            "drug_molecules_right_id",
            "context_features_id",
            "predictions",
            "model_name",
            "drug_name_left",
            "drug_name_right",
            "cancer_cell_name",
            "targets",
            "drug_pair_idx",
        ]
        + additional_cols,
    ]
    df_sub = pd.concat(
        (
            df_sub,
            df_sub.rename(
                columns={
                    "drug_molecules_left_id": "drug_molecules_right_id",
                    "drug_molecules_right_id": "drug_molecules_left_id",
                    "drug_name_left": "drug_name_right",
                    "drug_name_right": "drug_name_left",
                }
            ),
        )
    )

    return df_sub


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


def enrich_model_predictions(model_names, pred_dfs, task):
    """
    Enrich prediction dataframe with vocabulary and meta information on both drug and cancer cell line level.
    Parameters:
        model_names:
        pred_dfs:
        task: classification "clf" or regression "reg". Value determines what loss is added.
    Returns:
        pred_dfs (List[pd.DataFrame]) List of dataframes each dataframe holding the prediction from one model.
    """
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
        new_pred_dfs.append(df)
    return new_pred_dfs
