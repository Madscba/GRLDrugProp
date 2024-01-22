from graph_package.configs.directories import Directories

from graph_package.src.error_analysis.err_utils.err_utils import (
    generate_error_plots_per_entity,
    get_drug_level_df,
    enrich_model_pred,
    enrich_df_w_metric,
    generate_difference_df,
    ENTITY_ERR_DICT,
)
from graph_package.src.error_analysis.err_utils.err_utils_load import (
    get_saved_pred,
)
import pandas as pd


def load_and_prepare_pred(model_names, entity, comparison, err_configs):
    """
    Load model predictions and prepare them for comparison

    Args:
        model_names:
        entity:
        comparison (str): individual, concatenate or difference
        err_configs:

    Returns:

    """
    # load predictions (triplets, predictions, targets) from trained model(s),
    pred_dfs = get_saved_pred(err_configs)
    task = err_configs[0]['task']
    # enrich predictions with vocabularies and meta data
    pred_dfs = enrich_model_pred(model_names, pred_dfs, task)

    metric_name = "AUC_ROC" if task == "clf" else "MSE"
    group_by_column = ENTITY_ERR_DICT[entity]["group_by"]
    x_label_col_name = ENTITY_ERR_DICT[entity]["x_label_column_name"]

    if entity == "drug":
        if len(pred_dfs) > 1:
            pred_dfs = [
                get_drug_level_df([pred_dfs[i]], task) for i in range(len(pred_dfs))
            ]
        else:
            pred_dfs = [get_drug_level_df(pred_dfs, task)]

    # prepare df(s) for relevant comparison
    if comparison == "individual":
        grouped_dfs = enrich_df_w_metric(
            pred_dfs[0], group_by_column, metric_name
        )
        x_labels = (
            pred_dfs[0].groupby(group_by_column)[x_label_col_name].max().reset_index()
        )
        grouped_dfs = grouped_dfs.merge(x_labels, on=group_by_column)

    elif comparison == "concatenate":
        assert (
            len(model_names) > 1
        ), "Concatenation of predictions requires more than one model"
        pred_dfs = pd.concat(pred_dfs)
        grouped_dfs = enrich_df_w_metric(
            pred_dfs, group_by_column, metric_name
        )
        # check that x_labels work as they should in this case
        x_labels = (
            pred_dfs.groupby(group_by_column)[x_label_col_name].max().reset_index()
        )
        grouped_dfs = grouped_dfs.merge(x_labels, on=group_by_column)

    elif comparison == "difference":
        assert (
            len(model_names) > 1
        ), "Difference comparison requires more than one model"
        grouped_dfs = [
            enrich_df_w_metric(
                pred_df, group_by_column, metric_name
            )
            for pred_df in pred_dfs
        ]
        x_labels = [
            pred_df.groupby(group_by_column)[x_label_col_name].max().reset_index()
            for pred_df in pred_dfs
        ]
        grouped_dfs = [
            grouped_df.merge(x_label, on=group_by_column)
            for grouped_df, x_label in zip(grouped_dfs, x_labels)
        ]
        grouped_dfs = generate_difference_df(
            group_by_column, grouped_dfs, metric_name, model_names, x_label_col_name
        )
    else:
        raise ValueError(f"Comparison {comparison} not supported")

    return grouped_dfs


def main_error_diagnostics(err_configs, comparison, model_names, entities):
    if comparison == "individual":
        for i in range(len(model_names)):
            for entity in entities:
                df = load_and_prepare_pred(
                    [model_names[i]],
                    entity,
                    comparison,
                    {0: err_configs[i]},
                )
                generate_error_plots_per_entity(
                    df,
                    {0: err_configs[i]},
                    entity,
                    comparison,
                    model_names[i],
                )

    else:
        for entity in entities:
            dfs = load_and_prepare_pred(
                model_names, entity, comparison, err_configs
            )
            generate_error_plots_per_entity(
                dfs, {0: err_configs[0]}, entity, comparison, "&".join(model_names)
            )


if __name__ == "__main__":
    model_1_config = {
        "task": "reg",
        "target": "zip_mean",
        "day_of_prediction": "21_01_2024",
        "prediction_file_name": "gnn_model_pred_dict_1j75xr94dxyat0e9lixmwb8e.pkl",
        "bar_plot_config": {"add_bar_info": True},
    }
    model_2_config = {
        "task": "reg",
        "target": "zip_mean",
        "day_of_prediction": "21_01_2024",
        "prediction_file_name": "gnn_model_pred_dict_bo9n6s0v6g7oa3tvy0s1nmvy.pkl",
        "bar_plot_config": {"add_bar_info": True},
    }

    # Note that if multiple model_configs are given and the comparison is not individual,
    # the first models plotting config will be used.

    err_configs = {
        0: model_1_config,
        1: model_2_config,
    }
    # todo validate: that models names can be 1 or more. check that every comparison works. And that each entity works
    comparison = "difference"  # "individual", "concatenate" or "difference"
    model_names = ["rescal", "deepdds"]
    entities = ["drug"] # drug_pair, drug "disease", "tissue", "cancer_cell", "drug_target"]

    assert len(model_names) == len(
        err_configs
    ), "Number of models and configs must be equal"

    main_error_diagnostics(err_configs, comparison, model_names, entities)
