from graph_package.configs.directories import Directories

from graph_package.src.error_analysis.err_utils.err_utils import (
    generate_error_plots_per_entity,
    get_drug_level_df,
    enrich_model_predictions,
    enrich_df_w_metric_nexp_meantarget_per_group,
    generate_difference_df,
    ENTITY_ERR_DICT,
)
from graph_package.src.error_analysis.err_utils.err_utils_load import (
    get_saved_pred,
)
import pandas as pd


def run_err_diag(
    pred_df, model_names, path_to_prediction_folder, task, entity, comparison
):
    """
    Load predictions, enrich with entity information and provide diagnostic bar plots on entity level:
    Single drugs, drug pairs, triplets, disease, tissue, cancer cell line, drug target

    Parameters:
        model_names List[str]: Models for which to generate err diagnostic plots.

    Returns:
        None
    """
    entity_args = ENTITY_ERR_DICT[entity]
    metric_name = "AUC_ROC" if task == "clf" else "MSE"
    run_name = path_to_prediction_folder.name

    generate_error_plots_per_entity(
        df_list,
        plt_legend,  # legend_list[idx],
        group_by_columns=entity_args["group_by"],
        title=f"{entity}_{comparison}",
        xlabel_col_name=entity_args["x_label_column_name"],
        run_name=run_name,
        metric_name=metric_name,
        task=task,
    )

    ##Investigate single drug
    drug_titles = [f"drug_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        if len(df_list) > 1:
            df_drug_without_dupl = [
                get_drug_level_df([df_list[i]], task) for i in range(len(df_list))
            ]
        else:
            df_drug_without_dupl = [get_drug_level_df(df_list, task)]
        generate_error_plots_per_entity(
            df_drug_without_dupl,
            legend_list[idx],
            ["drug_molecules_left_id"],
            drug_titles[idx],
            "drug_name_left",
            run_name=run_name,
            metric_name=metric_name,
            task=task,
        )


def load_and_prepare_predictions_for_comp(model_names, entity, comparison, err_configs):
    # load predictions (triplets, predictions, targets) from trained model(s),
    pred_dfs = get_saved_pred(err_configs)

    # enrich predictions with vocabularies and meta data
    pred_dfs = enrich_model_predictions(model_names, pred_dfs, task)

    metric_name = "AUC_ROC" if task == "clf" else "MSE"
    group_by_column = ENTITY_ERR_DICT[entity]["group_by"]
    x_label_col_name = ENTITY_ERR_DICT[entity]["x_label_column_name"]

    # todo add a check for entity = "drug":
    if entity == "drug":
        if len(pred_dfs) > 1:
            df_drug_without_dupl = [
                get_drug_level_df([pred_dfs[i]], task) for i in range(len(pred_dfs))
            ]
        else:
            df_drug_without_dupl = [get_drug_level_df(pred_dfs, task)]

    # prepare df(s) for relevant comparison
    if comparison == "individual":
        grouped_dfs = enrich_df_w_metric_nexp_meantarget_per_group(
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
        grouped_dfs = enrich_df_w_metric_nexp_meantarget_per_group(
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
            enrich_df_w_metric_nexp_meantarget_per_group(
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
                df = load_and_prepare_predictions_for_comp(
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
            dfs = load_and_prepare_predictions_for_comp(
                model_names, entity, comparison, err_configs
            )
            generate_error_plots_per_entity(
                dfs, {0: err_configs[0]}, entity, comparison, "&".join(model_names)
            )


if __name__ == "__main__":
    task = "reg"
    target = "zip_mean"
    day_of_prediction = "10_12_2023"
    path_to_prediction_folder = (
        Directories.OUTPUT_PATH
        / "model_predictions"
        / day_of_prediction
        / "_".join([task, target])
    )

    model_1_config = {
        "task": "reg",
        "target": "zip_mean",
        "day_of_prediction": "10_12_2023",
        "prediction_file_name": f"rescal_model_pred_dict.pkl",
        "bar_plot_config": {"add_bar_info": True},
    }

    # Note that if multiple model_configs are given and the comparison is not individual,
    # the first models plotting config will be used.

    err_configs = {
        0: model_1_config,
        # 1: model_1_config,
    }
    # todo validate: that models names can be 1 or more. check that every comparison works. And that each entity works
    comparison = "individual"  # "individual", "concatenate" or "difference"
    model_names = ["deepdds"]
    entities = ["tissue"] # drug_pair, drug "disease", "tissue", "cancer_cell", "drug_target"]

    assert len(model_names) == len(
        err_configs
    ), "Number of models and configs must be equal"

    main_error_diagnostics(err_configs, comparison, model_names, entities)
