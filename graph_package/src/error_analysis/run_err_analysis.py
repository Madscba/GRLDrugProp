from graph_package.configs.directories import Directories

from graph_package.src.error_analysis.err_utils.err_utils import (
    generate_barplot_w_performance_metric_grouped_by_entity,
    get_drug_level_df,
    enrich_model_predictions,
    enrich_df_w_metric_nexp_meantarget_per_group,
    generate_difference_df,
)
from graph_package.src.error_analysis.err_utils.err_utils_load import (
    get_saved_pred,
)
import pandas as pd

entity_err_config = {
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


def error_diagnostics_plot(
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
    entity_args = entity_err_config[entity]
    metric_name = "AUC_ROC" if task == "clf" else "MSE"
    run_name = path_to_prediction_folder.name

    ##Investigate drug pairs, "drug_pair_name"
    for idx, df_list in enumerate(df_lists):
        generate_barplot_w_performance_metric_grouped_by_entity(
            df_list,
            plt_legend,  # legend_list[idx],
            group_by_columns=entity_args["group_by"],
            title=f"{entity}_{comparison}",
            xlabel_col_name=entity_args["x_label_column_name"],
            run_name=run_name,
            metric_name=metric_name,
            task=task,
        )

    ##Investigate cancer cell line, "cancer_cell_name"
    cancer_cell_line_titles = [f"cancer_cell_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        generate_barplot_w_performance_metric_grouped_by_entity(
            df_list,
            legend_list[idx],
            ["context_features_id"],
            cancer_cell_line_titles[idx],
            "cancer_cell_name",
            run_name=run_name,
            metric_name=metric_name,
            task=task,
        )

    ##Investigate drug target, "drug_targets_name"
    drug_targets_titles = [f"drug_targets_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        generate_barplot_w_performance_metric_grouped_by_entity(
            df_list,
            legend_list[idx],
            ["drug_targets_idx"],
            drug_targets_titles[idx],
            "drug_targets_name",
            run_name=run_name,
            metric_name=metric_name,
            task=task,
        )

    # Investigate disease id, "disease_id"
    disease_titles = [f"disease_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        generate_barplot_w_performance_metric_grouped_by_entity(
            df_list,
            legend_list[idx],
            ["disease_idx"],
            disease_titles[idx],
            "disease_id",
            run_name=run_name,
            metric_name=metric_name,
            task=task,
        )

    # Investigate tissue, "name"
    tissue_titles = [f"tissue_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        generate_barplot_w_performance_metric_grouped_by_entity(
            df_list,
            legend_list[idx],
            ["tissue_id"],
            tissue_titles[idx],
            "tissue_name",
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
        generate_barplot_w_performance_metric_grouped_by_entity(
            df_drug_without_dupl,
            legend_list[idx],
            ["drug_molecules_left_id"],
            drug_titles[idx],
            "drug_name_left",
            run_name=run_name,
            metric_name=metric_name,
            task=task,
        )


def load_and_prepare_predictions_for_comp(
    model_names, entity, comparison, path_to_prediction_folder, task
):
    # load predictions (triplets, predictions, targets) from trained model(s),
    pred_dfs = get_saved_pred(model_names, task, path_to_prediction_folder)

    # enrich predictions with vocabularies and meta data
    pred_dfs = enrich_model_predictions(model_names, pred_dfs, task)

    metric_name = "AUC_ROC" if task == "clf" else "MSE"
    group_by_column = entity_err_config[entity]["group_by"]
    x_label_col_name = entity_err_config[entity]["x_label_column_name"]

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


if __name__ == "__main__":
    task = "reg"
    target = "zip_mean"
    day_of_prediction = "10_12_2023"
    task_target = "_".join([task, target])
    path_to_prediction_folder = (
        Directories.OUTPUT_PATH / "model_predictions" / day_of_prediction / task_target
    )

    # todo validate: that models names can be 1 or more. check that every comparison works. And that each entity works
    comparison = "difference"  # "individual", "concatenate" or "difference"
    model_names = ["rescal", "deepdds"]
    entities = ["drug_pair", "drug", "disease", "tissue", "cancer_cell", "drug_target"]

    if comparison == "individual":
        for i in range(len(model_names)):
            for entity in entities:
                df = load_and_prepare_predictions_for_comp(
                    [model_names[i]],
                    entity,
                    comparison,
                    path_to_prediction_folder,
                    task,
                )
                error_diagnostics_plot(
                    df,
                    model_names[i],
                    path_to_prediction_folder,
                    task,
                    entity,
                    comparison,
                )
    else:
        for entity in entities:
            dfs = load_and_prepare_predictions_for_comp(
                model_names, entity, comparison, path_to_prediction_folder, task
            )
            error_diagnostics_plot(
                dfs, model_names, path_to_prediction_folder, task, entity, comparison
            )
