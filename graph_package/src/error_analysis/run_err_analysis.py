from graph_package.configs.directories import Directories

from graph_package.src.error_analysis.err_utils.err_utils import (
    generate_barplot_w_performance_metric_grouped_by_entity,
    get_drug_level_df,
    enrich_model_predictions,
)
from graph_package.src.error_analysis.err_utils.err_utils_load import (
    get_saved_pred,
)


def error_diagnostics_plots(model_names, path_to_prediction_folder, task):
    """
    Load predictions, enrich with entity information and provide diagnostic bar plots on entity level:
    Single drugs, drug pairs, triplets, disease, tissue, cancer cell line, drug target

    Parameters:
        model_names List[str]: Models for which to generate err diagnostic plots.

    Returns:
        None
    """
    # load predictions from trained model(s)
    pred_dfs = get_saved_pred(model_names, task, path_to_prediction_folder)
    # enrich predictions with vocabularies and meta data
    combined_df, pred_dfs = enrich_model_predictions(model_names, pred_dfs, task)

    # prepare lists for error diagnostics on individual models and a combined analysis
    df_lists = [pred_dfs, combined_df]
    title_suffix = ["single_model", "both"]
    combined_legend = ["&".join(model_names)]
    legend_list = [model_names, combined_legend]

    metric_name = "AUC_ROC" if task == "clf" else "MSE"
    run_name = path_to_prediction_folder.name

    ##Investigate drug pairs, "drug_pair_name"
    drug_pair_titles = [f"drug_pair_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        generate_barplot_w_performance_metric_grouped_by_entity(
            df_list,
            legend_list[idx],
            ["drug_pair_idx"],
            drug_pair_titles[idx],
            "drug_pair_name",
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


if __name__ == "__main__":
    task = "reg"
    target = "zip_mean"
    day_of_prediction = "10_12_2023"
    task_target = "_".join([task, target])
    path_to_prediction_folder = (
        Directories.OUTPUT_PATH / "model_predictions" / day_of_prediction / task_target
    )
    models = ["rescal", "deepdds"]
    error_diagnostics_plots(models, path_to_prediction_folder, task)
