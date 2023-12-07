import pandas as pd
from graph_package.configs.directories import Directories
from graph_package.src.main import (
    load_data,
    get_dataloaders,
)
from graph_package.src.error_analysis.utils import (
    find_best_model_ckpt,
    barplot_aucroc_grouped_by_entity,
    sort_df_by_metric,
    map_to_index,
    get_prediction_dataframe,
    get_node_degree,
    enrich_model_predictions,
)
import hydra
import numpy as np
from graph_package.configs.definitions import model_dict
from pytorch_lightning import Trainer
import torch
import matplotlib.pyplot as plt


@hydra.main(
    config_path=str(Directories.CONFIG_PATH / "hydra_configs"),
    config_name="config.yaml",
)
def main(config):
    """
    Load checkpoint and run err. diagnostics for rescal model. remove comment line 34 to run deepdds (and outcomment 35,36)

    Be aware that the test dataset is generated differently than in main.

    Args:
        config:

    Returns:

    """
    # model_name = get_model_name(config, sys_args=sys.argv)
    model_name = "rescal"
    config.update({"model": {"dim": 100, "ent_tot": "", "rel_tot": ""}})

    dataset = load_data(model=model_name, dataset=config.dataset)

    if model_name == "rescal":
        update_dict = {
            "ent_tot": int(dataset.num_entity.numpy()),
            "rel_tot": int(dataset.num_relation.numpy()),
        }
        config.model.update(update_dict)

    check_point_path_folder = Directories.CHECKPOINT_PATH / model_name
    best_ckpt = find_best_model_ckpt(check_point_path_folder)

    model = model_dict[model_name].load_from_checkpoint(best_ckpt, **config.model)

    generator1 = torch.Generator().manual_seed(42)
    split_lengths = [
        int(np.ceil(len(dataset) * frac))
        if idx == 0
        else int(np.floor(len(dataset) * frac))
        for idx, frac in enumerate([0.8, 0.2])
    ]
    train_set, test_set = torch.utils.data.random_split(
        dataset, split_lengths, generator=generator1
    )

    data_loaders = get_dataloaders(
        [train_set, test_set], batch_sizes=config.batch_sizes
    )
    loggers = []
    trainer = Trainer(logger=loggers, **config.trainer)

    trainer.test(
        model,
        dataloaders=data_loaders["test"],
        ckpt_path=best_ckpt,
    )
    print(trainer.callback_metrics)

    error_diagnostics_plots()


def error_diagnostics_plots(model_names, path_to_prediction_folder):
    """
    Load predictions and provide diagnostics bar plots on entity level:

    Single drugs, drug pairs, triplets, disease, tissue, cancer cell line, drug target
    Parameters:
        model_names List[str]: Models for which to generate err diagnostic plots.

    Returns:
        None
    """
    # load predictions from trained model(s)
    pred_file_names = [f"{model}_model_pred_dict.pkl" for model in model_names]
    pred_dfs = [
        get_prediction_dataframe(pred_file, save_path=path_to_prediction_folder)
        for pred_file in pred_file_names
    ]
    run_name = path_to_prediction_folder.name

    # enrich predictions with vocabularies and meta data
    combined_legend = ["&".join(model_names)]

    combined_df, pred_dfs = enrich_model_predictions(model_names, pred_dfs)
    df_lists = [pred_dfs, [combined_df]]
    title_suffix = ["single_model", "both"]
    legend_list = [model_names, combined_legend]

    ##Investigate triplet (drug,drug, cell line), "triplet_name"
    triplet_titles = [f"triplet_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        barplot_aucroc_grouped_by_entity(
            df_list,
            legend_list[idx],
            ["triplet_idx"],
            triplet_titles[idx],
            "triplet_name",
            add_bar_info=False,
            run_name=run_name,
        )

    ##Investigate drug pairs, "drug_pair_name"
    drug_pair_titles = [f"drug_pair_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        barplot_aucroc_grouped_by_entity(
            df_list,
            legend_list[idx],
            ["drug_pair_idx"],
            drug_pair_titles[idx],
            "drug_pair_name",
            run_name=run_name,
        )

    ##Investigate cancer cell line, "rel_name"
    cancer_cell_line_titles = [f"cancer_cell_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        barplot_aucroc_grouped_by_entity(
            df_list,
            legend_list[idx],
            ["context_features_id"],
            cancer_cell_line_titles[idx],
            "rel_name",
            run_name=run_name,
        )

    ##Investigate drug target, "drug_targets_name"
    drug_targets_titles = [f"drug_targets_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        barplot_aucroc_grouped_by_entity(
            df_list,
            legend_list[idx],
            ["drug_targets_idx"],
            drug_targets_titles[idx],
            "drug_targets_name",
            run_name=run_name,
        )

    # Investigate disease id, "disease_id"
    disease_titles = [f"disease_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        barplot_aucroc_grouped_by_entity(
            df_list,
            legend_list[idx],
            ["disease_idx"],
            disease_titles[idx],
            "disease_id",
            run_name=run_name,
        )

    # Investigate tissue, "name"
    tissue_titles = [f"tissue_{title}" for title in title_suffix]
    for idx, df_list in enumerate(df_lists):
        barplot_aucroc_grouped_by_entity(
            df_list,
            legend_list[idx],
            ["tissue_id"],
            tissue_titles[idx],
            "name",
            run_name=run_name,
        )

    ##Investigate single drug
    drug_titles = [f"drug_{title}" for title in title_suffix]
    save_path = Directories.OUTPUT_PATH / "err_diagnostics"
    for idx, df_list in enumerate(df_lists):
        if len(df_list) > 1:
            df_drug_without_dupl = [
                get_drug_level_df([df_list[i]]) for i in range(len(df_list))
            ]
        else:
            df_drug_without_dupl = [get_drug_level_df(df_list)]
        barplot_aucroc_grouped_by_entity(
            df_drug_without_dupl,
            legend_list[idx],
            ["drug_molecules_left_id"],
            drug_titles[idx],
            "drug_name_left",
            run_name=run_name,
        )


def get_drug_level_df(df_list):
    df = df_list[0]
    df_sub = df.loc[
        :,
        [
            "drug_molecules_left_id",
            "drug_molecules_right_id",
            "context_features_id",
            "predictions",
            "correct_pred",
            "model_name",
            "drug_name_left",
            "drug_name_right",
            "rel_name",
            "pred_prob",
            "targets",
            "drug_pair_idx",
        ],
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


if __name__ == "__main__":
    # load_dotenv(".env")
    # main()

    task = "clf"
    target = "zip_mean"
    day_of_prediction = "03_12_2023"
    task_target = "_".join([task, target])
    path_to_prediction_folder = (
        Directories.OUTPUT_PATH / "model_predictions" / day_of_prediction / task_target
    )
    models = ["rescal", "deepdds"]
    error_diagnostics_plots(models, path_to_prediction_folder)
