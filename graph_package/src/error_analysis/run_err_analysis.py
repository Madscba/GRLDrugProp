import pandas as pd
from graph_package.configs.directories import Directories
from graph_package.src.main import (
    load_data,
    get_dataloaders,
)
from graph_package.src.error_analysis.utils import (
    find_best_model_ckpt,
    barplot_mean_correct_prediction_grouped_by_entity,
    sort_df_by_mean_correct_pred,
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


def error_diagnostics_plots(model_names):
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
    pred_dfs = [get_prediction_dataframe(pred_file) for pred_file in pred_file_names]

    # enrich predictions with vocabularies and meta data
    combined_df, pred_dfs = enrich_model_predictions(model_names, pred_dfs)
    combined_legend = "&".join(model_names)

    ##Investigate triplet (drug,drug, cell line)
    barplot_mean_correct_prediction_grouped_by_entity(
        pred_dfs, model_names, ["triplet_idx"], title="triplet_single_model"
    )
    barplot_mean_correct_prediction_grouped_by_entity(
        [combined_df], combined_legend, ["triplet_idx"], title="triplet"
    )

    ##Investigate drug pairs
    barplot_mean_correct_prediction_grouped_by_entity(
        pred_dfs, model_names, ["drug_pair_idx"], title="drug_pair_single_model"
    )
    barplot_mean_correct_prediction_grouped_by_entity(
        [combined_df], combined_legend, ["drug_pair_idx"], title="drug_pair"
    )

    ##Investigate cancer cell line
    barplot_mean_correct_prediction_grouped_by_entity(
        pred_dfs, model_names, ["context_features_id"], title="cancer_cell_single_model"
    )
    barplot_mean_correct_prediction_grouped_by_entity(
        [combined_df], combined_legend, ["context_features_id"], title="cancer_cell"
    )

    ##Investigate drug target
    combined_df["drug_targets_idx"] = combined_df.apply(
        map_to_index, axis=1, cols=["target_type", "target_type_right"]
    )
    barplot_mean_correct_prediction_grouped_by_entity(
        [combined_df], combined_legend, ["drug_targets_idx"], title="drug_target_types"
    )

    # Investigate disease id
    barplot_mean_correct_prediction_grouped_by_entity(
        [combined_df], combined_legend, ["disease_id"], title="disease_id"
    )

    # Investigate tissue
    barplot_mean_correct_prediction_grouped_by_entity(
        [combined_df], combined_legend, ["tissue_id"], title="tissue_id"
    )

    ##Investigate single drug
    save_path = Directories.OUTPUT_PATH / "err_diagnostics"
    for idx, df in enumerate(pred_dfs):
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
        df_sub_new = df_sub[
            ["drug_molecules_left_id", "drug_molecules_right_id"]
        ].drop_duplicates()
        df_pairs_new = df_sub.drug_molecules_left_id.value_counts()
        df_pairs_new.plot(kind="bar")
        plt.legend([model_names[idx]])
        plt.savefig(save_path / "n_experiments_per_drug_in_testset")
        plt.show()

        df_pairs_without_dupl = df_sub.iloc[df_sub_new.index]
        df_pairs_pred = df_pairs_without_dupl.groupby("drug_molecules_left_id").agg(
            {"correct_pred": "mean"}
        )
        df_pairs_pred_sorted = sort_df_by_mean_correct_pred(df_pairs_pred)
        df_pairs_pred_sorted.plot(kind="bar")
        plt.legend([model_names[idx]])
        plt.savefig(save_path / "drug_mean_correct_pred_bar_plot")
        plt.show()

        df_node_degrees = pd.DataFrame(get_node_degree(), columns=["node_degree"])
        df_drug_degree = df_pairs_without_dupl.merge(
            df_node_degrees, left_on="drug_molecules_left_id", right_index=True
        )
        df_pairs_pred = df_drug_degree.groupby("node_degree").agg(
            {"correct_pred": "mean"}
        )
        df_drug_degree_sorted = sort_df_by_mean_correct_pred(df_pairs_pred)
        df_drug_degree_sorted.plot(kind="bar")
        plt.legend([model_names[idx]])
        plt.savefig(save_path / "drug_degree_correct_pred_bar_plot")
        plt.show()


if __name__ == "__main__":
    # load_dotenv(".env")
    # main()
    models = ["rescal", "deepdds"]
    error_diagnostics_plots(models)
