import pandas as pd

from graph_package.configs.directories import Directories
from graph_package.src.main import (
    load_data,
    init_model,
    get_model_name,
    get_dataloaders,
    split_dataset,
)
from graph_package.src.error_analysis.utils import get_model_pred, get_err_analysis_path
import hydra
import json
from os import listdir
import sys
import os, numpy as np
from graph_package.configs.definitions import model_dict
from pytorch_lightning import Trainer
import torch
from scipy.special import expit
from sklearn.metrics import log_loss
import seaborn as sns
import matplotlib.pyplot as plt


def find_best_model_ckpt(ckpt_folder):
    checkpoints, scores = [], []
    for path, subdirs, files in os.walk(ckpt_folder):
        for name in files:
            scores.append(name[-11:-5])
            checkpoints.append(name)
    best_idx = np.where(np.array(scores) == max(scores))[0][0]
    max_checkpoint = ckpt_folder / f"fold_{best_idx}" / checkpoints[best_idx]
    return max_checkpoint


@hydra.main(
    config_path=str(Directories.CONFIG_PATH / "hydra_configs"),
    config_name="config.yaml",
)
def main(config):
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
    # ckpt_folder = save_dir / 'default' / 'version_{}'.format(version) / 'checkpoints'
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

    test = "abc"


def error_diagnostics_plots():
    df = get_prediction_dataframe()

    if True:  # If model is rescal, we need to apply a sigmoid first.
        df["pred_prob"] = df["predictions"].apply(expit)
        df["pred_thresholded"] = df["pred_prob"].apply(lambda x: 1 if x > 0.5 else 0)
        df["correct_pred"] = np.isclose(df["pred_thresholded"], df["targets"]).astype(
            int
        )

    df["bce_loss"] = -(
        df["targets"] * np.log(df["pred_prob"])
        + (1 - df["targets"]) * np.log(1 - df["pred_prob"])
    )

    #
    # enrich
    # 1. first load entitiy vocabs
    #     and relation vocabs
    gold_root_path = Directories.DATA_PATH / "gold" / "oneil"
    vocabs_file = ["entity_vocab.json", "relation_vocab.json"]
    df_ent_vocab = pd.DataFrame(
        load_json(gold_root_path / vocabs_file[0]).items(),
        columns=["drug_name", "drug_id"],
    )
    df_rel_vocab = pd.DataFrame(
        load_json(gold_root_path / vocabs_file[1]).items(),
        columns=["rel_id", "rel_name"],
    )
    df_rel_vocab["rel_id"] = df_rel_vocab["rel_id"].apply(pd.to_numeric)

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

    # 2. secondly load cell-line dict
    #    and drug-dict
    bronze_root_path = Directories.DATA_PATH / "bronze" / "drugcomb"
    meta_data_files = ["cell_line_dict.json", "drug_dict.json"]
    cell_line_meta_data = pd.DataFrame(
        load_json(bronze_root_path / meta_data_files[0])
    ).T
    drug_meta_data = pd.DataFrame(load_json(bronze_root_path / meta_data_files[1])).T

    df["drug_pair"] = df["drug_name_left"] + df["drug_name_right"]

    # Subpopulations of interest:
    # understand the distribution of variables, detect patterns, and identify potential trends within each subpopulation
    # descriptive statistics for each subpopulation. This may include measures such as mean, median, mode, standard deviation, and percentiles
    # data visualization

    ###1. Across entities
    # See data exploration for code to plot across
    # - Drugs
    # - Cell lines
    # - Drug-Drug
    # -Drug-Cell line
    # -Tissue

    ###2. Node statistics
    # -Degrees
    # -Centrality

    ###3. Node Features
    # See data exploration for networkx code
    # -
    for idx, category in enumerate(["drug_pair", "context_features_id"]):
        create_violin_plot(df, category, "correct_pred", idx)
    # merge meta data
    a = 123


def get_prediction_dataframe():
    pred_dict = get_model_pred()
    dataframes = [
        pd.DataFrame(pred_dict[key]) for key in ["batch", "predictions", "targets"]
    ]
    df = pd.concat(dataframes, axis=1)
    df.columns = [
        "drug_molecules_left_id",
        "drug_molecules_right_id",
        "context_features_id",
        "label_id",
        "predictions",
        "targets",
    ]
    return df


#


def load_json(filepath):
    with open(filepath) as f:
        file = json.load(f)
    return file


def create_violin_plot(df, category_col, value_col, name):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=category_col, y=value_col, data=df)
    plt.title(f"Violin Plot of {value_col} by {category_col}")
    save_path = get_err_analysis_path("")
    plt.savefig(save_path / f"{name}_violin.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # load_dotenv(".env")
    main()
