"""main module."""

from pytorch_lightning.loggers import WandbLogger
from graph_package.src.main_utils import (
    reset_wandb_env,
    load_data,
    init_model,
    get_model_name,
    get_checkpoint_path,
    get_dataloaders,
    split_dataset,
    update_model_kwargs,
    pretrain_single_model,
    get_cv_splits,
)
from graph_package.configs.definitions import model_dict, dataset_dict
from graph_package.src.etl.dataloaders import KnowledgeGraphDataset
from graph_package.src.pl_modules.callbacks import TestDiagnosticCallback
from graph_package.configs.directories import Directories
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import hydra
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split as train_val_split
import os
from pytorch_lightning import Trainer
import sys
import wandb
import shutil
import warnings


from copy import deepcopy
import logging
import numpy as np

# from numpy import dot, array, zeros, setdiff1d
from numpy.linalg import norm
from numpy.random import shuffle
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_curve, auc
from graph_package.src.rescal_als.rescal import rescal_als
from graph_package.src.rescal_als.examples.kinships import (
    predict_rescal_als,
    precision_recall_curve,
    innerfold,
)
import scipy.sparse as sp
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="hydra")
warnings.filterwarnings(
    "ignore", category=PossibleUserWarning, module="pytorch_lightning"
)


def construct_T_from_torch(K):
    # Assuming your sparse tensor 'K' has layout torch.sparse_coo convert into lil_matrix format

    # Step 1: Extract indices, values, and size information
    sparse_tensor = K.coalesce()
    indices = sparse_tensor.indices().numpy()
    values = sparse_tensor.values().numpy()
    size = sparse_tensor.size()

    # Step 2: Create lil_matrix objects based on the last index
    last_indices = indices[-1, :]

    unique_last_indices = np.unique(last_indices)
    lil_matrices = []

    for last_index in unique_last_indices:
        # select the rows corresponding to the current last index /corresponding to cell line
        mask = last_indices == last_index
        rows = indices[:, mask]
        data = values[mask]

        # convert to a lil_matrix for the current last index / current cell line
        lil_matrix = sp.lil_matrix((size[0], size[1]))
        lil_matrix[rows[0], rows[1]] = data

        lil_matrices.append(lil_matrix)

    return lil_matrices


@hydra.main(
    config_path=str(Directories.CONFIG_PATH / "hydra_configs"),
    config_name="config.yaml",
    version_base="1.1",
)
def main(config):
    if config.wandb:
        wandb.login()

    model_name = get_model_name(config, sys_args=sys.argv)
    dataset = load_data(dataset_config=config.dataset, task=config.task)
    update_model_kwargs(config, model_name, dataset)

    splits = get_cv_splits(dataset, config)

    for k, (train_idx, test_idx) in enumerate(splits):
        loggers = []
        if config.wandb:
            reset_wandb_env()
            project = "GRLDrugProp"
            entity = "master-thesis-dtu"
            wandb.init(
                group=config.run_name,
                project=project,
                entity=entity,
                name=f"fold_{k}",
                config=dict(config),
            )
            loggers.append(WandbLogger())

        call_backs = [TestDiagnosticCallback(model_name=model_name, config=config)]

        train_set, test_set = split_dataset(
            dataset, split_method="custom", split_idx=(train_idx, test_idx)
        )

        train_idx, val_idx = train_val_split(
            train_set.indices,
            test_size=0.1,
            random_state=config.seed,
            stratify=dataset.get_labels(train_set.indices),
        )

        train_set, val_set = split_dataset(
            dataset, split_method="custom", split_idx=(train_idx, val_idx)
        )

        # We take undirected adjacency matrix, so inverse should automatically be generated
        # inv_indices = dataset.make_inv_triplets(train_set.indices)
        # train_set.indices = train_set.indices + inv_indices

        data_loaders = get_dataloaders(
            [train_set, val_set, test_set], batch_sizes=config.batch_sizes
        )

        logging.basicConfig(level=logging.INFO)
        _log = logging.getLogger("Example Kinships")

        K = (
            train_set.dataset.graph.undirected()
            .edge_mask(np.append(train_set.indices, test_set.indices))
            .adjacency
        )  # This adjacency matrix corresponds to all the triplets that has been used to train and evaluate our SGD version of Rescal
        e, k = K.shape[0], K.shape[2]
        SZ = e * e * k

        idx_train = list(train_set.indices)
        idx_test = list(test_set.indices)

        # All the triplets seen by our SGD Rescal are not synergetic, so we need to modify the adjacency matrix, such that non synergetic relations are not set to 1
        train_syn = (
            train_set.dataset.data_df.iloc[train_set.indices]["synergy_zip_mean"] >= 5
        ).astype(int)
        test_syn = (
            test_set.dataset.data_df.iloc[test_set.indices]["synergy_zip_mean"] >= 5
        ).astype(int)

        T = construct_T_from_torch(K)

        # modify to only have ones, where it is synergetic (zip_mean > 5)
        train_idx_3dim = np.unravel_index(idx_train, (e, e, k))
        test_idx_3dim = np.unravel_index(idx_test, (e, e, k))

        # a triplet in our graph are not necessarily equal to a label being 1, so we have to modify.
        # Note that missing values will be interpreted just as our negative samples.
        for i, idx in enumerate(train_syn.index.values):
            T[train_idx_3dim[2][i]][
                train_idx_3dim[0][i], train_idx_3dim[1][i]
            ] = train_syn[idx]
            T[train_idx_3dim[2][i]][
                train_idx_3dim[1][i], train_idx_3dim[0][i]
            ] = train_syn[idx]

        for i, idx in enumerate(test_syn.index.values):
            T[test_idx_3dim[2][i]][test_idx_3dim[0][i], test_idx_3dim[1][i]] = test_syn[
                idx
            ]
            T[test_idx_3dim[2][i]][test_idx_3dim[1][i], test_idx_3dim[0][i]] = test_syn[
                idx
            ]

        GROUND_TRUTH = deepcopy(
            T
        )  # Not used in this impl. -> We use test_syn as the true labels.

        # hyperparams to test ALS over.
        ranks = [6, 7, 8, 9, 10, 15, 20, 36]
        metrics = ["AUC_ROC_train", "AUC_ROC_test", "AUC_PR_train", "AUC_PR_test"]
        reg_scales = [0, 1, 5, 10]

        # create result dataframe
        index = pd.MultiIndex.from_product(
            [metrics, ranks, reg_scales], names=["metric", "rank", "reg_scale"]
        )
        df_metric = pd.DataFrame(index=index)

        for rank in ranks:
            for reg_scale in reg_scales:
                conf = {"reg": reg_scale, "rank": rank}

                AUC_PR_test, AUC_ROC_test = innerfold(
                    T, idx_test, idx_test, e, k, SZ, GROUND_TRUTH, conf, test_syn
                )

                _log.info("AUC-PR Test Mean: %f" % (AUC_PR_test))
                _log.info("AUC-ROC Test Mean: %f" % (AUC_ROC_test))

                df_metric["AUC_PR_test", rank, reg_scale] = AUC_PR_test
                df_metric["AUC_ROC_test", rank, reg_scale] = AUC_ROC_test
        results = df_metric.iloc[0]
        a = 2


if __name__ == "__main__":
    load_dotenv(".env")
    main()
