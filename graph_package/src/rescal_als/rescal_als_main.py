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
        # Select the rows corresponding to the current last index
        mask = last_indices == last_index
        rows = indices[:, mask]
        data = values[mask]

        # Create a lil_matrix for the current last index
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

        data_loaders = get_dataloaders(
            [train_set, val_set, test_set], batch_sizes=config.batch_sizes
        )

        logging.basicConfig(level=logging.INFO)
        _log = logging.getLogger("Example Kinships")

        # mat = loadmat(r'C:\Users\Mads-\Documents\Universitet\Kandidat\5_semester\thesis\GRLDrugProp\graph_package\src\rescal_als\data\alyawarradata.mat')
        # K = np.array(mat['Rs'], np.float32)
        # e, k = K.shape[0], K.shape[2]
        # SZ = e * e * k

        K = train_set.dataset.graph.edge_mask(dataset.indices).adjacency
        e, k = K.shape[0], K.shape[2]
        SZ = e * e * k

        # copy ground truth before preprocessing
        # GROUND_TRUTH = K.copy()

        GROUND_TRUTH = deepcopy(K.to_dense().numpy())

        # construct array for rescal
        # T = [lil_matrix(K[:, :, i]) for i in range(k)]
        # T = [train_set.dataset.graph.edge_mask(train_set.indices).adjacency[:e,:e,i] for i in range(k)]
        T = construct_T_from_torch(K)
        # _log.info('Datasize: %d x %d x %d | No. of classes: %d' % (T[0].shape + (len(T),) + (k,)))

        # Do cross-validation
        # FOLDS = 10
        # IDX = list(range(SZ))
        # shuffle(IDX)

        FOLDS = 10
        IDX = list(range(SZ))
        shuffle(IDX)

        fsz = int(SZ / FOLDS)
        ranks = [1, 5, 10, 20, 35]
        metrics = ["AUC_ROC_train", "AUC_ROC_test", "AUC_PR_train", "AUC_PR_test"]
        reg_scales = [0, 1, 5, 10]
        index = pd.MultiIndex.from_product(
            [metrics, ranks, reg_scales], names=["metric", "rank", "reg_scale"]
        )
        data = np.random.rand(len(index))
        df_metric = pd.DataFrame(data, index=index, columns=["random_data"])
        for rank in ranks:
            for reg_scale in reg_scales:
                offset = 0
                conf = {"reg": reg_scale, "ran": rank}

                AUC_PR_train = np.zeros(FOLDS)
                AUC_ROC_train = np.zeros(FOLDS)
                AUC_PR_test = np.zeros(FOLDS)
                AUC_ROC_test = np.zeros(FOLDS)
                for f in range(FOLDS):
                    idx_test = IDX[offset : offset + fsz]
                    idx_train = np.setdiff1d(IDX, idx_test)
                    shuffle(idx_train)
                    idx_train = idx_train[:fsz].tolist()  # original
                    # idx_train = idx_train[::2].tolist()
                    _log.info("Train Fold %d" % f)
                    AUC_PR_train[f], AUC_ROC_train[f] = innerfold(
                        T, idx_train + idx_test, idx_train, e, k, SZ, GROUND_TRUTH, conf
                    )
                    _log.info("Test Fold %d" % f)
                    AUC_PR_test[f], AUC_ROC_test[f] = innerfold(
                        T, idx_test, idx_test, e, k, SZ, GROUND_TRUTH, conf
                    )

                    offset += fsz

                _log.info(
                    "AUC-PR Test Mean / Std: %f / %f"
                    % (AUC_PR_test.mean(), AUC_PR_test.std())
                )
                _log.info(
                    "AUC-ROC Test Mean / Std: %f / %f"
                    % (AUC_ROC_test.mean(), AUC_ROC_test.std())
                )
                _log.info(
                    "AUC-PR Train Mean / Std: %f / %f"
                    % (AUC_PR_train.mean(), AUC_PR_train.std())
                )
                _log.info(
                    "AUC-ROC Train Mean / Std: %f / %f"
                    % (AUC_ROC_train.mean(), AUC_ROC_train.std())
                )

                df_metric["AUC_PR_test", rank, reg_scale] = AUC_PR_test.mean()
                df_metric["AUC_ROC_test", rank, reg_scale] = AUC_ROC_test.mean()
                df_metric["AUC_ROC_train", rank, reg_scale] = AUC_PR_train.mean()
                df_metric["AUC_PR_train", rank, reg_scale] = AUC_ROC_train.mean()
        a = 2
        b = a**2
        # checkpoint_callback = ModelCheckpoint(
        #     dirpath=get_checkpoint_path(model_name, k), **config.checkpoint_callback
        # )
        # call_backs.append(checkpoint_callback)

        # if (model_name == "hybridmodel") and config.model.pretrain_model:
        #     check_point = pretrain_single_model(config, data_loaders, k)
        #     config.model.update({"ckpt_path": check_point})

        # model = init_model(
        #     model=model_name,
        #     task=config.task,
        #     model_kwargs=config.model,
        #     target=config.dataset.target,
        # )

        # trainer = Trainer(
        #     logger=loggers,
        #     callbacks=call_backs,
        #     **config.trainer,
        # )

        # trainer.validate(model, dataloaders=data_loaders["val"])

        # trainer.fit(
        #     model,
        #     train_dataloaders=data_loaders["train"],
        #     val_dataloaders=data_loaders["val"],
        # )
        # trainer.test(
        #     model,
        #     dataloaders=data_loaders["test"],
        #     ckpt_path=checkpoint_callback.best_model_path,
        # )
        # if config.wandb:
        #     wandb.config.checkpoint_path = checkpoint_callback.best_model_path
        #     wandb.finish()


if __name__ == "__main__":
    load_dotenv(".env")
    main()
