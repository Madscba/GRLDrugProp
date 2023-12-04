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

warnings.filterwarnings("ignore", category=UserWarning, module="hydra")
warnings.filterwarnings(
    "ignore", category=PossibleUserWarning, module="pytorch_lightning"
)


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

        call_backs = [TestDiagnosticCallback(model_name=model_name)]

        train_set, test_set = split_dataset(
            dataset, split_method="custom", split_idx=(train_idx, test_idx)
        )
        train_set, val_set = train_val_split(
            train_set,
            test_size=0.1,
            random_state=config.seed,
            stratify=dataset.get_labels(train_set.indices),
        )
        data_loaders = get_dataloaders(
            [train_set, val_set, test_set], batch_sizes=config.batch_sizes
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=get_checkpoint_path(model_name, k), **config.checkpoint_callback
        )
        call_backs.append(checkpoint_callback)

        if (model_name == "hybridmodel") and config.model.pretrain_model:
            check_point = pretrain_single_model(config, data_loaders, k)
            config.model.update({"ckpt_path": check_point})

        model = init_model(
            model=model_name,
            task=config.task,
            model_kwargs=config.model,
            target=config.dataset.target,
        )

        trainer = Trainer(
            logger=loggers,
            callbacks=call_backs,
            **config.trainer,
        )

        trainer.validate(model, dataloaders=data_loaders["val"])

        trainer.fit(
            model,
            train_dataloaders=data_loaders["train"],
            val_dataloaders=data_loaders["val"],
        )

        trainer.test(
            model,
            dataloaders=data_loaders["test"],
            ckpt_path=checkpoint_callback.best_model_path,
        )
        wandb.finish()


if __name__ == "__main__":
    load_dotenv(".env")
    main()
