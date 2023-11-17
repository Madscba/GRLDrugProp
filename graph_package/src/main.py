"""main module."""

from pytorch_lightning.loggers import WandbLogger
from graph_package.utils.helpers import (
    reset_wandb_env,
    load_data,
    init_model,
    get_model_name,
    get_checkpoint_path,
    get_dataloaders,
    split_dataset,
    update_model_kwargs
)
from graph_package.configs.definitions import model_dict, dataset_dict
from graph_package.src.etl.dataloaders import KnowledgeGraphDataset
from graph_package.configs.directories import Directories
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import hydra
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold, train_test_split as train_val_split
import os
from pytorch_lightning import Trainer
import sys
import wandb
import shutil




def pretrain_single_model(config, data_loaders, k):
    model_name = config.model.pretrain_model
    check_point_path = Directories.CHECKPOINT_PATH / model_name

    if os.path.isdir(check_point_path):
        shutil.rmtree(check_point_path)

    checkpoint_callback = ModelCheckpoint(
            dirpath=get_checkpoint_path(model_name, k), **config.checkpoint_callback
        )

    model = init_model(model=model_name, model_kwargs=config.model[model_name])

    trainer = Trainer(
        logger=[],
        callbacks=[checkpoint_callback],
        **config.trainer,
    )

    trainer.fit(
        model,
        train_dataloaders=data_loaders["train"],
        val_dataloaders=data_loaders["val"],
    )

    return checkpoint_callback.best_model_path
    

@hydra.main(
    config_path=str(Directories.CONFIG_PATH / "hydra_configs"),
    config_name="config.yaml",
)
def main(config):
    if config.wandb:
        wandb.login()

    model_name = get_model_name(config, sys_args=sys.argv)
    dataset = load_data(dataset=config.dataset)
    update_model_kwargs(config, model_name, dataset)

    kfold = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )

    if config.remove_old_checkpoints:
        check_point_path = Directories.CHECKPOINT_PATH / model_name
        if os.path.isdir(check_point_path):
            shutil.rmtree(check_point_path)
    


    for k, (train_idx, test_idx) in enumerate(
        kfold.split(dataset, dataset.get_labels(dataset.indices))
    ):
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

        call_backs = []

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
            check_point = pretrain_single_model(config,data_loaders,k)
            config.model.update({"ckpt_path": check_point})

        model = init_model(model=model_name, model_kwargs=config.model)

        trainer = Trainer(
            logger=loggers,
            callbacks=call_backs,
            **config.trainer,
        )

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
