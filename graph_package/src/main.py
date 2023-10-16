"""main module."""

from typing import List, Tuple, Union
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import WandbLogger
from models import RESCAL
from graph_package.configs.definitions import model_dict, dataset_dict
from graph_package.configs.definitions import model_dict, dataset_dict
from graph_package.configs.directories import Directories
from torch.utils.data import random_split, Subset
from models import DeepDDS, RESCAL
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import hydra
from dotenv import load_dotenv
from sklearn.model_selection import KFold, train_test_split as train_val_split
from sklearn.model_selection import KFold, train_test_split as train_val_split
from graph_package.utils.meter import WandbMeter
from torchdrug.data import DataLoader
from torchdrug.data import DataLoader
import os
from pytorch_lightning import Trainer
import sys
import wandb 
import shutil


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def load_data(model: str = "deepdds", dataset: str = "oneil"):
    """Fetch formatted data depending on modelling task"""
    dataset_key =dataset.lower()+ "_" + model.lower() 
    dataset = dataset_dict[dataset_key]()
    return dataset


def init_model(model: str = "deepdds", model_kwargs: dict = {}):
    """Load model from registry"""
    model = model_dict[model.lower()](**model_kwargs)
    return model

def get_model_name(config: dict, sys_args: List[str]):
    for arg in sys_args:
        if arg.startswith("model="):
            return arg.split("=")[1]
    return "deepdds"

def get_checkpoint_path(model_name: str,k: int):
    checkpoint_path = Directories.CHECKPOINT_PATH / model_name / f"fold_{k}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_path)

def get_dataloaders(datasets: List[DataLoader], batch_size: int):
    dataloaders = []
    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        dataloaders.append(dataloader)
    return dataloaders
def get_checkpoint_path(model_name: str,k: int):
    checkpoint_path = Directories.CHECKPOINT_PATH / model_name / f"fold_{k}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_path)

def get_dataloaders(datasets: List[DataLoader], batch_size: int):
    dataloaders = []
    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        dataloaders.append(dataloader)
    return dataloaders

def split_dataset(
    dataset,
    split_method: str = "custom",
    split_idx: Tuple[List[int], List[int]] = None,
):
    if split_method == "custom":
        train_set = Subset(dataset, split_idx[0])
        val_set = Subset(dataset, split_idx[1])

    return train_set, val_set


@hydra.main(config_path=str(Directories.CONFIG_PATH / "hydra_configs"),config_name='config.yaml')
def main(config):


    if config.wandb:
        wandb.login()
    model_name = get_model_name(config, sys_args = sys.argv)
    dataset = load_data(model=model_name, dataset=config.dataset)
    kfold= KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    
    if config.remove_old_checkpoints:
        shutil.rmtree(Directories.CHECKPOINT_PATH / model_name)

    for k, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        loggers = []
        if config.wandb:
            reset_wandb_env()
            wandb.init(group = config.run_name,name=f"fold_{k}", config=dict(config))
            loggers.append(WandbLogger()) 
        call_backs = []

        train_set, test_set = split_dataset(dataset, split_method="custom", split_idx=(train_idx, test_idx))
        train_set, val_set = train_val_split(train_set, test_size=0.1, random_state=config.seed)
        train_loader, val_loader, test_loader = get_dataloaders([train_set, val_set, test_set], batch_size=config.batch_size)
        
        checkpoint_callback = ModelCheckpoint(dirpath=get_checkpoint_path(model_name,k),**config.checkpoint_callback)
        call_backs.append(checkpoint_callback)
        
        model = init_model(model=model_name, model_kwargs=config.model)

        trainer = Trainer(
            logger=loggers,
            callbacks=call_backs,
            **config.trainer,
        )
    
        trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)
        trainer.test(model,dataloaders=test_loader,ckpt_path=checkpoint_callback.best_model_path)
        wandb.finish()





if __name__ == "__main__":
    load_dotenv(".env")
    main()
