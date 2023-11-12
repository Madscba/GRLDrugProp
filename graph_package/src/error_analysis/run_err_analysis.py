from graph_package.configs.directories import Directories
from graph_package.src.main import (
    load_data,
    init_model,
    get_model_name,
    get_dataloaders,
    split_dataset,
)
import hydra
from os import listdir
import sys
import os, numpy as np
from graph_package.configs.definitions import model_dict
from pytorch_lightning import Trainer
import torch


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
    test = "abc"


if __name__ == "__main__":
    # load_dotenv(".env")
    # main()

    # fetch saved predictions
    # enrich data with side information
    # plot histograms and other error
    pass
