# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from graph_package.src.models.distmult import DistMult
from graph_package.src.pl_modules import BasePL
import torch
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from graph_package.configs.directories import Directories
import seaborn as sns

# %%
base_path =Directories.REPO_PATH / "plots/explainability/distmult/visualize_cell_line_emb"
base_path.mkdir(parents=True, exist_ok=True)

state_dict_fold_0 = torch.load(Directories.REPO_PATH / 'checkpoints/explainability/distmult-16/fold_0/epoch=149-val_mse=11.4737.ckpt')['state_dict']
state_dict_fold_1 = torch.load(Directories.REPO_PATH / 'checkpoints/explainability/distmult-16/fold_1/epoch=149-val_mse=11.3706.ckpt')['state_dict']
state_dicts = [state_dict_fold_0, state_dict_fold_1]

sns.set()

random_cell_lines={
    139:"LOVO",
    162:"SNB-75",
    149:"OV90",
    50:"L-1236",
    56:"MDA-MB-175-VII",
    82:"NCI-H838"}


for i,state_dict in enumerate(state_dicts):
    print(f'fold {i} model \n')
    save_path = Path(base_path / f"fold_{i}")
    save_path.mkdir(parents=True, exist_ok=True)


    # train a distmultmodel
    embeddings = state_dict[f"model.relation"].detach().numpy()
    n_cell_lines = embeddings.shape[0]
    num_cell_lines = embeddings.shape[0]
    grid_size = 3

    cell_line_ids = np.arange(n_cell_lines)
    np.random.seed(0)
    np.random.shuffle(cell_line_ids)
    np.save(base_path / f"fold_{i}" / "cell_line_ids.npy", cell_line_ids)
    cell_line_ids_16 = cell_line_ids[:-(len(cell_line_ids)%16)]
    cell_line_ids_16 = cell_line_ids_16.reshape(-1,16)


    emb_dims = range(embeddings.shape[1])
    #for j,random_cell_lines in enumerate(cell_line_ids_16):
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    fig.tight_layout(pad=2.0)
    for i, cell_line_idx in enumerate(random_cell_lines.keys()):
        row = i // grid_size
        col = i % grid_size
        axs[row, col].bar(emb_dims, embeddings[cell_line_idx])
        axs[row, col].set_ylim(-1.5, 1.5)
        axs[row, col].set_title(f"{random_cell_lines[cell_line_idx]}")
    fig.text(0.5, 0.03, "Embedding Dimension", ha="center", va="center")
    fig.text(0.03, 0.5, "Size of Embedding", ha="center", va="center", rotation="vertical")
    plt.subplots_adjust(bottom=0.1, top=0.9)
    plt.subplots_adjust(left=0.1, right=0.9)
    # plt.savefig(save_path / f'cell_line_emb_{j}.png')
    plt.savefig(save_path / f'cell_line_emb_best_worst.png')
    fig.clear()

# %%
