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

# %%

base_path =Directories.REPO_PATH / "plots/explainability/distmult/visualize_drug_embeddings"
base_path.mkdir(parents=True, exist_ok=True)
state_dict_fold_0 = torch.load(Directories.REPO_PATH / 'checkpoints/explainability/distmult-16/fold_0/epoch=149-val_mse=11.4737.ckpt')['state_dict']
state_dict_fold_1 = torch.load(Directories.REPO_PATH / 'checkpoints/explainability/distmult-16/fold_1/epoch=149-val_mse=11.3706.ckpt')['state_dict']
state_dicts = [state_dict_fold_0, state_dict_fold_1]
n_drugs_viz=30

for i,state_dict in enumerate(state_dicts):
    print(f'fold {i} model \n')
    save_path = Path(base_path / f"fold_{i}")
    save_path.mkdir(parents=True, exist_ok=True)
    embeddings = state_dict[f"model.entity"].detach().numpy()
    
    n_drugs = embeddings.shape[0]
    grid_size = 4
    cm = plt.cm.get_cmap("tab20")

    drug_ids = np.arange(n_drugs)
    np.random.seed(0)
    np.random.shuffle(drug_ids)
    drug_ids_50 = drug_ids[:-(len(drug_ids)%n_drugs_viz)]
    drug_ids_50 = drug_ids_50.reshape(-1,n_drugs_viz)


    cm = plt.cm.get_cmap("tab20")
    colors = cm(np.linspace(0,1,n_drugs_viz))

    for j,random_drugs in enumerate(drug_ids_50):
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        fig.tight_layout(pad=2.0)
        for i in range(16):
            row = i // grid_size
            col = i % grid_size
            axs[row, col].scatter(embeddings[random_drugs, i], embeddings[random_drugs, row],c=colors)
            axs[row, col].set_xlabel(f"Embedding {i}")
            axs[row, col].set_ylabel(f"Embedding {row}")
        plt.savefig(save_path / f'drug_emb_{j}.png')


    # %%
