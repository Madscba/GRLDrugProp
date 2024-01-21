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

save_path = Directories.REPO_PATH / "plots/explainability/distmult/visualize_drug_emb"
save_path.mkdir(parents=True, exist_ok=True)
file_path = 'drug_emb.png'
scatter_path = save_path / file_path

# train a distmultmodel
state_dict_path = (
    Directories.REPO_PATH
    / "checkpoints/distmult-16/fold_0/epoch=110-val_mse=9.7478.ckpt"
)
state_dict = torch.load(state_dict_path)["state_dict"]

# state_dict = torch.load("checkpoints/distmult-256/fold_0/epoch=22-val_mse=8.0689.ckpt")[
#     "state_dict"
# ]
embedding_type = "entity"  # entity or relation
embeddings = state_dict[f"model.{embedding_type}"].detach().numpy()
num_drug = embeddings.shape[0]
grid_size = 4

cm = plt.cm.get_cmap("tab20")
colors = cm(np.linspace(0,1,num_drug))
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
fig.tight_layout(pad=2.0)
emb_dims = range(embeddings.shape[1])

for i in range(16):
    row = i // grid_size
    col = i % grid_size
    axs[row, col].scatter(embeddings[:, i], embeddings[:, row],c=colors,alpha=0.5)
    axs[row, col].set_xlabel(f"Embedding {i}")
    axs[row, col].set_ylabel(f"Embedding {row}")

plt.savefig(scatter_path)


# %%
