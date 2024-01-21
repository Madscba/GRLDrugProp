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

overwrite = True
base_path = Directories.REPO_PATH / "plots/explainability/distmult"
save_path = Path(base_path / "visualize_cell_line_emb")
save_path.mkdir(parents=True, exist_ok=True)
file_path = (
    "cell_line_emb.png"
    if not overwrite
    else "cell_line_emb_{}.png".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
)
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
embedding_type = "relation"  # entity or relation
plot_title = "distmult-256-drug-std.png"
embeddings = state_dict[f"model.{embedding_type}"].detach().numpy()
num_cell_lines = embeddings.shape[0]
grid_size = 4

random_cell_lines = (
    np.random.choice(num_cell_lines, size=grid_size**2, replace=False)
    if grid_size**2 < num_cell_lines
    else np.arange(num_cell_lines)
)


fig, axs = plt.subplots(4, 4, figsize=(12, 12))
fig.tight_layout(pad=2.0)
emb_dims = range(embeddings.shape[1])

for i, cell_line_idx in enumerate(random_cell_lines):
    row = i // grid_size
    col = i % grid_size
    axs[row, col].bar(emb_dims, embeddings[cell_line_idx])
    axs[row, col].set_ylim(-1.5, 1.5)
    axs[row, col].set_title(f"Cell Line {cell_line_idx}")

fig.text(0.5, 0.04, "Cell Line Index", ha="center", va="center")
fig.text(0.06, 0.5, "Size of Embedding", ha="center", va="center", rotation="vertical")
plt.subplots_adjust(bottom=0.1, top=0.9)
plt.subplots_adjust(left=0.1, right=0.9)

plt.savefig(scatter_path)

# %%
