#%%
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from graph_package.src.models.distmult import DistMult
from graph_package.src.pl_modules import BasePL
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg
from pathlib import Path
from graph_package.configs.directories import Directories

base_path =Directories.REPO_PATH / "plots/explainability/distmult"
save_path = Path(base_path /'pairwise_product_pca'/'pca_16.png')
save_path.parent.mkdir(parents=True, exist_ok=True)

#state_dict = torch.load('checkpoints/distmult-256/fold_0/epoch=22-val_mse=8.0689.ckpt')['state_dict']
state_dict = torch.load(Directories.REPO_PATH / 'checkpoints/distmult-16/fold_0/epoch=110-val_mse=9.7478.ckpt')['state_dict']

cell_line_embeddings = state_dict["model.relation"].detach().numpy()
drug_embeddings = state_dict["model.entity"].detach().numpy()
cov_per_cell_line = drug_embeddings * cell_line_embeddings.reshape(-1,1,cell_line_embeddings.shape[-1])

n_components = 4
n_cell_lines = cell_line_embeddings.shape[0]
n_drugs = drug_embeddings.shape[0]

pca_components = np.empty((n_cell_lines,n_components,16))

# Perform PCA analysis
for i in range(len(cell_line_embeddings)):
    pca = PCA(n_components=n_components)
    pca = pca.fit(cov_per_cell_line[i])
    print('explained variance of first cell line')
    print(pca.explained_variance_ratio_)
    pca_components[i] = pca.components_


# Make sure components have same sign 
for i in range(n_components):
    U,s,_ = linalg.svd(pca_components[:,i,:])
    explained_variance = (s ** 2) / np.sum(s ** 2)
    print(f'explained variance of pca on {i}.th components')
    print(explained_variance)
    pca_components[:,i,:] = pca_components[:,i,:]*np.sign(U[:,0]).reshape(-1,1)

random_cell_lines = np.random.choice(n_cell_lines, size=16, replace=False)

# Create a 4x4 plot grid
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
cm = plt.cm.get_cmap("tab20")
colors = cm(np.linspace(0,1,n_drugs))
# Plot the first and second principal components for each random cell line
pca = PCA(n_components=4)
for i, cell_line_idx in enumerate(random_cell_lines):
    row = i // 4
    col = i % 4
    pca = pca.fit(cov_per_cell_line[cell_line_idx])
    pca.components_ = pca_components[cell_line_idx]
    pca_scores = pca.transform(cov_per_cell_line[cell_line_idx])
    axs[row, col].scatter(pca_scores[:,0], pca_scores[:,1],c=colors)
    axs[row, col].set_xlabel("Principal Component 1")
    axs[row, col].set_ylabel("Principal Component 2")

plt.tight_layout()
plt.savefig(save_path)
# %%
