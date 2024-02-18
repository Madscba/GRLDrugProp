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
import seaborn as sns

base_path =Directories.REPO_PATH / "plots/explainability/distmult/pairwise_product_pca"
base_path.mkdir(parents=True, exist_ok=True)

# state_dict_fold_0 = torch.load(Directories.REPO_PATH / 'checkpoints/explainability/distmult/fold_0/epoch=06-val_mse=7.2576.ckpt')['state_dict']
# state_dict_fold_1 = torch.load(Directories.REPO_PATH / 'checkpoints/explainability/distmult/fold_1/epoch=06-val_mse=7.0433.ckpt')['state_dict']
state_dict_fold_0 = torch.load(Directories.REPO_PATH /'checkpoints/explainability/distmult-16/fold_0/epoch=149-val_mse=11.4737.ckpt')['state_dict']
state_dict_fold_1 = torch.load(Directories.REPO_PATH /'checkpoints/explainability/distmult-16/fold_1/epoch=149-val_mse=11.3706.ckpt')['state_dict']
state_dicts = [state_dict_fold_0, state_dict_fold_1]
n_drugs_viz = 25
sns.set()

random_cell_lines={
    139:"LOVO",
    162:"SNB-75",
    149:"OV90",
    50:"L-1236",
    56:"MDA-MB-175-VII",
    82:"NCI-H838"}


for k,state_dict in enumerate(state_dicts):
    print(f'fold {k} model \n')
    save_path = base_path / f'fold_{k}'
    save_path.mkdir(parents=True, exist_ok=True)
    cell_line_embeddings = state_dict["model.relation"].detach().numpy()
    drug_embeddings = state_dict["model.entity"].detach().numpy()
    cov_per_cell_line = drug_embeddings * cell_line_embeddings.reshape(-1,1,cell_line_embeddings.shape[-1])
    cov_per_cell_line = np.concatenate((cov_per_cell_line,drug_embeddings.reshape(1,*drug_embeddings.shape)),axis=0)
    n_components = 4
    n_cell_lines = cell_line_embeddings.shape[0]+1
    n_drugs = drug_embeddings.shape[0]

    pca_components = np.empty((n_cell_lines,n_components,16))
    explained_variance_list= []
    #Perform PCA analysis
    for i in range(n_cell_lines):
        pca = PCA(n_components=n_components)
        pca = pca.fit(cov_per_cell_line[i])
        explained_variance_16 = np.sum(pca.explained_variance_ratio_[:2])
        print(f'Explained variance of cell line {i}: {explained_variance_16}')
        pca_components[i] = pca.components_
        explained_variance_list.append(explained_variance_16)



    for i in range(n_components):
        U,s,_ = linalg.svd(pca_components[:,i,:])
        explained_variance = (s ** 2) / np.sum(s ** 2)
        explained_variance_16 = np.cumsum(explained_variance)[:16]
        print(f'explained variance of pca on {i}.th components {explained_variance_16}')
        #a = np.sign(U[:,0])
        pca_components[:,i,:] = pca_components[:,i,:]*np.sign(U[:,0]).reshape(-1,1)

    cell_line_ids = np.load(base_path.parent / f"visualize_cell_line_emb/fold_{k}/cell_line_ids.npy")
    cell_line_ids_16 = cell_line_ids[:-(len(cell_line_ids)%16)]
    cell_line_ids_16 = cell_line_ids_16.reshape(-1,16)
    
    # Create a 4x4 plot grid
    #fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    cm = plt.cm.get_cmap("tab20")
    colors = cm(np.linspace(0,1,n_drugs_viz))
    # Plot the first and second principal components for each random cell line
    pca = PCA(n_components=4)
    # for j,random_cell_lines in enumerate(cell_line_ids_16):
    #     fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i, cell_line_idx in enumerate(random_cell_lines.keys()):
        row = i // 3
        col = i % 3
        pca = pca.fit(cov_per_cell_line[cell_line_idx])
        pca.components_ = pca_components[cell_line_idx]
        pca_scores = pca.transform(cov_per_cell_line[cell_line_idx])
        axs[row, col].scatter(pca_scores[:n_drugs_viz,0], pca_scores[:n_drugs_viz,1],c=colors)
        axs[row, col].set_title(f'{random_cell_lines[cell_line_idx]} ({(explained_variance_list[cell_line_idx]*100):.2f}%)')
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    fig.text(0.5, 0.03, 'Principal Component 2', ha='center')
    fig.text(0.03, 0.5, 'Principal Component 1', va='center', rotation='vertical')
    plt.savefig(save_path / f'cell_lines_best_worst.png')

    # Adjust the layout to make room for the figure text
    
        #plt.savefig(save_path / f'cell_line_{j}.png')
    pca = pca.fit(cov_per_cell_line[-1])
    pca_components_new= pca_components[-1]
    pca_components_new[[0,1],:] = pca_components_new[[0,1],:]*-1
    pca.components_ = pca_components_new
    pca_scores = pca.transform(cov_per_cell_line[-1])
    plt.figure(figsize=(3,3))
    plt.scatter(pca_scores[:n_drugs_viz,0], pca_scores[:n_drugs_viz,1],c=colors)
    plt.xlabel("Principal Component 1")
    plt.title(f"Drug embeddings ({(explained_variance_list[cell_line_idx]*100):.2f}%)")
    plt.ylabel("Principal Component 2") 
    plt.savefig(save_path / f'drug_embeddings.png')


