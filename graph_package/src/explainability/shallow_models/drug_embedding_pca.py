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
from pathlib import Path
# directories
from graph_package.configs.directories import Directories

base_path = Directories.REPO_PATH / "plots/explainability/distmult"
save_path = Path(base_path /'embedding_pca')
save_path.mkdir(parents=True, exist_ok=True)

# train a distmultmodel 
state_dict_fold_0 = torch.load(Directories.REPO_PATH /'checkpoints/explainability/distmult-16/fold_0/epoch=149-val_mse=11.4737.ckpt')['state_dict']
state_dict_fold_1 = torch.load(Directories.REPO_PATH /'checkpoints/explainability/distmult-16/fold_1/epoch=149-val_mse=11.3706.ckpt')['state_dict']
state_dict = state_dict_fold_1
#state_dict = torch.load('checkpoints/distmult-256/fold_0/epoch=22-val_mse=8.0689.ckpt')['state_dict']
color_metric = 'std'
embedding_type='entity' # entity or relation
plot_title = f'distmult-16-{embedding_type}-{color_metric}.png'
save_path = save_path / plot_title
embeddings = state_dict[f"model.{embedding_type}"].detach().numpy()

# Perform PCA analysis
pca = PCA(n_components=4)
pca = pca.fit(embeddings)
print(np.sum(pca.explained_variance_ratio_))
principal_components = pca.fit_transform(embeddings)

data_csv = pd.read_csv(Directories.REPO_PATH / 'data/gold/drugcomb_filtered/drugcomb_filtered.csv')
df_inv = data_csv.copy()
df_inv["drug_1_name"], df_inv["drug_2_name"] = data_csv["drug_2_name"], data_csv["drug_1_name"]
df_inv["drug_1_id"], df_inv["drug_2_id"] = data_csv["drug_2_id"], data_csv["drug_1_id"]
data_df = pd.concat([data_csv, df_inv], ignore_index=True)
metric = data_df.groupby(['context_id' if embedding_type=='relation' else 'drug_1_id']).agg({'synergy_zip_mean': color_metric}).reset_index()

# Create a plot grid
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
n_drugs = 25
cm = plt.cm.get_cmap("tab20")
colors = cm(np.linspace(0,1,n_drugs))
# Plot the first four principal components
l = [1,2,2,3]
k=0
for i in range(2):
    for j in range(2):
        x_component_index = i
        y_component_index = l[k]
        axs[i, j].scatter(principal_components[:n_drugs, x_component_index], principal_components[:n_drugs, y_component_index],
                          c=colors)
        axs[i, j].set_xlabel(f"Principal Component {x_component_index + 1}")
        axs[i, j].set_ylabel(f"Principal Component {y_component_index + 1}")
        k+=1

plt.tight_layout()
# fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='viridis'),
#              ax=axs, orientation='vertical', label=color_metric)
plt.savefig(save_path)
# %%
