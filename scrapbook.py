#%%
import torch

# Create a batch of vectors and a batch of matrices
batch_size = 3
vector_dim = 4
matrix_dim = 4
vectors = torch.randn(batch_size, vector_dim)
matrices = torch.randn(batch_size, matrix_dim, matrix_dim)

# Calculate the matrix-vector product
result = torch.bmm(matrices, vectors.unsqueeze(2)).squeeze()

print(result)

# %%
