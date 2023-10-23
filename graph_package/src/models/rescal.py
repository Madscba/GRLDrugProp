import torch
import torch.nn as nn
from torchdrug import core


class RESCAL(nn.Module, core.Configurable):
    def __init__(self, ent_tot, rel_tot, dim=100, reg=0):
        super(RESCAL, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim = dim
        self.reg = reg

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_matrices = nn.Embedding(self.rel_tot, self.dim * self.dim)
        # self.rel_matrices = nn.Parameter(torch.Tensor(self.dim, self.dim)) # Adjusted for cont relations

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_matrices.weight.data)

    def _calc(self, h, t, r):
        t = t.view(-1, self.dim, 1)
        r = r.view(-1, self.dim, self.dim)
        tr = torch.matmul(r, t)
        tr = tr.view(-1, self.dim)
        return -torch.sum(h * tr, -1)

    def get_hrt(self, h_index, t_index, r_index):
        # Transform data to expected batch tensor format
        h = self.ent_embeddings(h_index.squeeze())
        t = self.ent_embeddings(t_index.squeeze())
        r = self.rel_matrices(r_index.squeeze())
        return h, r, t

    def forward(self, h_index, t_index, r_index):
        score = torch.zeros_like(h_index, dtype=torch.float32)
        for i in range(h_index.shape[1]):
            h, r, t = self.get_hrt(h_index[:, i], t_index[:, i], r_index[:, i])
            score[:, i] = self._calc(h, t, r)
        return score

    def regularization(self, data):
        h, r, t = self.get_hrt(data)
        regul = (torch.mean(h**2) + torch.mean(t**2) + torch.mean(r**2)) / 3
        return regul

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

    def __name__(self):
        return "RESCAL"
