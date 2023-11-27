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


    def get_hrt(self, h_index, t_index, r_index):
        # Transform data to expected batch tensor format
        h = self.ent_embeddings(h_index).view(-1, 1, self.dim)
        t = self.ent_embeddings(t_index).view(-1, self.dim, 1)
        r = self.rel_matrices(r_index).view(-1, self.dim, self.dim)
        return h, r, t

    def forward(self, inputs):
        h_index, t_index, r_index = map(lambda x: x.squeeze(),inputs.split(1, dim=1))
        h, r, t = self.get_hrt(h_index, t_index, r_index)
        score = -torch.bmm(h,torch.bmm(r,t)).squeeze()
        return score

    def regularization(self, data):
        h, r, t = self.get_hrt(data)
        regul = (torch.mean(h**2) + torch.mean(t**2) + torch.mean(r**2)) / 3
        return regul

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.cpu().numpy()

    def __name__(self):
        return "RESCAL"
