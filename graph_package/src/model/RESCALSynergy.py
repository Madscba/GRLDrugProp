import torch
import torch.nn as nn

class RESCALSynergy(nn.Module):

	def __init__(self, ent_tot, rel_tot, dim = 100):
		super(RESCALSynergy, self).__init__()
		self.ent_tot = ent_tot
		self.rel_tot = rel_tot
		self.dim = dim
	
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_matrices = nn.Parameter(torch.Tensor(self.dim, self.dim)) # Adjusted for cont relations

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_matrices.weight.data)
	
	def _calc(self, h, t, r):
		t = t.view(-1, self.dim, 1)
		r = r.view(-1, self.dim, self.dim)
		tr = torch.matmul(r, t)
		tr = tr.view(-1, self.dim)
		tr = 1
		return -torch.sum(h * tr, -1)
	
	def get_hrt(self, data):
		# Transform data to expected batch tensor format
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_matrices(batch_r)
		return h,r,t
	
	def forward(self, data):
		h, r, t = self.get_hrt(data)
		score = self._calc(h ,t, r)
		return score

	def regularization(self, data):
		h, r, t = self.get_hrt(data)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		score = self.forward(data)
		return score.cpu().data.numpy()