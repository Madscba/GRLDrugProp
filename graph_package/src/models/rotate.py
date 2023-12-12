import torch 
from torch import nn
from torchdrug.models import RotatE as rotate

class RotatE(rotate):
    """
    RotatE embedding proposed in `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_.

    .. _RotatE\: Knowledge Graph Embedding by Relational Rotation in Complex Space:
        https://arxiv.org/pdf/1902.10197.pdf

    Parameters:
        num_entity (int): number of entities
        num_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        max_score (float, optional): maximal score for triplets
    """

    def __init__(
        self,
        ent_tot,
        rel_tot, 
        dim,
        max_score=30
    ):
        super().__init__(
            num_entity=ent_tot, 
            num_relation=rel_tot, 
            embedding_dim=dim,
            max_score=max_score
        )
        nn.init.xavier_uniform_(self.entity)
        nn.init.xavier_uniform_(self.relation)

    def score_triplet(self, h_index, t_index, r_index):
        """
        RotatE score function from `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_.

        .. _RotatE\: Knowledge Graph Embedding by Relational Rotation in Complex Space:
            https://arxiv.org/pdf/1902.10197.pdf

        Parameters:
            entity (Tensor): entity embeddings of shape :math:`(|V|, 2d)`
            relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
            h_index (LongTensor): index of head entities
            t_index (LongTensor): index of tail entities
            r_index (LongTensor): index of relations
        """
        h = self.entity[h_index]
        r = self.relation[r_index]
        t = self.entity[t_index]

        h_re, h_im = h.chunk(2, dim=-1)
        r_re, r_im = torch.cos(r), torch.sin(r)
        t_re, t_im = t.chunk(2, dim=-1)

        x_re = h_re * r_re - h_im * r_im - t_re
        x_im = h_re * r_im + h_im * r_re - t_im
        x = torch.stack([x_re, x_im], dim=-1)
        score = x.norm(p=2, dim=-1).sum(dim=-1)
        return self.max_score - score
    
    def forward(self, inputs):
        h_index, t_index, r_index = map(lambda x: x.squeeze(),inputs.split(1, dim=1))
        score = self.score_triplet(h_index,t_index, r_index)
        return score 
    
    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.cpu().numpy()
    
    def __name__(self) -> str:
        return "RotatE"
