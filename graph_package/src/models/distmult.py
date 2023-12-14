from torch import nn
from torchdrug.models import DistMult as distmult

class DistMult(distmult):
    """
    DistMult embedding proposed in `Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_.

    .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases:
        https://arxiv.org/pdf/1412.6575.pdf

    Parameters:
        num_entity (int): number of entities
        num_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        l3_regularization (float, optional): weight for l3 regularization
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
            embedding_dim=dim
        )
        self.max_score = max_score
        nn.init.xavier_uniform_(self.entity)
        nn.init.xavier_uniform_(self.relation)

    def score_triplet(self, h_index, t_index, r_index):
        """
        DistMult score function from `Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_.

        .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases:
            https://arxiv.org/pdf/1412.6575.pdf

        Parameters:
            entity (Tensor): entity embeddings of shape :math:`(|V|, d)`
            relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
            h_index (LongTensor): index of head entities
            t_index (LongTensor): index of tail entities
            r_index (LongTensor): index of relations
        """
        h = self.entity[h_index]
        r = self.relation[r_index]
        t = self.entity[t_index]
        score = (h * r * t).sum(dim=-1)
        return score
    
    def forward(self, inputs):
        h_index, t_index, r_index = map(lambda x: x.squeeze(),inputs.split(1, dim=1))
        score = self.score_triplet(h_index,t_index, r_index)
        return score 
    
    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.cpu().numpy()
    
    def __name__(self) -> str:
        return "DistMult"