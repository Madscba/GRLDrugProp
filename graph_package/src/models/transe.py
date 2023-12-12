from torch import nn
from torchdrug.models import TransE as transe

class TransE(transe):
    """
    TransE embedding proposed in `Translating Embeddings for Modeling Multi-relational Data`_.

    .. _Translating Embeddings for Modeling Multi-relational Data:
        https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

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
        Compute the score for each triplet.

        Parameters:
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
        """
        h = self.entity[h_index]
        r = self.relation[r_index]
        t = self.entity[t_index]
        score = (h + r - t).norm(p=1, dim=-1)
        return self.max_score - score

    def forward(self, inputs):
        h_index, t_index, r_index = map(lambda x: x.squeeze(),inputs.split(1, dim=1))
        score = self.score_triplet(h_index,t_index, r_index)
        return score 
    
    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.cpu().numpy()
    
    def __name__(self) -> str:
        return "TransE"