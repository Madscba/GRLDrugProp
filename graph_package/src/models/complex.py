from torch import nn
from torchdrug.models import ComplEx as complex

class ComplEx(complex):
    """
    ComplEx embedding proposed in `Complex Embeddings for Simple Link Prediction`_.

    .. _Complex Embeddings for Simple Link Prediction:
        http://proceedings.mlr.press/v48/trouillon16.pdf

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
        max_score=30,
        l3_regularization=0
    ):
        super().__init__(
            num_entity=ent_tot, 
            num_relation=rel_tot, 
            embedding_dim=dim,
            l3_regularization=l3_regularization
        )
        self.max_score = max_score
        nn.init.xavier_uniform_(self.entity)
        nn.init.xavier_uniform_(self.relation)

    def score_triplet(self, h_index, t_index, r_index):
        """
        ComplEx score function from `Complex Embeddings for Simple Link Prediction`_.

        .. _Complex Embeddings for Simple Link Prediction:
            http://proceedings.mlr.press/v48/trouillon16.pdf

        Parameters:
            entity (Tensor): entity embeddings of shape :math:`(|V|, 2d)`
            relation (Tensor): relation embeddings of shape :math:`(|R|, 2d)`
            h_index (LongTensor): index of head entities
            t_index (LongTensor): index of tail entities
            r_index (LongTensor): index of relations
        """
        h = self.entity[h_index]
        r = self.relation[r_index]
        t = self.entity[t_index]

        h_re, h_im = h.chunk(2, dim=-1)
        r_re, r_im = r.chunk(2, dim=-1)
        t_re, t_im = t.chunk(2, dim=-1)

        x_re = h_re * r_re - h_im * r_im
        x_im = h_re * r_im + h_im * r_re
        x = x_re * t_re + x_im * t_im
        score = x.sum(dim=-1)
        return score
    
    def forward(self, inputs):
        h_index, t_index, r_index = map(lambda x: x.squeeze(),inputs.split(1, dim=1))
        score = self.score_triplet(h_index,t_index, r_index)
        return score 
    
    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.cpu().numpy()

    def __name__(self) -> str:
        return "ComplEx"