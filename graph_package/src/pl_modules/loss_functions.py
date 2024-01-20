import torch
from torch.nn import functional as F
from torch import log, pi


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MSECellLineVar(torch.nn.Module):
    def __init__(self, num_relation):
        super().__init__()
        self.var = torch.nn.Parameter(torch.ones(num_relation,device=device), requires_grad=False)

    def set_var_grad(self,requires_grad=True):
        self.var.requires_grad = requires_grad

    def forward(self, preds: torch.Tensor, target: torch.Tensor, cell_line_id: torch.Tensor):
        k = preds.shape[0]
        var = F.softplus(self.var[cell_line_id])
        err = target-preds
        log_detS = torch.sum(torch.log(var))
        negloglike =  1/2*log_detS+1/2*torch.dot(err,1/var*err)
        mean_negloglike = negloglike/k
        return mean_negloglike


