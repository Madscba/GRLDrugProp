from graph_package.src.error_analysis.utils import (
    save_performance_plots,
    save_model_pred,
)
from graph_package.src.main_utils import (
    init_model,
    save_pretrained_drug_embeddings
)
from graph_package.src.models.hybridmodel import remove_prefix_from_keys
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig
from pathlib import Path
from graph_package.src.etl.dataloaders import KnowledgeGraphDataset
import torch
import pandas as pd

class TestDiagnosticCallback(Callback):
    def __init__(self, model_name, config: DictConfig, graph: KnowledgeGraphDataset, fold: int, triplets: pd.DataFrame) -> None:
        self.model_name = model_name
        self.config = config
        self.graph = graph
        self.fold = fold
        self.triplets = triplets

    def on_test_end(self, trainer, pl_module):
        # save and perform err_diag
        (
            df_cm,
            metrics,
            preds,
            target,
            batch,
            batch_idx,
        ) = pl_module.test_step_outputs.values()
        print("conf_matrix:\n", df_cm)

        save_performance_plots(
            df_cm, metrics, preds, target, self.config, self.model_name, save_path=Path("")
        )
        save_model_pred(
            batch_idx, batch, preds, target, self.config, self.model_name, save_path=Path("")
        )
        if self.model_name=="gnn":
            model = init_model(
                model=self.model_name,
                config=self.config,
                graph=self.graph,
            ).model
            state_dict = remove_prefix_from_keys(
                torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"], "model."
            )
            model.load_state_dict(state_dict)
            model.eval()
            self._explain_attention(batch[0], model)
        # save pretrained drug embeddings
        if self.model_name in ["deepdds", "distmult"]:
            save_pretrained_drug_embeddings(model=pl_module,fold=self.fold)
        pl_module.test_step_outputs.clear()

    def _explain_attention(self,batch,model):
        n_nodes = self.graph.node_feature.shape[0]
        self_loop_edges = torch.stack([torch.arange(n_nodes, device=self.graph.device)], dim=1).view(-1, 1).repeat(1, 2)
        self_loop_edge_list = torch.cat([self_loop_edges, torch.ones(n_nodes,1,dtype=torch.int,device=self.graph.device)*self.graph.num_relation], dim=1)
        edge_list = torch.cat([self.graph.edge_list, self_loop_edge_list], dim=0)
        
        node_in, node_out, relation = edge_list.t()
        for layer in model.gnn_layers:
            _, attention = layer(self.graph, self.graph.node_feature, return_att=True)

        # Get top 10 attention weights and their indices
        top_indices = torch.topk(attention.squeeze(1), 10).indices

        # Retrieve corresponding triplets
        top_triplets = [(node_in[i].item(), node_out[i].item(), relation[i].item()) for i in top_indices]

        # Match with synergy scores
        top_triplets_with_scores = []
        for triplet in top_triplets:
            drug_1_id, drug_2_id, context_id = triplet
            synergy_score = self.triplets.loc[
                (self.triplets['drug_1_id'] == drug_1_id) & 
                (self.triplets['drug_2_id'] == drug_2_id) & 
                (self.triplets['context_id'] == context_id), 
                'synergy_zip_mean'
            ].values[0]
            top_triplets_with_scores.append((triplet, synergy_score))

        import matplotlib.pyplot as plt
        import seaborn as sns


        return
