import numpy as np
import torch
import math
from torch_geometric.data import Data

def relabel(g, edge_index):
        sub_nodes = torch.unique(edge_index)
        x = g.x[sub_nodes]
        batch = g.batch[sub_nodes]
        row, col = edge_index
        pos = None
        try:
            pos = g.pos[sub_nodes]
        except:
            pass

        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((g.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        edge_index = node_idx[edge_index]
        return x, edge_index, batch, pos

def pack_explanatory_subgraph(top_ratio=0.2, 
                                  graph=None, imp=None, relabel_flag=True):
        assert len(imp) == graph.num_edges, 'length mismatch'
        
        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]
        exp_subgraph = graph.clone()
        exp_subgraph.y = graph.y
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = min(max(math.ceil(top_ratio * Gi_n_edge), 1), Gi_n_edge)
            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
        # retrieval properties of the explanatory subgraph
        # .... the edge_attr.
        if graph.edge_attr is not None:
            exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        # .... the edge_index.
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        # .... the nodes.
        # exp_subgraph.x = graph.x
        if relabel_flag:
            exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch, exp_subgraph.pos = relabel(exp_subgraph, exp_subgraph.edge_index)
        
        return exp_subgraph

def evaluate_acc(gnn_model, top_ratio_list, graph=None, imp=None):
    acc = np.array([[]])
    prob = np.array([[]])
    y = graph.y
    for idx, top_ratio in enumerate(top_ratio_list):
        if top_ratio == 1.0:
            #emb, logits, pred = gnn_model(graph.x, graph.edge_index, batch = graph.batch)
            data = Data(x=graph.x, edge_index=graph.edge_index, batch = graph.batch)
            logits, pred, emb, sub_embs = gnn_model(data)
        else:
            exp_subgraph = pack_explanatory_subgraph(top_ratio, graph=graph, imp=imp)
            #emb, logits, pred = gnn_model(exp_subgraph.x, exp_subgraph.edge_index, batch = exp_subgraph.batch)
            data = Data(x=exp_subgraph.x, edge_index=exp_subgraph.edge_index, batch = exp_subgraph.batch)
            logits, pred, emb, sub_embs = gnn_model(data)
        res_acc = (y == logits.argmax(dim=1)).detach().cpu().float().view(-1, 1).numpy()
        res_prob = torch.nn.Softmax(dim=1)(logits)[:, y].detach().cpu().float().view(-1, 1).numpy()
        acc = np.concatenate([acc, res_acc], axis=1)
        prob = np.concatenate([prob, res_prob], axis=1)
    return acc, prob
