from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import ModuleList, Linear as Lin
from torch_geometric.nn import BatchNorm, ARMAConv
import math
import numpy as np
from torch.nn import functional as F

from torch_geometric.nn import MessagePassing
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import matplotlib.cm as cm
from torch_geometric.data import Data

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, act=nn.Tanh()):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
                ('lin1', Lin(in_channels, hidden_channels)),
                ('act', act),
                ('lin2', Lin(hidden_channels, out_channels))
                ]))
     
    def forward(self, x):
        return self.mlp(x)


class EdgeMaskNet(torch.nn.Module):

    def __init__(self,
                 n_in_channels,
                 e_in_channels,
                 hid=72, n_layers=3):
        super(EdgeMaskNet, self).__init__()

        self.node_lin = Lin(n_in_channels, hid)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(n_layers):
            conv = ARMAConv(in_channels=hid, out_channels=hid)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hid))

        if e_in_channels > 1:
            self.edge_lin1 = Lin(2 * hid, hid)
            self.edge_lin2 = Lin(e_in_channels, hid)
            self.mlp = MLP(2 * hid, hid, 1)
        else:
            self.mlp = MLP(2 * hid, hid, 1)
        self._initialize_weights()
        
    def forward(self, x, edge_index, edge_attr= None):

        x = torch.flatten(x, 1, -1)
        x = F.relu(self.node_lin(x))
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index))
            x = batch_norm(x)

        e = torch.cat([x[edge_index[0, :]], x[edge_index[1, :]]], dim=1)

        if edge_attr is not None and edge_attr.size(-1) > 1:
            e1 = self.edge_lin1(e)
            e2 = self.edge_lin2(edge_attr)
            e = torch.cat([e1, e2], dim=1)  # connection

        return self.mlp(e)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight) 


EPS = 1e-6

class InstanceLevelExplainer(nn.Module):
    coeffs = {
        'edge_size': 1e-3,
        'edge_ent': 1e-2,
    }
    graphtype_edgesnum = {"house":12, "star": 12, "cycle":12, "grid":24, "fan":22, "diamond":24}
    def __init__(self, args, gnn_model,
                 n_in_channels=14,
                 e_in_channels=3,
                 hid=50, n_layers=2,
                 n_label=2, gamma=1
                 ):
        super(InstanceLevelExplainer, self).__init__() 
        self.vis_dict = None
        self.device = args.device
        self.gnn_model = gnn_model
        self.dataset_name=args.dataset_name
        self.mask_net = EdgeMaskNet(
                n_in_channels,
                e_in_channels,
                hid=hid,
                n_layers=n_layers).to(args.device)

    def _set_masks(self, mask, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module._explain = True
                module._edge_mask = mask

    def _clear_masks(self, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module._explain = False
                module._edge_mask = None

    def _reparameterize(self, log_alpha, beta=1, training=True):
        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()
            
        return gate_inputs
    
    def get_pos_edge(self, graph, mask, ratio):
        num_edge = [0]
        num_node = [0]
        sep_edge_idx = []
        graph_map = graph.batch[graph.edge_index[0, :]]
        pos_idx = torch.LongTensor([])
        mask = mask.detach().cpu()
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = max(math.ceil(ratio * Gi_n_edge), 1)

            Gi_pos_edge_idx = np.argsort(-mask[edge_indicator])[:topk]

            pos_idx = torch.cat([pos_idx, edge_indicator[Gi_pos_edge_idx]])
            num_edge.append(num_edge[i] + Gi_n_edge)
            num_node.append(
                num_node[i] + (graph.batch == i).sum().long()
            )
            sep_edge_idx.append(Gi_pos_edge_idx)

        return pos_idx, num_edge, num_node, sep_edge_idx
    
    def _relabel(self, g, edge_index):
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


    def fidelity_loss(self, log_logits, mask, pred_label):
        pred = log_logits.softmax(dim=1)[0]
        logit = pred[pred_label.item()]
        pred_loss = -torch.log(logit)
        pred_loss = pred_loss
        # size
        self.mask_act = 'sigmoid'
        self.coff_size = 0.01
        self.coff_ent = 1.0
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(mask)
        elif self.mask_act == "ReLU":
            mask = nn.functional.relu(mask)
        #size_loss = self.coff_size * torch.sum(mask) #len(mask[mask > 0]) #torch.sum(mask)
        size_loss = torch.mean(mask)

        # entropy
        mask = mask *0.99+0.005     # maybe a .clone()
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss
    
    def calculate_entropy(self, x):
        ex = torch.exp(x)
        ex_norm = ex/torch.norm(ex, p=1)
        entropy = -sum(ex_norm*torch.log(ex_norm))
        return entropy


    def explain(self, graph, label_graphs, l=2):
        edge_mask = self.mask_net(
            graph.x,
            graph.edge_index,
            graph.edge_weight
        ).view(-1)
        edge_mask = self._reparameterize(edge_mask, training=False)
        self.edge_mask = edge_mask
        data = Data(x=graph.x, edge_index=graph.edge_index)
        self._set_masks(edge_mask, self.gnn_model) 
        G1_logits,  G1_pred, G1_emb, G1_node_embs = self.gnn_model(data)
        self._clear_masks(self.gnn_model)
        
        fid_loss = self.fidelity_loss(G1_logits, edge_mask, graph.y)

        #calculate cosine similarity with each cluster center
        graph_similarity = []
        for label_g_emb in label_graphs:
            sim = torch.cosine_similarity(G1_emb, torch.Tensor(label_g_emb).to(G1_emb.device))
            graph_similarity.append(sim)
        s_list = [1/s for s in graph_similarity]
        s_list.sort()
        label_loss = sum(s_list[:l])

        imp = edge_mask.detach().cpu().numpy() 
        self.last_result = (graph, imp)
        
        return edge_mask, fid_loss, label_loss, graph_similarity
    

    def get_explain_graph(self, graph, draw_graph=0):
        edge_mask = self.mask_net(
                graph.x,
                graph.edge_index,
                graph.edge_attr
            ).view(-1)
        edge_mask = self._reparameterize(edge_mask, training=False)
        imp = edge_mask.detach().cpu().numpy()
        self.last_result = (graph, imp)
        
        return edge_mask
 