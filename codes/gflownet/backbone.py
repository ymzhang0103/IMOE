import torch
from torch.nn import Linear, LeakyReLU, Parameter, Module
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data

class GFlowNetBackbone(Module):
    def __init__(self):
        super(GFlowNetBackbone, self).__init__()
        
    def forward(**kwargs):
        pass
    
    def _init_backbone(self):
        pass
    
    def _init_actionspace(self):
        pass
    

class GCNBackbone(GFlowNetBackbone):
    def __init__(self, num_features, num_gcn_hidden, num_mlp_hidden, init_actionspace=None):
        super(GCNBackbone, self).__init__()
        
        self.num_features = num_features
        self.num_gcn_hidden = num_gcn_hidden
        self.num_mlp_hidden = num_mlp_hidden
        
        self._init_backbone()
        if init_actionspace is not None:
            self.action_space = init_actionspace()
        else:
            self.action_space = self._init_actionspace()
        
        
    def forward(self, x, edge_index):
        
        # Concatenate node embeddings and edge_index with candidate sets
        combined_x = torch.cat([x, self.action_space.x.to(x.device)], dim=0)
        edge_index = torch.cat([edge_index, self.action_space.edge_index.to(edge_index.device)], dim=1)
        
        # Pass through GCN + MLP
        logits = self.model(combined_x, edge_index)
        
        # Mask for removing action candidates from starting node
        action_mask = torch.zeros_like(logits[:, 0]) # Num_Nodes x 1
        action_mask[:x.size(0)] = 1
        
        P_F_s, P_F_e, P_B_s, P_B_e = (
            logits[:, 0] * action_mask - 100 * (1 - action_mask),
            logits[:, 1], 
            logits[:, 2] * action_mask - 100 * (1 - action_mask),
            logits[:, 3]
        )
        return (P_F_s, P_F_e), (P_B_s, P_B_e)
    
    def _init_backbone(self):
        
        # GCN (7-32) (32-64) (64-128) (128-128)
        layers = []
        for i in range(len(self.num_gcn_hidden) + 1):
            if i == 0 :
                layers.append((GCNConv(self.num_features, self.num_gcn_hidden[i]), 'x, edge_index -> x'))
            elif i == len(self.num_gcn_hidden):
                layers.append((GCNConv(self.num_gcn_hidden[i-1], self.num_mlp_hidden[0]), 'x, edge_index -> x'))
            else:
                layers.append((GCNConv(self.num_gcn_hidden[i-1], self.num_gcn_hidden[i]), 'x, edge_index -> x'))
            layers.append(LeakyReLU(inplace=True))
            
        
        # MLP
        for i in range(len(self.num_mlp_hidden) - 1):
            layers.append((Linear(self.num_mlp_hidden[i], self.num_mlp_hidden[i+1]), 'x -> x'))
            layers.append(LeakyReLU(inplace=True))
        
        layers.append(Linear(self.num_mlp_hidden[-1], 4)) # output is shape: num_nodes x 4 where columns are P_f_s, P_f_e, P_b_s, P_b_e        
        
        self.model = Sequential('x, edge_index', layers)
        
        self.logZ = Parameter(torch.ones(1))
    
    def _init_actionspace(self):
        # Stop action is the first node in candidate list
        x = Data(
            x=torch.zeros(self.num_features + 1, self.num_features), # +1 adds stop action
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )

        for i in range(1, self.num_features + 1):
            x.x[i, i-1] = 1
        
        return x