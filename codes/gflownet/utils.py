import torch
from torch_geometric.utils import degree
from torch_geometric.data import Data
import numpy as np
import argparse
from typing import List
from collections import OrderedDict


### GRAPH UTILS ###
def check_edge(edge_index, new_edge):
    # Check if an edge is in the graph
    return bool(torch.all(torch.eq(edge_index, new_edge), dim=0).any())

def append_edge(edge_index, new_edge):
    return torch.cat((edge_index, new_edge), dim=1)

def get_init_state_graph(num_node_features, start_idx=-1):
    # Create initial graph with only the starting node
    g = Data(x=torch.zeros((1, num_node_features)), 
             edge_index=torch.zeros((2, 0), dtype=torch.long), 
             y=torch.torch.zeros((1, 1), dtype=torch.long),
             batch =  torch.zeros(1, dtype=torch.long)
        )
    # Assign starting node feature
    if start_idx == -1:
        start_idx = np.random.randint(0, num_node_features)

    g.x[0, start_idx] = 1
    g.y[0] = start_idx
    return g

### MUTAG UTILS ###
def check_valency_violation(G):
    valency_dict = {0: 4, 1: 3, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1}
    degree_values = degree(G.edge_index[0], num_nodes=G.num_nodes) + degree(G.edge_index[1], num_nodes=G.num_nodes)

    for i in range(G.num_nodes):
        atom_type = G.x[i].nonzero().item()
        if degree_values[i] > valency_dict[atom_type]:
            return True
    return False

def take_action_mutag(G, action):
    # Takes an action in the form (starting_node, ending_node) and 
    # returns the new graph, whether the action is valid, and whether the action is a stop action
    start, end = action
    G_new = G.clone()

    # If end node is stop action, return graph
    if end == G.x.size(0):
        return G, True, True

    # If end node is new candidate, add it to the graph
    if end > G.x.size(0): # changed from end > G.x.size(0) - 1 because now stop action is G.x.size(0)
        # Create new node
        candidate_idx = end - G.x.size(0) - 1
        new_feature = torch.zeros(1, G_new.x.size(1))
        new_feature[0, candidate_idx] = 1
        G_new.x = torch.cat([G_new.x, new_feature], dim=0)
        G_new.y = torch.cat([G_new.y, torch.zeros((1, 1))], dim=0)
        G_new.y[G_new.x.size(0)-1] = candidate_idx 
        end = G_new.x.size(0) - 1
    
    # Check if edge already exists
    if check_edge(G_new.edge_index, torch.tensor([[start], [end]])):
        # If edge exists, return original G 
        return G, False, False
    else:
        # Add edge from start to end
        G_new.edge_index = append_edge(G_new.edge_index, torch.tensor([[start], [end]]))
    
    # Check if valency is violated
    if check_valency_violation(G_new):
        return G, False, False
    else:
        return G_new, True, False
    
    
### LOGISTICS UTILS ###
def save_gflownet(gflownet, path, name):
    import os
    from datetime import datetime
    
    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(gflownet.backbone.state_dict(), os.path.join(path, name + '_backbone_' + dt_string + '.pt'))
    torch.save(gflownet.proxy.state_dict(), os.path.join(path, name + '_proxy_' + dt_string + '.pt'))
    
def load_gflownet(gflownet, path, name):
    import os
    
    backbone_path = os.path.join(path, name + '_backbone.pt')
    proxy_path = os.path.join(path, name + '_proxy.pt')
    
    gflownet.backbone.load_state_dict(torch.load(backbone_path))
    gflownet.proxy.load_state_dict(torch.load(proxy_path))
    
    return gflownet
### PLOTTING UTILS ###


def get_gnnModel_params():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument("--model_name", type=str, default='gcn', help="model name string")  
    parser.add_argument("--checkpoint", type=str, default='./checkpoint', help="checkpoint path string")  
    parser.add_argument("--readout", type=str, default='max', help=" the graph pooling method")  
    parser.add_argument("--model_path", type=str, default='', help="default path to save the model")  
    parser.add_argument('--concate', type=bool, default=False, help="whether to concate the gnn features before mlp")
    parser.add_argument('--adj_normlize', type=bool, default=False, help="the edge_weight normalization for gcn conv")
    parser.add_argument('--emb_normlize', type=bool, default=True, help="the l2 normalization after gnn layer")
    parser.add_argument('--latent_dim', type= List, default=[20,20,20], help="the hidden units for each gnn layer[20, 20, 20]      [128, 128, 128]")
    parser.add_argument('--mlp_hidden', type= List, default=[], help="the hidden units for mlp classifier")
    parser.add_argument('--gnn_dropout', type=float, default=0.0, help="the dropout after gnn layers")
    parser.add_argument('--dropout', type=float, default=0.6, help="the dropout after mlp layers")
    args, _ = parser.parse_known_args()
    return args


def load_gnnNets(gnnNets, ckpt_path):
    #ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    ckpt = torch.load(ckpt_path)
    new_state_dic = OrderedDict()
    for key, value in gnnNets.state_dict().items():
        if key == 'classify_layer.weight':
            old_key = 'model.module_15.weight'
        elif key == 'classify_layer.bias':
            old_key = 'model.module_15.bias'
        elif key == 'gnn_model.module_0.lin.weight':
            old_key = 'model.module_0.lin.weight'
        elif key == 'gnn_model.module_0.bias':
            old_key = 'model.module_0.bias'
        elif key == 'gnn_model.module_3.lin.weight':
            old_key = 'model.module_3.lin.weight'
        elif key == 'gnn_model.module_3.bias':
            old_key = 'model.module_3.bias'
        elif key == 'gnn_model.module_6.lin.weight':
            old_key = 'model.module_6.lin.weight'
        elif key == 'gnn_model.module_6.bias':
            old_key = 'model.module_6.bias'
        elif key == 'mlp_model.module_0.weight':
            old_key = 'model.module_11.weight'
        elif key == 'mlp_model.module_0.bias':
            old_key = 'model.module_11.bias'
        elif key == 'mlp_model.module_2.weight':
            old_key = 'model.module_13.weight'
        elif key == 'mlp_model.module_2.bias':
            old_key = 'model.module_13.bias'
        new_state_dic[key] = ckpt[old_key]
        '''if "gnn_layers" in old_key:
            new_state_dic[key] = ckpt['net'][old_key].T
        else:
            new_state_dic[key] = ckpt['net'][old_key]'''
    gnnNets.load_state_dict(new_state_dic)
    #gnnNets.load_state_dict(ckpt['net'])
    return gnnNets


