import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from datasets.gen_synthetic_dataset import *

def get_cycle_motif():
     #Cycle Motif
    cycle_num_nodes = 6
    feat_dim = 10
    row = torch.arange(cycle_num_nodes).view(-1, 1).repeat(1, 2).view(-1)
    col1 = torch.arange(-1, cycle_num_nodes - 1) % cycle_num_nodes
    col2 = torch.arange(1, cycle_num_nodes + 1) % cycle_num_nodes
    col = torch.stack([col1, col2], dim=1).sort(dim=-1)[0].view(-1)
    cycle_edge_indices=torch.stack([row, col], dim=0).numpy()
    cycle_edge_index = torch.cat((torch.tensor(cycle_edge_indices), torch.tensor([cycle_edge_indices[1], cycle_edge_indices[0]])), dim=1)
    cycle_structure = Data(
        num_nodes=cycle_num_nodes,
        x = torch.ones(cycle_num_nodes, feat_dim),
        edge_index=cycle_edge_index.contiguous(),
    )
    return cycle_structure

def get_grid_motif():
    grid_num_nodes = 9
    feat_dim = 10
    grid_edge_indices = [
            [0, 1],
            [0, 3],
            [1, 4],
            [3, 4],
            [1, 2],
            [2, 5],
            [4, 5],
            [3, 6],
            [6, 7],
            [4, 7],
            [5, 8],
            [7, 8],
            [1, 0],
            [3, 0],
            [4, 1],
            [4, 3],
            [2, 1],
            [5, 2],
            [5, 4],
            [6, 3],
            [7, 6],
            [7, 4],
            [8, 5],
            [8, 7],
        ]
    grid_edge_indices = np.array(grid_edge_indices).T
    grid_edge_index = torch.cat((torch.tensor(grid_edge_indices), torch.tensor([grid_edge_indices[1], grid_edge_indices[0]])), dim=1)
    grid_structure = Data(
            num_nodes=grid_num_nodes,
            x = torch.ones(grid_num_nodes, feat_dim),
            edge_index=grid_edge_index.contiguous(),
        )
    return grid_structure

def get_house_motif():
    house_num_nodes = 5
    feat_dim = 10
    house_edge_indices = [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 4],
            [1, 0],
            [2, 0],
            [2, 1],
            [3, 1],
            [4, 2],
            [4, 3],
        ]
    house_edge_indices = np.array(house_edge_indices).T
    house_edge_index = torch.cat((torch.tensor(house_edge_indices), torch.tensor([house_edge_indices[1], house_edge_indices[0]])), dim=1)
    house_structure = Data(
            num_nodes=house_num_nodes,
            x = torch.ones(house_num_nodes, feat_dim),
            edge_index=house_edge_index.contiguous(),
        )
    return house_structure


def get_emptyhouse_motif():
    house_num_nodes = 5
    feat_dim = 10
    house_edge_indices = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 4],
            [1, 0],
            [2, 0],
            [3, 1],
            [4, 2],
            [4, 3],
        ]
    house_edge_indices = np.array(house_edge_indices).T
    house_edge_index = torch.cat((torch.tensor(house_edge_indices), torch.tensor([house_edge_indices[1], house_edge_indices[0]])), dim=1)
    house_structure = Data(
            num_nodes=house_num_nodes,
            x = torch.ones(house_num_nodes, feat_dim),
            edge_index=house_edge_index.contiguous(),
        )
    return house_structure


def get_star_motif():
    star_num_nodes = 7
    feat_dim = 10
    star_edge_indices = []
    for k in range(1, star_num_nodes):
        star_edge_indices.append([0, k])
    star_edge_indices = np.array(star_edge_indices).T
    star_edge_index = torch.cat((torch.tensor(star_edge_indices), torch.tensor([star_edge_indices[1], star_edge_indices[0]])), dim=1)
    star_structure = Data(
            num_nodes=star_num_nodes,
            x = torch.ones(star_num_nodes, feat_dim),
            edge_index=star_edge_index.contiguous(),
        )
    return star_structure


def get_fan_motif():
    feat_dim = 10
    fan_motif, role_id = eval("fan")(start=0, nb_branches=6)
    fan_num_nodes = nx.number_of_nodes(fan_motif)
    #fan_edge_indices = torch.tensor(np.array(fan_motif.edges).T)
    fan_edge_indices = np.array(fan_motif.edges).T
    fan_edge_index = torch.cat((torch.tensor(fan_edge_indices), torch.tensor([fan_edge_indices[1], fan_edge_indices[0]])), dim=1)
    fan_structure = Data(
            num_nodes=fan_num_nodes,
            x = torch.ones(fan_num_nodes, feat_dim),
            edge_index=fan_edge_index.contiguous(),
        )
    return fan_structure


def get_diamond_motif():
    feat_dim = 10
    diamond_motif, role_id = eval("diamond")(start=0)
    diamond_num_nodes = nx.number_of_nodes(diamond_motif)
    #diamond_edge_index = torch.tensor(np.array(diamond_motif.edges).T)
    diamond_edge_indices = np.array(diamond_motif.edges).T
    diamond_edge_index = torch.cat((torch.tensor(diamond_edge_indices), torch.tensor([diamond_edge_indices[1], diamond_edge_indices[0]])), dim=1)
    diamond_structure = Data(
            num_nodes=diamond_num_nodes,
            x = torch.ones(diamond_num_nodes, feat_dim),
            edge_index=diamond_edge_index.contiguous(),
        )
    return diamond_structure