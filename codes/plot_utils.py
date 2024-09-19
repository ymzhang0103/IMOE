import numpy as np
import pickle as pkl
import networkx as nx
import sys
import torch
import matplotlib.pyplot as plt
from textwrap import wrap
import rdkit.Chem as Chem
from torch_geometric.datasets import MoleculeNet
from PIL import Image
import io
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


class PlotUtils():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    '''
    def plot(self, graph, nodelist, figname, **kwargs):
        """ plot function for different dataset """
        if self.dataset_name.lower() == 'BA_2motifs'.lower():
            self.plot_ba2motifs(graph, nodelist, figname=figname)
        elif self.dataset_name.lower() in ['bbbp', 'mutag']:
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, figname=figname)
        elif self.dataset_name.lower() == 'ba_shapes':
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bashapes(graph, nodelist, y, node_idx, figname=figname)
        elif self.dataset_name.lower() == 'ba_community':
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bacommunity(graph, nodelist, y, node_idx, figname=figname)
        elif self.dataset_name.lower() in ['grt_sst2_BERT_Identity'.lower()]:
            words = kwargs.get('words')
            self.plot_sentence(graph, nodelist, words=words, figname=figname)
        else:
            raise NotImplementedError
    '''
    def plot(self, graph, nodelist, figname, **kwargs):
        """ plot function for different dataset """
        if self.dataset_name.lower() in ['ba_2motifs', 'ba_lrp']:
            self.plot_ba2motifs(graph, nodelist, figname=figname)
        elif self.dataset_name.lower() in ['mutag'] + list(MoleculeNet.names.keys()):
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, figname=figname)
        elif self.dataset_name.lower() in ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle']:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bashapes(graph, nodelist, y, node_idx, figname=figname)
        elif self.dataset_name.lower() in ['graph-sst2', 'graph-sst5', 'graph-twitter']:
            words = kwargs.get('words')
            self.plot_sentence(graph, nodelist, words=words, figname=figname)
        else:
            raise NotImplementedError

    def plot_new(self, graph, nodelist, edgelist, figname, **kwargs):
        """ plot function for different dataset """
        if self.dataset_name.lower() in ['ba_shapes', 'ba_community']:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
            if self.dataset_name.lower() == 'ba_shapes':
                node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
            elif self.dataset_name.lower() == 'ba_community':
                node_color = ['#FFA500', '#4970C6', '#FE0000', 'green', '#B08B31', '#00F5FF', '#EE82EE', 'blue']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
            self.plot_subgraph_new(graph, node_idx, nodelist=None, edgelist=edgelist, colors=colors, figname=figname, subgraph_edge_color='black')
        elif self.dataset_name.lower() in ['graph_sst2', 'graph_sst5', 'graph_twitter']:
            words = kwargs.get('words')
            self.plot_sentence_new(graph, nodelist=nodelist, edgelist=edgelist, words=words, figname=figname)
        elif 'mutagenicity' in self.dataset_name.lower():
            x = kwargs.get('x')
            self.plot_mutagenicity_new(graph, nodelist=nodelist, edgelist=edgelist, x=x, figname=figname)
        elif self.dataset_name.lower() == 'nci1':
            x = kwargs.get('x')
            self.plot_NCI1(graph, nodelist=nodelist, edgelist=edgelist, x=x, figname=figname)
        elif self.dataset_name.lower() in ['mutag'] + list(MoleculeNet.names.keys()):
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, figname=figname)
        elif self.dataset_name.lower() == "ba_4motifs":
            graphid = kwargs.get('graphid')
            if graphid < 500:   #house
                basis_width = 20
            elif graphid >= 500 and graphid<1000:   #star
                basis_width = 18
            elif graphid>=1000 and graphid<1500:   #cycle
                basis_width = 19
            elif graphid>=1500 and graphid<2000:   #grid
                basis_width = 16
            x = kwargs.get('x')
            colors = ["#14F5FF" if v < basis_width else "#FF93EB" for v in range(x.shape[0]) ]    # #0093FF, #FF1493
            self.plot_syn_graph(graph, nodelist=None, edgelist=edgelist, colors=colors, figname=figname, subgraph_edge_color='black')
        else:
            raise NotImplementedError


    def plot_mutagenicity(self, graph, nodelist, edgelist, x, figname):
        node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                        8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
        node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
        node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
        #node_color = ['orange','red','lime','green','blue','orchid','darksalmon','darkslategray','gold','bisque','tan','lightseagreen','indigo','navy']
        #colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        node_color = ['cyan','red','bisque','yellowgreen','royalblue','yellow','orchid','orange','gold','pink','tan','lightseagreen','lime','navy']
        colors = [node_color[v] for k, v in node_idxs.items()]

        if edgelist is None and nodelist is not None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=800)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='gray', arrows=False)

        if nodelist is not None:
            pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color='black',
                               arrows=False)

        if node_labels is not None:
            nx.draw_networkx_labels(graph, pos, node_labels, font_size=18)

        plt.axis('off')
        if figname is not None:
            plt.savefig(figname)
        # plt.show()
        plt.close('all')
    

    def plot_mutagenicity_new(self, graph, nodelist, edgelist, x, figname):
        node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                        8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
        node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
        node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
        #node_color = ['orange','red','lime','green','blue','orchid','darksalmon','darkslategray','gold','bisque','tan','lightseagreen','indigo','navy']
        #colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        node_color = ['cyan','red','bisque','yellowgreen','royalblue','yellow','orchid','orange','gold','pink','tan','lightseagreen','lime','navy']
        
        if edgelist is None  and nodelist is not None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)

        '''colors = [node_color[v] for k, v in node_idxs.items()]
        nx.draw_networkx_nodes(graph, pos,
                               nodelist=nodelist,
                               node_color=colors,
                               node_size=800)'''
        
        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color='silver',
                               node_size=800)
        
        if node_labels is not None:
            nx.draw_networkx_labels(graph, pos, node_labels, font_size=18, font_color="gray")
        
        nx.draw_networkx_edges(graph, pos, width=2, edge_color='gray', arrows=False)

        if nodelist is not None:
            imp_node_idxs = {k: v for k, v in node_idxs.items() if k in nodelist}
            imp_node_labels = {k: node_dict[v] for k, v in imp_node_idxs.items()}
            imp_colors = [node_color[v] for k, v in imp_node_idxs.items()]
            nx.draw_networkx_nodes(graph, pos,
                                nodelist=nodelist,
                                node_color=imp_colors,
                                node_size=800)
            
            pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color='black',
                               arrows=False)

            nx.draw_networkx_labels(graph, pos, imp_node_labels, font_size=18, font_color="k")

        plt.axis('off')
        if figname is not None:
            plt.savefig(figname)
        # plt.show()
        plt.close('all')


    def plot_atom(self, graph, figname):
        node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                        8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
        node_color = ['cyan','red','bisque','yellowgreen','royalblue','gold','orchid','orange','yellow','pink','tan','lightseagreen','lime','navy']
        x =  list(range(13))
        node_idxs = {k: int(v) for k, v in enumerate(x)}
        node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
        colors = [node_color[v] for k, v in node_idxs.items()]
        pos = nx.kamada_kawai_layout(graph)

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=x,
                               node_color=colors,
                               node_size=600)
        if node_labels is not None:
            nx.draw_networkx_labels(graph, pos, node_labels, font_size=15)

        plt.axis('off')
        if figname is not None:
            plt.savefig(figname)
        # plt.show()
        plt.close('all')
    


    def plot_NCI1(self, graph, nodelist, edgelist, x, figname):
        #node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
        #                8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
        node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
        #node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
        node_labels = {k: Chem.rdchem.PeriodicTable.GetElementSymbol(Chem.rdchem.GetPeriodicTable(), int(v))
                           for k, v in node_idxs.items()}
        node_color = ['white', 'green', 'maroon',  '#4970C6', 'brown', 'indigo', 'orange', 'blue', 'red', 'orchid', '#F0EA00', 'tan','lime','blue','#E49D1C','darksalmon','darkslategray','gold','bisque','lightseagreen','navy']
        #node_color = ['orange','red','lime','green','blue','orchid','darksalmon','darkslategray','gold','bisque','tan','lightseagreen','indigo','navy']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='gray', arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color='black',
                               arrows=False)

        if node_labels is not None:
            nx.draw_networkx_labels(graph, pos, node_labels, font_size=14)

        plt.axis('off')
        if figname is not None:
            plt.savefig(figname)
        # plt.show()
        plt.close('all')

    def plot_syn_graph(self, graph, nodelist=None, edgelist=None, colors='#FFA500', labels=None, edge_color='gray',
                                subgraph_edge_color='black', title_sentence=None, figname=None):
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        if nodelist is None:
            nodelist=[]
            for (n_frm, n_to) in edgelist:
                nodelist.append(n_frm)
                nodelist.append(n_to)
            nodelist = list(set(nodelist))

        pos = nx.kamada_kawai_layout(graph) # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)
        
        nx.draw_networkx_edges(graph, pos, width=1, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=1,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        plt.close('all')


    def plot_subgraph_new(self, graph, node_idx, nodelist=None, edgelist=None, colors='#FFA500', labels=None, edge_color='gray',
                                subgraph_edge_color='black', title_sentence=None, figname=None):
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        if nodelist is None:
            nodelist=[]
            for (n_frm, n_to) in edgelist:
                nodelist.append(n_frm)
                nodelist.append(n_to)
            nodelist = list(set(nodelist))

        pos = nx.kamada_kawai_layout(graph) # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        node_idx = int(node_idx)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors
        nx.draw_networkx_nodes(graph, pos=pos,
                            nodelist=[node_idx],
                            node_color=node_idx_color,
                            node_size=600)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        plt.close('all')

    def plot_sentence_new(self, graph, nodelist, words, edgelist=None, figname=None):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        nx.draw_networkx_nodes(graph, pos, node_color="#C4F9FF", nodelist=list(graph.nodes()), node_size=300) #D4F1EF
        nx.draw_networkx_labels(graph, pos, words_dict, font_size=14)
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist,
                                   node_color='#96EEC6',  #6B8BF5, #C0C2DE
                                   node_shape='o',
                                   node_size=500)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]

        nx.draw_networkx_edges(graph, pos, width=3, edge_color='grey')
        nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=3, edge_color='black')

        plt.axis('off')
        plt.title('\n'.join(wrap(' '.join(words), width=50)))
        if figname is not None:
            plt.savefig(figname)
        plt.close('all')


    def plot_subgraph(self, graph, nodelist, colors='#FFA500', labels=None, edge_color='gray',
                    edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        # plt.show()
        plt.close('all')

    def plot_subgraph_with_nodes(self, graph, nodelist, node_idx, colors='#FFA500', labels=None, edge_color='gray',
                                edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):
        node_idx = int(node_idx)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]

        pos = nx.kamada_kawai_layout(graph) # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors

        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[node_idx],
                               node_color=node_idx_color,
                               node_size=600)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        plt.close('all')

    def plot_ba2motifs(self, graph, nodelist, edgelist=None, figname=None):
        return self.plot_subgraph(graph, nodelist, edgelist=edgelist, figname=figname)

    def plot_molecule(self, graph, nodelist, x, edgelist=None, figname=None):
        # collect the text information and node color
        if self.dataset_name.lower() == 'mutag':
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        elif self.dataset_name.lower() == 'bbbp':
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                           for k, v in element_idxs.items()}
            node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
            colors = [node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, nodelist, colors=colors, labels=node_labels,
                           edgelist=edgelist, edge_color='gray',
                           subgraph_edge_color='black',
                           title_sentence=None, figname=figname)

    def plot_sentence(self, graph, nodelist, words, edgelist=None, figname=None):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist,
                                   node_color='yellow',
                                   node_shape='o',
                                   node_size=500)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)
        nx.draw_networkx_labels(graph, pos, words_dict)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color='grey')
        nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=3, edge_color='black')

        plt.axis('off')
        plt.title('\n'.join(wrap(' '.join(words), width=50)))
        if figname is not None:
            plt.savefig(figname)
        plt.close('all')

    def plot_bashapes(self, graph, nodelist, y, node_idx, edgelist=None, figname=None):
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph, nodelist, node_idx, colors, edgelist=edgelist, figname=figname,
                           subgraph_edge_color='black')

    def plot_bacommunity(self, graph, nodelist, y, node_idx, edgelist=None, figname=None):
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green', '#B08B31', '#00F5FF', '#EE82EE', 'blue']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph, nodelist, node_idx, colors, edgelist=edgelist, figname=figname,
                           subgraph_edge_color='black')


    def e_map_mutag(self, bond_type, reverse=False):
        if not reverse:
            if bond_type == Chem.BondType.SINGLE:
                return 0
            elif bond_type == Chem.BondType.DOUBLE:
                return 1
            elif bond_type == Chem.BondType.AROMATIC:
                return 2
            elif bond_type == Chem.BondType.TRIPLE:
                return 3
            else:
                raise Exception("No bond type found")

        if bond_type == 0:
            return Chem.BondType.SINGLE
        elif bond_type == 1:
            return Chem.BondType.DOUBLE
        elif bond_type == 2:
            return Chem.BondType.AROMATIC
        elif bond_type == 3:
            return Chem.BondType.TRIPLE
        else:
            raise Exception("No bond type found")


    def graph_to_mol(self, X, edge_index, edge_attr):
        mol = Chem.RWMol()
        if "Mutagenicity" in self.dataset_name:
            node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                            8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
            X = [ Chem.Atom(node_dict[x]) for x in X ]
        else:
            X = [ Chem.Atom(Chem.rdchem.PeriodicTable.GetElementSymbol(Chem.rdchem.GetPeriodicTable(), int(x))) for x in X ]

        E = edge_index
        for x in X:
            mol.AddAtom(x)
        for (u, v), attr in zip(E, edge_attr):
            if type(attr) is int or type(attr) is float:          #2024.1.10
                #attr = self.e_map_mutag(0, reverse=True)    
                attr = self.e_map_mutag(attr, reverse=True)
            else:
                attr = self.e_map_mutag(attr.index(1), reverse=True)

            if mol.GetBondBetweenAtoms(u, v):
                continue
            mol.AddBond(u, v, attr)
        return mol

    def visualize(self, graph, nodelist, edgelist, figname):
        num_nodes = graph.x.shape[0]
        num_edges = graph.edge_index.shape[1]
        vis_dict = {
            'mutag': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
            'Tox21Net': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
            'BA3MotifNet': {'node_size': 300, 'linewidths': 1, 'font_size': 10, 'width': 3},
            'TR3MotifNet': {'node_size': 300, 'linewidths': 1, 'font_size': 10, 'width': 5},
            'GraphSST2Net': {'node_size': 400, 'linewidths': 1, 'font_size': 12, 'width': 3},
            'MNISTNet': {'node_size': 100, 'linewidths': 1, 'font_size': 10, 'width': 2},
            'defult': {'node_size': 200, 'linewidths': 1, 'font_size': 10, 'width': 2}
        }
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(list(graph.edge_index.cpu().numpy().T))
        
        if  "Mutagenicity" in self.dataset_name or  self.dataset_name == 'NCI1' or self.dataset_name == 'mutag':
            x = torch.topk(graph.x, 1)[1].squeeze().detach().cpu().tolist()
            edge_index = graph.edge_index.T.detach().cpu().tolist()
            #if graph.edge_label is None:      #2024.1.9
            if hasattr(graph, "edge_label"):
                edge_label = graph.edge_label.detach().cpu().tolist()
            else:
                edge_label =  [0]* graph.edge_index.shape[1]
            mol = self.graph_to_mol(x, edge_index, edge_label)
            print("smile", Chem.MolToSmiles(mol))
            d = rdMolDraw2D.MolDraw2DCairo(1024, 1024)
            def add_atom_index(mol):
                atoms = mol.GetNumAtoms()
                for i in range( atoms ):
                    mol.GetAtomWithIdx(i).SetProp(
                        'molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx()))
                return mol

            hit_bonds=[]
            if edgelist is not None:
                for (u, v) in edgelist:
                    hit_bonds.append(mol.GetBondBetweenAtoms(int(u), int(v)).GetIdx())
                rdMolDraw2D.PrepareAndDrawMolecule(
                    d, mol, highlightAtoms=nodelist, highlightBonds=hit_bonds,
                    #highlightAtomColors={i:(1, 0, 0) for i in nodelist},
                    #highlightBondColors={i:(1, 0, 0) for i in hit_bonds}
                    )
            else:
                rdMolDraw2D.PrepareAndDrawMolecule(
                    d, mol, 
                    #highlightAtoms=nodelist, highlightBonds=hit_bonds,
                    #highlightAtomColors={i:(1, 0, 0) for i in nodelist},
                    #highlightBondColors={i:(1, 0, 0) for i in hit_bonds}
                    )
            d.FinishDrawing()
            bindata = d.GetDrawingText()
            iobuf = io.BytesIO(bindata)
            im = Image.open(iobuf)
            #im.show()
            im.save(figname)
            

    def visualize_new(self, graph, nodelist, edgelist, figname):
        num_nodes = graph.x.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(list(graph.edge_index.cpu().numpy().T))
        
        if  "Mutagenicity" in self.dataset_name or  self.dataset_name == 'NCI1' or self.dataset_name == 'mutag':
            x = torch.topk(graph.x, 1)[1].squeeze().detach().cpu().tolist()
            edge_index = graph.edge_index.T.detach().cpu().tolist()
            if hasattr(graph, "edge_label"):
                edge_label = graph.edge_label.detach().cpu().tolist()
            else:
                edge_label =  [0]* graph.edge_index.shape[1]
            mol = self.graph_to_mol(x, edge_index, edge_label)
            print("smile", Chem.MolToSmiles(mol))
            d = rdMolDraw2D.MolDraw2DCairo(1024, 1024)
            opts = rdMolDraw2D.MolDrawOptions()
            # set edge size
            opts.bondLineWidth = 10
            # set atom letter size
            opts.minFontSize = 60
            d.SetDrawOptions(opts)

            hit_bonds=[]
            if edgelist is not None:
                for (u, v) in edgelist:
                    hit_bonds.append(mol.GetBondBetweenAtoms(int(u), int(v)).GetIdx())
                rdMolDraw2D.PrepareAndDrawMolecule(
                    d, mol, highlightAtoms=nodelist, highlightBonds=hit_bonds,
                    #highlightAtomColors={i:(1, 0, 0) for i in nodelist},
                    #highlightBondColors={i:(1, 0, 0) for i in hit_bonds}
                    )
            else:
                rdMolDraw2D.PrepareAndDrawMolecule(
                    d, mol, 
                    #highlightAtoms=nodelist, highlightBonds=hit_bonds,
                    #highlightAtomColors={i:(1, 0, 0) for i in nodelist},
                    #highlightBondColors={i:(1, 0, 0) for i in hit_bonds}
                    )
            d.FinishDrawing()
            bindata = d.GetDrawingText()
            iobuf = io.BytesIO(bindata)
            im = Image.open(iobuf)
            #im.show()
            im.save(figname)

