import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch import Tensor
from math import sqrt
from torch_geometric.nn import MessagePassing

from typing import List, Union



class MaskoutMetric:
    def __init__(
        self,
        model, 
        prog_args
    ):
        self.model = model
        self.model.eval()
        self.prog_args= prog_args

    def GnnNets_NC2value_func_new(self, gnnNets_NC, node_idx):
        def value_func(data):
            with torch.no_grad():
                logits, probs, _, _ = gnnNets_NC(data=data)
                # select the corresponding node prob through the node idx on all the sampling graphs
                batch_size = data.batch.max() + 1
                probs = probs.reshape(batch_size, -1, probs.shape[-1])
                scores = probs[:, node_idx]
                return scores
        return value_func

    def GnnNets_GC2value_func_new(self, gnnNets):
        def value_func(batch):
            with torch.no_grad():
                logits, probs,_,_ = gnnNets(data=batch)
                #_,logits, probs = gnnNets(x=batch.x, edge_index=batch.edge_index)
                #probs = F.softmax(logits, dim=-1)
                score = probs.squeeze()
            return score
        return value_func

    def gnn_prob(self, coalition: list, data: Data, value_func: str, subgraph_building_method='zero_filling') -> torch.Tensor:
        """ the value of subgraph with selected nodes """
        num_nodes = data.num_nodes
        subgraph_build_func = self.get_graph_build_func(subgraph_building_method)
        mask = torch.zeros(num_nodes).type(torch.float32).to(data.x.device)
        mask[coalition] = 1.0
        ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
        mask_data = Data(x=ret_x, edge_index=ret_edge_index)
        mask_data = Batch.from_data_list([mask_data])
        score = value_func(mask_data)
        # get the score of predicted class for graph or specific node idx
        return score

    def get_graph_build_func(self, build_method):
        if build_method.lower() == 'zero_filling':
            return self.graph_build_zero_filling
        elif build_method.lower() == 'split':
            return self.graph_build_split
        else:
            raise NotImplementedError

    def graph_build_zero_filling(self, X, edge_index, node_mask: np.array):
        """ subgraph building through masking the unselected nodes with zero features """
        ret_X = X * node_mask.unsqueeze(1)
        return ret_X, edge_index

    def graph_build_split(X, edge_index, node_mask: np.array):
        """ subgraph building through spliting the selected nodes from the original graph """
        row, col = edge_index
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        ret_edge_index = edge_index[:, edge_mask]
        return X, ret_edge_index

    def calculate_selected_nodes(self, edge_index, edge_mask, top_k):
        threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])
        hard_mask = (edge_mask > threshold).cpu()
        edge_idx_list = torch.where(hard_mask == 1)[0]
        selected_nodes = []
        #edge_index = data.edge_index.cpu().numpy()
        for edge_idx in edge_idx_list:
            selected_nodes += [edge_index[0][edge_idx].item(), edge_index[1][edge_idx].item()]
        selected_nodes = list(set(selected_nodes))
        return selected_nodes

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor = None):
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = 0.0
        #self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        #std = torch.nn.init.calculate_gain('sigmoid') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = True
                module._edge_mask = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None, and set the explain flag to :obj:`False` """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = False
                module._edge_mask = None
        #self.node_feat_masks = None
        self.edge_mask = None

    def evaluate_adj_new(self, node_idx_new, features, adj):
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.prog_args.device)
            sub_edge_index = torch.nonzero(adj).t()
            new_edge_mask = adj[sub_edge_index[0], sub_edge_index[1]]
            self.__clear_masks__()
            self.__set_masks__(x, sub_edge_index, new_edge_mask) 
            data = Data(x=features, edge_index=sub_edge_index)
            logit, pred, _ = self.model(data.to(self.prog_args.device))
            _, pred_label = torch.max(pred, 1)
            self.__clear_masks__()
        return pred_label[node_idx_new], pred[node_idx_new]

    def evaluate_adj_new_GC(self, features, adj):
        with torch.no_grad():
            sub_edge_index = torch.nonzero(adj).t()
            new_edge_mask = adj[sub_edge_index[0], sub_edge_index[1]]
            self.__clear_masks__()
            self.__set_masks__(features, sub_edge_index, new_edge_mask) 
            data = Data(x=features, edge_index=sub_edge_index, batch = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device))
            _, pred, _ = self.model(data)
            _, pred_label = torch.max(pred, 1)
            self.__clear_masks__()
        return pred_label, pred[0]

    def predict(self, features, edge_index, edge_mask):
        self.__clear_masks__()
        self.__set_masks__(features, edge_index, edge_mask)    
        data = Data(x=features, edge_index=edge_index, batch = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device))
        self.model.eval()
        mask_logits, mask_preds, embed, node_embs = self.model(data)
        mask_pred = mask_preds.squeeze()
        self.__clear_masks__()
        return mask_pred

    '''def predict(self, features, edge_index, edge_mask):
        #self.__clear_masks__()
        #self.__set_masks__(features, edge_index, edge_mask)    
        data = Data(x=features, edge_index=edge_index, edge_weight = edge_mask, batch = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device))
        self.model.eval()
        embed, _, mask_preds = self.model(x=features, edge_index=edge_index, edge_weight = edge_mask)
        mask_pred = mask_preds.squeeze()
        #self.__clear_masks__()
        return mask_pred'''


    def metric_del_edges_GC(self, topk_arr, sub_feature, edge_mask, sub_edge_index, origin_pred, masked_pred, label):
        origin_label = torch.argmax(origin_pred)
        #edge_mask = mask[sub_edge_index[0], sub_edge_index[1]]

        x = sub_feature.to(self.prog_args.device)
        related_preds_dict = dict()
        attack_status = 0
        for top_k in topk_arr:
            select_k = round(top_k/100 * len(sub_edge_index[0]))

            selected_impedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[:select_k]#按比例选择top_k%的重要边
            other_notimpedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[select_k:]        #按比例选择top_k%的重要边
            sparsity_edges = 1- len(selected_impedges_idx) / sub_edge_index.shape[1]

            maskimp_edge_mask = torch.ones(len(edge_mask)).to(self.prog_args.device) 
            maskimp_edge_mask[selected_impedges_idx] = 1-edge_mask[selected_impedges_idx]#重要的边，权重置为1-mask
            maskimp_pred = self.predict(x, sub_edge_index, maskimp_edge_mask)

            masknotimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(self.prog_args.device) 
            masknotimp_edge_mask[other_notimpedges_idx] = edge_mask[other_notimpedges_idx]      #除了重要的top_k%之外的其他边置为mask
            masknotimp_pred = self.predict(x, sub_edge_index, masknotimp_edge_mask)

            delimp_edge_mask = torch.ones(len(edge_mask)).to(self.prog_args.device) 
            delimp_edge_mask[selected_impedges_idx] = 0.0    #remove important edges
            delimp_pred = self.predict(x, sub_edge_index, delimp_edge_mask)

            retainimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(self.prog_args.device) 
            retainimp_edge_mask[other_notimpedges_idx] = 0.0   #remove not important edges
            retainimp_pred = self.predict(x, sub_edge_index, retainimp_edge_mask)

            #delete nodes
            selected_nodes = self.calculate_selected_nodes(sub_edge_index, edge_mask, select_k)
            maskout_nodes_list = [node for node in range(sub_feature.shape[0]) if node not in selected_nodes]
            value_func = self.GnnNets_GC2value_func_new(self.model)
            data = Data(x=sub_feature, edge_index=sub_edge_index, edge_attr=edge_mask, batch = torch.zeros(sub_feature.shape[0], dtype=torch.int64, device=sub_feature.device))
            maskimp_pred_nodes = self.gnn_prob(maskout_nodes_list, data, value_func, subgraph_building_method='zero_filling')
            retainimp_pred_nodes = self.gnn_prob(selected_nodes, data, value_func, subgraph_building_method='zero_filling')
            sparsity_nodes = 1 - len(selected_nodes) / sub_feature.shape[0]

            related_preds = [{
                'label': label,
                'origin_label': origin_label,
                'origin': origin_pred,
                'origin_l': origin_pred[label],
                'origin_ol': origin_pred[origin_label],
                'masked': masked_pred,
                'masked_l': masked_pred[label],
                'masked_ol': masked_pred[origin_label],
                'maskimp': maskimp_pred,
                'maskimp_l': maskimp_pred[label],
                'maskimp_ol': maskimp_pred[origin_label],
                'masknotimp': masknotimp_pred,
                'masknotimp_l': masknotimp_pred[label],
                'masknotimp_ol': masknotimp_pred[origin_label],
                'delimp':delimp_pred,
                'delimp_l':delimp_pred[label],
                'delimp_ol':delimp_pred[origin_label],
                'retainimp':retainimp_pred,
                'retainimp_l':retainimp_pred[label],
                'retainimp_ol':retainimp_pred[origin_label],
                'sparsity_edges': sparsity_edges,
                'maskimp_nodes': maskimp_pred_nodes,
                'maskimp_nodes_l':maskimp_pred_nodes[label],
                'maskimp_nodes_ol':maskimp_pred_nodes[origin_label],
                'retainimp_nodes': retainimp_pred_nodes,
                'retainimp_nodes_l':retainimp_pred_nodes[label],
                'retainimp_nodes_ol':retainimp_pred_nodes[origin_label],
                'sparsity_nodes': sparsity_nodes
            }]
            related_preds_dict[top_k] = related_preds

        pred_mask = [edge_mask.cpu().detach().numpy()]
        return pred_mask, related_preds_dict
    
    



def fidelity(ori_probs: torch.Tensor, maskout_probs: torch.Tensor) -> float:
    drop_probability = abs(ori_probs - maskout_probs)
    return drop_probability.mean().item()

def fidelity_complete(ori_probs, maskout_probs):
    drop_prob_complete = [ori_probs[i] - maskout_probs[i] for i in range(len(ori_probs))]
    result = np.mean([sum(abs(i)).item() for i in drop_prob_complete])
    return result

class XCollector(object):
    def __init__(self, sparsity=None):
        self.__related_preds, self.__targets = {'zero': [], 'origin': [], 'masked': [], 'maskimp': [], 'masknotimp':[], 'delimp':[], 'retainimp':[], 'sparsity_edges': [], 'maskimp_nodes':[], 'retainimp_nodes':[], 'sparsity_nodes':[], \
       'origin_l': [], 'masked_l': [], 'maskimp_l': [], 'masknotimp_l':[], 'delimp_l':[], 'retainimp_l':[], 'maskimp_nodes_l':[], 'retainimp_nodes_l':[], 'origin_ol': [], 'masked_ol': [], 'maskimp_ol': [], 'masknotimp_ol':[], 'delimp_ol':[], 'retainimp_ol':[], 'maskimp_nodes_ol':[], 'retainimp_nodes_ol':[], 'label': [], 'origin_label': [], 'attack_status':[], 'fake_pred':[], 'fake_pred_l':[], 'fake_pred_ol':[]}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__sparsity_edges = sparsity
        self.__simula, self.__simula_origin, self.__simula_complete = None, None, None
        self.__fidelity, self.__fidelity_origin, self.__fidelity_complete, self.__fidelityminus, self.__fidelityminus_origin, self.__fidelityminus_complete= None, None, None, None, None, None
        self.__del_fidelity, self.__del_fidelity_origin, self.__del_fidelity_complete, self.__del_fidelityminus, self.__del_fidelityminus_origin, self.__del_fidelityminus_complete = None, None, None, None, None, None
        self.__fake_fidelityminus, self.__fake_fidelityminus_origin, self.__fake_fidelityminus_complete = None, None, None
        self.__fidelity_nodes, self.__fidelity_origin_nodes, self.__fidelity_complete_nodes, self.__fidelityminus_nodes, self.__fidelityminus_origin_nodes, self.__fidelityminus_complete_nodes = None, None, None, None, None, None
        self.__sparsity_nodes = None
        self.__attack_acc = None
    @property
    def targets(self) -> list:
        return self.__targets

    def new(self):
        self.__related_preds = {'zero': [], 'origin': [], 'masked': [], 'maskimp': [], 'masknotimp':[], 'delimp':[], 'retainimp':[], 'sparsity_edges': [], 'maskimp_nodes':[], 'retainimp_nodes':[], 'sparsity_nodes':[], \
       'origin_l': [], 'masked_l': [], 'maskimp_l': [], 'masknotimp_l':[], 'delimp_l':[], 'retainimp_l':[], 'maskimp_nodes_l':[], 'retainimp_nodes_l':[], 'origin_ol': [], 'masked_ol': [], 'maskimp_ol': [], 'masknotimp_ol':[], 'delimp_ol':[], 'retainimp_ol':[], 'maskimp_nodes_ol':[], 'retainimp_nodes_ol':[], 'label': [], 'origin_label': [], 'attack_status':[], 'fake_pred':[], 'fake_pred_l':[], 'fake_pred_ol':[]}
        self.__targets = []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__simula, self.__simula_origin, self.__simula_complete = None, None, None
        self.__fidelity, self.__fidelity_origin, self.__fidelity_complete, self.__fidelityminus, self.__fidelityminus_origin, self.__fidelityminus_complete= None, None, None, None, None, None
        self.__del_fidelity, self.__del_fidelity_origin, self.__del_fidelity_complete, self.__del_fidelityminus, self.__del_fidelityminus_origin, self.__del_fidelityminus_complete = None, None, None, None, None, None
        self.__fake_fidelityminus, self.__fake_fidelityminus_origin, self.__fake_fidelityminus_complete = None, None, None
        self.__fidelity_nodes, self.__fidelity_origin_nodes, self.__fidelity_complete_nodes, self.__fidelityminus_nodes, self.__fidelityminus_origin_nodes, self.__fidelityminus_complete_nodes = None, None, None, None, None, None
        self.__attack_acc = None

    def collect_data(self,
                     masks: List[Tensor],
                     related_preds: List[dict],
                     label: int = 0) -> None:
        r"""
        The function is used to collect related data. After collection, we can call fidelity, fidelity_inv, sparsity
        to calculate their values.

        Args:
            masks (list): It is a list of edge-level explanation for each class.
            related_preds (list): It is a list of dictionary for each class where each dictionary
            includes 4 type predicted probabilities and sparsity.
            label (int): The ground truth label. (default: 0)
        """
        if not np.isnan(label):
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)
            for key in self.__related_preds.keys():
                if key not in related_preds[0].keys():
                    self.__related_preds[key].append(None)
            self.__targets.append(label)
            self.masks.append(masks)

    @property
    def attack_acc(self):
        if self.__attack_acc:
            return self.__attack_acc
        elif None in self.__related_preds['attack_status']:
            return None
        else:
            #use prop of correct label
            attack_status = torch.tensor(self.__related_preds['attack_status'])
            self.__attack_acc = attack_status.sum().item()/attack_status.shape[0]
            return self.__attack_acc

    @property
    def simula(self):
        if self.__simula:
            return self.__simula
        elif None in self.__related_preds['masked_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            masked_preds, origin_preds = torch.tensor(self.__related_preds['masked_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__simula = fidelity(origin_preds, masked_preds)
            return self.__simula

    @property
    def simula_origin(self):
        if self.__simula_origin:
            return self.__simula_origin
        elif None in self.__related_preds['masked_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            masked_preds, origin_preds = torch.tensor(self.__related_preds['masked_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__simula_origin = fidelity(origin_preds, masked_preds)
            return self.__simula_origin

    @property
    def simula_complete(self):
        if self.__simula_complete:
            return self.__simula_complete
        elif None in self.__related_preds['masked'] or None in self.__related_preds['origin']:
            return None
        else:
            masked_preds, origin_preds= self.__related_preds['masked'], self.__related_preds['origin']
            self.__simula_complete = fidelity_complete(origin_preds, masked_preds)
            return self.__simula_complete

    @property
    def fidelity(self):
        if self.__fidelity:
            return self.__fidelity
        elif None in self.__related_preds['maskimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelity = fidelity(origin_preds, maskout_preds)
            return self.__fidelity

    @property
    def fidelity_origin(self):
        if self.__fidelity_origin:
            return self.__fidelity_origin
        elif None in self.__related_preds['maskimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelity_origin = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_origin

    @property
    def fidelity_complete(self):
        if self.__fidelity_complete:
            return self.__fidelity_complete
        elif None in self.__related_preds['maskimp'] or None in self.__related_preds['origin']:
            return None
        else:
            maskout_preds, origin_preds = self.__related_preds['maskimp'], self.__related_preds['origin']
            self.__fidelity_complete = fidelity_complete(origin_preds, maskout_preds)
            return self.__fidelity_complete

    @property
    def fidelityminus(self):
        if self.__fidelityminus:
            return self.__fidelityminus
        elif None in self.__related_preds['masknotimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            masknotimp_preds, origin_preds = torch.tensor(self.__related_preds['masknotimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelityminus = fidelity(origin_preds, masknotimp_preds)
            return self.__fidelityminus

    @property
    def fidelityminus_origin(self):
        if self.__fidelityminus_origin:
            return self.__fidelityminus_origin
        elif None in self.__related_preds['masknotimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            masknotimp_preds, origin_preds = torch.tensor(self.__related_preds['masknotimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelityminus_origin = fidelity(origin_preds, masknotimp_preds)
            return self.__fidelityminus_origin

    @property
    def fidelityminus_complete(self):
        if self.__fidelityminus_complete:
            return self.__fidelityminus_complete
        elif None in self.__related_preds['masknotimp'] or None in self.__related_preds['origin']:
            return None
        else:
            masknotimp_preds, origin_preds = self.__related_preds['masknotimp'], self.__related_preds['origin']
            self.__fidelityminus_complete = fidelity_complete(origin_preds,masknotimp_preds)
            return self.__fidelityminus_complete
    
    @property
    def del_fidelity(self):
        if self.__del_fidelity:
            return self.__del_fidelity
        elif None in self.__related_preds['delimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            delimp_preds, origin_preds = torch.tensor(self.__related_preds['delimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__del_fidelity = fidelity(origin_preds, delimp_preds)
            return self.__del_fidelity

    @property
    def del_fidelity_origin(self):
        if self.__del_fidelity_origin:
            return self.__del_fidelity_origin
        elif None in self.__related_preds['delimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            delimp_preds, origin_preds = torch.tensor(self.__related_preds['delimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__del_fidelity_origin = fidelity(origin_preds, delimp_preds)
            return self.__del_fidelity_origin

    @property
    def del_fidelity_complete(self):
        if self.__del_fidelity_complete:
            return self.__del_fidelity_complete
        elif None in self.__related_preds['delimp'] or None in self.__related_preds['origin']:
            return None
        else:
            delimp_preds, origin_preds = self.__related_preds['delimp'], self.__related_preds['origin']
            self.__del_fidelity_complete = fidelity_complete(origin_preds, delimp_preds)
            return self.__del_fidelity_complete

    @property
    def del_fidelityminus(self):
        if self.__del_fidelityminus:
            return self.__del_fidelityminus
        elif None in self.__related_preds['retainimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__del_fidelityminus = fidelity(origin_preds, retainimp_preds)
            return self.__del_fidelityminus

    @property
    def del_fidelityminus_origin(self):
        if self.__del_fidelityminus_origin:
            return self.__del_fidelityminus_origin
        elif None in self.__related_preds['retainimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__del_fidelityminus_origin = fidelity(origin_preds, retainimp_preds)
            return self.__del_fidelityminus_origin

    @property
    def del_fidelityminus_complete(self):
        if self.__del_fidelityminus_complete:
            return self.__del_fidelityminus_complete
        elif None in self.__related_preds['retainimp'] or None in self.__related_preds['origin']:
            return None
        else:
            retainimp_preds, origin_preds = self.__related_preds['retainimp'], self.__related_preds['origin']
            self.__del_fidelityminus_complete = fidelity_complete(origin_preds, retainimp_preds)
            return self.__del_fidelityminus_complete

    @property
    def fake_fidelityminus(self):
        if self.__fake_fidelityminus:
            return self.__fake_fidelityminus
        elif None in self.__related_preds['fake_pred_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            fake_preds, origin_preds = torch.tensor(self.__related_preds['fake_pred_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fake_fidelityminus = fidelity(origin_preds, fake_preds)
            return self.__fake_fidelityminus

    @property
    def fake_fidelityminus_origin(self):
        if self.__fake_fidelityminus_origin:
            return self.__fake_fidelityminus_origin
        elif None in self.__related_preds['fake_pred_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            fake_preds, origin_preds = torch.tensor(self.__related_preds['fake_pred_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fake_fidelityminus_origin = fidelity(origin_preds, fake_preds)
            return self.__fake_fidelityminus_origin

    @property
    def fake_fidelityminus_complete(self):
        if self.__fake_fidelityminus_complete:
            return self.__fake_fidelityminus_complete
        elif None in self.__related_preds['fake_pred'] or None in self.__related_preds['origin']:
            return None
        else:
            fake_preds, origin_preds = self.__related_preds['fake_pred'], self.__related_preds['origin']
            self.__fake_fidelityminus_complete = fidelity_complete(origin_preds, fake_preds)
            return self.__fake_fidelityminus_complete

    @property
    def sparsity_edges(self):
        r"""
        Return the Sparsity value.
        """
        if self.__sparsity_edges:
            return self.__sparsity_edges
        elif None in self.__related_preds['sparsity_edges']:
            return None
        else:
            return torch.tensor(self.__related_preds['sparsity_edges']).mean().item()

    @property
    def fidelity_nodes(self):
        if self.__fidelity_nodes:
            return self.__fidelity_nodes
        elif None in self.__related_preds['maskimp_nodes_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_nodes_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelity_nodes = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_nodes

    @property
    def fidelity_origin_nodes(self):
        if self.__fidelity_origin_nodes:
            return self.__fidelity_origin_nodes
        elif None in self.__related_preds['maskimp_nodes_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_nodes_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelity_origin_nodes = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_origin_nodes

    @property
    def fidelity_complete_nodes(self):
        if self.__fidelity_complete_nodes:
            return self.__fidelity_complete_nodes
        elif None in self.__related_preds['maskimp_nodes'] or None in self.__related_preds['origin']:
            return None
        else:
            maskout_preds, origin_preds = self.__related_preds['maskimp_nodes'], self.__related_preds['origin']
            self.__fidelity_complete_nodes = fidelity_complete(origin_preds, maskout_preds)
            return self.__fidelity_complete_nodes
    
    @property
    def fidelityminus_nodes(self):
        if self.__fidelityminus_nodes:
            return self.__fidelityminus_nodes
        elif None in self.__related_preds['retainimp_nodes_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_nodes_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelityminus_nodes = fidelity(origin_preds, retainimp_preds)
            return self.__fidelityminus_nodes

    @property
    def fidelityminus_origin_nodes(self):
        if self.__fidelityminus_origin_nodes:
            return self.__fidelityminus_origin_nodes
        elif None in self.__related_preds['retainimp_nodes_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_nodes_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelityminus_origin_nodes = fidelity(origin_preds, retainimp_preds)
            return self.__fidelityminus_origin_nodes

    @property
    def fidelityminus_complete_nodes(self):
        if self.__fidelityminus_complete_nodes:
            return self.__fidelityminus_complete_nodes
        elif None in self.__related_preds['retainimp_nodes'] or None in self.__related_preds['origin']:
            return None
        else:
            retainimp_preds, origin_preds = self.__related_preds['retainimp_nodes'], self.__related_preds['origin']
            self.__fidelityminus_complete_nodes = fidelity_complete(origin_preds, retainimp_preds)
            return self.__fidelityminus_complete_nodes
     
    @property
    def sparsity_nodes(self):
        if self.__sparsity_nodes:
            return self.__sparsity_nodes
        elif None in self.__related_preds['sparsity_nodes']:
            return None
        else:
            return torch.tensor(self.__related_preds['sparsity_nodes']).mean().item()


