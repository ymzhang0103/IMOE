import torch
import argparse
from tqdm import tqdm

from codes.InstanceLevel_Explainer import InstanceLevelExplainer
from codes.evaluate import evaluate_acc
import numpy as np
import os
import os.path as osp
from codes.metrics import MaskoutMetric, XCollector
import time
from codes.GNNmodels import GnnNets
from codes.ModelLevelExplainer.utils import get_gnnModel_params
from codes.load_dataset import get_dataset, get_dataloader
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from codes.plot_utils import PlotUtils


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ReFine")
    parser.add_argument('--cuda', type=int, default=0,
                        help='GPU device.')
    parser.add_argument('--dataset_root', type=str, default='datasets')
    parser.add_argument('--dataset_name', type=str, default='BA_4Motifs',
                        choices=['BA_4Motifs', 'Mutagenicity_full', 'NCI1', 'PROTEINS'])
    parser.add_argument('--result_dir', type=str, default="results/",
                        help='Result directory.')
    parser.add_argument('--lr', type=float, default=0.05,    
                        help='Fine-tuning learning rate.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Fine-tuning rpoch.')
    parser.add_argument('--ratio', type=float, default=0.1)  

    parser.add_argument('-o', '--prototype_dir', type=str, default='./prototype', help='Path to prototype directory')
    parser.add_argument('-g', '--num_graphs', type=int, default=100, help='Number of graphs to generate')
    return parser.parse_args()


def test():
    tik = time.time()
    ft_log = open(f"{log_dir}/testlog_min.txt", "w")
    
    instanceLevelExplainer.mask_net.load_state_dict(torch.load(osp.join(model_dir, f'bestF.pt')))
    instanceLevelExplainer.mask_net.eval()
    test_G_acc = []
    allnode_related_preds_dict = dict()
    allnode_mask_dict = dict()
    plotutils = PlotUtils(dataset_name=args.dataset_name)
    gid_index = 0
    for g in test_loader:
        graphid = test_loader.dataset.indices[gid_index]
        gid_index = gid_index + 1
        if graphid not in test_indices:
            continue
        g = g.to(args.device)
        label = g.y.item()
        edge_mask = instanceLevelExplainer.get_explain_graph(g)
        #print("imp, ", imp)
        acc, prob = evaluate_acc(gnn_model, top_ratio_list, graph=g, imp=edge_mask.detach().cpu().numpy())
        test_G_acc.append(acc.squeeze().tolist())

        data = Data(x=g.x, edge_index=g.edge_index, batch = g.batch)
        origin_logits, origin_pred, origin_emb, sub_embs = gnn_model(data)
        instanceLevelExplainer._set_masks(edge_mask, gnn_model) 
        masked_logits,  masked_pred, masked_emb, masked_node_embs = gnn_model(data)
        instanceLevelExplainer._clear_masks(gnn_model)

        origin_pred = origin_pred[0]
        masked_pred = masked_pred[0]
        pred_mask, related_preds_dict = metric.metric_del_edges_GC(args.topk_arr, g.x, edge_mask, g.edge_index, origin_pred, masked_pred, label)
        allnode_related_preds_dict[graphid] = related_preds_dict
        allnode_mask_dict[graphid] = pred_mask
        if plot_flag:
            if not os.path.exists (log_dir+"/case"):
                os.makedirs(log_dir+"/case")
            if args.dataset_name =="Mutagenicity" or args.dataset_name =="Mutagenicity_full":
                visual_imp_edge_count = 8
            else:
                if args.dataset_name =="BA_4Motifs":
                    if graphid>=1500:
                        visual_imp_edge_count = 24
                    else:
                        visual_imp_edge_count = 12
            edges_idx_desc = torch.tensor(pred_mask[0]).sort(descending=True).indices
            important_nodelist = []
            important_edgelist = []
            for idx in edges_idx_desc:
                if len(important_edgelist)<visual_imp_edge_count:
                    if (g.edge_index[0][idx].item(), g.edge_index[1][idx].item()) not in important_edgelist:
                        important_nodelist.append(g.edge_index[0][idx].item())
                        important_nodelist.append(g.edge_index[1][idx].item())
                        important_edgelist.append((g.edge_index[0][idx].item(), g.edge_index[1][idx].item()))
                        important_edgelist.append((g.edge_index[1][idx].item(), g.edge_index[0][idx].item()))
            important_nodelist = list(set(important_nodelist))
            if args.dataset_name =="Mutagenicity" or args.dataset_name =="Mutagenicity_full" or args.dataset_name =="NCI1":
                plotutils.visualize(data, nodelist=important_nodelist, edgelist=important_edgelist, 
                                figname=os.path.join(log_dir+"/case", f"example_{graphid}_6.pdf"))
            else:
                ori_graph = to_networkx(data, to_undirected=True)
                if args.dataset_name =="BA_4Motifs":
                    plotutils.plot_new(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, x=data.x, graphid = graphid,
                                figname=os.path.join(log_dir+"/case", f"example_{graphid}_6.pdf"))
                else:
                    if hasattr(dataset, 'supplement'):
                        words = dataset.supplement['sentence_tokens'][str(graphid)]
                        plotutils.plot_new(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, words=words,
                                figname=os.path.join(log_dir+"/case", f"example_{graphid}.pdf"))
                    else:
                        plotutils.plot_new(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, x=data.x,
                                figname=os.path.join(log_dir+"/case", f"example_{graphid}.pdf"))
    auc = np.average(np.array(test_G_acc), axis=0)
    print(" ACC-AUC: ", auc)

    one_simula_arr = []
    one_simula_origin_arr = []
    one_simula_complete_arr = []
    one_fidelity_arr = []
    one_fidelity_origin_arr = []
    one_fidelity_complete_arr = []
    one_fidelityminus_arr = []
    one_fidelityminus_origin_arr = []
    one_fidelityminus_complete_arr = []
    one_finalfidelity_complete_arr = []
    one_fvaluefidelity_complete_arr = []
    one_del_fidelity_arr = []
    one_del_fidelity_origin_arr = []
    one_del_fidelity_complete_arr = []
    one_del_fidelityminus_arr = []
    one_del_fidelityminus_origin_arr = []
    one_del_fidelityminus_complete_arr = []
    one_del_finalfidelity_complete_arr = []
    one_del_fvaluefidelity_complete_arr = []
    one_sparsity_edges_arr = []
    one_fidelity_nodes_arr = []
    one_fidelity_origin_nodes_arr = []
    one_fidelity_complete_nodes_arr = []
    one_fidelityminus_nodes_arr = []
    one_fidelityminus_origin_nodes_arr = []
    one_fidelityminus_complete_nodes_arr = []
    one_finalfidelity_complete_nodes_arr = []
    one_sparsity_nodes_arr = []
    for top_k in args.topk_arr:
        print("top_k: ", top_k)
        x_collector = XCollector()
        gid_index = 0
        for g in test_loader:
            graphid = test_loader.dataset.indices[gid_index]
            gid_index = gid_index + 1
            if graphid not in test_indices:
                continue
            related_preds = allnode_related_preds_dict[graphid][top_k]
            mask = allnode_mask_dict[graphid]
            x_collector.collect_data(mask, related_preds, label=0)

            ft_log.write("graphid,{}\n".format(graphid))
            ft_log.write("mask,{}\n".format(mask))
            ft_log.write("related_preds,{}\n".format(related_preds))

        one_simula_arr.append(round(x_collector.simula, 4))
        one_simula_origin_arr.append(round(x_collector.simula_origin, 4))
        one_simula_complete_arr.append(round(x_collector.simula_complete, 4))
        one_fidelity_arr.append(round(x_collector.fidelity, 4))
        one_fidelity_origin_arr.append(round(x_collector.fidelity_origin, 4))
        one_fidelity_complete_arr.append(round(x_collector.fidelity_complete, 4))
        one_fidelityminus_arr.append(round(x_collector.fidelityminus, 4))
        one_fidelityminus_origin_arr.append(round(x_collector.fidelityminus_origin, 4))
        one_fidelityminus_complete_arr.append(round(x_collector.fidelityminus_complete, 4))
        one_finalfidelity_complete_arr.append(round(x_collector.fidelity_complete - x_collector.fidelityminus_complete, 4))
        F_fidelity = 2/(1/x_collector.fidelity_complete +1/(1/x_collector.fidelityminus_complete))
        one_fvaluefidelity_complete_arr.append(round(F_fidelity, 4))
        one_del_fidelity_arr.append(round(x_collector.del_fidelity, 4))
        one_del_fidelity_origin_arr.append(round(x_collector.del_fidelity_origin, 4))
        one_del_fidelity_complete_arr.append(round(x_collector.del_fidelity_complete, 4))
        one_del_fidelityminus_arr.append(round(x_collector.del_fidelityminus, 4))
        one_del_fidelityminus_origin_arr.append(round(x_collector.del_fidelityminus_origin, 4))
        one_del_fidelityminus_complete_arr.append(round(x_collector.del_fidelityminus_complete, 4))
        one_del_finalfidelity_complete_arr.append(round(x_collector.del_fidelity_complete - x_collector.del_fidelityminus_complete, 4))
        del_F_fidelity = 2/(1/x_collector.del_fidelity_complete +1/(1/x_collector.del_fidelityminus_complete))
        one_del_fvaluefidelity_complete_arr.append(round(del_F_fidelity, 4))
        one_sparsity_edges_arr.append(round(x_collector.sparsity_edges, 4))
        one_fidelity_nodes_arr.append(round(x_collector.fidelity_nodes, 4))
        one_fidelity_origin_nodes_arr.append(round(x_collector.fidelity_origin_nodes, 4))
        one_fidelity_complete_nodes_arr.append(round(x_collector.fidelity_complete_nodes, 4))
        one_fidelityminus_nodes_arr.append(round(x_collector.fidelityminus_nodes, 4))
        one_fidelityminus_origin_nodes_arr.append(round(x_collector.fidelityminus_origin_nodes, 4))
        one_fidelityminus_complete_nodes_arr.append(round(x_collector.fidelityminus_complete_nodes, 4))
        one_finalfidelity_complete_nodes_arr.append(round(x_collector.fidelity_complete_nodes - x_collector.fidelityminus_complete_nodes, 4))
        one_sparsity_nodes_arr.append(round(x_collector.sparsity_nodes, 4))

    print("one_simula_arr =", one_simula_arr)
    print("one_simula_origin_arr =", one_simula_origin_arr)
    print("one_simula_complete_arr =", one_simula_complete_arr)
    print("one_fidelity_arr =", one_fidelity_arr)
    print("one_fidelity_origin_arr =", one_fidelity_origin_arr)
    print("one_fidelity_complete_arr =", one_fidelity_complete_arr)
    print("one_fidelityminus_arr=", one_fidelityminus_arr)
    print("one_fidelityminus_origin_arr=", one_fidelityminus_origin_arr)
    print("one_fidelityminus_complete_arr=", one_fidelityminus_complete_arr)
    print("one_finalfidelity_complete_arr=", one_finalfidelity_complete_arr)
    print("one_fvaluefidelity_complete_arr=", one_fvaluefidelity_complete_arr)
    print("one_del_fidelity_arr =", one_del_fidelity_arr)
    print("one_del_fidelity_origin_arr =", one_del_fidelity_origin_arr)
    print("one_del_fidelity_complete_arr =", one_del_fidelity_complete_arr)
    print("one_del_fidelityminus_arr=", one_del_fidelityminus_arr)
    print("one_del_fidelityminus_origin_arr=", one_del_fidelityminus_origin_arr)
    print("one_del_fidelityminus_complete_arr=", one_del_fidelityminus_complete_arr)
    print("one_del_finalfidelity_complete_arr=", one_del_finalfidelity_complete_arr)
    print("one_del_fvaluefidelity_complete_arr=", one_del_fvaluefidelity_complete_arr)
    print("one_sparsity_edges_arr =", one_sparsity_edges_arr)
    print("one_fidelity_nodes_arr =", one_fidelity_nodes_arr)
    print("one_fidelity_origin_nodes_arr =", one_fidelity_origin_nodes_arr)
    print("one_fidelity_complete_nodes_arr =", one_fidelity_complete_nodes_arr)
    print("one_fidelityminus_nodes_arr=", one_fidelityminus_nodes_arr)
    print("one_fidelityminus_origin_nodes_arr=", one_fidelityminus_origin_nodes_arr)
    print("one_fidelityminus_complete_nodes_arr=", one_fidelityminus_complete_nodes_arr)
    print("one_finalfidelity_complete_nodes_arr=", one_finalfidelity_complete_nodes_arr)
    print("one_sparsity_nodes_arr =", one_sparsity_nodes_arr)

    tok = time.time()
    ft_log.write("one_auc={}".format(auc) + "\n")
    ft_log.write("one_simula={}".format(one_simula_arr) + "\n")
    ft_log.write("one_simula_orign={}".format(one_simula_origin_arr) + "\n")
    ft_log.write("one_simula_complete={}".format(one_simula_complete_arr) + "\n")
    ft_log.write("one_fidelity={}".format(one_fidelity_arr) + "\n")
    ft_log.write("one_fidelity_orign={}".format(one_fidelity_origin_arr) + "\n")
    ft_log.write("one_fidelity_complete={}".format(one_fidelity_complete_arr) + "\n")
    ft_log.write("one_fidelityminus={}".format(one_fidelityminus_arr)+"\n")
    ft_log.write("one_fidelityminus_origin={}".format(one_fidelityminus_origin_arr)+"\n")
    ft_log.write("one_fidelityminus_complete={}".format(one_fidelityminus_complete_arr)+"\n")
    ft_log.write("one_finalfidelity_complete={}".format(one_finalfidelity_complete_arr)+"\n")
    ft_log.write("one_fvaluefidelity_complete={}".format(one_fvaluefidelity_complete_arr)+"\n")
    ft_log.write("one_del_fidelity={}".format(one_del_fidelity_arr) + "\n")
    ft_log.write("one_del_fidelity_orign={}".format(one_del_fidelity_origin_arr) + "\n")
    ft_log.write("one_del_fidelity_complete={}".format(one_del_fidelity_complete_arr) + "\n")
    ft_log.write("one_del_fidelityminus={}".format(one_del_fidelityminus_arr)+"\n")
    ft_log.write("one_del_fidelityminus_origin={}".format(one_del_fidelityminus_origin_arr)+"\n")
    ft_log.write("one_del_fidelityminus_complete={}".format(one_del_fidelityminus_complete_arr)+"\n")
    ft_log.write("one_del_finalfidelity_complete={}".format(one_del_finalfidelity_complete_arr)+"\n")
    ft_log.write("one_del_fvaluefidelity_complete={}".format(one_del_fvaluefidelity_complete_arr)+"\n")
    ft_log.write("one_sparsity_edges={}".format(one_sparsity_edges_arr) + "\n")
    ft_log.write("one_fidelity_nodes={}".format(one_fidelity_nodes_arr) + "\n")
    ft_log.write("one_fidelity_origin_nodes={}".format(one_fidelity_origin_nodes_arr) + "\n")
    ft_log.write("one_fidelity_complete_nodes={}".format(one_fidelity_complete_nodes_arr) + "\n")
    ft_log.write("one_fidelityminus_nodes={}".format(one_fidelityminus_nodes_arr)+"\n")
    ft_log.write("one_fidelityminus_origin_nodes={}".format(one_fidelityminus_origin_nodes_arr)+"\n")
    ft_log.write("one_fidelityminus_complete_nodes={}".format(one_fidelityminus_complete_nodes_arr)+"\n")
    ft_log.write("one_finalfidelity_complete_nodes={}".format(one_finalfidelity_complete_nodes_arr)+"\n")
    ft_log.write("one_sparsity_nodes={}".format(one_sparsity_nodes_arr) + "\n")
    ft_log.write("test time,{}".format(tok-tik))
    ft_log.close()


test_flag = False
plot_flag = False
seed = 2023
torch.manual_seed(seed)
#top_ratio_list=[0.1*i for i in range(1,11)]
args = parse_args()
args.topk_arr = list(range(10))+list(range(10,101,5))
top_ratio_list = [i * 0.01 for i in args.topk_arr]
args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
#args.device="cpu"
args.cluster_algorithm = "DBSCAN"   #KMeans, DBSCAN
args.km_n = 6

if args.dataset_name == "BA_4Motifs":
    args.lr = 0.005
    args.ratio = 0.05


lr_str = str(args.lr).replace(".","")
ratio_str = str(args.ratio).replace(".","")

model_dir = f"models/instance_level_explainer/{args.dataset_name}_lr{lr_str}_epoch{args.epoch}_k{ratio_str}_sizemean_seed{seed}_{args.cluster_algorithm}/"
if args.cluster_algorithm == "KMeans":
    model_dir = f"{model_dir}/km{args.km_n}/"
os.makedirs(model_dir, exist_ok=True)

log_dir = f"log/instance_level_explanation_{args.dataset_name}_lr{lr_str}_epoch{args.epoch}_k{ratio_str}_sizemean_seed{seed}_{args.cluster_algorithm}"
if args.cluster_algorithm == "KMeans":
    log_dir = f"{log_dir}/km{args.km_n}"
os.makedirs(log_dir, exist_ok=True)

dataset = get_dataset(args.dataset_root, args.dataset_name)
dataset.data.x = dataset.data.x.float()
dataset.data.y = dataset.data.y.squeeze().long()
#if args.graph_classification:
args.batch_size = 1
args.random_split_flag = True
args.data_split_ratio =  [0.8, 0.1, 0.1]  #None
args.seed = seed
dataloader_params = {'batch_size': args.batch_size,
                        'random_split_flag': args.random_split_flag,
                        'data_split_ratio': args.data_split_ratio,
                        'seed': args.seed}
loader = get_dataloader(dataset, **dataloader_params)
train_loader = loader['train']
test_loader = loader['test']
val_loader = loader['eval']
e_in_channels=1
train_instances = train_loader.dataset.indices
test_indices = test_loader.dataset.indices
eval_indices = val_loader.dataset.indices
if args.dataset_name == "Mutagenicity" or args.dataset_name == "Mutagenicity_full":
    test_indices = [i for i in test_indices if dataset.data.y[i]==0]
    train_instances = [i for i in train_instances if dataset.data.y[i]==0]
    eval_indices = [i for i in eval_indices if dataset.data.y[i]==0]

if args.dataset_name == "BA_4Motifs":
    node_feature_size = 10
elif  args.dataset_name == "Mutagenicity_full":
    node_feature_size = 14
elif args.dataset_name == "NCI1":
    node_feature_size = 38
elif args.dataset_name == "PROTEINS":
    node_feature_size = 3

n_classes_dict = { 'BA_4Motifs':2,  'Mutagenicity_full':2,  'NCI1':2, 'PROTEINS':2 }

model_args = get_gnnModel_params()
model_args.device = args.device
gnn_model = GnnNets(input_dim=node_feature_size,  output_dim=n_classes_dict[args.dataset_name], model_args=model_args)
gnn_model.to_device()
ckpt = torch.load(f'models/gnnModel/{args.dataset_name}/gcn_best.pth')
gnn_model.load_state_dict(ckpt['net'])
gnn_model.eval()

instanceLevelExplainer = InstanceLevelExplainer(args, gnn_model, 
                n_in_channels=torch.flatten(train_loader.dataset[0].x, 1, -1).size(1),
                e_in_channels=e_in_channels,    
                n_label=n_classes_dict[args.dataset_name])

metric = MaskoutMetric(gnn_model, args)
if not test_flag:
    args.prototype_dir =  osp.join(args.prototype_dir, f"{args.dataset_name}")
    # get cluster center
    cluster_center_dic = {}
    graphs_dic = torch.load(f"{args.prototype_dir}/{args.dataset_name}-{args.num_graphs}.json")
    for label in range(n_classes_dict[args.dataset_name]):
        l_index = f"l{label}"
        cluster_dir = f"{args.prototype_dir}/{args.dataset_name}-{args.num_graphs}-label{label}-cluster.json"
        if args.cluster_algorithm == "KMeans":
            cluster_dir = f"{args.prototype_dir}/{args.dataset_name}-{args.num_graphs}-label{label}-cluster-km{args.km_n}.json"
        cluster_dic = torch.load(cluster_dir)

        if args.cluster_algorithm == "KMeans":
            km = cluster_dic["km"]
            cluster_center = km.cluster_centers_
        elif args.cluster_algorithm == "DBSCAN":
            graph_embs = graphs_dic[label]["graph_embs"]
            db = cluster_dic["db"]
            n_clusters = len(set(db.labels_))-(1 if -1 in db.labels_ else 0)
            cluster_center = []
            for i in range(n_clusters):
                cluster_center.append(np.mean(np.array([graph_embs[k] for k in np.where(db.labels_ ==i)[0]]), axis=0))
            cluster_center = np.array(cluster_center)
        cluster_center_dic[label] = cluster_center

    imp_log = open(f"{log_dir}/imp.txt", "w")
    ft_log = open(f"{log_dir}/loss.txt", "w")
    
    best_acc = 0
    best_F_fidelity = 0
    instanceLevelExplainer.train()
    optimizer = torch.optim.Adam(instanceLevelExplainer.mask_net.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        fid_loss = 0
        label_loss = 0
        imp_log.write("epoch,{}\n".format(epoch))
        #print(self.mask_net.state_dict())
        gid_index = 0
        for g in tqdm(iter(train_loader), total=len(train_loader)):
            g = g.to(args.device)
            graphid = train_loader.dataset.indices[gid_index]
            gid_index = gid_index + 1
            if graphid not in train_instances:
                continue
            label_graphs = cluster_center_dic[g.y.item()]
            edge_mask, fl, ll, sim = instanceLevelExplainer.explain(g, label_graphs, ratio=args.ratio)
            #print("imp, ", imp)
            imp_log.write("imp,{}\n".format(edge_mask.detach().cpu().numpy() ))
            imp_log.write("sim,{}\n".format(sim))
            #imp_log.write("pos_idx,{}\n".format(pos_idx))
            fid_loss = fid_loss + fl
            label_loss = label_loss + ll
            #print("ll, ", ll)
            #ft_log.write("ll,{}, ".format(ll))
        print("epoch, ", epoch, "label_loss, ", label_loss, "fid_loss, ", fid_loss)
        ft_log.write("\n epoch,{}".format(epoch) + ", label_loss,{}".format(label_loss.item()) + ", fid_loss,{}\n".format(fid_loss))
        loss = label_loss + fid_loss
        loss.backward()
        optimizer.step()

        val_G_acc = []
        fidelity_complete = []
        fidelityminus_complete = []
        del_fidelity_complete = []
        del_fidelityminus_complete = []
        gid_index = 0
        for g in val_loader:
            g = g.to(args.device)
            graphid = val_loader.dataset.indices[gid_index]
            gid_index = gid_index + 1
            if graphid not in eval_indices:
                continue
            label = g.y.item()
            edge_mask = instanceLevelExplainer.get_explain_graph(g)
            #print("imp, ", imp)
            acc, prob = evaluate_acc(gnn_model, [args.ratio], graph=g, imp=edge_mask.detach().cpu().numpy())
            val_G_acc.append(acc.squeeze().tolist())

            #origin_emb, origin_logits,  origin_pred = gnn_model(g.x, edge_index=g.edge_index, batch=g.batch)
            #masked_emb, masked_logits,  masked_pred = gnn_model(g.x, edge_index=g.edge_index, edge_weight=edge_mask, batch=g.batch)
            data = Data(x=g.x, edge_index=g.edge_index, batch = g.batch)
            origin_logits, origin_pred, origin_emb, sub_embs = gnn_model(data)
            instanceLevelExplainer._set_masks(edge_mask, gnn_model) 
            masked_logits,  masked_pred, masked_emb, masked_node_embs = gnn_model(data)
            instanceLevelExplainer._clear_masks(gnn_model)
            
            origin_pred = origin_pred[0]
            masked_pred = masked_pred[0]
            topk = args.ratio*100
            pred_mask, related_preds_dict = metric.metric_del_edges_GC([topk], g.x, edge_mask, g.edge_index, origin_pred, masked_pred, label)
            maskimp_probs = related_preds_dict[topk][0]["maskimp"]
            fidelity_complete_onenode = sum(abs(origin_pred - maskimp_probs)).item()
            fidelity_complete.append(fidelity_complete_onenode)
            masknotimp_probs = related_preds_dict[topk][0]["masknotimp"]
            fidelityminus_complete_onenode = sum(abs(origin_pred - masknotimp_probs)).item()
            fidelityminus_complete.append(fidelityminus_complete_onenode)
            delimp_probs = related_preds_dict[topk][0]["delimp"]
            del_fidelity_complete_onenode = sum(abs(origin_pred - delimp_probs)).item()
            del_fidelity_complete.append(del_fidelity_complete_onenode)
            retainimp_probs = related_preds_dict[topk][0]["retainimp"]
            del_fidelityminus_complete_onenode = sum(abs(origin_pred - retainimp_probs)).item()
            del_fidelityminus_complete.append(del_fidelityminus_complete_onenode)
            
        if np.array(val_G_acc).mean() > best_acc:
            print("saving best ACC model......", "new best ACC, ", np.array(val_G_acc).mean(), "old best ACC, ", best_acc)
            best_acc = np.array(val_G_acc).mean()
            ft_log.write("best_acc,{}\n".format(best_acc))
            ft_log.write("saving best ACC model......\n")
            torch.save(instanceLevelExplainer.mask_net.state_dict(), osp.join(model_dir, f'bestACC.pt'))

        fidelityplus = np.mean(fidelity_complete)
        fidelityminus = np.mean(fidelityminus_complete)
        decline = torch.sub(fidelityplus, fidelityminus)
        F_fidelity = 2/(1/fidelityplus +1/(1/fidelityminus))
        if F_fidelity >= best_F_fidelity:
            best_F_fidelity = F_fidelity
            print("epoch", epoch, "best_F_fidelity", best_F_fidelity, "saving best F_fidelity model...")
            ft_log.write("best_F_fidelity,{}\n".format(best_F_fidelity))
            ft_log.write("saving best F_fidelity model...\n")
            torch.save(instanceLevelExplainer.mask_net.state_dict(), osp.join(model_dir, f'bestF.pt'))

    torch.save(instanceLevelExplainer.mask_net.state_dict(), os.path.join(model_dir, f'last.pt'))
    imp_log.close()
    ft_log.close()

    test()
else:
   test()
 