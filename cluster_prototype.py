import torch
import argparse

from codes.gflownet.environment import Environment, class_prob_reward
from codes.gflownet.backbone import GCNBackbone
from codes.gflownet.gflownet_wrapper import GFlowNet
import networkx as nx
import os
import os.path as osp
from pathlib import Path
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN, KMeans
from torch_geometric.utils import to_networkx
from umap import UMAP
from sklearn.cluster import DBSCAN, KMeans
from codes.GNNmodels import GnnNets
from codes.gflownet.utils import get_gnnModel_params
import matplotlib
matplotlib.use("Agg")

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Test")
    parser.add_argument('--cuda', type=int, default=0, help='GPU device.')
    parser.add_argument('--dataset_name', type=str, default='BA_4Motifs',
                        choices=['BA_4Motifs', 'Mutagenicity_full', 'NCI1', 'PROTEINS'])
    parser.add_argument('-o', '--output_dir', type=str, default='prototype', help='Path to output directory')
    parser.add_argument('-g', '--num_graphs', type=int, default=100, help='Number of graphs to generate')
    parser.add_argument('-m', '--max_actions', type=int, default=10, help='Maximum number of actions to take')
    parser.add_argument('-n', '--num_nodes', type=int, default=3, help='Minimum number of nodes in a graph')
    parser.add_argument('-idx', '--iteration', type=int, default=0, help='index of GNN model')

    parser.add_argument('--ratio', type=float, default=0.4)
    return parser.parse_args()


def generate_nx_plot(args, graph):
    if  args.dataset_name == "Mutagenicity":
        atoms = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
        colors = {0: 'green', 1: 'red', 2: 'orange', 3:'lime', 4: 'blue', 5: 'yellow', 6: 'pink', 7: 'orchid', 8: 'gold', 9: 'purple', 10:'tan', 11: 'lightseagreen',12: 'indigo', 13: 'navy'}
        #colors = ['orange', 'red', 'lime', 'green', 'blue', 'orchid', 'darksalmon', 'darkslategray', 'gold', 'bisque', 'tan', 'lightseagreen', 'indigo', 'navy']
    labelsdict = {i:atoms[int(graph.y[i].item())] for i in range(graph.num_nodes)}
    
    fig = plt.figure(figsize=(4, 4))
    _colors = [colors[int(graph.y[i].item())] for i in range(graph.num_nodes)]
    G_nx = to_networkx(graph, to_undirected=True)
    nx.draw_networkx(G_nx, pos=nx.spring_layout(G_nx, seed=42), with_labels=True, node_color=_colors, labels=labelsdict)
    plt.close()
    return fig


def cluster_1(all_graph_emb, all_graph_label, args, OUTPUT_DIR):
    all_graph_emb_arr = all_graph_emb.detach().cpu().numpy()

    km =  KMeans(n_clusters=4).fit(all_graph_emb_arr)
    print("KMeans cluster index of graph", km.labels_)
    print("KMeans cluster center", km.cluster_centers_)

    db = DBSCAN(eps=1500, min_samples=2).fit(all_graph_emb_arr)
    print("DBSCAN cluster index of graph", db.labels_)
    print("DBSCAN cluster core ", db.core_sample_indices_)
    n_clusters = len(set(db.labels_))-(1 if -1 in db.labels_ else 0)
    print("DBSCAN count of clusters", n_clusters)

    emb_umap = UMAP().fit_transform(all_graph_emb_arr)
    km_2 =  KMeans(n_clusters=4).fit(emb_umap)
    db_2 =  DBSCAN(eps=4, min_samples=2).fit(emb_umap)
    print("UMAP KMeans cluster index of graph", km_2.labels_)
    print("UMAP KMeans cluster center", km_2.cluster_centers_)
    print("UMAP DBSCAN cluster index of graph", db_2.labels_)
    print("UMAP DBSCAN cluster core ", db_2.core_sample_indices_)
    n_clusters_2 = len(set(db_2.labels_))-(1 if -1 in db_2.labels_ else 0)
    print("UMAP DBSCAN count of clusters", n_clusters_2)

    if not os.path.exists(OUTPUT_DIR+"-cluster"):
        os.makedirs(OUTPUT_DIR+"-cluster")
    for label in graphs_dic.keys():
        graphs = graphs_dic[label]["graphs"]
        pred_label_arr = graphs_dic[label]["pred_label_arr"]
        for i in range(len(graphs)):
            graph = graphs[i]
            pred_label = pred_label_arr[i]
            if args.dataset_name == "":
                fig = generate_nx_plot(graph)
                fig.savefig(os.path.join(OUTPUT_DIR+"-cluster", Path(r'%s-label%d-graph%s-predlabel%d-kml%d-dbl%d-ukml%d-udbl%d.png' % (args.dataset_name, label, str(i), pred_label, 
                                                                                                     km.labels_[i], db.labels_[i], km_2.labels_[i], db_2.labels_[i]))))
            
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(2,3,1)
    plt.scatter(emb_umap[:,0], emb_umap[:,1], c=all_graph_label)
    ax1.set_title("origin graph")
    ax2 = plt.subplot(2,3,2)
    plt.scatter(emb_umap[:,0], emb_umap[:,1], c=km.labels_)
    ax2.set_title("KMeans  origin emb")
    ax3 = plt.subplot(2,3,3)
    plt.scatter(emb_umap[:,0], emb_umap[:,1], c=km_2.labels_)
    ax3.set_title("KMeans  UMAP emb")
    ax4 = plt.subplot(2,3,5)
    plt.scatter(emb_umap[:,0], emb_umap[:,1], c=db.labels_)
    ax4.set_title("DBSCAN  origin emb")
    ax5 = plt.subplot(2,3,6)
    plt.scatter(emb_umap[:,0], emb_umap[:,1], c=db_2.labels_)
    ax5.set_title("DBSCAN  UMAP emb")
    plt.savefig(f"{args.output_dir}/{args.dataset_name}-cluster-{args.num_graphs}.png")
    plt.show()

    cluster_dic = {"km":km, "db":db, "km_2":km_2, "db_2":db_2}
    torch.save(cluster_dic, f"{args.output_dir}/{args.dataset_name}-{args.num_graphs}-cluster-1.json")
    #print("graph_label", all_graph_label)
    #print("graphs_emb", graphs_emb)
    #print("graphs_emb_umap", emb_umap)



def cluster_km(cur_label, graph_emb, trans_emb, args, km_n):
    km =  KMeans(n_clusters=km_n).fit(graph_emb)
    print("KMeans cluster index of graph", km.labels_)
    print("KMeans cluster center", km.cluster_centers_)

    km_2 =  KMeans(n_clusters=km_n).fit(trans_emb)
    print("UMAP KMeans cluster index of graph", km_2.labels_)
    print("UMAP KMeans cluster center", km_2.cluster_centers_)

    cluster_dic = {"km":km, "km_2":km_2}
    torch.save(cluster_dic, f"{args.output_dir}/{args.dataset_name}-{args.num_graphs}-label{cur_label}-cluster-km{km_n}.json")


def cluster_DBSCAN(cur_label, graph_emb, trans_emb, args, eps=1, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(graph_emb)   
    print("DBSCAN cluster index of graph", db.labels_)
    print("DBSCAN cluster core ", db.core_sample_indices_)
    n_clusters = len(set(db.labels_))-(1 if -1 in db.labels_ else 0)
    print("DBSCAN count of clusters", n_clusters)

    db_2 =  DBSCAN(eps=eps, min_samples=min_samples).fit(trans_emb)
    #print("UMAP DBSCAN cluster index of graph", db_2.labels_)
    #print("UMAP DBSCAN cluster core ", db_2.core_sample_indices_)
    n_clusters_2 = len(set(db_2.labels_))-(1 if -1 in db_2.labels_ else 0)
    #print("UMAP DBSCAN count of clusters", n_clusters_2)

    cluster_dic = {"db":db, "db_2":db_2}
    #torch.save(cluster_dic, f"{args.output_dir}/{args.dataset_name}-{args.num_graphs}-label{cur_label}-cluster-eps{eps}-minneib{min_samples}.json")
    return cluster_dic
    

def cluster(cur_label, graph_emb, trans_emb, args, all_graph_label):
    km_n = 4
    km =  KMeans(n_clusters=km_n).fit(graph_emb)
    print("KMeans cluster index of graph", km.labels_)
    print("KMeans cluster center", km.cluster_centers_)

    km_2 =  KMeans(n_clusters=km_n).fit(trans_emb)
    print("UMAP KMeans cluster index of graph", km_2.labels_)
    print("UMAP KMeans cluster center", km_2.cluster_centers_)

    #db = DBSCAN(eps=100, min_samples=2).fit(graph_emb)   #MUTAG
    db = DBSCAN(eps=1, min_samples=5).fit(graph_emb)   
    print("DBSCAN cluster index of graph", db.labels_)
    print("DBSCAN cluster core ", db.core_sample_indices_)
    n_clusters = len(set(db.labels_))-(1 if -1 in db.labels_ else 0)
    print("DBSCAN count of clusters", n_clusters)

    db_2 =  DBSCAN(eps=1, min_samples=5).fit(trans_emb)
    print("UMAP DBSCAN cluster index of graph", db_2.labels_)
    print("UMAP DBSCAN cluster core ", db_2.core_sample_indices_)
    n_clusters_2 = len(set(db_2.labels_))-(1 if -1 in db_2.labels_ else 0)
    print("UMAP DBSCAN count of clusters", n_clusters_2)

    cluster_dic = {"km":km, "db":db, "km_2":km_2, "db_2":db_2}
    torch.save(cluster_dic, f"{args.output_dir}/{args.dataset_name}-{args.num_graphs}-label{cur_label}-cluster-km{km_n}.json")
    cluster_plot(cur_label, trans_emb, cluster_dic, args, all_graph_label)


def cluster_plot(cur_label, trans_emb, cluster_dic, args, all_graph_label):
    km = cluster_dic["km"]
    km_2 = cluster_dic["km_2"]
    db = cluster_dic["db"]
    db_2 = cluster_dic["db_2"]

    #plt.figure(figsize=(10, 10))
    plt.figure()
    ax1 = plt.subplot(2,3,1)
    plt.scatter(trans_emb[:,0], trans_emb[:,1], c = all_graph_label)
    ax1.set_title("origin graph")
    
    ax2 = plt.subplot(2,3,2)
    plt.scatter(trans_emb[:,0], trans_emb[:,1], c=km.labels_)
    ax2.set_title("KMeans  origin emb")
    ax3 = plt.subplot(2,3,3)
    plt.scatter(trans_emb[:,0], trans_emb[:,1], c=km_2.labels_)
    ax3.set_title("KMeans  UMAP emb")
    ax4 = plt.subplot(2,3,5)
    plt.scatter(trans_emb[:,0], trans_emb[:,1], c=db.labels_)
    ax4.set_title("DBSCAN  origin emb")
    ax5 = plt.subplot(2,3,6)
    plt.scatter(trans_emb[:,0], trans_emb[:,1], c=db_2.labels_)
    ax5.set_title("DBSCAN  UMAP emb")
    plt.savefig(f"{args.output_dir}/{args.dataset_name}-target{cur_label}-cluster-{args.num_graphs}.png")


def gen_graph(args):
    args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    #args.device = "cpu"

    if args.dataset_name == "BA_4Motifs":
        node_feature_size = 10
    elif  args.dataset_name == "Mutagenicity_full":
        node_feature_size = 14
    elif args.dataset_name == "NCI1":
        node_feature_size = 38
    elif args.dataset_name == "PROTEINS":
        node_feature_size = 3

    n_classes_dict = { 'BA_4Motifs':2, 'Mutagenicity_full':2, 'NCI1':2, 'PROTEINS':2 }

    cfg = {
        'node_feature_size': node_feature_size,
        'alpha': 1.0,
        'threshold': 0.5,
    }    
    cfg['device'] = args.device

    model_args = get_gnnModel_params()
    model_args.device = args.device
    gnn_model = GnnNets(input_dim=node_feature_size,  output_dim=n_classes_dict[args.dataset_name], model_args=model_args)
    gnn_model.to_device()
    ckpt = torch.load(f'models/gnnModel/{args.dataset_name}/gcn_best.pth')
    gnn_model.load_state_dict(ckpt['net'])
    gnn_model.eval()

    ACTIONS_LIMIT = args.max_actions
    MIN_NODE_COUNT = args.num_nodes
    OUTPUT_DIR = osp.join(args.output_dir, f"{args.dataset_name}-{args.num_graphs}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    graphs_dic = {}
    #graphs_emb_dic = {}
    all_graph_emb = None
    all_graph_label = []
    for label in range(n_classes_dict[args.dataset_name]):
        cfg['target'] = label
        env = Environment(args.dataset_name, reward_fn=class_prob_reward, gnn_model=gnn_model, config=cfg)
        backbone = GCNBackbone(num_features=node_feature_size, num_gcn_hidden=[32, 64, 128], num_mlp_hidden=[128, 512])
        gflownet = GFlowNet(backbone, env, args.device)
        gflownet.load(f'models/model_level_explainer/{args.dataset_name}_n{args.num_nodes}_act{args.max_actions}/{args.dataset_name}_gflownet_backbone_target{label}_{args.pretrain_model_type}.pt')
        gflownet.backbone.to(args.device)
        gflownet.eval()

        graphs = []
        graph_embs = []
        pred_label_arr = []
        label_arr = []
        num_graphs = 0
        while 1==1:
            graph = gflownet.generate(MIN_NODE_COUNT, ACTIONS_LIMIT)
            logits, pred, graph_emb, node_embs = gnn_model(graph)
            print("num_graphs: ", num_graphs,"pred label: ",torch.argmax(pred), ", label: ",label)
            if torch.argmax(pred)==label:
                graphs.append(graph)
                graph_embs.append(graph_emb.cpu().detach().numpy()[0])
                num_graphs = num_graphs + 1
                if "Mutagenicity" in args.dataset_name:
                    fig = generate_nx_plot(args, graph)
                    fig.savefig(os.path.join(OUTPUT_DIR, f'{args.dataset_name}_seed{args.seed}_target{label}_graph{num_graphs}_pred{pred[0][label]}.png'))
                if all_graph_emb is None:
                    all_graph_emb = graph_emb
                else:
                    all_graph_emb = torch.cat([all_graph_emb,graph_emb], dim=0)
                all_graph_label.append(label)
                pred_label_arr.append(torch.argmax(pred))
                label_arr.append(label)
            if num_graphs == args.num_graphs:
                break
            
            torch.save( {"graphs":graphs, "graph_embs":graph_embs, "pred_label_arr":pred_label_arr, "label_arr":label_arr},  f"{args.output_dir}/{args.dataset_name}-{args.num_graphs}-label{label}.json")
        graphs_dic[label] = {"graphs":graphs, "graph_embs":graph_embs, "pred_label_arr":pred_label_arr, "label_arr":label_arr}
        
    torch.save({"all_graph_emb":all_graph_emb, "all_graph_label":all_graph_label},  f"{args.output_dir}/{args.dataset_name}-{args.num_graphs}-allgraphemb.json")

    print("starting UMAP.......")
    #trans_emb_dic = {}
    reducer =  UMAP(n_neighbors=10, n_components=2)
    for label in graphs_dic.keys():
        graph_embs = graphs_dic[label]["graph_embs"]
        trans_emb = reducer.fit_transform(graph_embs)
        #trans_emb_dic[label] = trans_emb
        graphs_dic[label]["trans_emb"] =trans_emb
    torch.save(graphs_dic,  f"{args.output_dir}/{args.dataset_name}-{args.num_graphs}.json")



#generate model-level explanations
args = parse_args()
args.pretrain_model_type = "BESTLoss"
args.seed=2023

if args.dataset_name == "BA_4Motifs":
    args.num_nodes = 5
    args.max_actions = 30
elif args.dataset_name == "PROTEINS":
    args.num_nodes = 3
    args.max_actions = 30
elif args.dataset_name == "NCI1":
    args.num_nodes = 3
    args.max_actions = 10
elif args.dataset_name == "Mutagenicity_full":
    args.num_nodes = 3
    args.max_actions = 10

args.output_dir =  osp.join(args.output_dir, f"{args.dataset_name}_{args.pretrain_model_type}")
#args.output_dir =  osp.join(args.output_dir, f"{args.dataset_name}_{args.pretrain_model_type}_n{args.num_nodes}_act{args.max_actions}")
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
#print("generate graphs......")
gen_graph(args)
#cluster for each class
graphs_dic = torch.load(f"{args.output_dir}/{args.dataset_name}-{args.num_graphs}.json")
for label in graphs_dic.keys():
    trans_emb = graphs_dic[label]["trans_emb"]
    label_arr = graphs_dic[label]["label_arr"]
    graph_embs = graphs_dic[label]["graph_embs"]
    cluster_dic = cluster_DBSCAN(label, graph_embs, trans_emb, args, eps=1, min_samples=5)  

    db = cluster_dic["db"]
    n_clusters = len(set(db.labels_))-(1 if -1 in db.labels_ else 0)
    print("label", label, "n_clusters",  n_clusters)
    