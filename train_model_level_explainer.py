import torch
import tqdm
import copy
import numpy as np

from codes.gflownet.environment import Environment, class_prob_reward
from codes.gflownet.backbone import GCNBackbone
from codes.gflownet.gflownet_wrapper import GFlowNet

from codes.GNNmodels import GnnNets 
from codes.gflownet.utils import get_gnnModel_params
import os




EPOCHS = 100000
ACTIONS_LIMIT = 30
MIN_NODE_COUNT = 3
CLIP = -1
UPDATE_FREQ = 2
LR_1 = 1e-3
LR_2 = 1e-4
dataset_name = "BA_4Motifs"
seed = 2023
#device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

if dataset_name == "BA_4Motifs":
    node_feature_size = 10
    MIN_NODE_COUNT = 5
elif dataset_name == "Mutagenicity_full":
    node_feature_size = 14
elif dataset_name == "NCI1":
    node_feature_size = 38
    ACTIONS_LIMIT = 10
elif dataset_name == "PROTEINS":
    node_feature_size = 3


n_classes_dict = {'BA_4Motifs':2, 'Mutagenicity_full':2, 'NCI1':2, 'PROTEINS':2}

model_args = get_gnnModel_params()
model_args.device = device
gnn_model = GnnNets(input_dim=node_feature_size,  output_dim=n_classes_dict[dataset_name], model_args=model_args)
gnn_model.to_device()
ckpt = torch.load(f'models/gnnModel/{dataset_name}/gcn_best.pth')
gnn_model.load_state_dict(ckpt['net'])
gnn_model.eval()


cfg = {
    'node_feature_size': node_feature_size,
    'alpha': 10.0,
    'threshold': 0.5,
    'target': 0,
    'device': device,
}
target = cfg['target']

backbone = GCNBackbone(num_features=node_feature_size, num_gcn_hidden=[32, 64, 128], num_mlp_hidden=[128, 512])
env = Environment(dataset_name, reward_fn=class_prob_reward, gnn_model=gnn_model, config=cfg)

gflownet = GFlowNet(backbone, env, device)

opt = torch.optim.Adam(gflownet.backbone.parameters(), lr=LR_1)

# training loop 
actions_taken = minibatch_loss = 0
best_loss = np.inf
best_minibatch_loss = np.inf
best_reward = 0.0
best_model = None
model_level_dir = f"models/model_level_explainer/{dataset_name}_n{MIN_NODE_COUNT}_act{ACTIONS_LIMIT}_seed{seed}/"
os.makedirs(model_level_dir, exist_ok=True)
f_train = open(model_level_dir + f"target{target}_trainlog.txt", "w")

pbar = tqdm.tqdm(range(EPOCHS), desc=f"Epoch: {0}, Loss: {0:.4f}", unit="episode")
for episode in pbar:
    actions_taken = 0
    # reset gflownet flows
    gflownet.zero_flows()
    
    # initialize state
    state = gflownet.new(start_idx=-1)
    
    while actions_taken < ACTIONS_LIMIT:
        # sample action
        action, P_Forward = gflownet.sample_action(state)
        
        # take action 
        new_state, _, valid, stop = gflownet.take_action(state, action)
        
        # calculate forward flow of action
        forward_flow = gflownet.calculate_forward_flow(action, P_Forward)
        
        # calculate backward flow
        _, P_Backward = gflownet.calculate_flow_dist(new_state)
        backward_flow = gflownet.calculate_backward_flow(action, P_Backward)
        
        # if valid action, update flows
        if valid:
            gflownet.update_flows(forward_flow, backward_flow)
        else:
            gflownet.update_flows(torch.tensor(0.0), torch.tensor(0.0))
        
        state = new_state
        actions_taken += 1
        
        if stop and state.size() >= MIN_NODE_COUNT:
            break

    # calculate reward for completed graph
    reward = gflownet.calculate_reward(state)

    # calculate loss
    loss = (gflownet.logZ + gflownet.total_forward_flow - torch.log(reward).clip(CLIP) - gflownet.total_backward_flow).pow(2)
    minibatch_loss += loss

    if episode % 100 ==0:
        print("logZ:", gflownet.logZ.item(), "forward_flow:", gflownet.total_forward_flow.item(), "reward:", torch.log(reward).clip(CLIP).item(), "backward_flow:", gflownet.total_backward_flow.item())
        #pbar.set_description(f"Epoch: {episode}, Loss: {minibatch_loss.item():.4f}")
        print("Epoch: ", episode, "Loss:", minibatch_loss.item())    
    
    f_train.write("logZ:{}".format(gflownet.logZ.item())+", forward_flow:{}".format(gflownet.total_forward_flow.item())+", reward:{}".format(torch.log(reward).clip(CLIP).item()) + ", backward_flow:{}".format(gflownet.total_backward_flow.item())+"\n")
    f_train.write("Epoch:{}".format(episode)+", loss:{}".format(loss.item())+", minibatch_loss:{}".format(minibatch_loss.item()) +"\n")
    
    if reward > best_reward:
        f_train.write("saving best reward model\n")
        best_reward = reward
        torch.save(gflownet.backbone.state_dict(), model_level_dir + f"{dataset_name}_gflownet_backbone_target{target}_BESTReward.pt")
        #torch.save(gflownet.backbone.state_dict(), model_level_dir + f"{dataset_name}_gflownet_backbone_target{target}_BESTReward_epoch{math.floor(episode / 5000)*5000}.pt")
        
    if loss < best_loss:
        f_train.write("saving best model\n")
        best_loss = loss
        best_model = copy.deepcopy(gflownet.backbone)
        torch.save(best_model.state_dict(), model_level_dir + f"{dataset_name}_gflownet_backbone_target{target}_BESTLoss.pt")
        #torch.save(gflownet.backbone.state_dict(), model_level_dir + f"{dataset_name}_gflownet_backbone_target{target}_BESTLoss_epoch{math.floor(episode / 5000)*5000}.pt")
       
    if episode % UPDATE_FREQ == 0:
        if minibatch_loss < best_minibatch_loss:
            f_train.write("saving best minibatch model\n")
            best_minibatch_loss = minibatch_loss
            best_model = copy.deepcopy(gflownet.backbone)
            torch.save(best_model.state_dict(), model_level_dir + f"{dataset_name}_gflownet_backbone_target{target}_BESTMiniLoss.pt")
            #torch.save(gflownet.backbone.state_dict(), model_level_dir + f"{dataset_name}_gflownet_backbone_target{target}_BESTMiniLoss_epoch{math.floor(episode / 5000)*5000}.pt")
       
        minibatch_loss.backward()
        opt.step()
        opt.zero_grad()
        minibatch_loss = 0

print("Best Loss: ", best_loss.item(), "Best minibatch Loss: ", best_minibatch_loss.item())
# save best model
torch.save(gflownet.backbone.state_dict(), model_level_dir + f"{dataset_name}_gflownet_backbone_target{target}_LAST.pt")