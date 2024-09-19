import torch
from torch.distributions import Categorical

from codes.gflownet.backbone import GFlowNetBackbone
from codes.gflownet.environment import *


class BaseGFlowNet(object):
    def __init__(self, backbone: GFlowNetBackbone, environment: Environment, device):
        
        self.backbone = backbone
        self.environment = environment
        self.device = device
        self.gnn_model = environment.gnn_model
        self.logZ = self.backbone.logZ
        
        self.curr_flows = None
        self.prev_flows = None
        self.total_forward_flow = torch.tensor(0.0)
        self.total_backward_flow = torch.tensor(0.0)
        
    def __call__(self, **kwargs):
        self.calculate_flow_dist(**kwargs)
    
    def new():
        pass
    
    def forward(self, **kwargs):
        return self.backbone.forward(**kwargs)

    def sample_action(self, state, edge_index, action):
        pass
    
    def take_action(self, state, edge_index, action):
        pass
    
    def calculate_flow_dist(self, state):
        pass
    
    def calculate_forward_flow(self, action, p_forward):
        pass
    
    def calculate_backward_flow(self, action, p_backward):
        pass
    
    def update_flows(self, forward_flow, backward_flow):
        pass
    
    def zero_flows(self):
        pass
    
    def generate(self, min_nodes, max_actions):
        pass
    
    def load(self, path):
        pass


class GFlowNet(BaseGFlowNet):
    def __init__(self, backbone: GFlowNetBackbone, environment: Environment, device):
        super(GFlowNet, self).__init__(backbone, environment, device)
        
    def __call__(self, state):
        return self.calculate_flow_dist(state)

    def sample_action(self, state: MutagState):
        P_forward, _ = self.calculate_flow_dist(state)
        
        P_start, P_end = P_forward
        
        # Sample start node
        start_dist = Categorical(logits=P_start)
        start = start_dist.sample()
        
        # Mask out starting node 
        mask = torch.ones_like(P_end)
        mask[start] = 0
        
        P_end = P_end * mask - 100 * (1 - mask)
        
        # Sample end node
        end_dist = Categorical(logits=P_end)
        end = end_dist.sample()
        
        return MutagAction("add_node", start, end), (P_start, P_end)
    
    def take_action(self, state, action):
        return self.environment.step(state, action)
    
    def new(self, start_idx=-1):
        return self.environment.new(start_idx)   
    
    def calculate_reward(self, state: MutagState):
        return self.environment.calculate_reward(state)
    
    def calculate_flow_dist(self, state):
        return self.backbone(state.value.x.to(self.device), state.value.edge_index.to(self.device))
    
    def calculate_forward_flow(self, action, p_forward):
        return Categorical(logits=p_forward[0]).log_prob(action.start) + Categorical(logits=p_forward[1]).log_prob(action.end)
    
    def calculate_backward_flow(self, action, p_backward):
        return Categorical(logits=p_backward[0]).log_prob(action.start) + Categorical(logits=p_backward[1]).log_prob(action.end)
    
    def update_flows(self, forward_flow, backward_flow):
        self.total_forward_flow += forward_flow
        self.total_backward_flow += backward_flow
        
    def zero_flows(self):
        self.total_forward_flow = torch.tensor(0.0).to(self.device)
        self.total_backward_flow = torch.tensor(0.0).to(self.device)
        
    def save(self, path):
        torch.save(self.backbone.state_dict(), path)
    
    def load(self, path):
        self.backbone.load_state_dict(torch.load(path))
        
    def eval(self):
        self.backbone.eval()
        
    def train(self):
        self.backbone.train()
        
    def generate(self, min_nodes, max_actions) -> Data:
        self.backbone.eval()

        actions_taken = 0
        state = self.new()
        while actions_taken < max_actions:
            
            action, _ = self.sample_action(state)
                
            state, _, _, stop = self.take_action(state, action)
            
            if stop and state.size() >= min_nodes:
                break

            actions_taken += 1
                
        return state.value
            
            
        
class BATree4Motifs_GFlowNet(GFlowNet):
    def __init__(self, backbone: GFlowNetBackbone, environment: BATree4MotifsEnvironment, device):
        super(BATree4Motifs_GFlowNet, self).__init__(backbone, environment, device)

    def __call__(self, state):
        return self.calculate_flow_dist(state)

    def sample_action(self, state: BATree4MotifsState):
        P_forward, _ = self.calculate_flow_dist(state)
        P_start, P_end = P_forward
        # Sample start node
        start_dist = Categorical(logits=P_start)
        start = start_dist.sample()
        # Mask out starting node 
        mask = torch.ones_like(P_end)
        mask[start] = 0
        P_end = P_end * mask - 100 * (1 - mask)
        # Sample end node
        end_dist = Categorical(logits=P_end)
        end = end_dist.sample()
        return BATree4MotifsAction("add_node", start, end), (P_start, P_end)

    
    def take_action(self, state, action):
        return self.environment.step(state, action)
    
    def new(self, start_idx=-1):
        return self.environment.new(start_idx)   
    
    def calculate_reward(self, state: BATree4MotifsState):
        return self.environment.calculate_reward(state)
    
    def calculate_flow_dist(self, state):
        return self.backbone(state.value.x.to(self.device), state.value.edge_index.to(self.device))
    
    def calculate_forward_flow(self, action, p_forward):
        return Categorical(logits=p_forward[0]).log_prob(action.start) + Categorical(logits=p_forward[1]).log_prob(action.end)
    
    def calculate_backward_flow(self, action, p_backward):
        return Categorical(logits=p_backward[0]).log_prob(action.start) + Categorical(logits=p_backward[1]).log_prob(action.end)
    
    def update_flows(self, forward_flow, backward_flow):
        self.total_forward_flow += forward_flow
        self.total_backward_flow += backward_flow
        
    def zero_flows(self):
        self.total_forward_flow = torch.tensor(0.0).to(self.device)
        self.total_backward_flow = torch.tensor(0.0).to(self.device)
        
    def save(self, path):
        torch.save(self.backbone.state_dict(), path)
    
    def load(self, path):
        self.backbone.load_state_dict(torch.load(path))
        
    def eval(self):
        self.backbone.eval()
        
    def train(self):
        self.backbone.train()
        
    def generate(self, min_nodes, max_actions) -> Data:
        self.backbone.eval()
        actions_taken = 0
        state = self.new()
        while actions_taken < max_actions:
            action, _ = self.sample_action(state)
            state, _, _, stop = self.take_action(state, action)
            if stop and state.size() >= min_nodes:
                break
            actions_taken += 1
        return state.value
    

    