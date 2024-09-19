import codes.gflownet.utils as utils
import torch
from torch_geometric.data import Data


### ACTION CLASSES ###
class Action():
    def __init__(self, action_type):
        self.type = action_type

class MutagAction(Action):
    def __init__(self, action_type, start, end):
        super(MutagAction, self).__init__(action_type)
        
        self.start = start
        self.end = end     

class BATree4MotifsAction(Action):
    def __init__(self, action_type, start, end):
        super(BATree4MotifsAction, self).__init__(action_type)
        
        self.start = start
        self.end = end        

### STATE CLASSES ###
class State():
    def __init__(self, value):
        self.value = value
    
    def size(self):
        return self.value.size()
        
class MutagState(State):
    def __init__(self, value):
        super(MutagState, self).__init__(value)
    
    def size(self):
        return self.value.x.size(0)

class BATree4MotifsState(State):
    def __init__(self, value):
        super(BATree4MotifsState, self).__init__(value)
    
    def size(self):
        return self.value.x.size(0)


### REWARD FUNCTIONS ###
def class_prob_reward(gnn_model, graph, target: int, alpha: float=1.0,  threshold: float=0.5):
    data = Data(x=graph.x, edge_index=graph.edge_index, batch=graph.batch)
    pred = gnn_model(data)[1].squeeze(0)
    if torch.argmax(pred) == target:
        return alpha * pred[target]
    else:
        return torch.tensor(0.0)
    
                
### ENVIRONMENT CLASSES ###
class BaseEnvironment():
    def __init__(self, env_name, reward_fn, gnn_model, config):
        self.env_name = env_name
        self.reward_fn = reward_fn
        self.gnn_model = gnn_model
        self.config = config
    
    def new(self) -> State:
        pass
    
    def step(self, state: State, action: Action):
        pass
    
    
class Environment(BaseEnvironment):
    def __init__(self, env_name, reward_fn, gnn_model, config):
        super(Environment, self).__init__(env_name, reward_fn, gnn_model, config)
            
        self.state_type = "MutagGraph"
        self.node_feature_size = config['node_feature_size']
        self.alpha = config['alpha']
        self.threshold = config['threshold']
        self.target = config['target']
        
        self.action_space = self._init_action_space()
        
        
    def new(self, start_idx=-1) -> MutagState:
        return MutagState(value=utils.get_init_state_graph(self.node_feature_size, start_idx))
    
    def step(self, state: MutagState, action: MutagAction):
        new_state, valid, stop = self._take_action_mutag(state.value, (action.start, action.end))
        new_state = MutagState(new_state)
        #G = new_state.value
        reward = self.calculate_reward(new_state)
        return new_state, reward, valid, stop
    
    def calculate_reward(self, state: MutagState):
        return self.reward_fn(self.gnn_model, state.value.to(self.config['device']), self.target, self.alpha, self.threshold)
        
    def _init_action_space(self):
        # Stop action is the first node in candidate list
        x = Data(
            x=torch.zeros(self.node_feature_size + 1, self.node_feature_size), # +1 adds stop action
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )

        for i in range(1, self.node_feature_size + 1):
            x.x[i, i-1] = 1
        return x

    def _take_action_mutag(self, G, action):
        # Takes an action in the form (starting_node, ending_node) and 
        # returns the new graph, whether the action is valid, and whether the action is a stop action
        start, end = action

        G_new = G.clone()
        
        if start == end:
            return G, False, False

        # If end node is stop action, return graph
        if end == G.x.size(0):
            return G, True, True

        # If end node is new candidate, add it to the graph
        if end > G.x.size(0): # changed from end > G.x.size(0) - 1 because now stop action is G.x.size(0)
            # Create new node
            candidate_idx = end - G.x.size(0) - 1
            new_feature = torch.zeros(1, G_new.x.size(1)).to(G_new.x.device)
            new_feature[0, candidate_idx] = 1
            G_new.x = torch.cat([G_new.x, new_feature], dim=0)
            G_new.y = torch.cat([G_new.y, torch.zeros((1, 1)).to(G_new.y.device)], dim=0)
            G_new.y[G_new.x.size(0)-1] = candidate_idx 
            end = G_new.x.size(0) - 1

        if G_new.edge_index.size(1)!=0 and G_new.edge_index.max() >= G_new.x.size(0):
            print("G.edge_index.max()", G_new.edge_index.max(), "G.x.size(0)", G_new.x.size(0))

        # Check if edge already exists
        if utils.check_edge(G_new.edge_index, torch.tensor([[start], [end]]).to(G_new.edge_index.device)):
            # If edge exists, return original G 
            return G, False, False
        else:
            # Add edge from start to end
            G_new.edge_index = utils.append_edge(G_new.edge_index, torch.tensor([[start], [end]]).to(G_new.edge_index.device))
            G_new.edge_index = utils.append_edge(G_new.edge_index, torch.tensor([[end], [start]]).to(G_new.edge_index.device))
        
        if G_new.edge_index.max() >= G_new.num_nodes:
            return G, False, False
        else:
            return G_new, True, False
        
    
class BATree4MotifsEnvironment(Environment):
    def __init__(self, env_name, reward_fn, proxy, config):
        super(BATree4MotifsEnvironment, self).__init__(env_name, reward_fn, proxy, config)
            
        self.state_type = "BATree4MotifsGraph"
        self.node_feature_size = config['node_feature_size']
        self.alpha = config['alpha']
        self.threshold = config['threshold']
        self.target = config['target']
        
        self.action_space = self._init_action_space()
        
    def new(self, start_idx=-1) -> BATree4MotifsState:
        return BATree4MotifsState(value=utils.get_init_state_graph(self.node_feature_size, start_idx))
    
    def step(self, state: BATree4MotifsState, action: BATree4MotifsAction):
        new_state, valid, stop = self._take_action_BATree4Motifs(state.value, (action.start, action.end))
        new_state = BATree4MotifsState(new_state)
        reward = self.calculate_reward(new_state)
        return new_state, reward, valid, stop
    
    def calculate_reward(self, state: BATree4MotifsState):
        return self.reward_fn(self.proxy, state.value.to(self.config['device']), self.target, self.alpha, self.threshold)
        
    def _init_action_space(self):
        # Stop action is the first node in candidate list
        x = Data(
            x=torch.zeros(self.node_feature_size + 1, self.node_feature_size), # +1 adds stop action
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        for i in range(1, self.node_feature_size + 1):
            x.x[i, i-1] = 1
        return x

    def _take_action_BATree4Motifs(self, G, action):
        # Takes an action in the form (starting_node, ending_node) and 
        # returns the new graph, whether the action is valid, and whether the action is a stop action
        start, end = action
        G_new = G.clone()
        
        if start == end:
            return G, False, False

        # If end node is stop action, return graph
        if end == G.x.size(0):
            return G, True, True

        # If end node is new candidate, add it to the graph
        if end > G.x.size(0): # changed from end > G.x.size(0) - 1 because now stop action is G.x.size(0)
            # Create new node
            candidate_idx = end - G.x.size(0) - 1
            new_feature = torch.zeros(1, G_new.x.size(1)).to(G_new.x.device)
            new_feature[0, candidate_idx] = 1
            G_new.x = torch.cat([G_new.x, new_feature], dim=0)
            G_new.y = torch.cat([G_new.y, torch.zeros((1, 1)).to(G_new.y.device)], dim=0)
            G_new.y[G_new.x.size(0)-1] = candidate_idx 
            end = G_new.x.size(0) - 1

        if G_new.edge_index.size(1)!=0 and G_new.edge_index.max() >= G_new.x.size(0):
            print("G.edge_index.max()", G_new.edge_index.max(), "G.x.size(0)", G_new.x.size(0))

        # Check if edge already exists
        if utils.check_edge(G_new.edge_index, torch.tensor([[start], [end]]).to(G_new.edge_index.device)):
            #print("111")
            # If edge exists, return original G 
            return G, False, False
        else:
            #print("222")
            # Add edge from start to end
            G_new.edge_index = utils.append_edge(G_new.edge_index, torch.tensor([[start], [end]]).to(G_new.edge_index.device))
        
        if G_new.edge_index.max() >= G_new.num_nodes:
            #print("555")
            return G, False, False
        else:
            return G_new, True, False
            