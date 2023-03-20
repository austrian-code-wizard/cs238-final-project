import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

import mdp
from tqdm import tqdm_notebook
import numpy as np
from collections import deque
#discount factor for future utilities
DISCOUNT_FACTOR = 0.99

#number of episodes to run
NUM_EPISODES = 1000

#max steps per episode
MAX_STEPS = 10000

#score agent needs for environment to be solved
SOLVED_SCORE = 195

HORIZON = 5 * 12 * 8
UNITS_TO_SELL = 250
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

#device to run model on 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):
    
    #Takes in observations and outputs actions
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(9, 128)
        self.output_layer = nn.Linear(128, 10)
    
    #forward pass
    def forward(self, x):
        #input states
        x = self.input_layer(x)
        print(x)
        #relu activation
        x = F.relu(x)
        
        #actions
        actions = self.output_layer(x)
        print(actions)
        #get softmax for a probability distribution
        action_probs = F.softmax(actions, dim=1)
        
        return action_probs
#Using a neural network to learn state value
class StateValueNetwork(nn.Module):
    
    #Takes in state
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()
        
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, 1)
        
    def forward(self, x):
        #input layer
        x = self.input_layer(x)
        
        #activiation relu
        x = F.relu(x)
        
        #get state value
        state_value = self.output_layer(x)
        
        return state_value
def select_action(network, state):
    ''' Selects an action given current state
    Args:
    - network (Torch NN): network to process state
    - state (Array): Array of action space in an environment
    
    Return:
    - (int): action that is selected
    - (float): log probability of selecting that action given state and network
    '''
    
    #convert state to float tensor, add 1 dimension, allocate tensor on device
    def parse_state(state):
        data = torch.FloatTensor(np.stack([
            state["low"].to_numpy(),
            state["high"].to_numpy(),
            state["close"].to_numpy(),
            state["open"].to_numpy(),
            state["volume"].to_numpy(),
        ])).T
        return torch.concat([
            data,
            torch.repeat_interleave(torch.FloatTensor([[state["units_sold"]]]), 6, 0),
            torch.repeat_interleave(torch.FloatTensor([[state["cost_basis"]]]), 6, 0),
            torch.repeat_interleave(torch.FloatTensor([[state["steps_left"]]]), 6, 0),
            torch.repeat_interleave(torch.FloatTensor([[state["time"]]]), 6, 0)
        ], dim=1)[..., -1, :]

    state = parse_state(state)
    
    #use network to predict action probabilities
    action_probs = network(state)
    state = state.detach()
    
    #sample an action using the probability distribution
    m = Categorical(action_probs)
    action = m.sample()
    print(m)
    print(m.log_prob(action))
    #return action
    return action.item(), m.log_prob(action)
#Make environment
env = mdp.TradeExecutionEnv()
trade_sizes = {0: 0, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 250}
env = mdp.DiscreteTradeSizeWrapper(env, trade_sizes)

#Init network
policy_network = PolicyNetwork(9, 10).to(DEVICE)
stateval_network = StateValueNetwork(9).to(DEVICE)

#Init optimizer
policy_optimizer = optim.SGD(policy_network.parameters(), lr=0.001)
stateval_optimizer = optim.SGD(stateval_network.parameters(), lr=0.001)
#track scores
scores = []

writer = SummaryWriter('logs/ac')

#track recent scores
recent_scores = deque(maxlen = 100)

#run episodes
for episode in range(NUM_EPISODES):
    
    #init variables
    state = env.reset(UNITS_TO_SELL, HORIZON, GLOBAL_SEED)
    done = False
    score = 0
    I = 1
    
    #run episode, update online
    for step in range(MAX_STEPS):
        
        #get action and log probability
        action, lp = select_action(policy_network, state)
        
        #step with action
        new_state, reward, done, _,_ = env.step(action)
        
        #update episode score
        score += reward
        
        #get state value of current state
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        state_val = stateval_network(state_tensor)
        
        #get state value of next state
        new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(0).to(DEVICE)        
        new_state_val = stateval_network(new_state_tensor)
        
        #if terminal state, next state val is 0
        if done:
            new_state_val = torch.tensor([0]).float().unsqueeze(0).to(DEVICE)
        
        #calculate value function loss with MSE
        val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val, state_val)
        val_loss *= I
        
        #calculate policy loss
        advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
        policy_loss = -lp * advantage
        policy_loss *= I
        
        #Backpropagate policy
        policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        policy_optimizer.step()
        
        #Backpropagate value
        stateval_optimizer.zero_grad()
        val_loss.backward()
        stateval_optimizer.step()
        
        if done:
            break
            
        #move into new state, discount I
        state = new_state
        I *= DISCOUNT_FACTOR
    
    #append episode score 
    scores.append(score)
    recent_scores.append(score)
    
    #early stopping if we meet solved score goal
    if np.array(recent_scores).mean() >= SOLVED_SCORE:
        break
        
        
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# import numpy as np

# sns.set()

# plt.plot(scores)
# plt.ylabel('score')
# plt.xlabel('episodes')
# plt.title('Training score of CartPole Actor-Critic TD(0)')

# reg = LinearRegression().fit(np.arange(len(scores)).reshape(-1, 1), np.array(scores).reshape(-1, 1))
# y_pred = reg.predict(np.arange(len(scores)).reshape(-1, 1))
# plt.plot(y_pred)
# plt.show()