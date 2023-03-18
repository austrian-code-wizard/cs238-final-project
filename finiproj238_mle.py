#import matplotlib.pyplot as plt    
import numpy as np
#from numpy import exp
#from scipy.special import factorial
import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D
#import statsmodels.api as sm
#from statsmodels.api import Poisson
from scipy import stats
from scipy.stats import norm
#from statsmodels.iolib.summary2 import summary_col

import mdptoolbox.mdp



UNITS_TO_SELL = 240
HORIZON = 5 * 12 * 8
SEED = 42

trade_sizes = {
  0: 0,
  1: 1,
  2: 2,
  3: 4,
  4: 8,
  5: 16,
  6: 32,
  7: 64,
  8: 128,
  9: 240
}

#states are keys in dictionaraies, converting the individual pandas slices into a numpy matrix

features = np.array([state["open"], state["high"], state["low"], state["close"], state["volume"], state["units_to_sell"] - state["units_sold"], trade_sizes[action], 1])

transition_count
states = ["open", "close", "high", "low", "volume"]

actions = [""]#discrete options of amounts to sell

#keep a general sum of the reward as we sell for each state and action pair (when we sell taking action a in state s)
#every time we sell, update transition counts --> at every step, check if we need to sell --> if we sell update the transition count from state to action to next state
#if it's a sell
# update the count for going from state, to state' update count for state to action and run equation 16.1

def epsilon_greedy_policy(state, theta, num_actions, epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, num_actions)
    else:
        return np.argmax([Q(state, a, theta) for a in range(num_actions)])

def increment_transition_counts(dict, state, action, next_state):
    #create an item for the specific transition
    if (state, action) not in transition_prob_dict.keys():
        dict[(state, action)] = np.zeros(3)
    
    #increment the value

    transition_num =  

def MDP_transiion_counts(env, num_episodes, epsilon=0.1, buff_size=1000, batch_size=32):
    state = env.reset(UNITS_TO_SELL, HORIZON, SEED) 
    #initialize dictionary
    transition_prob_dict = {}
    #initialize dictionary with all the state, action pairs (tuple) and all the possible states as a vector)
    while not done:
        action == epsilon_greedy_policy(state, theta, env.action_space.n, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.add((state, action, reward, next_state, done))
        batch = replay_buffer.sample(batch_size)
        rewards += reward
        increment_transition_counts(transition_prob_dict, state, action, next_state)




#if action == sell:
#increment reward by the selling price
#increment transition count for state, with action, to state'
#increment transition count for state with action
#run equation 16.1
#def lookahead(MDP, s, a):
#    stateSpace, valFunction =  
#    n = sum()


