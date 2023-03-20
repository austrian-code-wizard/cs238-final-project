import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorboardX import SummaryWriter
from tensorflow.keras import layers

import mdp

SEED = 42
HORIZON = 5 * 12 * 8
UNITS_TO_SELL = 250

# Configuration parameters for the whole setup
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = mdp.TradeExecutionEnv()
trade_sizes = {0: 0, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 250}
env = mdp.DiscreteTradeSizeWrapper(env, trade_sizes)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 9
num_actions = 10
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax", trainable=False)(common)
critic = layers.Dense(1, trainable=False)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

def parse_state(state):
    data = np.vstack([
        state["low"].to_numpy(),
        state["high"].to_numpy(),
        state["close"].to_numpy(),
        state["open"].to_numpy(),
        state["volume"].to_numpy(),
    ]).T[-1,:]
    return np.concatenate([data, np.array([state["units_sold"]]), np.array([state["cost_basis"]]),np.array([state["time"]]), np.array([1])])

writer = SummaryWriter('logs/ac')

while True:  # Run until solved
    state = env.reset(UNITS_TO_SELL, HORIZON, SEED)
    episode_reward = 0
    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.

        state = parse_state(state)
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        # Predict action probabilities and estimated future rewards
        # from environment state
        action_probs, critic_value = model(state)
        critic_value_history.append(critic_value[0, 0])

        # Choose action greedily
        print(action_probs)
        action = tf.argmax(action_probs[0]).numpy()

        # Apply the selected action in our environment
        state, reward, done, _, _= env.step(action)
        rewards_history.append(reward)
        episode_reward += reward

        if done:
            break

    # Update running reward to check condition for solving
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with


    # Print episode information
    print('Episode {}\tReward: {:.2f}'.format(
        episode_count, episode_reward))

    # Check if task is solved (running_reward is average of 100 episodes)
    if running_reward > 200:
        print("Solved!")
        break

    episode_count += 1

