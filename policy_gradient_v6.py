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
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
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

episode_reward_history = []
while True:  # Run until solved
    state = env.reset(UNITS_TO_SELL, HORIZON, SEED)
    episode_reward = 0
    with tf.GradientTape() as tape:
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

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _, _= env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break


        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    episode_reward_history.append(episode_reward)
    writer.add_scalar('episode reward', episode_reward, global_step=episode_count)
    print('Ep {}\tMoving average score: {:.2f}\t'.format(episode_count, episode_reward))
    if np.mean(episode_reward_history[-3:]) >= -0.2:
            print("Done")
            break

episode_count = 0
episode_reward_history_train = []
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
        action = tf.argmax(action_probs[0]).numpy()

        # Apply the selected action in our environment
        state, reward, done, _, _= env.step(action)
        rewards_history.append(reward)
        episode_reward += reward

        if done:
            break

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with


    # Print episode information
    print('Episode {}\tReward: {:.2f}'.format(
        episode_count, episode_reward))
    episode_reward_history_train.append(episode_reward)
    # Check if task is solved (running_reward is average of 100 episodes)
    if episode_count > 5:
        print("Solved!")
        break

    episode_count += 1


print("Learning Episode Reward History: ", episode_reward_history)
print("Test Reward History: ", episode_reward_history_train)