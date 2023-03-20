import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorboardX import SummaryWriter
from tensorflow.keras import layers
from mdp import TradeExecutionEnv, DiscreteTradeSizeWrapper

SEED = 42
HORIZON = 5 * 12 * 2
UNITS_TO_SELL = 64
BATCH_SIZE = 32
EPOCHS = 1000
EVAL_EPOCHS = 50
EPSILON = 0.1
GAMMA = 0.99

env = TradeExecutionEnv()

trade_sizes = {
  i: i*2 for i in range(33)
}
env = DiscreteTradeSizeWrapper(env, trade_sizes)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = len(env.observation_space)
num_actions = len(trade_sizes)
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
    return np.concatenate([data, np.array([state["steps_left"]]), np.array([state["units_sold"]]), np.array([state["cost_basis"]])])

writer = SummaryWriter('logs/ac')

episode_reward_history = []
for e in range(EPOCHS):  # Run until solved
    state = env.reset(UNITS_TO_SELL, HORIZON, (BATCH_SIZE * EPOCHS)+e)
    episode_reward = 0
    with tf.GradientTape() as tape:
        done = False
        while not done:
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


        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + GAMMA * discounted_sum
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

episode_count = 0
reward_same_mdp = []
for e in range(EVAL_EPOCHS):  # Run until solved
    state = env.reset(UNITS_TO_SELL, HORIZON, SEED)
    episode_reward = 0
    done = False
    while not done:
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

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with


    # Print episode information
    reward_same_mdp.append(episode_reward)
    # Check if task is solved (running_reward is average of 100 episodes)

    episode_count += 1
print(f"Mean test reward: {np.mean(reward_same_mdp)}")


episode_count = 0
reward_train_mdp = []
for e in range(EVAL_EPOCHS):  # Run until solved
    state = env.reset(UNITS_TO_SELL, HORIZON, seed=(BATCH_SIZE * EPOCHS)+e)
    episode_reward = 0
    done = False
    while not done:
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

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with


    # Print episode information
    reward_train_mdp.append(episode_reward)
    # Check if task is solved (running_reward is average of 100 episodes)

    episode_count += 1
print(f"Mean test reward: {np.mean(reward_train_mdp)}")


episode_count = 0
reward_test_mdp = []
for e in range(EVAL_EPOCHS):  # Run until solved
    state = env.reset(UNITS_TO_SELL, HORIZON, seed=SEED+e, test=True)
    episode_reward = 0
    done = False
    while not done:
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

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with


    # Print episode information
    reward_test_mdp.append(episode_reward)
    # Check if task is solved (running_reward is average of 100 episodes)

    episode_count += 1
print(f"Mean test reward: {np.mean(reward_test_mdp)}")


import json

EXP_NAME = "actor_critic_meta_mdp_train.json"
with open(f"./results/{EXP_NAME}", "w+") as f:
    json.dump({
        "train_rewards": episode_reward_history,
        "same_mdp_eval_rewards": reward_same_mdp,
        "train_mdp_eval_rewards": reward_train_mdp,
        "test_mdp_eval_rewards": reward_test_mdp
    }, f)