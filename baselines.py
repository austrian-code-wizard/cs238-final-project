import numpy as np
from mdp import TradeExecutionEnv

SEED = 42
HORIZON = 5 * 12 * 2
UNITS_TO_SELL = 64
EVAL_EPOCHS = 50
EPOCHS = 300
BATCH_SIZE = 32


def average_policy(env):
    done = False
    reward = 0
    i = 0
    rewards = []
    for i in range(EVAL_EPOCHS):
        env.reset(UNITS_TO_SELL, HORIZON, SEED+i, test=True)
        episode_reward = 0
        while not done:
            # Assumes that horizon is divisible by units_to_sell
            action = 1 if i % (HORIZON // UNITS_TO_SELL) == 0 else 0
            i += 1
            _, reward, done, _, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)


def random_policy(env):
    rewards = []

    # Iterate over 100 episodes to get the average reward
    for i in range(EVAL_EPOCHS):
        done = False
        reward = 0
        obs = env.reset(UNITS_TO_SELL, HORIZON, SEED+i, test=True)
        episode_rewards = 0
        while not done:
            np.random.seed(i)
            action = np.random.randint(0, env.units_to_sell - env.units_sold +1)

            # Force random policy to sell all units
            if env.current_step + 1 >= env.horizon:
                action = env.units_to_sell - env.units_sold
            obs, reward, done, _, _ = env.step(action)
            episode_rewards += reward
        rewards.append(episode_rewards)
    return np.mean(rewards)


def optimal_policy(_):
    return 0


def worst_policy(_):
    return -1


if __name__ == "__main__":
    env = TradeExecutionEnv()
    print("Average policy reward: ", average_policy(env))
    print("Random policy reward: ", random_policy(env))
    print("Optimal policy reward: ", optimal_policy(env))
    print("Worst policy reward: ", worst_policy(env))