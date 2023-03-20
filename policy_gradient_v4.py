
import numpy as np
import mdp
import matplotlib.pyplot as plt

SEED = 42
HORIZON = 5 * 12 * 8
UNITS_TO_SELL = 250

# def parse_state(state):
#     return np.vstack([
#         state["low"].to_numpy(),
#         state["high"].to_numpy(),
#         state["close"].to_numpy(),
#         state["open"].to_numpy(),
#         state["volume"].to_numpy(),
#     ]).T[-1,:]

def parse_state(state):
    data = np.vstack([
        state["low"].to_numpy(),
        state["high"].to_numpy(),
        state["close"].to_numpy(),
        state["open"].to_numpy(),
        state["volume"].to_numpy(),
    ]).T[-1,:]
    return np.concatenate([data, np.array([state["units_sold"]]), np.array([state["cost_basis"]]),np.array([state["time"]]), np.array([1])])


def actor_critic(alpha, gamma, n_episodes, max_grad_norm):
    env = mdp.TradeExecutionEnv()
    trade_sizes = {0: 0, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 250}
    env = mdp.DiscreteTradeSizeWrapper(env, trade_sizes)

    n_states = 9
    n_actions = env.action_space.n
    actor_weights = np.random.randn(n_states, n_actions)
    critic_weights = np.random.randn(n_states, 1)

    def log_prob(action, logits):
        return logits[action] - np.log(np.sum(np.exp(logits)))

    def grad_log_prob(action, probs):
        grad = -probs.reshape(-1, 1)
        grad[action] += 1
        return grad
    
    for i_episode in range(n_episodes):
        state = env.reset(UNITS_TO_SELL, HORIZON, SEED)

        done = False
        episode_reward = 0
        t = 0

        observations = []
        actions = []
        rewards = []
        probs = []

        while not done:
            observation = parse_state(observation)
            observations.append(observation)
            
            action, prob = policy.act(observation)
            observation, reward, done, _,_ = env.step(action)
            
            totalreward += reward
            rewards.append(reward)
            actions.append(action)
            probs.append(prob)

    # generate policy from actor weights
    policy = lambda state: np.argmax(np.dot(parse_state(state), actor_weights))
    # print(policy)


    total_reward = 0
    n_runs = n_episodes
    for i in range(n_runs):
        state = env.reset(UNITS_TO_SELL, HORIZON, SEED)
        done = False
        print(i)
        while not done:
            action = policy(state)
            state, reward, done, _,_ = env.step(action)
            total_reward += reward

    expected_reward = total_reward / n_runs
    print("Expected reward on training set: {}".format(expected_reward))

    return actor_weights, critic_weights


if __name__ == "__main__":
    actor_critic(alpha=0.05, gamma=0.99, n_episodes=100,max_grad_norm=100)
