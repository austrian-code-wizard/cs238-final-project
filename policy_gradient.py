import numpy as np
import mdp
import matplotlib.pyplot as plt

SEED = 69
HORIZON = 5 * 12 * 8
UNITS_TO_SELL = 250
np.random.seed(SEED)

def parse_state(state):
    return np.vstack([
        state["low"].to_numpy(),
        state["high"].to_numpy(),
        state["close"].to_numpy(),
        state["open"].to_numpy(),
        state["volume"].to_numpy(),
    ]).T[-1,:]

def actor_critic(alpha, gamma, n_episodes, max_grad_norm):
    loss = list()
    env = mdp.TradeExecutionEnv()
    trade_sizes = {0: 0, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 250}
    env = mdp.DiscreteTradeSizeWrapper(env, trade_sizes)

    n_states = 5
    n_actions = env.action_space.n
    actor_weights = np.random.randn(n_states, n_actions)
    critic_weights = np.random.randn(n_states, 1)

    for i_episode in range(n_episodes):
        state = env.reset(UNITS_TO_SELL, HORIZON, SEED)

        done = False
        episode_reward = 0
        t = 0

        while not done:
            processed_state = parse_state(state)

            logits = np.dot(processed_state, actor_weights)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            action = np.random.choice(np.arange(n_actions), p=probs)

            next_state, reward, done, _,_ = env.step(action)
            episode_reward += reward

            processed_next_state = parse_state(next_state)

            td_error = reward + gamma * critic_weights.T.dot(processed_next_state) - critic_weights.T.dot(processed_state)
            loss.append(td_error)
            critic_weights += alpha * td_error * processed_state[:, np.newaxis]

            if not done:
                advantage = td_error
                actor_grad = np.zeros_like(actor_weights)
                actor_grad[:, action] = advantage * processed_state
                clipped_actor_grad = np.clip(actor_grad, -max_grad_norm, max_grad_norm)
                actor_weights += alpha * clipped_actor_grad

            state = next_state
            t += 1
        print("Episode {}: Reward = {}".format(i_episode, episode_reward))
    plt.plot(range(len(loss)),loss)
    plt.show()

if __name__ == "__main__":
    actor_critic(alpha=0.1, gamma=0.99, n_episodes=200,max_grad_norm=5)
