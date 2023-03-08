import numpy as np
from tqdm import tqdm
from mdp import TradeExecutionEnv, DiscreteTradeSizeWrapper


SEED = 42
HORIZON = 5 * 12 * 8
UNITS_TO_SELL = 250

env = TradeExecutionEnv()

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
  9: 250
}
env = DiscreteTradeSizeWrapper(env, trade_sizes)

def parse_state(state, action):
    data = np.vstack([
        state["low"].to_numpy(),
        state["high"].to_numpy(),
        state["close"].to_numpy(),
        state["open"].to_numpy(),
        state["volume"].to_numpy(),
    ]).T[-1,:]
    return np.concatenate([data, np.array([action]), np.array([state["units_sold"]]), np.array([state["cost_basis"]]), np.array([1])])

def Q(state, action, theta):
    return theta @ parse_state(state, action)

def Q_gradient(state, action):
    return parse_state(state, action)

def epsilon_greedy_policy(state, theta, num_actions, epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, num_actions)
    else:
        return np.argmax([Q(state, a, theta) for a in range(num_actions)])


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
    
    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), min(len(self.buffer), batch_size), replace=False)
        return np.asarray(self.buffer)[idx]


def evaluate_fixed_target_Q_learning(env, theta, num_episodes):
    rewards = []
    for _ in tqdm(range(num_episodes)):
        done = False
        state = env.reset(UNITS_TO_SELL, HORIZON, SEED)
        while not done:
            action = epsilon_greedy_policy(state, theta, env.action_space.n, 0)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
        rewards.append(reward)
    return np.mean(rewards)


def fixed_target_Q_learning(env, num_episodes, alpha=0.001, gamma=1, epsilon=0.1, buff_size=1000, batch_size=32):
    theta = np.zeros(9)
    theta_prime = theta.copy()
    replay_buffer = ReplayBuffer(buff_size)
    t = 0
    for _ in tqdm(range(num_episodes)):
        done = False
        state = env.reset(UNITS_TO_SELL, HORIZON, SEED)
        while not done:
            action = epsilon_greedy_policy(state, theta, env.action_space.n, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, done))
            batch = replay_buffer.sample(batch_size)
            for s, a, r, s_prime, d in batch:
                if d:
                    y = r
                else:
                    y = r + gamma * np.max([Q(s_prime, a_prime, theta_prime) for a_prime in range(env.action_space.n)])
                theta += alpha * (y - Q(s, a, theta)) * Q_gradient(s, a)
            t += 1
            if t % 100 == 0:
                theta_prime = theta.copy()
            state = next_state
    return theta

if __name__ == "__main__":
    theta = fixed_target_Q_learning(env, 2000)
    print(f"Mean reward: {evaluate_fixed_target_Q_learning(env, theta, 20)}")
    print(f"Theta = {theta}")