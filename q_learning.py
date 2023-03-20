import json
import numpy as np
from tqdm import tqdm
from mdp import TradeExecutionEnv, DiscreteTradeSizeWrapper


SEED = 42
HORIZON = 5 * 12 * 2
UNITS_TO_SELL = 64
ALPHA=0.001
GAMMA=0.99
EPSILON=0.1
BUFF_SIZE=20000
BATCH_SIZE=32
EPOCHS=600
EVAL_EPOCHS = 50
EXP_NAME = "q_learning_multi_mdp_train.json"

env = TradeExecutionEnv()

trade_sizes = {
  i: i*2 for i in range(33)
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
    return np.concatenate([data, np.array([2*(action / len(trade_sizes)) - 1]), np.array([state["steps_left"]]), np.array([state["units_sold"]]), np.array([state["cost_basis"]]), np.array([1])])

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



theta = np.zeros(len(env.observation_space) + 2)
theta_prime = theta.copy()
replay_buffer = ReplayBuffer(BUFF_SIZE)
t = 0
train_rewards = []
for e in tqdm(range(EPOCHS)):
    done = False
    state = env.reset(UNITS_TO_SELL, HORIZON, SEED+e)
    episode_rewards = 0
    while not done:
        action = epsilon_greedy_policy(state, theta, env.action_space.n, EPSILON)
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.add((state, action, reward, next_state, done))
        t += 1
        if t % 100 == 0:
            theta_prime = theta.copy()
        state = next_state
        episode_rewards += reward
    batch = replay_buffer.sample(BATCH_SIZE)
    for s, a, r, s_prime, d in batch:
        if d:
            y = r
        else:
            y = r + GAMMA * np.max([Q(s_prime, a_prime, theta_prime) for a_prime in range(env.action_space.n)])
        theta += ALPHA * (y - Q(s, a, theta)) * Q_gradient(s, a)
    train_rewards.append(episode_rewards)
    if e % 10 == 0:
        print('Epoch: {}, Average reward: {}'.format(e, np.mean(train_rewards[-10:])))

same_mdp_eval_rewards = []
for epoch in range(EVAL_EPOCHS):
    state = env.reset(UNITS_TO_SELL, HORIZON, SEED)
    episode_reward = 0
    done = False
    while not done:
        action = np.argmax([Q(state, a, theta) for a in range(env.action_space.n)])
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
        state = next_state
    same_mdp_eval_rewards.append(episode_reward)
print('Average reward: {}'.format(np.mean(same_mdp_eval_rewards)))

train_mdp_eval_rewards = []
for epoch in range(EVAL_EPOCHS):
    state = env.reset(UNITS_TO_SELL, HORIZON, SEED+epoch)
    episode_reward = 0
    done = False
    while not done:
        action = np.argmax([Q(state, a, theta) for a in range(env.action_space.n)])
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
        state = next_state
    train_mdp_eval_rewards.append(episode_reward)
print('Average reward: {}'.format(np.mean(train_mdp_eval_rewards)))

test_mdp_eval_rewards = []
for epoch in range(EVAL_EPOCHS):
    state = env.reset(UNITS_TO_SELL, HORIZON, SEED+epoch, test=True)
    episode_reward = 0
    done = False
    while not done:
        action = np.argmax([Q(state, a, theta) for a in range(env.action_space.n)])
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
        state = next_state
    test_mdp_eval_rewards.append(episode_reward)
print('Average reward: {}'.format(np.mean(test_mdp_eval_rewards)))

with open(f"./results/{EXP_NAME}", "w+") as f:
    json.dump({
        "train_rewards": train_rewards,
        "same_mdp_eval_rewards": same_mdp_eval_rewards,
        "train_mdp_eval_rewards": train_mdp_eval_rewards,
        "test_mdp_eval_rewards": test_mdp_eval_rewards
    }, f)