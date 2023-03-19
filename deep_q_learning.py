"""Dueling DQN taken from here: https://github.com/gouxiangchen/dueling-DQN-pytorch/blob/master/dueling_dqn.py"""

import torch
from torch import nn
import numpy as np
import random
from collections import deque
from itertools import count
import torch.nn.functional as F
from mdp import TradeExecutionEnv, DiscreteTradeSizeWrapper
from tensorboardX import SummaryWriter


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64, 256)
        self.fc_adv = nn.Linear(64, 256)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, action_size)

    def forward(self, state):
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()


class QRNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QRNNetwork, self).__init__()

        self.rnn = nn.LSTM(state_size, 32, 1, batch_first=True)
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64, 64)
        self.fc_adv = nn.Linear(64, 64)

        self.value = nn.Linear(64, 1)
        self.adv = nn.Linear(64, action_size)

    def _parse_state(self, state):
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
        ], dim=1)

    def forward(self, state):
        state = torch.stack([self._parse_state(state)] if isinstance(state, dict) else [self._parse_state(s) for s in state])
        state = state.to(device)
        y, _ = self.rnn(state)
        y = self.relu(self.fc1(y[:, -1, :]))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))
        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage
        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


onlineQNetwork = QRNNetwork(len(env.observation_space), len(trade_sizes)).to(device)
targetQNetwork = QRNNetwork(len(env.observation_space), len(trade_sizes)).to(device)
targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)
writer = SummaryWriter('logs/dqn')

GAMMA = 0.99
EXPLORE = 4000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 20000
BATCH = 16
EPOCHS = 5000
EVAL_EPOCHS = 100

UPDATE_STEPS = 4

memory_replay = Memory(REPLAY_MEMORY)

epsilon = INITIAL_EPSILON
learn_steps = 0
begin_learn = False

episode_reward = 0

# onlineQNetwork.load_state_dict(torch.load('ddqn-policy.para'))
for epoch in range(EPOCHS):

    state = env.reset(UNITS_TO_SELL, HORIZON, SEED)
    episode_reward = 0
    done = False
    while not done:
        p = random.random()
        if p < epsilon:
            action = random.randint(0, 1)
        else:
            action = onlineQNetwork.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        memory_replay.add((state, next_state, action, reward, done))
        if memory_replay.size() > 128:
            if begin_learn is False:
                print('learn begin!')
                begin_learn = True
            learn_steps += 1
            if learn_steps % UPDATE_STEPS == 0:
                targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
            batch = memory_replay.sample(BATCH, False)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

            with torch.no_grad():
                onlineQ_next = onlineQNetwork(batch_next_state)
                targetQ_next = targetQNetwork(batch_next_state)
                online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
                y = batch_reward + (1 - batch_done) * GAMMA * targetQ_next.gather(1, online_max_action.long())

            loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), global_step=learn_steps)

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        if done:
            break
        state = next_state
    writer.add_scalar('episode reward', episode_reward, global_step=epoch)
    if epoch % 10 == 0:
        torch.save(onlineQNetwork.state_dict(), 'ddqn-policy.para')
        print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))

rewards = []
for epoch in range(EVAL_EPOCHS):
    state = env.reset(UNITS_TO_SELL, HORIZON, SEED)
    episode_reward = 0
    done = False
    while not done:
        action = onlineQNetwork.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
        state = next_state
    rewards.append(episode_reward)
print('Average reward: {}'.format(np.mean(rewards)))