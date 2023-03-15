import os
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from mdp import TradeExecutionEnv, DiscreteTradeSizeWrapper

SEED = 42
HORIZON = 5 * 12 * 8
UNITS_TO_SELL = 240

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
  9: 240
}
env = DiscreteTradeSizeWrapper(env, trade_sizes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_size = 8
action_size = env.action_space.n
lr = 0.0001

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, action_size)

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
        ], dim=1)[..., -1, :]
    
    def forward(self, state):
        state = torch.stack([self._parse_state(state)] if isinstance(state, dict) else [self._parse_state(s) for s in state])
        state = state.to(device)
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)
    
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
        ], dim=1)[..., -1, :]

    def forward(self, state):
        state = torch.stack([self._parse_state(state)] if isinstance(state, dict) else [self._parse_state(s) for s in state])
        state = state.to(device)
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value
    

def parse_state(state):
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
        ], dim=1)[..., -1, :]

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
        state = env.reset(UNITS_TO_SELL, HORIZON, SEED)
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        # env.reset()

        for i in count():
            # env.render()
            # state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            next_state, reward, done, _, _ = env.step(action.item())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print(i)
                print('Iteration: {}, Reward: {}'.format(iter, sum(rewards)))
                break


        # next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()


if __name__ == '__main__':
    # if os.path.exists('model/actor.pkl'):
    #     actor = torch.load('model/actor.pkl')
    #     print('Actor Model loaded')
    # else:
    actor = Actor(state_size, action_size).to(device)
    # if os.path.exists('model/critic.pkl'):
    #     critic = torch.load('model/critic.pkl')
    #     print('Critic Model loaded')
    # else:
    critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, n_iters=1000)