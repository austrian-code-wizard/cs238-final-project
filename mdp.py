import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box

class TradeExecutionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path: str = "./data/5m_intraday_data.csv"):
        self._data = pd.read_csv(data_path)
        self.action_space = Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        self.observation_space = {
            "open": Box(low=0, high=np.inf, shape=(1,)),
            "high": Box(low=0, high=np.inf, shape=(1,)),
            "low": Box(low=0, high=np.inf, shape=(1,)),
            "close": Box(low=0, high=np.inf, shape=(1,)),
            "volume": Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            "units_to_sell": Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            "units_sold": Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            "total_income": Box(low=0, high=np.inf, shape=(1,))
        }
        self.units_sold = None
        self.units_to_sell = None
        self.total_income = None
        self.horizon = None
        self.current_step = None

    
    def reset(self, units_to_sell, horizon, seed=0):
        self.units_to_sell = units_to_sell
        self.units_sold = 0
        self.total_income = 0
        np.random.seed(seed)
        self._start = np.random.randint(0, len(self._data) - horizon)
        self.current_step = self._start
        self.horizon = horizon + self.current_step
        return self._get_observation()

    def step(self, action):
        if isinstance(action, int):
            action = np.array([action]).astype(np.int32)
        assert self.current_step is not None, "You must call reset() before calling step()"
        assert self.action_space.contains(action)
        self.current_step += 1
        if self.current_step >= self.horizon:
            done = True
        else:
            done = False
        
        self.units_sold += int(action)
        self.total_income += int(action) * self._data.iloc[self.current_step]["Close"]
        if self.units_sold >= self.units_to_sell:
            done = True
        return self._get_observation(), self._get_reward(done), done, False, {}

    def _get_observation(self):
        return {
            "open": self._data.iloc[self.current_step]["Open"],
            "high": self._data.iloc[self.current_step]["High"],
            "low": self._data.iloc[self.current_step]["Low"],
            "close": self._data.iloc[self.current_step]["Close"],
            "volume": self._data.iloc[self.current_step]["Volume"],
            "units_to_sell": self.units_to_sell,
            "units_sold": self.units_sold,
            "total_income": self.total_income
        }

    def _get_reward(self, done):
        if not done: return 0
        if self.units_sold != self.units_to_sell:
            return -10
        max_outcome = self.units_to_sell * self._data.iloc[self._start+1:self.horizon+1]["Close"].max()
        min_outcome = self.units_to_sell * self._data.iloc[self._start+1:self.horizon+1]["Close"].min()
        return -(max_outcome - self.total_income) / (max_outcome - min_outcome)