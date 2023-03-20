import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box

class TradeExecutionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path: str = "./data/5m_intraday_data.csv"):
        self._data = pd.read_csv(data_path).interpolate()
        self.action_space = Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        self.observation_space = {
            "open": Box(low=0, high=np.inf, shape=(6,)),
            "high": Box(low=0, high=np.inf, shape=(6,)),
            "low": Box(low=0, high=np.inf, shape=(6,)),
            "close": Box(low=0, high=np.inf, shape=(6,)),
            "volume": Box(low=0, high=np.inf, shape=(6,)),
            "units_sold": Box(low=0, high=np.inf, shape=(1,)),
            "cost_basis": Box(low=0, high=np.inf, shape=(1,)),
            "steps_left": Box(low=0, high=np.inf, shape=(1,)),
            "time": Box(low=0, high=np.inf, shape=(1,))
        }
        self.units_sold = None
        self.units_to_sell = None
        self.total_income = None
        self.horizon = None
        self.current_step = None
        self.max_price = max(self._data["High"].max(), self._data["Low"].max(), self._data["Open"].max(), self._data["Close"].max())
        self.min_price = min(self._data["High"].min(), self._data["Low"].min(), self._data["Open"].min(), self._data["Close"].min())
        self.max_volume = self._data["Volume"].max()
        self.min_volume = self._data["Volume"].min()
        self.min_time = self._data["Timestamp"].min()
        self.max_time = self._data["Timestamp"].max()
        self.time = None

    
    def reset(self, units_to_sell, horizon, seed=0):
        self.units_to_sell = units_to_sell
        self.units_sold = 0
        self.total_income = 0
        np.random.seed(seed)
        self._start = np.random.randint(6, len(self._data) - horizon)
        self.current_step = self._start
        self.horizon = horizon + self.current_step
        self.max_price = max(self._data.iloc[int(self._start-6):int(self.horizon+1)]["High"].max(), self._data.iloc[int(self._start-6):int(self.horizon+1)]["Low"].max(), self._data.iloc[int(self._start-6):int(self.horizon+1)]["Open"].max(), self._data.iloc[int(self._start-6):int(self.horizon+1)]["Close"].max())
        self.min_price = min(self._data.iloc[int(self._start-6):int(self.horizon+1)]["High"].min(), self._data.iloc[int(self._start-6):int(self.horizon+1)]["Low"].min(), self._data.iloc[int(self._start-6):int(self.horizon+1)]["Open"].min(), self._data.iloc[int(self._start-6):int(self.horizon+1)]["Close"].min())
        self.max_volume = self._data.iloc[int(self._start-6):int(self.horizon+1)]["Volume"].max()
        self.min_volume = self._data.iloc[int(self._start-6):int(self.horizon+1)]["Volume"].min()
        self.min_time = self._data["Timestamp"].min()
        self.max_time = self._data["Timestamp"].max()
        self.time = self._data.iloc[int(self.horizon+1)]["Timestamp"]
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

    def normalize_time(self, time):
        return (time-self.min_time)/(self.max_time-self.min_time)

    def normalize_price(self, price):
        return 2 * (price - self.min_price) / (self.max_price - self.min_price) - 1

    def normalize_volume(self, volume):
        return 2 * (volume - self.min_volume) / (self.max_volume - self.min_volume) - 1

    def _get_observation(self):
        return {
            "open": self.normalize_price(self._data.iloc[self.current_step - 6 : self.current_step]["Open"]),
            "high": self.normalize_price(self._data.iloc[self.current_step - 6 : self.current_step]["High"]),
            "low": self.normalize_price(self._data.iloc[self.current_step - 6 : self.current_step]["Low"]),
            "close": self.normalize_price(self._data.iloc[self.current_step - 6 : self.current_step]["Close"]),
            "volume": self.normalize_volume(self._data.iloc[self.current_step - 6 : self.current_step]["Volume"]),
            "units_sold": self.units_sold / self.units_to_sell,
            "cost_basis": self.normalize_price(self.total_income / self.units_sold) if self.units_sold > 0 else 0,
            "steps_left": (self.horizon - self.current_step) / (self.horizon - self._start),
            "time": self.normalize_time(self._data.iloc[self.current_step]["Timestamp"])
        }

    def _get_reward(self, done):
        if not done: return 0
        if self.units_sold != self.units_to_sell:
            return -10
        max_outcome = self.units_to_sell * self._data.iloc[int(self._start+1):int(self.horizon+1)]["Close"].max()
        min_outcome = self.units_to_sell * self._data.iloc[int(self._start+1):int(self.horizon+1)]["Close"].min()
        return -(max_outcome - self.total_income) / (max_outcome - min_outcome)


class DiscreteTradeSizeWrapper(gym.Wrapper):
    def __init__(self, env, trade_sizes):
        super().__init__(env)
        self.trade_sizes = trade_sizes
        self.action_space = gym.spaces.Discrete(len(trade_sizes))

    def step(self, action):
        return self.env.step(self.trade_sizes[action])

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
