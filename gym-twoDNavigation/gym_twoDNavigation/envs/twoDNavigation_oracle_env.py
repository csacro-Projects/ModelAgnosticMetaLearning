# The environment implementation is based on https://github.com/cbfinn/maml_rl/ (point_env_randgoal_oracle.py).

import numpy as np
from gym import spaces

from gym_twoDNavigation.envs.twoDNavigation_env import TwoDNavigationEnv


class TwoDNavigationOracleEnv(TwoDNavigationEnv):

    def __init__(self):
        super(TwoDNavigationOracleEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

    def step(self, action):
        next_observation, reward, done, self.goal = super(TwoDNavigationOracleEnv, self).step(action)
        return np.r_[next_observation, np.copy(self.goal)], reward, done, {}

    def reset(self, reset_goal=None):
        observation = super(TwoDNavigationOracleEnv, self).reset(reset_goal)
        return np.r_[observation, np.copy(self.goal)]
