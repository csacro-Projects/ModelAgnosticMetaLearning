# The environment implementation is based on https://github.com/cbfinn/maml_rl/ (point_env_randgoal.py).
# Additionally, we added normalization of the action space directly in the environment itself
# and implemented the 'plot' mode in the render method.

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from typing import Tuple


class TwoDNavigationEnv(gym.Env):
    metadata = {'render.modes': ['plot', 'print_only']}

    def __init__(self, goal=None, state=(0, 0)):
        self.goal = goal
        self.state = state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        self.action_space_scaling_factor = 10.0
        self.actual_action_space = spaces.Box(low=-1.0 / self.action_space_scaling_factor,
                                              high=1.0 / self.action_space_scaling_factor, shape=(2,), dtype=np.float64)
        self.figure = None
        self.prev_state = None

    def step(self, action: Tuple[float, float]):
        # scale and clip action to action_space boundaries
        action = tuple(act / self.action_space_scaling_factor for act in action)
        action = tuple(min(max(act, self.actual_action_space.low[0]), self.actual_action_space.high[0]) for act in action)
        self.state = self.state + np.array(action)
        x, y = self.state
        x -= self.goal[0]
        y -= self.goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self.state)
        return next_observation, reward, done, self.goal

    def reset(self, reset_goal=None):
        if reset_goal is not None:
            self.goal = reset_goal
        elif self.goal is None:
            self.goal = np.random.uniform(-0.5, 0.5, size=(2,))
        self.state = (0, 0)
        observation = np.copy(self.state)
        return observation

    def render(self, mode='plot'):
        if mode is 'plot':
            if self.figure is None:
                plt.ion()
                self.figure, ax = plt.subplots()
                plt.plot(self.state[0], self.state[1], 'bo')
            plt.plot(self.goal[0], self.goal[1], 'ro')
            if self.prev_state is not None:
                plt.plot([self.prev_state[0], self.state[0]], [self.prev_state[1], self.state[1]], 'b')
            self.figure.canvas.draw()
            self.prev_state = self.state
        print('current state:', self.state, 'goal:', self.goal)

    def seed(self, seed=None):
        return [0]

    @staticmethod
    def sample_goals(num_goals: int, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(-0.5, 0.5, size=(num_goals, 2,))
