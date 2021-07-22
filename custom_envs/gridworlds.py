import gym
from gym import spaces
import numpy as np


class SimpleGridworldEnv(gym.Env):
    def __init__(self, height=5, width=5):
        self.height = height
        self.width = width
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.height*self.width)
        
        # self.observation_space = spaces.Tuple((
        #         spaces.Discrete(self.height),
        #         spaces.Discrete(self.width)
        #         ))
        self.moves = {
                0: (-1, 0),  # up
                1: (0, 1),   # right
                2: (1, 0),   # down
                3: (0, -1),  # left
                }

        # begin in start state
        self.reset()

    def _one_hot(self, sample):
        index = sample[0]*self.width + sample[1]
        features = np.zeros(self.height * self.width)
        features[index]=1
        
        return features
    
    def step(self, action):

        x, y = self.moves[action]
        self.S = self.S[0] + x, self.S[1] + y

        self.S = max(0, self.S[0]), max(0, self.S[1])
        self.S = (min(self.S[0], self.height - 1),
                  min(self.S[1], self.width - 1))

        if self.S == (1, 4):
            return self._one_hot(self.S), -1, True, {}
        else:
            return self._one_hot(self.S), -1, False, {}

    def reset(self):
        self.S = (3, 0)
        return self._one_hot(self.S)


class WindyGridworldEnv(gym.Env):
    def __init__(self, height=7, width=10):
        self.height = height
        self.width = width
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.height*self.width)
        
        # self.observation_space = spaces.Tuple((
        #         spaces.Discrete(self.height),
        #         spaces.Discrete(self.width)
        #         ))
        self.moves = {
                0: (-1, 0),  # up
                1: (0, 1),   # right
                2: (1, 0),   # down
                3: (0, -1),  # left
                }

        # begin in start state
        self.reset()

    def _one_hot(self, sample):
        index = sample[0]*self.width + sample[1]
        features = np.zeros(self.height * self.width)
        features[index]=1
        
        return features
    
    def step(self, action):
        if self.S[1] in (3, 4, 5, 8):
            self.S = self.S[0] - 1, self.S[1]
        elif self.S[1] in (6, 7):
            self.S = self.S[0] - 2, self.S[1]

        x, y = self.moves[action]
        self.S = self.S[0] + x, self.S[1] + y

        self.S = max(0, self.S[0]), max(0, self.S[1])
        self.S = (min(self.S[0], self.height - 1),
                  min(self.S[1], self.width - 1))

        if self.S == (3, 7):
            return self._one_hot(self.S), -1, True, {}
        else:
            return self._one_hot(self.S), -1, False, {}

    def reset(self):
        self.S = (3, 0)
        return self._one_hot(self.S)


