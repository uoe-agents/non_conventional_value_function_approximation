'''
Code adapted from: https://github.com/podondra/gym-gridworlds
'''

import gym
from gym import spaces
import numpy as np


class SimpleGridworldEnv(gym.Env):
    '''
    A Gym environment class that represents a simple gridworld.
    
    Attributes
    ----------
    action_space: gym.Space
        action space from Gym
    observation_space: gym.Space  
        state space from Gym
    height: int
        height of the gridworld
    width: int
        width of the gridworld
    
    Methods
    -------
    step():
        Returns an environment state given an action
    reset():
        Resets the environment
    '''
    def __init__(self, height=5, width=5):
        self.height = height
        self.width = width
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.height*self.width)
        
        self.moves = {
                0: (-1, 0),  # up
                1: (0, 1),   # right
                2: (1, 0),   # down
                3: (0, -1),  # left
                }

        # begin in start state
        self.reset()

    def _one_hot(self, sample):
        '''
        Returns a one hot vector 
        
        Parameters
        ----------
        sample: object
            environment state in coordinate form
        
        Returns
        -------
        vector: np.array
            one hot vector in list format
        '''
        index = sample[0]*self.width + sample[1]
        features = np.zeros(self.height * self.width)
        features[index]=1
        
        return features
    
    def step(self, action):
        '''
        Takes as input an action and returns an environment state.
        
        Parameters
        ----------
        action: object
            represents an environment action
        
        Returns
        -------
        next_observation: tuple
            a tuple consisting of the next_state (in one-hot vector form), reward (always equals -1) and done (boolean value representing whether episode terminates) of the environment
        ''' 
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
        '''
        Resets the environment.

        Returns
        -------
        starting_state: object
            represents the starting state of the environment (in one-hot vector form)
        '''
        self.S = (3, 0)
        return self._one_hot(self.S)


class WindyGridworldEnv(gym.Env):
    '''
    A Gym environment class that represents WindyGridworld.
    
    WindyGridworld reference: http://www.incompleteideas.net/book/the-book-2nd.html
    
    Attributes
    ----------
    action_space: gym.Space
        action space from Gym
    observation_space: gym.Space  
        state space from Gym
    height: int
        height of the gridworld
    width: int
        width of the gridworld
    
    Methods
    -------
    step():
        Returns an environment state given an action
    reset():
        Resets the environment
    '''
    def __init__(self, height=7, width=10):
        self.height = height
        self.width = width
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.height*self.width)
        
        self.moves = {
                0: (-1, 0),  # up
                1: (0, 1),   # right
                2: (1, 0),   # down
                3: (0, -1),  # left
                }

        # begin in start state
        self.reset()

    def _one_hot(self, sample):
        '''
        Returns a one hot vector 
        
        Parameters
        ----------
        sample: object
            environment state in coordinate form
        
        Returns
        -------
        vector: np.array
            one hot vector in list format
        '''
        index = sample[0]*self.width + sample[1]
        features = np.zeros(self.height * self.width)
        features[index]=1
        
        return features
    
    def step(self, action):
        '''
        Takes as input an action and returns an environment state.
        
        Parameters
        ----------
        action: object
            represents an environment action
        
        Returns
        -------
        next_observation: tuple
            a tuple consisting of the next_state (in one-hot vector form), reward (always equals -1) and done (boolean value representing whether episode terminates) of the environment
        ''' 
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
        '''
        Resets the environment.

        Returns
        -------
        starting_state: object
            represents the starting state of the environment (in one-hot vector form)
        '''
        self.S = (3, 0)
        return self._one_hot(self.S)


