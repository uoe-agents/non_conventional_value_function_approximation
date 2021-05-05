from abc import ABC, abstractmethod
import gym
import numpy as np

class Agent(ABC):

    def __init__(
        self,
        action_space,
        observation_space
    ):

        self.action_space = action_space
        self.observation_space = observation_space

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        ...

    @abstractmethod
    def update(self):
        ...
