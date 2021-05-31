from copy import deepcopy
import gym
import numpy as np
import os.path
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam, SGD
from typing import Dict, Iterable, List
from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        gamma: float,
        epsilon: float,
        target_update_freq: int,
    ):

        self.action_space = action_space
        self.observation_space = observation_space

        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_counter = 0


    def schedule_hyperparameters(self, timestep, max_timestep):
        
        max_deduct, decay = 0.97, 0.1
        self.epsilon = 1.0 - (min(1.0, timestep/(decay * max_timestep))) * max_deduct


    def act(self, obs, explore):
        
        if explore and np.random.random_sample() < self.epsilon:
            # Sample a random action from the action space
            action = self.action_space.sample()
        else:
            # Obtain the action values given the current observations from the Critics Network
            actions = self.model(Tensor(obs))
            # Select the action with the highest action value given the current observations
            action = torch.argmax(actions).item()

        return action


    def update(self, batch):

        # Obtain the action values given the current states in the batch from critics network
        Q = self.model(batch.states)    
        # Obtain the action values of the actions selected in the batch
        q_current = Q.gather(1, batch.actions.long())

        # Obtain the action values given the next states in the batch from critics_target network
        Q_next = self.target_model(batch.next_states)
        # obtain the target value: maximum action value over next states action values
        q_target = (1-batch.done) * Q_next.detach().max(1)[0].unsqueeze(1)
        # calculate the value of y
        y = batch.rewards + self.gamma * q_target

        # calculate the mse loss between y and q_current
        q_loss = F.mse_loss(q_current, y) 
        # zeroise the gradients of the optimiser
        self.model_optim.zero_grad()      
        # perform a backward pass
        q_loss.backward()
        # perform an optimisation step of the parameters of the critic network
        self.model_optim.step()
          
        # increase update counter
        self.update_counter += 1

        # check for update condition
        if self.update_counter % self.target_update_freq == 0:
            # if update condition is met, hard update the parameters of the target network
            self.target_model.hard_update(self.model)

        # print(q_loss)      
        return {"q_loss": q_loss.item()}


class DQNAgent(Agent):

    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
        gamma: float,
        epsilon: float,    
        target_update_freq: int,
        fa,
        learning_rate: float,
        hidden_size: Iterable[int],
        batch_size: int,  
        **kwargs
    ):
        
        super().__init__(
            action_space, 
            observation_space,
            gamma,
            epsilon,
            target_update_freq
        )

        self.model = fa((observation_space.shape[0], *hidden_size, action_space.n))
        if torch.cuda.is_available():
            self.model.cuda()

        self.target_model = deepcopy(self.model)
        self.model_optim = Adam(self.model.parameters(), lr=learning_rate, eps=1e-3)


class LinearAgent(Agent):

    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
        gamma: float,
        epsilon: float,    
        target_update_freq: int,
        fa,
        learning_rate: float,
        batch_size: int,
        poly_degree: int,  
        **kwargs
    ):
        
        super().__init__(
            action_space, 
            observation_space,
            gamma,
            epsilon,
            target_update_freq
        )

        self.model = fa(observation_space.shape[0], action_space.n, poly_degree)
        if torch.cuda.is_available():
            self.model.cuda()

        self.target_model = deepcopy(self.model)
        self.model_optim = SGD(self.model.parameters(), lr=learning_rate)
