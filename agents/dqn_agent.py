from agents.agent import Agent
from function_approximators.dqn_fa import FCNetwork

from copy import deepcopy
import gym
import numpy as np
import os.path
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List

class DQN_Agent(Agent):

    def __init__(
        action_space,
        observation_space,
        learning_rate,
        hidden_sizes,
        target_update_freq,
        batch_size,
        gamma,
        epsilon
    ):

        super().__init__(action_space, observation_space)

        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.epsilon = epsilon

        self.model = FCNetwork((observation_space.shape[0], *hidden_sizes, action_space.n))
        self.target_model = deepcopy(self.model)
        self.model_optim = Adam(self.model.parameters(), lr=learning_rate, eps=1e-3)

    def schedule_hyperparameters(self, timestep, max_timestep):
        
        pass
        # max_deduct, decay = 0.97, 0.1
        # self.epsilon = 1.0 - (min(1.0, timestep/(decay * max_timestep))) * max_deduct

    def act(self, obs, explore):
        
        if explore and np.random.random_sample() < self.epsilon:
            # Sample a random action from the action space
            action = self.action_space.sample()
        else:
            # Obtain the action values given the current observations from the Critics Network
            actions = self.model(Tensor(obs))
            # Select the action with the highest action value given the current observations
            action = torch.argmax(actions).item()

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
   
        # print(f"Q: {Q}")
        # print(f"Q_next: {Q_next}")
        # print(f"q_target: {q_target}")
        # print(f"y: {y}")
        # print(f"q_loss: {q_loss}")
        
        # increase update counter
        self.update_counter += 1

        # check for update condition
        if self.update_counter % self.target_update_freq == 0:
            # if update condition is met, hard update the parameters of the target network
            self.target_model.hard_update(self.model)

        # print(q_loss)      
        return {"q_loss": q_loss.item()}