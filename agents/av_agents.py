from copy import deepcopy
import gym
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR

from typing import Dict, Iterable, List
from abc import ABC, abstractmethod
from collections import deque


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


    def act(self, obs, explore):
        
        if explore and np.random.random_sample() < self.epsilon:
            # Sample a random action from the action space
            action = self.action_space.sample()
        else:
            # Obtain the action values given the current observations from the Critics Network
            actions = self.model(Tensor(obs))
            # Select the action with the highest action value given the current observations
            action = torch.argmax(actions).item()

        # if self.update_counter % 1000 == 0:
        #     print(action)
        
        return action


    def update(self, batch):

        # Obtain the action values given the current states in the batch from critics network
        Q = self.model(batch.states)   
        # Obtain the action values of the actions selected in the batch
        q_current = Q.gather(1, batch.actions.long())

        # Obtain the action values given the next states in the batch from critics_target network
        Q_next = self.target_model(batch.next_states)
        # obtain the target value: maximum action value over next states action values
        q_target = (1-batch.done) * Q_next.max(1)[0].unsqueeze(1)
        # calculate the value of y
        y = batch.rewards + self.gamma * q_target
        
        # calculate the mse loss between y and q_current
        q_loss = F.mse_loss(q_current, y) 
        
        # if self.update_counter % 1000 == 0:
        #     print(f"Weights: {self.model.model[0].weight}")
        #     print(f"States: {batch.states}")
        #     print(f"Actions: {batch.actions}")
        #     print(f"Q: {Q}") 
        #     print(f"q_current: {q_current}")
        #     print(f"Q_next: {Q_next}")
        #     print(f"y: {y}")
            # print(f"q_loss: {q_loss}")
        
        
        # zeroise the gradients of the optimiser
        self.model_optim.zero_grad()      
        # perform a backward pass
        q_loss.backward()
        # perform an optimisation step of the parameters of the critic network
        self.model_optim.step()
        
        self.scheduler.step()
        # print(self.model_optim.param_groups[0]['lr'])
          
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
        max_deduct: float,
        decay: float,
        lr_step_size: int,
        lr_gamma: float, 
        **kwargs
    ):
        
        super().__init__(
            action_space, 
            observation_space,
            gamma,
            epsilon,
            target_update_freq
        )

        self.max_deduct = max_deduct
        self.decay = decay

        if observation_space.shape == ():
            input_size = observation_space.n
        else:
            input_size = observation_space.shape[0]

        self.model = fa((input_size, *hidden_size, action_space.n))


        self.target_model = deepcopy(self.model)
        self.model_optim = Adam(self.model.parameters(), lr=learning_rate, eps=1e-3)
        self.scheduler = StepLR(self.model_optim, step_size=lr_step_size, gamma=lr_gamma)

    def schedule_hyperparameters(self, timestep, max_timestep):
        self.epsilon = 1.0 - (min(1.0, timestep/(self.decay * max_timestep))) * self.max_deduct


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
        # tiling_specs: list,  
        max_deduct: float,
        decay: float,
        lr_step_size: int,
        lr_gamma: float,
        **kwargs
    ):
        
        super().__init__(
            action_space, 
            observation_space,
            gamma,
            epsilon,
            target_update_freq
        )

        self.max_deduct = max_deduct
        self.decay = decay
        
        if observation_space.shape == ():
            input_size = observation_space.n
        else:
            input_size = observation_space.shape[0]

        self.model = fa(input_size, action_space.n, poly_degree)
        
        if torch.cuda.is_available():
            self.model.cuda()

        self.target_model = deepcopy(self.model)
        self.model_optim = Adam(self.model.parameters(), lr=learning_rate, eps=1e-3)
        self.scheduler = StepLR(self.model_optim, step_size=lr_step_size, gamma=lr_gamma)
   
    def schedule_hyperparameters(self, timestep, max_timestep):
        self.epsilon = 1.0 - (min(1.0, timestep/(self.decay * max_timestep))) * self.max_deduct


class NonParametricAgent():
    
    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
        gamma: float,
        epsilon: float,    
        fa,
        max_deduct: float,
        decay: float,
        update_freq: int,
        model_save_freq: int,
        model_save_capacity: int,
        model_params, 
        **kwargs
    ):

        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_deduct = max_deduct
        self.decay = decay
        self.model_save_freq = model_save_freq
        self.update_freq = update_freq

        self.step_counter = 0
        self.fitted = False
        self.encoded_actions = self._encode_actions()

        self.model = fa(**model_params)
        self.models = deque([self.model], maxlen=model_save_capacity)


    def _one_hot(self, length, index):
        vector = np.zeros(length)
        vector[index]=int(1)
        return list(vector)

    def _encode_actions(self):
        length = self.action_space.n
        return [self._one_hot(length, i) for i in range(length)] 

    def _predict(self, inputs):
        l = len(self.models)
        out = []
        for i, f in enumerate(self.models):
            out.append(f.predict(inputs)*(i+1)/(sum(range(l+1))))
        # out.append(self.model.predict(inputs)*(l+1)/sum(range(l+2)))
        return np.sum(out, 0)

    def act(self, obs, explore):
        if (explore and np.random.random_sample() < self.epsilon) or (not self.fitted):
            action = self.action_space.sample()
        else:       
            Q = [self._predict(np.concatenate([obs, self.encoded_actions[i]],-1).reshape(1,-1)) for i in range(self.action_space.n)]
            action = np.argmax(Q)
        return action

    def update(self, batch):
        self.step_counter += 1
        
        if self.step_counter % self.update_freq == 0:
            
            inputs = np.concatenate([batch.states, [self.encoded_actions[int(i.item())] for i in batch.actions]], -1)
            preds = []
            
            for i in range(self.action_space.n):
                next_inputs = np.concatenate([batch.next_states, np.zeros((batch.actions.size()[0], 1)) + self.encoded_actions[i]], -1)
                preds.append(self._predict(next_inputs))
            
            preds = np.array(preds).T
            outputs = np.array(batch.rewards + self.gamma * (1-batch.done) * np.max(preds, 1).reshape(-1,1)).reshape(-1)  
            # print(inputs)
            # print(outputs[0])
            self.model.fit(inputs, outputs) 

            # check for update condition
            if self.step_counter % self.model_save_freq == 0:
                # if update condition is met, save current model
                self.models.append(deepcopy(self.model))
        
        else:
            pass

    def initial_fit(self, batch):
        inputs = np.concatenate([batch.states, [self.encoded_actions[int(i.item())] for i in batch.actions]], -1)
        outputs = np.array(batch.rewards).reshape(-1)
        self.model.fit(inputs, outputs)
        self.fitted = True

    def schedule_hyperparameters(self, timestep, max_timestep):
        self.epsilon = 1.0 - (min(1.0, timestep/(self.decay * max_timestep))) * self.max_deduct