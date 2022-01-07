from copy import deepcopy
import gym
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from typing import Iterable
from abc import ABC, abstractmethod
from collections import deque


class Agent(ABC):  
    '''
    A base class used by the DQN and Linear VFA agent classes.
    
    Attributes
    ----------
    action_space: gym.Space
        action space from Gym
    observation_space: gym.Space  
        state space from Gym
    gamma: float
        parameter of the Markov Decision Process
    epsilon: float
        exploration parameter
    target_update_freq: int
        parameter that defines the model update frequency
    update_counter: int
        parameter that counts the number of model updates

    Methods
    -------
    act():
        Returns an action given a state (observation) of the environment.
    update():
        Updates model given a batch of (state, action, next_state, reward, done) tuples
    '''

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        gamma: float,
        epsilon: float,
        target_update_freq: int,
        update_counter: int
    ):

        self.action_space = action_space
        self.observation_space = observation_space

        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_counter = 0


    def act(self, obs, explore):
        '''
        Takes as input an environment observation and returns an action.
        
        Parameters
        ----------
        obs: object
            represents an environment state
        explore: bool
            determines whether exploration happens
        
        Returns
        -------
        action: object
            represents an environment action
        '''    
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
        '''
        Takes as input a batch of tuples and updates the model parameters
        
        Parameters
        ----------
        batch: collections.namedtuple
            batch of (state, action, next_state, reward, done) tuples
        
        Returns
        -------
        q_loss: float
            mse loss of update
        '''
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
        
        # zeroise the gradients of the optimiser
        self.model_optim.zero_grad()      
        # perform a backward pass
        q_loss.backward()
        # perform an optimisation step of the parameters of the critic network
        self.model_optim.step()  
        self.scheduler.step() 
        # increase update counter
        self.update_counter += 1

        # check for update condition
        if self.update_counter % self.target_update_freq == 0:
            # if update condition is met, hard update the parameters of the target network
            self.target_model.hard_update(self.model)
   
        return {"q_loss": q_loss.item()}


class DQNAgent(Agent):
    '''
    A class that represents the Reinforcement Learning agent of the DQN model.
    
    Attributes
    ----------
    action_space: gym.Space
        action space from Gym
    observation_space: gym.Space  
        state space from Gym
    gamma: float
        parameter of the Markov Decision Process
    epsilon: float
        exploration parameter
    target_update_freq: int
        parameter that defines the model update frequency
    fa: function_approximators.model
        function approximation model
    learning_rate: float
        parameter of the neural network
    hidden_size: Iterable[int]
        parameter of the neural network
    max_deduct: float
        parameter for reducing epsilon over time
    decay: float
        parameter for reducing epsilon over time
    lr_step_size: int
        parameter for reducing learning rate over time
    lr_gamma: float
        parameter for reducing learning rate over time

    Methods
    -------
    schedule_hyperparameters():
        adjusts hyperparameter values over time
    '''

    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
        gamma: float,
        epsilon: float,    
        target_update_freq: int,
        fa,
        learning_rate: float,
        hidden_size: Iterable[int],
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
        '''
        Adjusts values of hyperparameters over time
        
        Parameters
        ----------
        timestep: int
            current timestep
        max_timestep: int
            max timestep
        '''
        # reducing epsilon over time
        self.epsilon = 1.0 - (min(1.0, timestep/(self.decay * max_timestep))) * self.max_deduct


class LinearAgent(Agent):
    '''
    A class that represents the Reinforcement Learning agent of the Linear model.
    
    Attributes
    ----------
    action_space: gym.Space
        action space from Gym
    observation_space: gym.Space  
        state space from Gym
    gamma: float
        parameter of the Markov Decision Process
    epsilon: float
        exploration parameter
    target_update_freq: int
        parameter that defines the model update frequency
    fa: function_approximators.model
        function approximation model
    learning_rate: float
        parameter of the linear model
    poly_degree: Iterable[int]
        parameter of the linear model
    max_deduct: float
        parameter for reducing epsilon over time
    decay: float
        parameter for reducing epsilon over time
    lr_step_size: int
        parameter for reducing learning rate over time
    lr_gamma: float
        parameter for reducing learning rate over time

    Methods
    -------
    schedule_hyperparameters():
        adjusts hyperparameter values over time
    '''

    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
        gamma: float,
        epsilon: float,    
        target_update_freq: int,
        fa,
        learning_rate: float,
        poly_degree: int,
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
        '''
        Adjusts values of hyperparameters over time
        
        Parameters
        ----------
        timestep: int
            current timestep
        max_timestep: int
            max timestep
        '''       
        # reducing epsilon over time
        self.epsilon = 1.0 - (min(1.0, timestep/(self.decay * max_timestep))) * self.max_deduct


class FQIAgent():  
    '''
    A class that represents the Reinforcement Learning agent used for all models implemented under the Fitted-Q Iteration framework.
    
    Attributes
    ----------
    action_space: gym.Space
        action space from Gym
    observation_space: gym.Space  
        state space from Gym
    gamma: float
        parameter of the Markov Decision Process
    epsilon: float
        exploration parameter
    fa: function_approximators.model
        function approximation model
    max_deduct: float
        parameter for reducing epsilon over time
    decay: float
        parameter for reducing epsilon over time
    update_freq: int
        parameter of the FQI framework
    model_save_freq: int
        parameter of the FQI framework
    model_save_capacity: int
        parameter of the FQI framework
    model_params: list
        function approximation model parameters

    Methods
    -------
    schedule_hyperparameters():
        adjusts hyperparameter values over time
    act():
        Returns an action given a state (observation) of the environment.
    update():
        Updates model given a batch of (state, action, next_state, reward, done) tuples
    initial_fit():
        Initialises function approximation model
    '''
    
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
        if self.model_save_freq > 0:
            self.models = deque([self.model], maxlen=model_save_capacity)


    def _one_hot(self, length, index):
        '''
        Returns a one hot vector 
        
        Parameters
        ----------
        length: int
            length of the vector
        index: int
            position of 1
        
        Returns
        -------
        vector: list
            one hot vector in list format
        '''
        vector = np.zeros(length)
        vector[index]=int(1)
        return list(vector)

    def _encode_actions(self):
        '''
        Returns a one hot vector for each action in the environment
        
        Returns
        -------
        encoded_actions: list
            list of one hot vectors for each action in the environment
        '''
        length = self.action_space.n
        return [self._one_hot(length, i) for i in range(length)] 

    def _predict(self, inputs):
        '''
        Returns a weighted average prediction of the stored models given an observation
        
        Parameters
        ----------
        inputs: list
            environment observation
        
        Returns
        -------
        prediction: float
            weigted average value of predicted q values from stored models
        '''
        l = len(self.models)
        out = []
        for i, f in enumerate(self.models):
            out.append(f.predict(inputs)*(i+1)/(sum(range(l+2))))
        out.append(self.model.predict(inputs)*(l+1)/sum(range(l+2)))
        return np.sum(out, 0)

    def act(self, obs, explore):
        '''
        Takes as input an environment observation and returns an action.
        
        Parameters
        ----------
        obs: object
            represents an environment state
        explore: bool
            determines whether exploration happens
        
        Returns
        -------
        action: object
            represents an environment action
        ''' 
        if (explore and np.random.random_sample() < self.epsilon) or (not self.fitted):
            action = self.action_space.sample()
        else:   
            if self.model_save_freq > 0:    
                Q = [self._predict(np.concatenate([obs, self.encoded_actions[i]],-1).reshape(1,-1)) for i in range(self.action_space.n)]
            else:
                Q = [self.model.predict(np.concatenate([obs, self.encoded_actions[i]],-1).reshape(1,-1)) for i in range(self.action_space.n)]
            action = np.argmax(Q)
        
        return action

    def update(self, batch):
        '''
        Takes as input a batch of tuples and updates the model parameters
        
        Parameters
        ----------
        batch: collections.namedtuple
            batch of (state, action, next_state, reward, done) tuples
        '''
        self.step_counter += 1
        if self.step_counter % self.update_freq == 0:
            
            inputs = np.concatenate([batch.states, [self.encoded_actions[int(i.item())] for i in batch.actions]], -1)
            preds = []
            
            for i in range(self.action_space.n):
                next_inputs = np.concatenate([batch.next_states, np.zeros((batch.actions.size()[0], 1)) + self.encoded_actions[i]], -1)
                if self.model_save_freq > 0:
                    preds.append(self._predict(next_inputs))
                else:
                    preds.append(self.model.predict(next_inputs))
                
            preds = np.array(preds).T
            outputs = np.array(batch.rewards + self.gamma * (1-batch.done) * np.max(preds, 1).reshape(-1,1)).reshape(-1)  
            self.model.fit(inputs, outputs) 

            # check for update condition
            if self.model_save_freq > 0:
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
        '''
        Adjusts values of hyperparameters over time
        
        Parameters
        ----------
        timestep: int
            current timestep
        max_timestep: int
            max timestep
        '''
        self.epsilon = 1.0 - (min(1.0, timestep/(self.decay * max_timestep))) * self.max_deduct


class OnlineGaussianProccessAgent(): 
    '''
    A class that represents the Reinforcement Learning agent used for the Online Gaussian Process model.
    
    Attributes
    ----------
    action_space: gym.Space
        action space from Gym
    observation_space: gym.Space  
        state space from Gym
    gamma: float
        parameter of the Markov Decision Process
    epsilon: float
        exploration parameter
    fa: function_approximators.model
        function approximation model
    max_deduct: float
        parameter for reducing epsilon over time
    decay: float
        parameter for reducing epsilon over time
    model_params: list
        function approximation model parameters

    Methods
    -------
    schedule_hyperparameters():
        adjusts hyperparameter values over time
    act():
        Returns an action given a state (observation) of the environment.
    update():
        Updates model given a batch of (state, action, next_state, reward, done) tuples
    '''
    
    def __init__(self,
        action_space: gym.Space,
        observation_space: gym.Space,
        gamma: float,
        epsilon: float,    
        fa,
        max_deduct: float,
        decay: float,
        model_params, 
        **kwargs
    ):

        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_deduct = max_deduct
        self.decay = decay

        self.step_counter = 0
        self.encoded_actions = self._encode_actions()
        self.model = fa(**model_params)

        if observation_space.shape == ():
            input_size = observation_space.n
        else:
            input_size = observation_space.shape[0]
        
        self.X = np.zeros((1, input_size + self.action_space.n))
    
    def _one_hot(self, length, index):
        '''
        Returns a one hot vector 
        
        Parameters
        ----------
        length: int
            length of the vector
        index: int
            position of 1
        
        Returns
        -------
        vector: list
            one hot vector in list format
        '''
        vector = np.zeros(length)
        vector[index]=int(1)
        return list(vector)

    def _encode_actions(self):
        '''
        Returns a one hot vector for each action in the environment
        
        Returns
        -------
        encoded_actions: list
            list of one hot vectors for each action in the environment
        '''
        length = self.action_space.n
        return [self._one_hot(length, i) for i in range(length)] 

    def act(self, obs, explore):     
        '''
        Takes as input an environment observation and returns an action.
        
        Parameters
        ----------
        obs: object
            represents an environment state
        explore: bool
            determines whether exploration happens
        
        Returns
        -------
        action: object
            represents an environment action
        '''
        if (explore and np.random.random_sample() < self.epsilon):
            action = self.action_space.sample()
        else:       
            Q = [self.model.predict(self.X, np.concatenate([obs, self.encoded_actions[i]],-1).reshape(1,-1)) for i in range(self.action_space.n)]
            action = np.argmax(Q)
        return action 
    
    def update(self, obs, next_obs, reward, action, done):
        '''
        Takes as input a batch of tuples and updates the model parameters
        
        Parameters
        ----------
        obs: object
            environment state
        next_obs: object
            environment next state
        reward: float
            reward
        action: object
            action
        done: bool
            whether the environment should be reset
        '''     
        q_values = [self.model.predict(self.X, np.concatenate([next_obs, self.encoded_actions[i]],-1).reshape(1,-1), return_sigma=True) for i in range(self.action_space.n)]
        Q = [q[0] + 2*q[1] for q in q_values]
        # Q = [q[0] for q in q_values]
        Q_max = np.max(Q)
        x = np.concatenate([obs, self.encoded_actions[action]],-1).reshape(1,-1)
        Q_prev = self.model.predict(self.X, x).item()

        y = (reward + self.gamma * (1-done) * Q_max - Q_prev)
        add = self.model.update(self.X, x, y) 
        if add:
            self.X = np.vstack([self.X, x])

    def schedule_hyperparameters(self, timestep, max_timestep):
        '''
        Adjusts values of hyperparameters over time
        
        Parameters
        ----------
        timestep: int
            current timestep
        max_timestep: int
            max timestep
        '''
        self.epsilon = 1.0 - (min(1.0, timestep/(self.decay * max_timestep))) * self.max_deduct
