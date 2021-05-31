import gym
from random import random
from torch import nn
import numpy as np

env = gym.make("CartPole-v1")


class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.model = nn.Linear(4,2)
    
    def forward(self, x):
        return self.model(x)



def train(env, num_steps, num_episodes, optimiser):

    for episode in range(num_episodes):
        
        state=env.reset()
        
        for step in range(num_steps):

            action = act(state)
            state, _, done, _ = env.step(action)
            if done:
                break
        
        loss = _
        print(episode, step, loss.item)
        


def act(model, state):

    if random() < 0.03:
        action = env.action_space.sample()
    else:
        action_values = model(state)
        action = np.argmax(action_values)

    return action