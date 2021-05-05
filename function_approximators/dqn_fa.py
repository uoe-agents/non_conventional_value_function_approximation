import torch.nn as nn
import gym
import numpy as np 
from collections import namedtuple
import torch


Transition = namedtuple(
    "Transition", ("states", "actions", "next_states", "rewards", "done")
)

class ReplayBuffer:

    def __init__(self, capacity):

        self.capacity = capacity
        self.memory = None
        self.count = 0

    def __len__(self): 

        return min(self.count, self.capacity)

    def init_memory(self, transition):

        self.memory = Transition(
            *[np.zeros([self.capacity, elem.size], dtype=elem.dtype) for elem in transition]
        )

    def push(self, *args):

        # create namedTuple
        transition = Transition(*args)
        
        # checks
        for t in transition:
            assert t.ndim == 1
        
        assert transition.states.size == transition.next_states.size
        assert transition.rewards.size == 1
        
        # initialise replay buffer
        if not self.memory:
            self.init_memory(transition)

        # push transition in replay buffer
        index = self.count % self.capacity
        for i, arg in enumerate(args):
            self.memory[i][index, :] = arg

        # update count
        self.count += 1

    def sample(self, batch_size, device = "cpu"):

        indices = np.random.randint(0, high=len(self), size=batch_size)

        batch = Transition(
            *[
                torch.from_numpy(np.take(d, indices, axis=0)).to(device)
                for d in self.memory
            ]
        )
        return batch        


class FCNetwork(nn.Module):

    def __init__(self, layer_dims):

        # self.input_dim = input_dim
        # self.output_dim = output_dim

        super().__init__()
        self.model = self.compile_fcn(layer_dims)

    def compile_fcn(self, dims):
        
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if (i < len(dims) - 1):
                layers.append(nn.ReLU())

        model = nn.Sequential(*layers)
        
        return model

    def forward(self, x):
        
        return self.model(x)

    
    def update(self, target_net):
        for param, target_param in zip(self.parameters(), target_net.parameters()):
            param.data.copy_(target_param.data)

        
