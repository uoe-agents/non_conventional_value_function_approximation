import torch.nn as nn
import gym
import numpy as np 
from collections import namedtuple
import torch


class FA_model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return self.model(x)

    def hard_update(self, target_net):
        
        for param, target_param in zip(self.parameters(), target_net.parameters()):
            param.data.copy_(target_param.data)


class NeuralNetwork(FA_model):

    def __init__(self, layer_dims):

        super().__init__()
        self.model = self._compile_fcn(layer_dims)

    def _compile_fcn(self, dims):
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if (i < len(dims) - 1):
                layers.append(nn.ReLU())

        model = nn.Sequential(*layers)
        return model


class LinearModel(FA_model):

    def __init__(self, input_dim, output_dim):

        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)
        