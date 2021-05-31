import torch.nn as nn
import gym
import numpy as np 
from collections import namedtuple
import torch


class FA_model(nn.Module):

    def __init__(self):
        super().__init__()

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

    def forward(self, x): 
        return self.model(x)


class LinearModel(FA_model):

    def __init__(self, input_dim, output_dim, poly_degree):

        super().__init__()
        self.model = nn.Linear(input_dim*poly_degree, output_dim)
        self.poly_degree = poly_degree

    def _polynomial_features(self, x):
        return torch.cat([x ** i for i in range(1, self.poly_degree+1)], -1)
        
    def forward(self, x): 
        return self.model(self._polynomial_features(x))
        # return self.model(x)