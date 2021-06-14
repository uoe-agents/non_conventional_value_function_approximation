import torch.nn as nn
import numpy as np 
import torch
from sklearn.tree import DecisionTreeRegressor


class FAModel(nn.Module):

    def __init__(self):
        super().__init__()

    def hard_update(self, target_net):
        
        for param, target_param in zip(self.parameters(), target_net.parameters()):
            param.data.copy_(target_param.data)


class NeuralNetwork(FAModel):

    def __init__(self, layer_dims):

        super().__init__()
        self.model = self._compile_fcn(layer_dims)

    def _compile_fcn(self, dims):
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if (i < len(dims) - 2):
                # layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())

        model = nn.Sequential(*layers)
        return model

    def forward(self, x): 
        return self.model(x)


class LinearModel(FAModel):

    def __init__(self, input_dim, output_dim, poly_degree=1, tiling_specs=None):

        super().__init__()
       
        if poly_degree==1:
            self.poly=False
            k=0
        elif poly_degree>1: 
            self.poly=True
            k=1
        self.model = nn.Linear(input_dim**poly_degree*k+input_dim, output_dim)    

    def _polynomial_features_1(self, x):
        return torch.cat([x ** i for i in range(1, self.poly_degree+1)], -1)

    def _polynomial_features_2(self, x):
        
        if len(x.size()) == 2:
            f = []
            for xi in x:
                f.append([i*j for i in xi for j in xi])
            return torch.cat([x, torch.Tensor(f)], -1)

        elif len(x.size()) == 1:
            f = [i*j for i in x for j in x]
            return torch.cat([x, torch.Tensor(f)], -1)
              
    def forward(self, x): 
        
        # return self.model(self._tiling_features(x, self.lows, self.highs, self.specs))
        if self.poly:
            return self.model(self._polynomial_features_2(x))
        else:
            return self.model(x)


class DecisionTree():

    def __init__(self, max_depth, min_samples_split, min_samples_leaf):

        self.model = DecisionTreeRegressor(max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf)    

    def predict(self, inputs):
        return self.model.predict(inputs)

    def fit(self, inputs, outputs):
        self.model.fit(inputs, outputs)