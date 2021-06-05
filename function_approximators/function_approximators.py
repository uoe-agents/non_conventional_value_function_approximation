import torch.nn as nn
import numpy as np 
import torch
from torch.nn.functional import softmax


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

    def __init__(self, input_dim, output_dim, poly_degree, tiling_specs):

        super().__init__()

        self.poly_degree = poly_degree
        self.lows, self.highs, self.specs = tiling_specs

        # self.model = nn.Linear(input_dim, output_dim)
        self.model = nn.Linear(input_dim**poly_degree+input_dim, output_dim)
        # self.model = nn.Linear(self.specs[0][0][0]**2*len(self.specs)+2, output_dim)       

    
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

    
    def _create_grid(self, lower, upper, bins, offsets):
        return [np.linspace(lower[dim], upper[dim], bins[dim] + 1)[1:-1] + offsets[dim] for dim in range(len(bins))]

    def _create_tilings(self, lower, upper, specs):
        return [self._create_grid(lower, upper, bins, offsets) for bins, offsets in specs]

    def _discretize(self, sample, grid):
        return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))

    def _tile_encoding(self, sample, tilings):
        return [self._discretize(sample, grid) for grid in tilings]

    def _get_indices(self, tile_encoding, n_bins):
        n_tilings = len(tile_encoding)
        indices = [i*n_bins + j + n*(n_bins**2) for n, (i, j) in enumerate(tile_encoding)]
        features = np.zeros(n_bins**2*n_tilings)
        features[indices]=1

        return features

    def _tiling_features(self, x, lower, upper, specs):

        tilings = self._create_tilings(lower, upper, specs)
        
        if len(x.size()) == 1:
            tile_encoding = self._tile_encoding(x[[0,2]], tilings)
            features = self._get_indices(tile_encoding, specs[0][0][0])
            
            return torch.cat([x[[1,3]],torch.Tensor(features)], -1)
        
        elif len(x.size()) == 2:       
            features = []
            for xi in x:
                tile_encoding = self._tile_encoding(xi[[0,2]], tilings)
                features.append(self._get_indices(tile_encoding, specs[0][0][0]))

            return torch.cat([x[:,[1,3]],torch.Tensor(features)], -1)
          
        
    def forward(self, x): 
        
        # return self.model(self._tiling_features(x, self.lows, self.highs, self.specs))
        return self.model(self._polynomial_features_2(x))
        # return self.model(x)