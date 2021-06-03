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
        # self.model = nn.Linear(input_dim, output_dim)
        self.model = nn.Linear(input_dim*poly_degree, output_dim)
        # self.model = nn.Linear(50, output_dim)
        self.poly_degree = poly_degree

    
    def _polynomial_features_1(self, x):
        return torch.cat([x ** i for i in range(1, self.poly_degree+1)], -1)

    
    def _polynomial_features_2(self, x):
        
        try:
            f = []
            for xi in x:
                f.append([i*j for i in xi for j in xi])
            return torch.Tensor(f)

        except:
            return torch.Tensor([i*j for i in x for j in x])

    
    def _tiling_features(self, x):
          
        tiling_coords_theta = np.array([-0.18,-0.13,-0.08,-0.03,0.07,0.12,0.17,0.22])
        tiling_coords_pos = np.array([-3.2,-1.2,-0.7,0.3,0.8,2.8])

        a = []

        try:
            for i in range(3):
                f_tiling_theta = np.digitize(x[:,2], tiling_coords_theta+i*0.02)
                
                new_features_theta = np.zeros((x.shape[0],len(tiling_coords_theta)+1))
                new_features_theta[np.arange(x.shape[0]),f_tiling_theta]=1
                a.append(torch.Tensor(new_features_theta))
                
                f_tiling_pos = np.digitize(x[:,0], tiling_coords_pos+i*0.2)
                
                new_features_pos = np.zeros((x.shape[0],len(tiling_coords_pos)+1))
                new_features_pos[np.arange(x.shape[0]),f_tiling_pos]=1
                a.append(torch.Tensor(new_features_pos))
            
            return torch.cat([x[:,[1,3]],a[0],a[1],a[2],a[3],a[4],a[5]], -1)

        except:
            for i in range(3):
                f_tiling_theta = np.digitize(x[2], tiling_coords_theta+i*0.02)
                
                new_features_theta = np.zeros(len(tiling_coords_theta)+1)
                new_features_theta[f_tiling_theta]=1
                a.append(torch.Tensor(new_features_theta))
                
                f_tiling_pos = np.digitize(x[0], tiling_coords_pos+i*0.2)
                
                new_features_pos = np.zeros(len(tiling_coords_pos)+1)
                new_features_pos[f_tiling_pos]=1
                a.append(torch.Tensor(new_features_pos))
            
            return torch.cat([x[[1,3]],a[0],a[1],a[2],a[3],a[4],a[5]], -1)
            
        
    def forward(self, x): 
        # return self.model(self._tiling_features(x))
        return self.model(self._polynomial_features_1(x))
        # return self.model(x)