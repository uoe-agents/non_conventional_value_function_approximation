import torch.nn as nn
import numpy as np 
import torch
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import graphviz
from sklearn import tree


class ParametricModel(nn.Module):

    def __init__(self):
        super().__init__()

    def hard_update(self, target_net):
        
        for param, target_param in zip(self.parameters(), target_net.parameters()):
            param.data.copy_(target_param.data)


class NeuralNetwork(ParametricModel):

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


class LinearModel(ParametricModel):

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


class NonParametricModel():
   
    def predict(self, inputs):
        return self.model.predict(inputs)

    def fit(self, inputs, outputs):
        self.model.fit(inputs, outputs)


class DecisionTree(NonParametricModel):

    def __init__(self, criterion="mse", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1):

        super().__init__()
        self.model = DecisionTreeRegressor(criterion=criterion,
                                           splitter=splitter,
                                           max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf)

    def export_tree(self, feature_names, plot_name):
        dot_data = tree.export_graphviz(self.model, out_file=None, 
                                feature_names=feature_names,  
                                filled=True)
        # Draw graph
        graph = graphviz.Source(dot_data, format="png") 
        graph.render(plot_name)



class RandomForest(NonParametricModel):

    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf):

        self.model = RandomForestRegressor(n_estimators=n_estimators,
                                           max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf)    

class ExtraTrees(NonParametricModel):

    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf):

        self.model = ExtraTreesRegressor(n_estimators=n_estimators,
                                           max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf)

class GradientBoostingTrees(NonParametricModel):

    def __init__(self, loss="ls", learning_rate=0.1, n_estimators=100, max_depth=3, min_samples_split=2, min_samples_leaf=1):

        self.model = GradientBoostingRegressor(loss=loss,
                                           learning_rate=learning_rate,
                                           n_estimators=n_estimators,
                                           max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf)

class SupportVectorRegressor(NonParametricModel):

    def __init__(self, kernel, degree, C):

        self.model = SVR(kernel=kernel,
                         degree=degree, 
                         C=C)

class KNeighboursRegressor(NonParametricModel):

    def __init__(self, n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30):

        self.model = KNeighborsRegressor(n_neighbors=n_neighbors,
                                         weights=weights,
                                         algorithm=algorithm,
                                         leaf_size=leaf_size)

class GaussianProcess(NonParametricModel):

    def __init__(self, kernel, alpha=1e-10, n_restarts_optimizer=0, normalize_y=False):

        self.model = GaussianProcessRegressor(kernel,
                                              alpha=alpha,
                                              n_restarts_optimizer=n_restarts_optimizer,
                                              normalize_y=normalize_y)

class GaussianProcessOnline(NonParametricModel):

    def __init__(self, kernel, alpha=1e-10, n_restarts_optimizer=0, normalize_y=False):

        self.model = GaussianProcessRegressor(kernel,
                                              alpha=alpha,
                                              n_restarts_optimizer=n_restarts_optimizer,
                                              normalize_y=normalize_y)

    def predict(self, inputs, return_std=False):
        return self.model.predict(inputs, return_std=return_std)