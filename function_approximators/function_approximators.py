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
        self.model = nn.Linear(input_dim**poly_degree*k + input_dim, output_dim)    

    def _polynomial_features_2(self, x):
        return torch.cat([x ** i for i in range(1, self.poly_degree+1)], -1)

    def _polynomial_features(self, x):
        
        if len(x.size()) == 2:
            f = []
            for xi in x:
                f.append([i*j for i in xi for j in xi])
            return torch.cat([x, torch.Tensor(f)], -1)

        elif len(x.size()) == 1:
            f = [i*j for i in x for j in x]
            return torch.cat([x, torch.Tensor(f)], -1)
              
    def forward(self, x): 
        
        if self.poly:
            return self.model(self._polynomial_features(x))
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

class eGaussianProcess(NonParametricModel):

    def __init__(self, kernel, alpha=1e-10, n_restarts_optimizer=0, normalize_y=False):

        self.model = GaussianProcessRegressor(kernel,
                                              alpha=alpha,
                                              n_restarts_optimizer=n_restarts_optimizer,
                                              normalize_y=normalize_y)

    def predict(self, inputs, return_std=False):
        return self.model.predict(inputs, return_std=return_std)

class OnlineGaussianProcess():

    def __init__(self, kernel, sigma_0, init, epsilon_tol, basis_limit):

        self.kernel = kernel
        self.epsilon_tol = epsilon_tol
        self.basis_limit = basis_limit
        self.model = self.initialise(sigma_0, init)

    def initialise(self, sigma_0, init):
        self.sigma_0 = sigma_0
        self.alpha = np.array([[init]])
        self.C = np.array([[init]])
        # self.mew = 0.1
        # self.sigma = 0.1
        # self.r = 0.1
        self.e = np.array([[1]])
        self.Q = np.array([[1]])

    def predict(self, X, x, return_sigma=False):
        if return_sigma:
            kk = self.kernel(X,x)
            k = self.kernel(x,x)
            mew = self.alpha.T @ kk
            sigma = self.sigma_0**2 + k + kk.T @ self.C @ kk
            return (mew, sigma)
        else:
            return self.alpha.T @ self.kernel(X,x)

    def _inc_dim_v(self, v):
        return np.pad(v, ((0,1),(0,0)))

    def _inc_dim_m(self, m):
        return np.pad(m, ((0,1),(0,1)))

    # def _dec_dim_v(self, v, index):
    #     return np.delete(v, [index], axis=0)

    # def _dec_dim_m(self, m, index):
    #     m = np.delete(m, [index], axis=0)
    #     m = np.delete(m, [index], axis=1)
    #     return m

    def update(self, X, x, y):
        k = self.kernel(x,x)
        kk = self.kernel(X,x)
        
        _, sigma = self.predict(X, x, return_sigma=True)
        self.r = -1/sigma
        self.q = y/sigma

        self.gamma = k - kk.T @ self.Q @ kk
        self.e_hat = self.Q@kk

        if self.gamma < self.epsilon_tol:
            self.s = self.C@kk + self.e_hat
            add = False    
        else:
            self.e = np.vstack([[0], self.e])
            self.Q = self._inc_dim_m(self.Q) + self.gamma * (self._inc_dim_v(self.e_hat)-self.e) @ (self._inc_dim_v(self.e_hat)-self.e).T
            self.s = self._inc_dim_v(self.C@kk) + self.e
            self.C = self._inc_dim_m(self.C) + self.r*(self.s @ self.s.T)
            self.alpha = self._inc_dim_v(self.alpha) + self.q*self.s 
            add = True
        
        # index_del = None
        # if X.shape[0] >= self.basis_limit:
        #     epsilon = [np.absolute(self.alpha[i,0])/self.Q[i,i] for i in range(X.shape[0])]
        #     index_del = np.argmin(epsilon)
        #     self.e = self._dec_dim_v(self.e, 0)

        #     Q_star = (self._dec_dim_v(self.Q[:, index_del], index_del)).reshape(-1,1)
        #     # print(Q_star.shape)
        #     q_star = self.Q[index_del,index_del]
        #     # print(q_star.shape)
        #     alpha_star = self.alpha[index_del,0]
        #     # print(alpha_star.shape)
        #     c_star = self.C[index_del, index_del]
        #     # print(c_star.shape)
        #     C_star = (self._dec_dim_v(self.C[:, index_del], index_del)).reshape(-1,1)
        #     # print(C_star.shape)

        #     self.e_hat = - Q_star/q_star
        #     # print(self.e_hat)
        #     self.Q = self._dec_dim_m(self.Q, index_del) - (Q_star@Q_star.T)/q_star
        #     # print(self.Q.shape)
        #     self.s = self._dec_dim_v(self.s, index_del)
        #     # print(self.s.shape)
        #     self.C = self._dec_dim_m(self.C, index_del) + c_star*(Q_star@Q_star.T)/(q_star**2) - 1/q_star* (Q_star@C_star.T + C_star@Q_star.T)
        #     # print(self.C.shape)
        #     self.alpha = self._dec_dim_v(self.alpha, index_del) + alpha_star/q_star * Q_star
        #     # print(self.alpha.shape)

        return add