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
from sklearn.preprocessing import PolynomialFeatures


class ParametricModel(nn.Module):
    '''
    A base torch.nn.Module class used by the NeuralNetwork and LinearModel classes.
    
    Methods
    -------
    hard_update():
        Updates target model with current model's parameters
    '''
    def __init__(self):
        super().__init__()

    def hard_update(self, target_net):
        
        for param, target_param in zip(self.parameters(), target_net.parameters()):
            param.data.copy_(target_param.data)

class NeuralNetwork(ParametricModel):
    '''
    Represents a Neural Network model.
    
    Attributes
    ----------
    model: torch.nn.Sequential
        neural network model
    
    Methods
    -------
    forward():
        Updates the model parameters given a training example
    '''
    def __init__(self, layer_dims):

        super().__init__()
        self.model = self._compile_fcn(layer_dims)

    def _compile_fcn(self, dims):
        '''
        Creates the neural network given a set of dimensions for each layer
        
        Parameters
        ----------
        dims: list
            dimensions of each layer of the neural network
        
        Returns
        -------
        model: torch.nn.Sequential
            neural network model
        '''
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if (i < len(dims) - 2):
                layers.append(nn.ReLU())

        model = nn.Sequential(*layers)
        return model

    def forward(self, x): 
        '''
        Updates model parameters given a new example
        
        Parameters
        ----------
        x: torch.Tensor
            training example
        
        Returns
        -------
        model: torch.nn.Sequential
            neural network model
        '''
        return self.model(x)

class LinearModel(ParametricModel):
    '''
    Represents a Linear model.
    
    Attributes
    ----------
    model: torch.nn.Linear
        linear model
    poly: bool
        whether polynomial feature representation is used
    poly_degree: int
        degree of polynomial feature representation
    
    Methods
    -------
    forward():
        Updates the model parameters given a training example
    '''
    def __init__(self, input_dim, output_dim, poly_degree=1):

        super().__init__()

        if poly_degree==1:
            self.poly=False
            n = input_dim
        elif poly_degree>1: 
            self.poly=True
            self.trans = PolynomialFeatures(degree=2, include_bias=False)
            n = self.trans.fit_transform(np.zeros((1,input_dim))).shape[1]
        
        self.model = nn.Linear(n, output_dim)    

    def _polynomial_features(self, x):
        '''
        Performs a polynomial representation on a set of training examples
        
        Parameters
        ----------
        x: torch.Tensor
            training example
        
        Returns
        -------
        poly_x: torch.Tensor
            polynomial feature representation of training examples
        '''
        if len(x.size()) == 2:
            return torch.Tensor(self.trans.fit_transform(x))
        elif len(x.size()) == 1:
            return torch.Tensor(self.trans.fit_transform(x.reshape(1, -1)))

    def forward(self, x): 
        '''
        Updates model parameters given a new example
        
        Parameters
        ----------
        x: torch.Tensor
            training example
        
        Returns
        -------
        model: torch.nn.Linear
            linear model
        '''
        if self.poly:
            return self.model(self._polynomial_features(x))
        else:
            return self.model(x)

class NonParametricModel():
    '''
    A base class used by the DecisionTree, RandomForest, ExtraTrees, GradientBoostingTrees, SupportVectorRegressor, 
    KNeighboursRegressor, GaussianProcess and eGaussianProcess classes.
    
    Methods
    -------
    predict():
        Returns predicted outputs given a set of inputs
    fit():
        fits a supervised regression model given a training set of (inputs, outputs)
    
    '''
    
    def predict(self, inputs):
        '''
        Returns model's predictions given a set of inputs
        
        Parameters
        ----------
        inputs: torch.Tensor
            inputs
        
        Returns
        -------
        outputs: torch.Tensor
            model predictions
        
        '''
        return self.model.predict(inputs)

    def fit(self, inputs, outputs):
        '''
        Fits a supervised regression model given a training set of (inputs, outputs) pairs.
        
        Parameters
        ----------
        inputs: torch.Tensor
            training inputs
        outputs: torch.Tensor
            training outputs
        
        '''
        self.model.fit(inputs, outputs)

class DecisionTree(NonParametricModel):
    '''
    Represents a Decison Tree model.
        
    Attributes
    ----------
    model: sklearn.tree.DecisionTreeRegressor
        decision tree model
        
    Methods
    -------
    export_tree():
        Exports fitted tree and plots it
    
    '''
    def __init__(self, criterion="mse", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1):

        super().__init__()
        self.model = DecisionTreeRegressor(criterion=criterion,
                                           splitter=splitter,
                                           max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf)

    def export_tree(self, feature_names, plot_name):
        '''
        Exports the model's fitted tree and plots it using the graphviz library.
        
        Parameters
        ----------
        feature_names: list
            names of each of the model's features
        plot_name: string
            name of the plot
        
        '''
        dot_data = tree.export_graphviz(self.model, out_file=None, 
                                feature_names=feature_names,  
                                filled=True)
        # Draw graph
        graph = graphviz.Source(dot_data, format="png") 
        graph.render(plot_name)

class RandomForest(NonParametricModel):
    '''
    Represents a Random Forest model.
    
    Attributes
    ----------
    model: sklearn.ensemble.RandomForestRegressor
        random forest model
        
    '''
    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf):

        self.model = RandomForestRegressor(n_estimators=n_estimators,
                                           max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf)    

class ExtraTrees(NonParametricModel):
    '''
    Represents an Extremely Randomized Trees model.
    
    Attributes
    ----------
    model: sklearn.ensmble.ExtraTreesRegressor
        extremely randomized trees model   
    
    '''
    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf):

        self.model = ExtraTreesRegressor(n_estimators=n_estimators,
                                           max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf)

class GradientBoostingTrees(NonParametricModel):
    '''
    Represents a Gradient Boosting Trees model.
    
    Attributes
    ----------
    model: sklearn.ensmble.GradientBoostingRegressor
        gradient boosting trees model
    
    '''
    def __init__(self, loss="ls", learning_rate=0.1, n_estimators=100, max_depth=3, min_samples_split=2, min_samples_leaf=1):

        self.model = GradientBoostingRegressor(loss=loss,
                                           learning_rate=learning_rate,
                                           n_estimators=n_estimators,
                                           max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf)

class SupportVectorRegressor(NonParametricModel):
    '''
    Represents a Support Vector regressor model.
    
    Attributes
    ----------
    model: sklearn.svm.SVR
        support vector regression model
    
    '''
    def __init__(self, kernel, degree, C):

        self.model = SVR(kernel=kernel,
                         degree=degree, 
                         C=C)

class KNeighboursRegressor(NonParametricModel):
    '''
    Represents a K-Neighbours regressor model.
    
    Attributes
    ----------
    model: sklearn.neighbors.KNeighborsRegressor
        k-neighbours regressor model
    
    '''
    def __init__(self, n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30):

        self.model = KNeighborsRegressor(n_neighbors=n_neighbors,
                                         weights=weights,
                                         algorithm=algorithm,
                                         leaf_size=leaf_size)

class GaussianProcess(NonParametricModel):
    '''
    Represents a Gaussian Process model.
    
    Attributes
    ----------
    model: sklearn.gaussian_process.GaussianProcessRegressor
        Gaussian process model
    
    '''
    def __init__(self, kernel, alpha=1e-10, n_restarts_optimizer=0, normalize_y=False):

        self.model = GaussianProcessRegressor(kernel,
                                              alpha=alpha,
                                              n_restarts_optimizer=n_restarts_optimizer,
                                              normalize_y=normalize_y)

class eGaussianProcess(NonParametricModel):
    '''
    Represents a Gaussian Process model which uses the uncertainty estimates of each prediction to aid environment exploration.
    
    Attributes
    ----------
    model: sklearn.gaussian_process.GaussianProcessRegressor
        Gaussian process model
    
    Methods
    -------
    predict():
        Returns predicted outputs given a set of inputs
    
    '''
    def __init__(self, kernel, alpha=1e-10, n_restarts_optimizer=0, normalize_y=False):

        self.model = GaussianProcessRegressor(kernel,
                                              alpha=alpha,
                                              n_restarts_optimizer=n_restarts_optimizer,
                                              normalize_y=normalize_y)

    def predict(self, inputs, return_std=False):
        '''
        Returns model's predictions given a set of inputs
        
        Parameters
        ----------
        inputs: torch.Tensor
            inputs
        
        Returns
        -------
        outputs: torch.Tensor
            model predictions
        
        '''
        return self.model.predict(inputs, return_std=return_std)

class OnlineGaussianProcess():
    '''
    Represents an Online Gaussian Process model.
    
    Reference: https://www.researchgate.net/publication/11500673_Sparse_On-Line_Gaussian_Processes
    
    Attributes
    ----------
    kernel: sklearn.gaussian_process.kernels
        Gaussian process kernel
    epsilon_tol: float
        parameter that controls the sensitivity of adding new examples in the basis
    basis_limit: int
        parameter that controls the maximum number of examples that can be added in the basis
    
    Methods
    -------
    predict():
        Returns predicted outputs given a set of inputs
    update():
        Updates model parameters given a set of training examples
    
    '''
    def __init__(self, kernel, sigma_0, init, epsilon_tol, basis_limit):

        self.kernel = kernel
        self.epsilon_tol = epsilon_tol
        self.basis_limit = basis_limit
        self._initialise(sigma_0, init)

    def _initialise(self, sigma_0, init):
        '''
        Initialises model parameters
        
        '''
        self.sigma_0 = sigma_0
        self.alpha = np.array([[init]])
        self.C = np.array([[init]])
        self.e = np.array([[1]])
        self.Q = np.array([[1]])

    def predict(self, X, x, return_sigma=False):
        '''
        Returns model's predictions given a set of inputs
        
        Parameters
        ----------
        X: np.array
            current basis
        x: np.array
            input
        
        Returns
        -------
        output: np.array
            model prediction
        
        '''
        if return_sigma:
            kk = self.kernel(X,x)
            k = self.kernel(x,x)
            mew = self.alpha.T @ kk
            sigma = self.sigma_0**2 + k + kk.T @ self.C @ kk
            return (mew, sigma)
        else:
            return self.alpha.T @ self.kernel(X,x)

    def _inc_dim_v(self, v):
        '''
        Helper function: increases dimension of a given vector by padding with zero.
        
        '''
        return np.pad(v, ((0,1),(0,0)))

    def _inc_dim_m(self, m):
        '''
        Helper function: increases dimension of a given matrix by padding with zeros.
        
        '''
        return np.pad(m, ((0,1),(0,1)))

    def update(self, X, x, y):
        '''
        Updates model parameters given a new training (input, output) example 
        
        Parameters
        ----------
        X: np.array
            current basis
        x: np.array
            training input
        y: np.array
            training output
       
        Returns
        -------
        add: bool
            binary parameter indicating whether given training example should be added in the basis
        
        '''
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

        return add
