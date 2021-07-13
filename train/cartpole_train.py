import gym 
import numpy as np
import operator

from function_approximators.function_approximators import NeuralNetwork, LinearModel, DecisionTree, RandomForest, SupportVectorRegressor, KNeighboursRegressor, GaussianProcessRegressor, OnlineGaussianProcess
from utils.train_utils import train, solve

from agents.av_agents import DQNAgent, LinearAgent, FQIAgent, OnlineGaussianProccessAgent

RENDER = True
env = gym.make("CartPole-v1")


# DQN Config
CONFIG_DQN = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 10,
    "learning_rate": 0.00075,
    "hidden_size": (32,32),
    "target_update_freq": 200,
    "batch_size": 32,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "plot_loss": False,
    "epsilon": 1,
    "max_deduct": 0.97,
    "decay": 0.25,
    "lr_step_size": 1000,
    "lr_gamma": 0.95,
    "max_steps": 200,
    "non_param": False,
}

# Linear Config
CONFIG_LINEAR = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 10,
    "learning_rate": 0.02,
    "target_update_freq": 50,
    "batch_size": 32,
    "gamma": 0.99,
    "buffer_capacity": int(1e7),
    "plot_loss": False,
    "epsilon": 1,
    "max_steps": 200,
    "poly_degree": 2,
    "max_deduct": 0.97,
    "decay": 0.5,
    "lr_step_size": 1000,
    "lr_gamma": 0.99,
    "non_param": False,
}

# Decision Tree Config
CONFIG_DT = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 10,
    "model_save_freq": 1000,
    "model_save_capacity": 20,
    "update_freq": 1,
    "batch_size": 512,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.4,
    "max_steps": 200,
    "non_param": True,
    "model_params": {"criterion":"mse","max_depth": 10, "min_samples_split": 20, "min_samples_leaf": 5},
    "feature_names": ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity", "Action: Push Left", "Action: Push Right"],
    "plot_name": "dt_depth=8",
}

# Random Forest Config
CONFIG_RF = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 5,
    "model_save_freq": 1000,
    "model_save_capacity": 20,
    "update_freq": 5,
    "batch_size": 512,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.4,
    "max_steps": 200,
    "non_param": True,
    "model_params": {"n_estimators": 10,"max_depth": 10, "min_samples_split": 20, "min_samples_leaf": 5},
}

# Support Vector Regressor Config
CONFIG_SVR = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 10,
    "model_save_freq": 1000,
    "model_save_capacity": 20,
    "update_freq": 1,
    "batch_size": 512,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.3,
    "max_steps": 200,
    "non_param": True,
    "model_params": {"kernel":"rbf", "degree": 2, "C": 2},
}

# K-Neighbors Regressor Config
CONFIG_KNR = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 5,
    "model_save_freq": 1000,
    "model_save_capacity": 20,
    "update_freq": 1,
    "batch_size": 256,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.3,
    "max_steps": 200,
    "non_param": True,
    "model_params": {"n_neighbors":7, "weights": "distance", "algorithm": "auto", "leaf_size": 30},
}

# Gaussian Process Config
CONFIG_GP = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 5,
    "model_save_freq": 1000,
    "model_save_capacity": 20,
    "update_freq": 10,
    "batch_size": 512,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.3,
    "max_steps": 200,
    "non_param": True,
    "model_params": {"alpha": 1e-10, "normalize_y": False, "kernel":  RBF(length_scale=0.08, length_scale_bounds="fixed")},
}

# Online Gaussian Process Config
CONFIG_GP_Online = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 10,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "batch_size": 32,
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.3,
    "max_steps": 200,
    "non_param": True,
    "model_params": {"sigma_0": 0.5, "kernel":  rbf_kernel, "epsilon_tol": 0.05, "basis_limit": 1000},
}



function_approximators = [NeuralNetwork, LinearModel, DecisionTree, RandomForest, SupportVectorRegressor, KNeighboursRegressor, GaussianProcessRegressor, OnlineGaussianProcess]
agents = [DQNAgent, LinearAgent, *[FQIAgent]*5, OnlineGaussianProccessAgent]
configs = [CONFIG_DQN, CONFIG_LINEAR, CONFIG_DT, CONFIG_RF, CONFIG_SVR, CONFIG_KNR, CONFIG_GP, CONFIG_GP_Online]

legends = ["Neural Network", "Linear Model", "Decision Tree", "Random Forest", "Support Vectors", "K-Neighbours", "Gaussian Process", "Gaussian Process Online"]

n_seeds = 30

if __name__ == "__main__":

    for i in range(len(function_approximators)):
        
        returns = []
        for j in range(n_seeds):
            r, _ = train(env, 
                    configs[i], 
                    fa=function_approximators[i], 
                    agent = agents[i], 
                    render=RENDER,
                    online=False)
            env.close()
            returns.append(r)


        n_eps = []
        n_steps = []
        not_solved = []
        for j in range(n_seeds):
            print(f"\n Run: {i+1} \n")
            s, e, n = solve(env, 
                    CONFIG_LINEAR, 
                    fa=function_approximators[i], 
                    agent = agents[i],
                    target_return=195,
                    op=operator.ge, 
                    render=RENDER)
            env.close()
            n_eps.append(e)
            n_steps.append(s)
            not_solved.append(n)
