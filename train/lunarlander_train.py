import gym 
import operator
import numpy as np

from function_approximators.function_approximators import NeuralNetwork, LinearModel, DecisionTree, RandomForest, SupportVectorRegressor, KNeighboursRegressor, GaussianProcessRegressor, OnlineGaussianProcess
from utils.train_utils import train, solve, train_time

from agents.agents import DQNAgent, LinearAgent, FQIAgent, OnlineGaussianProccessAgent
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import rbf_kernel

RENDER = False
env = gym.make("LunarLander-v2")
environment = "lunarlander"


# DQN Config
CONFIG_DQN = {
    "episode_length": 500,
    "max_timesteps": 200000,
    "max_time": 30 * 60,
    "eval_freq": 10000, 
    "eval_episodes": 10,
    "learning_rate": 0.0015,
    "hidden_size": (256,128),
    "target_update_freq": 100,
    "batch_size": 64,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "plot_loss": False,
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.1,
    "lr_step_size": 1000,
    "lr_gamma": 0.99,
    "max_steps": 500,
    "non_param": False,
}

# Linear Config
CONFIG_LINEAR = {
    "episode_length": 500,
    "max_timesteps": 200000,
    "max_time": 30 * 60,
    "eval_freq": 10000, 
    "eval_episodes": 10,
    "learning_rate": 0.02,
    "target_update_freq": 50,
    "batch_size": 64,
    "gamma": 0.99,
    "buffer_capacity": int(1e5),
    "plot_loss": False,
    "epsilon": 1,
    "max_steps": 500,
    "poly_degree": 4,
    "max_deduct": 0.95,
    "decay": 0.1,
    "lr_step_size": 1000,
    "lr_gamma": 0.99,
    "non_param": False,
}

# Decision Tree Config
CONFIG_DT = {
    "episode_length": 500,
    "max_timesteps": 200000,
    "max_time": 30 * 60,
    "eval_freq": 10000, 
    "eval_episodes": 10,
    "model_save_freq": 0,
    "model_save_capacity": 0,
    "update_freq": 1,
    "batch_size": 128,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "epsilon": 1,
    "max_deduct": 0.9,
    "decay": 0.4,
    "max_steps": 500,
    "non_param": True,
    "model_params": {"criterion":"mse","max_depth": 20, "min_samples_split": 20, "min_samples_leaf": 5},
    "feature_names": ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity", "Action: Push Left", "Action: Push Right"],
    "plot_name": "dt_depth=8",
}

# Random Forest Config
CONFIG_RF = {
    "episode_length": 500,
    "max_timesteps": 200000,
    "max_time": 30 * 1000,
    "eval_freq": 10000, 
    "eval_episodes": 10,
    "model_save_freq": 0,
    "model_save_capacity": 0,
    "update_freq": 10,
    "batch_size": 128,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "epsilon": 1,
    "max_deduct": 0.9,
    "decay": 0.4,
    "max_steps": 500,
    "non_param": True,
    "model_params": {"n_estimators": 10,"max_depth": 20, "min_samples_split": 20, "min_samples_leaf": 5},
}

# Support Vector Regressor Config
CONFIG_SVR = {
    "episode_length": 500,
    "max_timesteps": 200000,
    "max_time": 30 * 1000,
    "eval_freq": 10000, 
    "eval_episodes": 10,
    "model_save_freq": 0,
    "model_save_capacity": 0,
    "update_freq": 200,
    "batch_size": 128,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.3,
    "max_steps": 500,
    "non_param": True,
    "model_params": {"kernel":"rbf", "degree": 2, "C": 1.2},
}


# K-Neighbors Regressor Config
CONFIG_KNR = {
    "episode_length": 500,
    "max_timesteps": 200000,
    "max_time": 30 * 1000,
    "eval_freq": 10000, 
    "eval_episodes": 10,
    "model_save_freq": 0,
    "model_save_capacity": 0,
    "update_freq": 100,
    "batch_size": 128,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "epsilon": 1,
    "max_deduct": 0.93,
    "decay": 0.4,
    "max_steps": 500,
    "non_param": True,
    "model_params": {"n_neighbors":10, "weights": "distance", "algorithm": "auto", "leaf_size": 30},
}

# Gaussian Process Config
CONFIG_GP = {
    "episode_length": 500,
    "max_timesteps": 200000,
    "max_time": 30 * 1000,
    "eval_freq": 10000, 
    "eval_episodes": 10,
    "model_save_freq": 0,
    "model_save_capacity": 0,
    "update_freq": 100,
    "batch_size": 128,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "epsilon": 1,
    "max_deduct": 0.9,
    "decay": 0.4,
    "max_steps": 500,
    "non_param": True,
    "model_params": {"alpha": 1e-10, "normalize_y": False, "kernel":  RBF(length_scale=0.3, length_scale_bounds="fixed")},
}

# Online Gaussian Process Config
CONFIG_GP_Online = {
    "episode_length": 500,
    "max_timesteps": 200000,
    "max_time": 30 * 1000,
    "eval_freq": 10000, 
    "eval_episodes": 10,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "batch_size": 32,
    "epsilon": 1,
    "max_deduct": 0.93,
    "decay": 0.4,
    "max_steps": 500,
    "non_param": True,
    "model_params": {"sigma_0": 0.3, "init": 0, "kernel":  rbf_kernel, "epsilon_tol": 0.075, "basis_limit": 1000},
}

CONFIGS = [CONFIG_DQN, CONFIG_LINEAR, CONFIG_DT, CONFIG_RF, CONFIG_SVR, CONFIG_KNR, CONFIG_GP, CONFIG_GP_Online]
onlines = [False, False, False, False, False, False, False, True]
function_approximators = [NeuralNetwork, LinearModel, DecisionTree, RandomForest, SupportVectorRegressor, KNeighboursRegressor, GaussianProcessRegressor, OnlineGaussianProcess]
agents = [DQNAgent, LinearAgent, *[FQIAgent]*5, OnlineGaussianProccessAgent]
legends = ["Neural Network", "Linear Model", "Decision Tree", "Random Forest", "Support Vectors", "K-Neighbours", "Gaussian Process", "Gaussian Process Online"]

n_seeds = 30

if __name__ == "__main__":

    times = []
    for j in range(len(function_approximators)):
        
        returns = []
        train_returns = []
        train_times = []
        for i in range(n_seeds):
            print(f"\n Run: {i+1} \n")
            r, _, t, times = train(env, 
                    CONFIGS[j], 
                    fa=function_approximators[j], 
                    agent = agents[j], 
                    render=RENDER,
                    online=onlines[j],
                    threshold = 0)
            env.close()
            returns.append(r)
            train_returns.append(t)
            train_times.append(times)

        with open(f'{environment}_eval_{legends[j]}.csv', 'ab') as eval:
            for i in range(n_seeds):
                np.savetxt(eval, [returns[i]], delimiter=',')
        
        with open(f'{environment}_train_{legends[j]}.csv', 'ab') as train:
            for i in range(n_seeds):
                np.savetxt(train, [train_returns[i]], delimiter=',')
                np.savetxt(train, [train_times[i]], delimiter=',')

        n_eps = []
        n_steps = []
        not_solved = []
        for i in range(n_seeds):
            print(f"\n Run: {i+1} \n")
            s, e, n = solve(env, 
                    CONFIGS[j], 
                    fa=function_approximators[j], 
                    agent = agents[j],
                    target_return=195,
                    op=operator.ge, 
                    render=RENDER,
                    online=onlines[j])
            env.close()
            n_eps.append(e)
            n_steps.append(s)
            not_solved.append(n)

        with open(f'{environment}_{legends[j]}.csv', 'ab') as se:
            np.savetxt(se, [n_eps], delimiter=',')
            np.savetxt(se, [n_steps], delimiter=',')
            np.savetxt(se, [not_solved], delimiter=',')

        time = train_time(env, 
                CONFIGS[j], 
                fa=function_approximators[j], 
                agent = agents[j],
                online=onlines[j])
        env.close()
        times.append(time)

    with open(f'{environment}_times.csv', 'ab') as t:
        np.savetxt(t, [times], delimiter=',')