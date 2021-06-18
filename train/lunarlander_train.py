import gym 

from function_approximators.function_approximators import DecisionTree, NeuralNetwork, LinearModel, RandomForest, ExtraTrees, SupportVectorRegressor 
from train_utils import train

from agents.agents import DQNAgent, LinearAgent, NonParametricAgent
from custom_envs.mountain_car import MountainCarEnv

RENDER = True
env = gym.make("LunarLander-v2")


# DQN Config
CONFIG_DQN = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 3,
    "learning_rate": 0.005,
    "hidden_size": (32,32),
    "target_update_freq": 200,
    "batch_size": 32,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "plot_loss": False,
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.5,
    "lr_step_size": 1000,
    "lr_gamma": 0.99,
    "max_steps": 200,
    "non_param": False,
}

# Linear Config
CONFIG_LINEAR = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 3,
    "learning_rate": 0.001,
    "target_update_freq": 100,
    "batch_size": 32,
    "gamma": 0.99,
    "buffer_capacity": int(1e7),
    "plot_loss": False,
    "epsilon": 1,
    "max_steps": 200,
    "poly_degree": 2,
    "max_deduct": 0.95,
    "decay": 0.5,
    "lr_step_size": 1000,
    "lr_gamma": 0.99,
    "non_param": False,
}

# Decision Tree Config
CONFIG_DT = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 60 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 3,
    "model_save_freq": 1000,
    "model_save_capacity": 20,
    "model_update_freq": 1,
    "batch_size": 512,
    "gamma": 0.99,
    "buffer_capacity": int(1e5),
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.5,
    "max_steps": 200,
    "non_param": True,
    "model_params": {"max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 3},
}

# Support Vector Regressor Config
CONFIG_SVR = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 3,
    "model_save_freq": 1000,
    "model_save_capacity": 10,
    "model_update_freq": 1,
    "batch_size": 512,
    "gamma": 0.99,
    "buffer_capacity": int(1e5),
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.3,
    "max_steps": 200,
    "non_param": True,
    "model_params": {"kernel":"rbf", "degree": 2, "C": 1},
}

function_approximators = [NeuralNetwork, LinearModel, DecisionTree, RandomForest, ExtraTrees, SupportVectorRegressor]
agents = [DQNAgent, LinearAgent, *[NonParametricAgent]*4]
configs = [CONFIG_DQN, CONFIG_LINEAR, CONFIG_DT, CONFIG_DT, CONFIG_DT, CONFIG_SVR]

if __name__ == "__main__":

    i=5
    print(configs[i])
    _ = train(env, 
            configs[i], 
            fa=function_approximators[i], 
            agent = agents[i], 
            render=RENDER)
    env.close()
