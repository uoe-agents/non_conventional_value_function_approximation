import gym 

from function_approximators.function_approximators import DecisionTree, NeuralNetwork, LinearModel, RandomForest, ExtraTrees, SupportVectorRegressor 
from train_utils import train

from agents.agents import DQNAgent, LinearAgent, NonParametricAgent
from custom_envs.mountain_car import MountainCarEnv

RENDER = True
# env = MountainCarEnv()
env = gym.make("Acrobot-v1")


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
    "model_params": {"max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 5},
}

function_approximators = [NeuralNetwork, LinearModel, DecisionTree, RandomForest, ExtraTrees, SupportVectorRegressor]
agents = [DQNAgent, LinearAgent, *[NonParametricAgent]*4]
configs = [CONFIG_DQN, CONFIG_LINEAR, CONFIG_DT]

if __name__ == "__main__":

    print(configs[2])
    _ = train(env, 
            configs[2], 
            fa=function_approximators[2], 
            agent = agents[2], 
            render=RENDER)
    env.close()
