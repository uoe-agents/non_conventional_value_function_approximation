import gym 

from function_approximators.function_approximators import DecisionTree, NeuralNetwork, LinearModel
from train_utils import train

from agents.agents import DQNAgent, LinearAgent, NonParametricAgent

RENDER = True
env = gym.make("CartPole-v1")


# DQN Config
CONFIG_DQN = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 10,
    "learning_rate": 0.001,
    "hidden_size": (16,32),
    "target_update_freq": 200,
    "batch_size": 32,
    "gamma": 0.99,
    "buffer_capacity": int(1e7),
    "plot_loss": False,
    "epsilon": 1,
    "max_deduct": 0.97,
    "decay": 0.5,
    "lr_step_size": 1000,
    "lr_gamma": 0.99,
    "max_steps": 200,
}

# Linear Config
CONFIG_LINEAR = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 5,
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
    # "tiling_specs": [[-2,-0.2], [2,0.2], [([20,20],[0,0]), ([20,20],[0.2,0.02]), ([20,20],[-0.2,-0.02])]]
}

# Decision Tree Config
CONFIG_DT = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 10,
    "model_save_freq": 1000,
    "batch_size": 512,
    "gamma": 0.99,
    "buffer_capacity": int(1e7),
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.5,
    "max_steps": 200,
    "non_param": True,
    "max_depth": 10, 
    "min_samples_split": 20, 
    "min_samples_leaf": 10,
}


function_approximators = [NeuralNetwork, LinearModel, DecisionTree]
agents = [DQNAgent, LinearAgent, NonParametricAgent]
configs = [CONFIG_DQN, CONFIG_LINEAR, CONFIG_DT]

if __name__ == "__main__":

    i = 2
    print(configs[i])
    _ = train(env, 
            configs[i], 
            fa=function_approximators[i], 
            agent = agents[i], 
            render=RENDER)
    env.close()
