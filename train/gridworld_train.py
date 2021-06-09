import gym 
from custom_envs.windy_gridworld import WindyGridworldEnv

from function_approximators.function_approximators import NeuralNetwork, LinearModel
from train_utils import train

from agents.agents import DQNAgent, LinearAgent

RENDER = False
env = WindyGridworldEnv()


# DQN Config
CONFIG_DQN = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 10,
    "learning_rate": 0.001,
    "hidden_size": (16,16),
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
    "eval_episodes": 10,
    "learning_rate": 0.005,
    "target_update_freq": 200,
    "batch_size": 32,
    "gamma": 0.99,
    "buffer_capacity": int(1e7),
    "plot_loss": False,
    "epsilon": 1,
    "max_deduct": 0.95,
    "decay": 0.5,
    "lr_step_size": 1000,
    "lr_gamma": 0.99,
    "max_steps": 200,
    "poly_degree": 1,
}


function_approximators = [NeuralNetwork, LinearModel]
agents = [DQNAgent, LinearAgent]
configs = [CONFIG_DQN, CONFIG_LINEAR]

if __name__ == "__main__":

    print(configs[0])
    _ = train(env, 
            configs[0], 
            fa=function_approximators[0], 
            agent = agents[0], 
            render=RENDER)
    env.close()
