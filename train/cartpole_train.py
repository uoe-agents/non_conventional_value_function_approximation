import gym 

from function_approximators.function_approximators import NeuralNetwork, LinearModel
from train_utils import train

from agents.agents import DQNAgent, LinearAgent

RENDER = False
env = gym.make("CartPole-v1")


# DQN Config
CONFIG_DQN = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 5,
    "learning_rate": 8e-4,
    "hidden_size": (128,64),
    "target_update_freq": 200,
    "batch_size": 10,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "plot_loss": False,
    "epsilon": 1,
    "max_steps": 200,
}

# Linear Config
CONFIG_LINEAR = {
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, 
    "eval_episodes": 10,
    "learning_rate": 0.01,
    "target_update_freq": 200,
    "batch_size": 32,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "plot_loss": False,
    "epsilon": 1,
    "max_steps": 200,
    "poly_degree": 2,
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
