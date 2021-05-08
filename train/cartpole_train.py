import gym 

from function_approximators.function_approximators import NeuralNetwork, LinearModel
from train_utils import play_episode, train

from agents.agents import DQN_Agent, Linear_Agent

RENDER = False
env = gym.make("CartPole-v1")

CONFIG = {
    "env": "CartPole-v1",
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


function_approximators = [NeuralNetwork, LinearModel]
agents = [DQN_Agent, Linear_Agent]

if __name__ == "__main__":

    _ = train(env, 
            CONFIG, 
            fa=function_approximators[1], 
            agent = agents[1], 
            render=RENDER)
    env.close()
