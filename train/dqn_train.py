import gym 
from train_utils import play_episode, train

RENDER = False

CARTPOLE_CONFIG = {
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
    "plot_loss": True,
    "epsilon": 1,
    "max_steps": 200,
}

LUNARLANDER_CONFIG = {
    "env": "LunarLander-v2",
    "episode_length": 500,
    "max_timesteps": 300000,
    "max_time": 120 * 60,
    "eval_freq": 5000,
    "eval_episodes": 5,  
    "learning_rate": 8e-4,
    "hidden_size": (256, 128),
    "target_update_freq": 100,
    "batch_size": 50,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "plot_loss": False,
    "epsilon": 1,
    "max_steps": 500,
}


if __name__ == "__main__":
    CONFIG = LUNARLANDER_CONFIG
    print(CONFIG)

    env = gym.make(CONFIG["env"])
    _ = train(env, CONFIG, render=RENDER)
    env.close()
