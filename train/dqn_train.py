import gym 
import numpy as numpy
import time
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt

from function_approximators.dqn_fa import ReplayBuffer
from agents.dqn_agent import DQN_Agent

RENDER = False

CARTPOLE_CONFIG = {
    "env": "CartPole-v1",
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, # HOW OFTEN WE EVALUATE (AND RENDER IF RENDER=TRUE)
    "eval_episodes": 5,
    "learning_rate": 8e-4,
    "hidden_size": (128,64),
    "target_update_freq": 200,
    "batch_size": 10,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "plot_loss": True, # SET TRUE FOR 3.3 (Understanding the Loss)
}


def play_episode(
    env,
    agent,
    replay_buffer,
    train=True,
    explore=True,
    render=False,
    max_steps=200,
    batch_size=64,
):

    obs = env.reset()
    done = False
    losses = []
    if render:
        env.render()

    episode_timesteps = 0
    episode_return = 0

    while not done:
        action = agent.act(obs, explore=explore)
        next_obs, reward, done, _ = env.step(action)
        if train:
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array([action], dtype=np.float32),
                np.array(next_obs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.update(batch)["q_loss"]
                losses.append(loss)

        episode_timesteps += 1
        episode_return += reward

        if render:
            env.render()

        if max_steps == episode_timesteps:
            break
        obs = next_obs

    return episode_timesteps, episode_return, losses


def train(env: gym.Env, config, output: bool = True) -> Tuple[List[float], List[float]]:

    timesteps_elapsed = 0
    agent = DQN_Agent(
        action_space=env.action_space, 
        observation_space=env.observation_space, 
        **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
    eval_times_all = []

    start_time = time.time()
    losses_all = []

    with tqdm(total=config["max_timesteps"]) as pbar:
        
        while timesteps_elapsed < config["max_timesteps"]:
            elapsed_seconds = time.time() - start_time
            
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break
            
            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
            episode_timesteps, _, losses = play_episode(
                env,
                agent,
                replay_buffer,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )
            
            timesteps_elapsed += episode_timesteps
            pbar.update(episode_timesteps)
            losses_all += losses

            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_returns = 0
                if config["env"] == "CartPole-v1":
                    max_steps = CARTPOLE_MAX_EPISODE_STEPS
                elif config["env"] == "LunarLander-v2":
                    max_steps = LUNARLANDER_MAX_EPISODE_STEPS
                else:
                    raise ValueError(f"Unknown environment {config['env']}")
                
                for _ in range(config["eval_episodes"]):
                    _, episode_return, _ = play_episode(
                        env,
                        agent,
                        replay_buffer,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=max_steps,
                        batch_size=config["batch_size"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]
                
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean returns of {eval_returns}"
                    )
                    pbar.write(f"Epsilon = {agent.epsilon}")
                eval_returns_all.append(eval_returns)
                eval_times_all.append(time.time() - start_time)
        
    if config["plot_loss"]:
        print("Plotting DQN loss...")
        losses = np.array(losses_all)
        x_values = config["batch_size"] + np.arange(len(losses))
        plt.plot(x_values, losses, "-", alpha=0.7, label=f"DQN loss")
        plt.legend(loc="best")
        plt.xlabel("Timesteps")
        plt.ylabel("Loss")
        plt.tight_layout(pad=0.3)
        plt.savefig(f"DQN_loss.pdf", format="pdf")
        plt.show()

    return np.array(eval_returns_all), np.array(eval_times_all)


if __name__ == "__main__":
    CONFIG = CARTPOLE_CONFIG
    print(CONFIG)
    env = gym.make(CONFIG["env"])
    print(env.action_space)
    _ = train(env, CONFIG)
    env.close()
