import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from copy import deepcopy

from function_approximators.replay import ReplayBuffer


def play_episode(
    env,
    agent,
    replay_buffer,
    non_param,
    max_steps,
    online=False,
    batch_size=32,
    train=True,
    explore=True,
    render=False,
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
           
        if train and not online:  
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array([action], dtype=np.float32),
                np.array(next_obs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                if non_param:
                    if not agent.fitted:
                        agent.initial_fit(batch)
                    else:
                        agent.update(batch)
                else:
                    loss = agent.update(batch)["q_loss"]
                    losses.append(loss)
        elif train:
            agent.update(obs, next_obs, reward, action, done)
        
        episode_timesteps += 1
        episode_return += reward

        if render:
            env.render()

        if max_steps == episode_timesteps:
            break
        obs = next_obs

    return episode_timesteps, episode_return, losses


def train(env, config, fa, agent, output = True, render=False, online=False):

    timesteps_elapsed = 0
    agent = agent(
        action_space = env.action_space, 
        observation_space = env.observation_space,
        fa=fa, 
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
                online=online,       
                non_param = config["non_param"],
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )
            
            timesteps_elapsed += episode_timesteps
            pbar.update(episode_timesteps)
            losses_all += losses   

            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_returns = 0

                for _ in range(config["eval_episodes"]):
                    episode_timesteps, episode_return, _ = play_episode(
                        env,
                        agent,
                        replay_buffer,
                        train=False,
                        explore=False,
                        render=render,
                        online=online,
                        non_param = config["non_param"],
                        max_steps = config["max_steps"],
                        batch_size=config["batch_size"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]

                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean returns of {eval_returns}"
                    )
                    pbar.write(f"Epsilon = {agent.epsilon}")
                    pbar.write(f"Support Points = {agent.X.shape[0]}")
                    if not config["non_param"]:
                        pbar.write(f"Learning rate = {agent.model_optim.param_groups[0]['lr']}")
                eval_returns_all.append(eval_returns)
                eval_times_all.append(time.time() - start_time)                  
       
    return np.array(eval_returns_all), np.array(eval_times_all)


def solve(env, config, fa, agent, target_return, op, render=False, online=False):

    timesteps_elapsed = 0
    agent = agent(
        action_space = env.action_space, 
        observation_space = env.observation_space,
        fa=fa, 
        **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"])
    n_eps = 0

    start_time = time.time()

    
    while timesteps_elapsed < config["max_timesteps"]:
        elapsed_seconds = time.time() - start_time
        
        if elapsed_seconds > config["max_time"]:
            # pbar.write(f"Training ended after {elapsed_seconds}s.")
            break
        
        agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
        episode_timesteps, _, _ = play_episode(
            env,
            agent,
            replay_buffer,
            online=online,
            train=True,
            explore=True,
            render=False,              
            non_param = config["non_param"],
            max_steps=config["episode_length"],
            batch_size=config["batch_size"],
        )
        
        timesteps_elapsed += episode_timesteps
        n_eps += 1

        eval_returns = 0
        for _ in range(config["eval_episodes"]):
            episode_timesteps, episode_return, _ = play_episode(
                env,
                agent,
                replay_buffer,
                online=online,
                train=False,
                explore=False,
                render=render,
                non_param = config["non_param"],
                max_steps = config["max_steps"],
                batch_size=config["batch_size"],
            )
            eval_returns += episode_return / config["eval_episodes"]

        if op(eval_returns, target_return):
            n=0
            print(f"Ep. timesteps: {episode_timesteps}")
            print(f"Total timesteps: {timesteps_elapsed}")
            print(f"Total episodes: {n_eps}")
            print(f"Evaluation mean return: {eval_returns}")
            # agent.model.export_tree(config["feature_names"], config["plot_name"])
            break
        else:
            n=1

    return timesteps_elapsed, n_eps, n