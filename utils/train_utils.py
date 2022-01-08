import numpy as np
from tqdm import tqdm
import time
from function_approximators.replay import ReplayBuffer
import matplotlib.pyplot as plt
from IPython import display

def play_episode(
    env,
    agent,
    replay_buffer,
    non_param,
    max_steps,
    online=False,
    threshold=-1,
    batch_size=32,
    train=True,
    explore=True,
    render=False,
):

    '''
    Plays an entire episode of a Markov Decision Process

    Parameters
    ----------
    env: gym.env
        environment from the Gym library
    agent: agents.agent
        reinforement learning agent
    replay_buffer: function_approximators.replay
        replay buffer 
    non_param: bool
        indicates whether a non-parameric model is used
    max_steps: int
        maximum steps in the episode
    online: bool
        indicates whether updates happen online
    threshold: float
        replay buffer parameter - determines threshold for adding new transitions
    batch_size: int
        batch size
    train: bool
        indicates whether training happens
    explore: bool
        indicates whether the agent explors (instead it just exploits its current knowledge)
    render: bool
        indicates whether the environment is rendered  
        
    Returns
    -------
    episode_timesteps: int
        number of timesteps in the episode before termination
    episode_return: float
        episodic return
    losses: list
        losses after each update
    
    '''  
    # resets environment
    obs = env.reset()
    # renders environment
    if render:
        env.render()
    
    # initialises parameters
    done = False
    losses = []
    episode_timesteps = 0
    episode_return = 0
 
    ## Plays episode until termination (done=1)
    while not done:
        # agent selects action
        action = agent.act(obs, explore=explore)
        # agent takes that action in the environment
        next_obs, reward, done, _ = env.step(action) 
        
        ## if training, update model parameters
        if train and not online:  
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array([action], dtype=np.float32),
                np.array(next_obs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
                if threshold > -1:
                    batch = replay_buffer.sample(replay_buffer.count)
                else:
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

        # render environment after each step
        if render:
            env.render()

        # break if maximum steps have been reached
        if max_steps == episode_timesteps:
            break
        obs = next_obs

    return episode_timesteps, episode_return, losses


def train(env, config, fa, agent, output = True, render=False, online=False, threshold=-1):

    '''
    Performs a training epoch by collecting data samples through running episodes and updating model parameters

    Parameters
    ----------
    env: gym.env
        environment from the Gym library
    agent: agents.agent
        reinforement learning agent
    config: dict
        hyperparameter values specified by user 
    fa: function_approximators.model
        function approximation model
    output: bool
        indicates whether informative messages are displayed during training
    online: bool
        indicates whether updates happen online
    threshold: float
        replay buffer parameter - determines threshold for adding new transitions
    render: bool
        indicates whether the environment is rendered  
        
    Returns
    -------
    eval_returns_all: np.array
        array with the evaluation return of each episode
    eval_times_all: np.array
        array with the evaluation times of each episode
    train_returns: np.array
        array with the training return of each episode
    train_timesteps: np.array
        array with the training times of each episode
    
    ''' 
    timesteps_elapsed = 0
    
    ## defines reinforcement learning agent and replay buffer
    agent = agent(
        action_space = env.action_space, 
        observation_space = env.observation_space,
        fa=fa, 
        **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"], threshold)

    ## initialises parameters
    eval_returns_all = []
    eval_times_all = []

    start_time = time.time()
    losses_all = []

    train_timesteps = []
    train_returns = []
    
    with tqdm(total=config["max_timesteps"]) as pbar:
        
        ## runs episodes until maximum number of timesteps is reached 
        while timesteps_elapsed < config["max_timesteps"]:
            elapsed_seconds = time.time() - start_time
            
            # breaks if maximum time has passed
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break
            
            # updates values of hyperparameters
            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
            
            ## plays an episode
            episode_timesteps, train_return, losses = play_episode(
                env,
                agent,
                replay_buffer,
                train=True,
                explore=True,
                render=False,       
                online=online,  
                threshold=threshold,     
                non_param = config["non_param"],
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )
            
            timesteps_elapsed += episode_timesteps

            train_timesteps.append(timesteps_elapsed)
            train_returns.append(train_return) 

            pbar.update(episode_timesteps)
            losses_all += losses   

            ## evaluates current policy every certain number of timesteps
            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_returns = 0

                ## runs an evaluation episode a certain number of times
                for _ in range(config["eval_episodes"]):
                    episode_timesteps, episode_return, _ = play_episode(
                        env,
                        agent,
                        replay_buffer,
                        train=False,
                        explore=False,
                        render=render,
                        online=online,
                        threshold=threshold,
                        non_param = config["non_param"],
                        max_steps = config["max_steps"],
                        batch_size=config["batch_size"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]

                ## displays informative messages during training
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean returns of {eval_returns}"
                    )
                    pbar.write(f"Epsilon = {agent.epsilon}")
                    if online:
                        pbar.write(f"Support Points = {agent.X.shape[0]}")
                    if not config["non_param"]:
                        pbar.write(f"Learning rate = {agent.model_optim.param_groups[0]['lr']}")
                    if threshold > -1:
                        pbar.write(f"Replay Buffer count: {replay_buffer.count}")
                
                ## stores evaluation returns after each evaluation frame
                eval_returns_all.append(eval_returns)
                eval_times_all.append(time.time() - start_time)                  
       
    return np.array(eval_returns_all), np.array(eval_times_all), np.array(train_returns), np.array(train_timesteps)


def solve(env, config, fa, agent, target_return, op, render=False, online=False, threshold=-1):
    '''
    Similar to the train() function but instead of updating the model parameters at each episode, it counts whether the MDP was solved

    Parameters
    ----------
    env: gym.env
        environment from the Gym library
    agent: agents.agent
        reinforement learning agent
    config: dict
        hyperparameter values specified by user 
    fa: function_approximators.model
        function approximation model
    op: operator
        operator used to check whether solving condition has been met
    online: bool
        indicates whether updates happen online
    threshold: float
        replay buffer parameter - determines threshold for adding new transitions
    render: bool
        indicates whether the environment is rendered  
    
    timesteps_elapsed: int
        number of timesteps elapsed in the training epoch
    n_eps: int
        number of episodes elapsed in the training epoch
    n: list
        losses after each update
    
    '''
    timesteps_elapsed = 0
    
    ## defines reinforcement learning agent and replay buffer
    agent = agent(
        action_space = env.action_space, 
        observation_space = env.observation_space,
        fa=fa, 
        **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"], threshold)
    
    n_eps = 0
    start_time = time.time()

    ## runs episodes until maximum number of timesteps has been reached
    while timesteps_elapsed < config["max_timesteps"]:
        elapsed_seconds = time.time() - start_time
        
        ## breaks if maximum time has elapsed
        if elapsed_seconds > config["max_time"]:
            break
        
        ## updates model hyperparameter values
        agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
        episode_timesteps, _, _ = play_episode(
            env,
            agent,
            replay_buffer,
            online=online,
            train=True,
            explore=True,
            render=False,    
            threshold=threshold,          
            non_param = config["non_param"],
            max_steps=config["episode_length"],
            batch_size=config["batch_size"],
        )
        
        timesteps_elapsed += episode_timesteps
        n_eps += 1
        eval_returns = 0
        
        ## runs an evaluation frame after each passed episode
        for _ in range(config["eval_episodes"]):
            episode_timesteps, episode_return, _ = play_episode(
                env,
                agent,
                replay_buffer,
                online=online,
                train=False,
                explore=False,
                render=render,
                threshold=threshold,
                non_param = config["non_param"],
                max_steps = config["max_steps"],
                batch_size=config["batch_size"],
            )
            eval_returns += episode_return / config["eval_episodes"]

        ## checks whether solving condition has been met
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


def train_time(env, config, fa, agent, online=False, threshold=-1):
    '''
    Measures the training time of each algorithm. Similar to the train() function but no evaluation is done.

    Parameters
    ----------
    env: gym.env
        environment from the Gym library
    agent: agents.agent
        reinforement learning agent
    config: dict
        hyperparameter values specified by user 
    fa: function_approximators.model
        function approximation model
    online: bool
        indicates whether updates happen online
    threshold: float
        replay buffer parameter - determines threshold for adding new transitions

    elapsed_seconds: float
        seconds elapsed since training started
    
    '''
    start_time = time.time() 
    timesteps_elapsed = 0
    
    ## defines reinforcement learning agent and replay buffer
    agent = agent(
        action_space = env.action_space, 
        observation_space = env.observation_space,
        fa=fa, 
        **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"], threshold)
   
    with tqdm(total=config["max_timesteps"]) as pbar:
    
        ## runs episodes until the maximum number of timesteps is reached
        while timesteps_elapsed < config["max_timesteps"]:
            
            ## updates hyperparameter values
            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
            
            ## plays an episode
            episode_timesteps, episode_returns, _ = play_episode(
                env,
                agent,
                replay_buffer,
                online=online,
                train=True,
                explore=True,
                render=False,   
                threshold=threshold,           
                non_param = config["non_param"],
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )

            pbar.update(episode_timesteps)
            
            timesteps_elapsed += episode_timesteps
            ## measures elapsed time
            elapsed_seconds = time.time() - start_time
    
    print(episode_returns)

    return elapsed_seconds
