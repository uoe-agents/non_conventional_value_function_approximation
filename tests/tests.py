import torch
import numpy as np
import gym

env = gym.make("CartPole-v1")

print(env.observation_space.low)



x=torch.Tensor([[1,2,3],[1,2,3]])
print(x)
# x = x.unsqueeze(1)
print(torch.cat([x ** i for i in range(1, 3)],-1))

def _tilings_features(env, n_tilings, bin_sizes, offsets):

    tilings = []

    upper_bounds = env.observation_space.high
    lower_bounds = env.observation_space.low
    n_features = env.observation_space.shape[0]

    for i in range(n_tilings):
        bin = bin_sizes[i]
        offset = offsets[i]
        tiling = []

        for j in range(n_features):
            f_tiling = np.linspace(lower_bounds[i], upper_bounds[i], bin[j]+1)[1:-1] + offset[j]
            tiling.append(f_tiling)
        
        tilings.append(tiling)

    return np.array(tilings)



