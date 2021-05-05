from torch import Tensor
from function_approximators.dqn import FCNetwork, ReplayBuffer
import numpy as np

model = FCNetwork([2,2])
print(model)
actions = model.forward(Tensor([0,0]))
print(actions)

replay = ReplayBuffer(10)
replay.push(
    np.array([0]),
    np.array([0,0]),
    np.array([0]),
    np.array([0]),
    np.array([0]))

print(replay.sample(1))

# for i in range(10):
#     print(i)
