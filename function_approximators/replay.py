# import torch.nn as nn
# import gym
import numpy as np 
from collections import namedtuple
import torch
import scipy.spatial.distance as dist


Transition = namedtuple(
    "Transition", ("states", "actions", "next_states", "rewards", "done")
)

class ReplayBuffer:

    def __init__(self, capacity, threshold=0):

        self.capacity = capacity
        self.memory = None
        self.count = 0
        self.threshold = threshold

    def __len__(self): 

        return min(self.count, self.capacity)

    def init_memory(self, transition):

        self.memory = Transition(
            *[np.zeros([self.capacity, elem.size], dtype=elem.dtype) for elem in transition]
        )

    def push(self, *args):

        # create namedTuple
        transition = Transition(*args)
        
        # checks
        for t in transition:
            assert t.ndim == 1
        
        assert transition.states.size == transition.next_states.size
        assert transition.rewards.size == 1
        
        # initialise replay buffer
        if not self.memory:
            self.init_memory(transition)
            # push transition in replay buffer
            index = self.count % self.capacity
            for i, arg in enumerate(args):
                self.memory[i][index, :] = arg

            # update count
            self.count += 1

        else:
            a = np.concatenate((args[0], args[1], args[2]))
            # print(a)
            distances = []
            for i in range(self.count):
                b = np.concatenate((self.memory[0][i], self.memory[1][i], self.memory[2][i]))
                distances.append(dist.euclidean(a,b))
            
            if min(distances) > self.threshold:
                # push transition in replay buffer
                index = self.count % self.capacity
                for i, arg in enumerate(args):
                    self.memory[i][index, :] = arg

                # update count
                self.count += 1

    def sample(self, batch_size, device = "cpu"):

        indices = np.random.randint(0, high=len(self), size=batch_size)

        batch = Transition(
            *[
                torch.from_numpy(np.take(d, indices, axis=0)).to(device)
                for d in self.memory
            ]
        )
        return batch        


