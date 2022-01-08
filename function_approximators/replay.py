import numpy as np 
from collections import namedtuple
import torch
import scipy.spatial.distance as dist


Transition = namedtuple(
    "Transition", ("states", "actions", "next_states", "rewards", "done")
)

class ReplayBuffer:
    '''
    Represents a buffer that can be used to store and sample (state, action, next_state, reward, done) tuples from past environment interactions.
    
    Attributes
    ----------
    capacity: int
        replay buffer capacity
    memory: collections.namedtuple  
        stored transition tuples
    count: int
        number of stored transitions
    threshold: float
        threshold for adding a new example in the buffer

    Methods
    -------
    init_memory():
        Initialises replay buffer memory.
    push():
        Stores a given transition in the replay buffer.
    sample():
        Returns a batch of transitions from the replay buffer
    
    '''
    def __init__(self, capacity, threshold=0):

        self.capacity = capacity
        self.memory = None
        self.count = 0
        self.threshold = threshold
        

    def __len__(self): 

        return min(self.count, self.capacity)

    def init_memory(self, transition):
        '''
        Initialises the replay buffer.
        
        Parameters
        ----------
        trasition: collections.namedtuple
            represents an environment transition
        
        '''  
        self.memory = Transition(
            *[np.zeros([self.capacity, elem.size], dtype=elem.dtype) for elem in transition]
        )
        self.m = transition[0].shape[0] + transition[1].shape[0]
        self.matrix = np.zeros((1, self.m))

    def push(self, *args):
        '''
        Stores a transition in the replay buffer memory
        
        Parameters
        ----------
        *args
            represent an environment transition in the form (state, action, next_state, reward, done)

        '''  
        # create namedTuple
        transition = Transition(*args)
        
        # checks
        for t in transition:
            assert t.ndim == 1
        
        assert transition.states.size == transition.next_states.size
        assert transition.rewards.size == 1
        
        if not self.memory:
            self.init_memory(transition)
        
        # initialise replay buffer
        if self.threshold <= -1:
            # push transition in replay buffer
            index = self.count % self.capacity
            for i, arg in enumerate(args):
                self.memory[i][index, :] = arg
            # update count
            self.count += 1

        else:
            a = np.concatenate((args[0], args[1])).reshape(1,self.m)
            distances = dist.cdist(a,self.matrix,'euclidean')
            
            if np.min(distances) > self.threshold: 
                self.matrix = np.concatenate((self.matrix, a), axis=0)
                # push transition in replay buffer
                index = self.count % self.capacity
                for i, arg in enumerate(args):
                    self.memory[i][index, :] = arg
                # update count
                self.count += 1

    def sample(self, batch_size, device = "cpu"):
        '''
        Returns a batch of transitions from the replay buffer.
        
        Parameters
        ----------
        batch_size: int
            batch size

        Returns
        -------
        batch: collections.namedtuple
            batch of transitions
        
        '''  
        indices = np.random.randint(0, high=len(self), size=batch_size)

        batch = Transition(
            *[
                torch.from_numpy(np.take(d, indices, axis=0)).to(device)
                for d in self.memory
            ]
        )
        return batch        


