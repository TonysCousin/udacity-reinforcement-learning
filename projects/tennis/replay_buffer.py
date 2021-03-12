# Defines an agent that uses the DDPG algorithm.
#
# This code is a copy of the code used for my continuous control (Reacher) project.
#
# This code is based on code provided by Udacity instructor staff
# for the DRL nanodegree program.

import numpy as np
import torch
import random
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int):  size of each training batch
            seed (float):      seed used for the random number generator
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    

    def sample(self):
        """Randomly sample a batch of experiences from memory.

           Return: tuple of the individual elements of experience:
                     states, actions, rewards, next_states, dones
                   Each element is a tensor with width of whatever that element needs to represent
                   a single agent.  Number of rows is num_agents * batch_size, so the tensor
                   is a stack of batches of [num_agents, element_width]
        """

        experiences = random.sample(self.memory, k=self.batch_size) #returns list of experiences

        if len(experiences) == self.batch_size:
            states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])) \
                                           .float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences]) \
                                               .astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""

        return len(self.memory)
