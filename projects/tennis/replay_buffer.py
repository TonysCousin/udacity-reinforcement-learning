# Provides a simple replay buffer (without priority) for randomly sampling past
# experiences.
#
# This code is a copy of the code used for my continuous control (Reacher) project.
#
# This code is based on code provided by Udacity instructor staff
# for the DRL nanodegree program, although the sample() method is completely
# redesigned to be more intuitive.

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
                   Each item in the tuple is a tensor of shape (b, a, x), where
                     b is the number of items in a training batch (same for all elements)
                     a is the number of agents (same for all elements)
                     x is the number of items in that element (different for each element)
                   Each "row" (set of x values) represents a single agent.
        """

        experiences = random.sample(self.memory, k=self.batch_size) #returns list of experiences
        e0 = experiences[0]
        num_agents = e0.state.shape[0] #assume this applies to all elements
        states = torch.zeros(self.batch_size, num_agents, e0.state.shape[1], dtype=torch.float)
        actions = torch.zeros(self.batch_size, num_agents, e0.action.shape[1], dtype=torch.float)
        rewards = torch.zeros(self.batch_size, num_agents, 1, dtype=torch.float)
        next_states = torch.zeros(self.batch_size, num_agents, e0.next_state.shape[1], dtype=torch.float)
        dones = torch.zeros(self.batch_size, num_agents, 1, dtype=torch.float)

        if len(experiences) == self.batch_size:

            # loop through all the experiences, assigning each to a layer in the output tensor
            for i, e in enumerate(experiences):
                #print("\n...i = ", i, ", experience =\n", e)
                states[i, :, :] = torch.from_numpy(e.state).to(device)
                actions[i, :, :] = torch.from_numpy(e.action).to(device)
                next_states[i, :, :] = torch.from_numpy(e.next_state).to(device)

                # reward and done are lists, not tensors
                rewards[i, :, :] = torch.tensor(e.reward).view(num_agents, -1)[:, :]
                dones[i, :, :]   = torch.tensor(e.done).view(num_agents, -1)[:, :]
                rewards.to(device)
                dones.to(device)

            #print("replay_buffer.sample: states = ", states.shape)
            #print(states)
            #print("                      actions = ", actions.shape)
            #print(actions)
            #print("                      rewards = ", rewards.shape)
            #print(rewards)

            return (states, actions, rewards, next_states, dones)

        else:
            print("\n///// ReplayBuffer.sample: unexpected experiences length = {} but batch size = {}"
                  .format(len(experiences), self.batch_size))
            return None


    def __len__(self):
        """Return the current size of internal memory."""

        return len(self.memory)
