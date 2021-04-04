# Provides a fixed-size replay buffer (without true priority) for randomly sampling past
# experiences. However, it recognizes experiences with positive rewards as valuable,
# so will retain them when they have hit the head of the queue.
# Assumes that the initial experiences added (until batch_size is reached) will be
# random garbage to prime the buffer, and will allow these primes to be pushed off the
# end, regardless of whether they have a good reward, and will not count them when
# reporting "good" experiences.

import numpy as np
import torch
import random
from collections import namedtuple, deque

REWARD_THRESHOLD = 0.0 # value above which is considered a "good" experience


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, prime_size, seed):
        """Initialize a ReplayBuffer object.
        Params
            buffer_size (int): maximum number of experiences that can be stored
            batch_size (int):  size of each training batch extracted during a sample
            prime_size (int):  initial number of experiences that are considered as random
                                 priming data (don't need to be kept once buffer is full)
            seed (float):      seed used for the random number generator
        """

        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.prime_size = prime_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
        self.rewards_exceed_threshold = 0
        self.num_primes = 0
    

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.

           Params:
               state, action, next_state (ndarray): [a, x] where a is number of agents, x is 
                                                      data width for that element
               reward, done (list): one item for each agent
        """

        # if the buffer is already full then (we don't want to lose good experiences)
        if len(self.memory) == self.buffer_size:

            # if we have already pushed the initial priming experiences off the end then
            if self.num_primes == 0:

                # if < 50% of the buffer's contents are good experiences then
                if self.rewards_exceed_threshold < self.buffer_size//2:

                    # while we have a desirable reward at the left end of the deque
                    while max(self.memory[0].reward) > REWARD_THRESHOLD:

                        # pop it off and push it back onto the right end to save it
                        self.memory.rotate(-1)

            # else (some priming experiences still exist)
            else:

                # reduce the prime count since one will be pushed off the left end
                self.num_primes -= 1

        # if number of entries < prime size, then count the entry as a prime
        if len(self.memory) < self.prime_size:
            self.num_primes += 1

        else:
            # if the incoming experience has a good reward, then increment the count
            if max(reward) > REWARD_THRESHOLD:
                self.rewards_exceed_threshold += 1
    
        # add the experience to the right end of the deque
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)


    def num_rewards_exceeding_threshold(self):
        """Returns the number of rewards in database that exceed the threshold of 'good' """

        return self.rewards_exceed_threshold


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

        # randomly sample enough experiences from the buffer for one batch
        experiences = []
        if len(self.memory) >= self.batch_size:
            experiences = random.sample(self.memory, k=self.batch_size) #returns list of experiences

            # create empty tensors to hold all of the experience data
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
                states[i, :, :] = torch.from_numpy(e.state)
                actions[i, :, :] = torch.from_numpy(e.action)
                next_states[i, :, :] = torch.from_numpy(e.next_state)

                # incoming reward and done are lists, not tensors
                rewards[i, :, :] = torch.tensor(e.reward).view(num_agents, -1)[:, :]
                dones[i, :, :]   = torch.tensor(e.done).view(num_agents, -1)[:, :]

            return (states, actions, rewards, next_states, dones)

        else:
            print("\n///// ReplayBuffer.sample: unexpected experiences length = {} but batch size = {}"
                  .format(len(experiences), self.batch_size))
            return None


    def __len__(self):
        """Return the current number of items in internal memory."""

        return len(self.memory)
