# Implements the MADDPG algorithm for multiple agents, where each agent has its own
# actor, but all agents share a critic.  The critic is a member of this class,
# but it is updated with all actions and observations from all agents, so
# each agent's critic is being simultaneously updated the same way.  This is
# not the most efficient way to do it, but for learning the basic approach it
# is fine.  A future enhancement would be to break out a common critic object
# that is truly separate from all actors.
#
# This code is heavily inspired by
# https://github.com/and-buk/Udacity-DRLND/blob/master/p_collaboration_and_competition/MADDPG.py

import numpy as np
import torch
import random

from ddpg_agent    import DdpgAgent

class Maddpg:
    """Manages the training and execution of multiple agents in the same environment"""

    def __init__(self, state_size, action_size, num_agents, random_seed=0, batch_size=32,
                 noise_decay=1.0, learn_every=20, learn_iter=1):
        """Initialize the one and only MADDPG manager

        Params
            state_size (int):     number of state values for each actor
            action_size (int):    number of action values for each actor
            random_seed (int):    random seed
            batch_size (int):     the size of each minibatch used for learning
            noise_decay (float):  multiplier on the magnitude of noise; decay is applied each time step (must be <= 1.0)
            learn_every (int):    number of time steps between learning sessions
            learn_iter (int):     number of learning iterations that get run during each learning session
        """

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        random.seed(random_seed)

        # define simple replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, batch_size, random_seed)

        # create a list of agent objects
        self.agents = [DdpgAgent(state_size, action_size, random_seed, batch_size,
                                 noise_decay, learn_every, learn_iter)
                       for a in range(num_agents)]


    def reset(self):
        """Resets all agents to an initial state to begin another episode"""

        for a in self.agents:
            a.reset()


    def act(self, states, add_noise=True):
        """Invokes each agent to perform an inference step using its current policy

           Params:
               states (tuple of float tensors):  the state values for all agents
               add_noise (bool):                 should noise be added to the results?

           Return:  ndarray of actions taken by all agents
        """

        print("Maddpg.act: states = ", states)
        actions = np.zeros(self.num_agents, self.action_size)
        print("Maddpg.act: actions = ", actions)
        for i, a in enumerate(self.agents):
            a[i, :] = a.act(states[i], add_noise)
        return a


    def step(self, obs, actions, rewards, next_obs, dones):
        """Stores a new experience from the environment in replay buffer and advances
           the agents by one time step, invoking learning if appropriate.

           Params:
               obs (tuple of float ndarray):      the current state values for all agents
               actions (ndarray of float):        the current actions from all agents
               rewards (ndarray of floats):       current rewards earned from all agents
               next_obs (tuple of float ndarray): est of next time step's states for all agents
               dones (ndarray of bool):           for each agent, is episode complete?

           Return:  none
        """

        # add the new experience to the replay buffer
        memory.add(obs, actions, rewards, next_obs, dones)

        # advance each agent
        for i, a in enumerate(self.agents):
            a.step(i)

