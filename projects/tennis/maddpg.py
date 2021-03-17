# Implements the MADDPG algorithm for multiple agents, where each agent has its own
# actor, but all agents effectively share a critic.  The critic in each agent
# is updated with all actions and observations from all agents, so
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

from replay_buffer import ReplayBuffer
from maddpg_agent  import MultiDdpgAgent

# initial probability of keeping "bad" episodes (until enough exist to start learning)
BAD_STEP_KEEP_PROB_INIT = 0.5


class Maddpg:
    """Manages the training and execution of multiple agents in the same environment"""

    def __init__(self, state_size, action_size, num_agents, bad_step_prob=0.5, random_seed=0,
                 batch_size=32, buffer_size=1000000, noise_decay=1.0, learn_every=20, learn_iter=1):
        """Initialize the one and only MADDPG manager

        Params
            state_size (int):     number of state values for each actor
            action_size (int):    number of action values for each actor
            num_agents (int):     number of agents in the environment (all will be trained)
            bad_step_prob (float):probability of keeping a time step experience if it generates
                                    no reward
            random_seed (int):    random seed
            batch_size (int):     the size of each minibatch used for learning
            buffer_size (int):    capacity of the experience replay buffer
            noise_decay (float):  multiplier on the magnitude of noise; decay is applied each time step (must be <= 1.0)
            learn_every (int):    number of time steps between learning sessions
            learn_iter (int):     number of learning iterations that get run during each learning session
        """

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.bad_step_keep_prob = min(bad_step_prob, 1.0)
        random.seed(random_seed)
        self.batch_size = batch_size

        # define simple replay memory common to all agents
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)
        self.learning_underway = False

        # create a list of agent objects
        self.agents = [MultiDdpgAgent(state_size, action_size, num_agents, random_seed, self.memory,
                                      batch_size, noise_decay, learn_every, learn_iter)
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

        actions = np.zeros((self.num_agents, self.action_size))
        for i, agent in enumerate(self.agents):
            actions[i, :] = agent.act(states[i], add_noise)
        return actions


    def step(self, obs, actions, rewards, next_obs, dones):
        """Stores a new experience from the environment in replay buffer, if appropriate,
           and advances the agents by one time step.

           Params:
               obs (ndarray of float):      the current state values for all agents, one row per agent
               actions (ndarray of float):  the current actions from all agents, one row per agent
               rewards (list of float):     current reward earned from each agent
               next_obs (ndarray of float): est of next time step's states, one row per agent
               dones (list of bool):        for each agent, is episode complete?

           Return:  none
        """

        #print("maddpg.step: obs = ", obs)
        #print("             actions = ", actions)
        #print("             rewards = ", rewards, end="")

        # set up probability of keeping bad experiences based upon whether the buffer is
        # full enough to start learning
        if len(self.memory) > self.batch_size:
            threshold = self.bad_step_keep_prob
            self.learning_underway = True #let's object owner know
        else:
            threshold = BAD_STEP_KEEP_PROB_INIT

        # if this step did not score any points, then use random draw to decide if it's a keeper
        if max(rewards) > 0.0  or  np.random.random() < threshold:

            # add the new experience to the replay buffer
            self.memory.add(obs, actions, rewards, next_obs, dones)

            # advance each agent
            for i, a in enumerate(self.agents):
                a.step(i)


    def get_memory_stats(self):
        """Gets statistics on the replay buffer memory contents.

           Return:  tuple of (size, good_exp), where size is total number
                      of items in the buffer, and good_exp is the number of those
                      items with a reward that exceeds the threshold of "good".
        """

        return (len(self.memory), self.memory.num_rewards_exceeding_threshold())


    def is_learning_underway(self):
        return self.learning_underway


    def checkpoint(self, path, tag, episode):
        """Saves checkpoint files for each of the networks.

           Params:
               path (string): directory path where the files will go (if not None, needs to end in /)
               tag (string):  an aribitrary tag to distinguish this set of networks (e.g. test ID)
               episode (int): the episode number
        """

        for i, a in enumerate(self.agents):
            torch.save(a.actor_local.state_dict(), "{}{}_actor{}_{:d}.pt"
                       .format(path, tag, i, episode))
            torch.save(a.critic_local.state_dict(), "{}{}_critic{}_{:d}.pt"
                       .format(path, tag, i, episode))
