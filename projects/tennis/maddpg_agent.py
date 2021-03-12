# Defines an agent that uses the DDPG algorithm in a MADDPG context.  That is, each agent
# has its own actor, but the critic looks at both actors and is essentially shared among
# the agents.
#
# This code is based on the code used for my continuous control (Reacher) project,
# but has been influenced by
# https://github.com/and-buk/Udacity-DRLND/blob/master/p_collaboration_and_competition/MADDPG.py
#
# This code is based on code provided by Udacity instructor staff
# for the DRL nanodegree program.

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import copy

from ou_noise      import OUNoise
from replay_buffer import ReplayBuffer
from model         import Actor, Critic


GAMMA = 0.99            # discount factor
TAU = 0.001             # for soft update of target parameters
LR_ACTOR = 0.0001       # learning rate of the actor
LR_CRITIC = 0.001       # learning rate of the critic
WEIGHT_DECAY = 1e-5     # L2 weight decay
NOISE_SCALE = 0.1       # scale factor applied to the raw noise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiDdpgAgent:
    """Interacts with and learns from the environment and other agents in the environment."""
    
    def __init__(self, state_size, action_size, random_seed, replay_buf, batch_size=32,
                 noise_decay=1.0, learn_every=20, learn_iter=1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int):          number of state values for this agent
            action_size (int):         number of action values for this agent
            random_seed (int):         seed for random number generator
            replay_buf (ReplayBuffer): buffer object holding experiences for replay (all agents)
            batch_size (int):          size of each minibatch used for learning
            noise_decay (float):       multiplier on the magnitude of noise; decay is applied
                                         each time step (must be <= 1.0)
            learn_every (int):         number of time steps between learning sessions
            learn_iter (int):          number of learning iterations that get run during each
                                         learning session
        """

        # don't seed the random package here; assume it has been done by caller
        # TODO:  understand why seeds are passed everywhere; isn't "random" a singleton that
        #        should only be seeded one time in the program?

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = replay_buf
        self.noise_mult = 1.0 #the noise multiplier that will get decayed
        self.noise_decay = min(noise_decay, 1.0) #guarantee that this won't make the noise grow
        self.learn_control = 0 #counts iterations between learning sessions
        self.learn_every = learn_every
        self.learn_iterations = learn_iter

        layer1_units = 400
        layer2_units = 256

        # Actor Network (w/ Target Network)
        self.actor_local  = Actor(state_size, action_size, random_seed,
                                  fc1_units=layer1_units, fc2_units=layer2_units).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed,
                                  fc1_units=layer1_units, fc2_units=layer2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local  = Critic(state_size, action_size, random_seed,
                                    fcs1_units=layer1_units, fc2_units=layer2_units).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed,
                                    fcs1_units=layer1_units, fc2_units=layer2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    
    def step(self, agent_id):
        """Use a random sample from buffer to learn.

           Params:
              agent_id (int): index of this agent in the replay arrays (values 0..N)
        """

        # Learn, if enough samples are available in memory, but only occasionally
        self.learn_control += 1
        if len(self.memory) > self.batch_size  and  self.learn_control > self.learn_every:
            self.learn_control = 0
            for j in range(self.learn_iterations):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, agent_id)


    def act(self, state, add_noise=True):
        """Returns actions for given state per current policy."""

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # add noise with gradual decay
        if add_noise:
            n = self.noise.sample() * NOISE_SCALE
            #print("act: action = ", action) #debug
            #print("     noise  = ", n)
            action += n*self.noise_mult
            self.noise_mult *= self.noise_decay
            if self.noise_mult < 0.001:
                self.noise_mult = 0.001

        return np.clip(action, -1, 1)

    def reset(self):
        """Reset the noise generator."""

        self.noise.reset()

    def learn(self, experiences, gamma, agent_id):
        """Update policy and value parameters using given batch of experience tuples.
           Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
           where:
               actor_target(state) -> action
               critic_target(state, action) -> Q-value

           Params
               experiences (Tuple of tensors): (s, a, r, s', done), where
                 each tensor is a stack batch_size high. Within each element of the
                 stack is num_agents rows; each row represents a single agent's info.
               gamma (float): discount factor
        """

        # extract the elements of the replayed experience row
        obs, actions, rewards, next_obs, dones = experiences
        print("learn: experiences =\n", experiences)
        print("learn: obs = ", obs.shape) #debug
        print("       rewards = ", rewards, rewards.shape)
        print("       next_obs = ", next_obs)
        print("       dones = ", dones, dones.shape)
        print("       actions = ", actions)
        print("       NOTE: actions don't get split out!")

        # split the state tensors out for each actor from their tensors
        states = torch.chunk(obs, 2, dim=1)
        next_states = torch.chunk(next_obs, 2, dim=1)
        print("learn: states = ", states) #debug

        # split out the rewards and dones for each actor from their tensors
        reward = rewards[:, agent_id].reshape(rewards[0], 1)
        done = dones[:, agent_id].rehsape(dones[0], 1)
        print("       reward = ", reward)


        # ---------------------------- update critic ---------------------------- #

        # Get predicted next-state actions and Q values from target models
        next_actions = [self.actor_target(s) for s in next_states]
        target_actions = torch.cat(next_actions, dim=1).to(device)
        Q_targets_next = self.critic_target(next_states, target_actions)
        print("       next_actions   = ", next_actions)   #debug
        print("       target_actions = ", target_actions)
        print("       Q_targets_next = ", Q_targets_next)

        # Compute Q targets for current states (y_i)
        Q_targets = reward + gamma*Q_targets_next*(1 - done)

        # Compute critic loss
        Q_expected = self.critic_local(obs, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        print("       critic_loss = ", critic_loss) #debug

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #

        # Compute actor loss
        ap = [self.actor_local(s) for s in states]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        print("       actions_pred = ", actions_pred) #debug
        actor_loss = -self.critic_local(obs, actions_pred).mean() #can't detach() here

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
           θ_target = τ*θ_local + (1 - τ)*θ_target

           Params
               local_model:  PyTorch model (weights will be copied from)
               target_model: PyTorch model (weights will be copied to)
               tau (float):  interpolation parameter 
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
