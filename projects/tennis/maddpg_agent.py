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
NOISE_SCALE = 0.5       # scale factor applied to the raw noise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiDdpgAgent:
    """Interacts with and learns from the environment and other agents in the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed, replay_buf,
                 batch_size=32, noise_decay=1.0, learn_every=20, learn_iter=1,
                 lr_actor=0.00001, lr_critic=0.000001, weight_decay=1.0e-5):
        """Initialize an Agent object.
        
        Params
            state_size (int):          number of state values for this agent
            action_size (int):         number of action values for this agent
            num_agents (int):          number of agents in the whole environment
            random_seed (int):         seed for random number generator
            replay_buf (ReplayBuffer): buffer object holding experiences for replay (all agents)
            batch_size (int):          size of each minibatch used for learning
            noise_decay (float):       multiplier on the magnitude of noise; decay is applied
                                         each time step (must be <= 1.0)
            learn_every (int):         number of time steps between learning sessions
            learn_iter (int):          number of learning iterations that get run during each
                                         learning session
            lr_actor (float):          learning rate for the actor network
            lr_critic (float):         learning rate for the critic network
            weight_decay (float):      weight decay rate for the critic optimizer
        """

        # don't seed the random package here; assume it has been done by caller
        # TODO:  understand why seeds are passed everywhere; isn't "random" a singleton that
        #        should only be seeded one time in the program?

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
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
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local  = Critic(num_agents*state_size, num_agents*action_size, random_seed,
                                    fcs1_units=layer1_units, fc2_units=layer2_units).to(device)
        self.critic_target = Critic(num_agents*state_size, num_agents*action_size, random_seed,
                                    fcs1_units=layer1_units, fc2_units=layer2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=lr_critic, weight_decay=weight_decay)

        # Noise process and latches for reporting on decay progress
        self.noise = OUNoise(action_size, random_seed)
        self.noise_level1_reported = False
        self.noise_level2_reported = False

        # storage for temp analysis of actions & noise;
        # allow room for 3 separate collection sequences
        self.ANAL_SIZE = 200 #number time steps in a sequence
        self.anal_actions = np.zeros((3*self.ANAL_SIZE, 2))
        self.anal_noise   = np.zeros((3*self.ANAL_SIZE, 2))
        self.anal_total_steps = 0 #total time steps since beginning of time
        self.anal_num = 0 #how many in the current sequence have been captured
        self.anal_seq_starts = [0, 100000, 200000] #near 10k & 20k episodes
        self.anal_cur_seq = 0

        # flag for printing the model weights on the indicated time step if > 0
        self.model_display_step = 0

    
    """Setters for the several hyperparameter "constants" defined at top of file."""

    def set_hp_gamma(self, gamma):
        global GAMMA
        GAMMA = gamma

    def set_hp_tau(self, tau):
        global TAU
        TAU = tau

    def set_hp_noise_scale(self, scale):
        global NOISE_SCALE
        NOISE_SCALE = scale

    def set_model_display_step(self, step):
        self.model_display_step = step


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

        #---------- the core functionality of this method ends here; the remainder
        #           is applying noise and facilitating debugging analysis

        # add noise with gradual decay
        if add_noise:
            n = self.noise.sample() * NOISE_SCALE

            # if a model display step is defined, then spill their guts
            if self.model_display_step > 0 \
               and  self.anal_total_steps == self.model_display_step:

               print("\n\nActor network weights:\n")
               for p in self.actor_local.parameters():
                   print(p.shape, ", max magnitude = ", torch.max(torch.abs(p)))
                   #print(p)
               print("\n\nCritic network weights:\n")
               for p in self.critic_local.parameters():
                   print(p.shape, ", max magnitude = ", torch.max(torch.abs(p)))
                   #print(p)

            # if current time step >= current seq start and num steps < seq size
            if self.anal_cur_seq < len(self.anal_seq_starts) \
               and  self.anal_total_steps >= self.anal_seq_starts[self.anal_cur_seq] \
               and  self.anal_num < self.ANAL_SIZE:

                # save the raw actions and noise values from this agent for analysis
                offset = self.anal_cur_seq*self.ANAL_SIZE
                for j in range(2):
                    self.anal_actions[offset + self.anal_num, j] = action[j]
                    self.anal_noise[offset + self.anal_num, j]   = n[j] 

                # increment the sequence time step counter
                self.anal_num += 1

                # if num steps == seq size then
                if self.anal_num == self.ANAL_SIZE:

                    # advance seq index and reset the num steps counter
                    self.anal_cur_seq += 1
                    self.anal_num = 0

            self.anal_total_steps += 1

            # apply the noise to the action output
            action += n*self.noise_mult

            # handle noise multiplier decay
            self.noise_mult *= self.noise_decay
            if self.noise_mult < 0.2:
                if not self.noise_level1_reported:
                    print(" *noise mult = 0.2")
                    self.noise_level1_reported = True
                if self.noise_mult < 0.0005:
                    self.noise_mult = 0.0005
                    if not self.noise_level2_reported:
                        print(" *noise mult = 0.0005")
                        self.noise_level2_reported = True

        return np.clip(action, -1, 1)


    def get_anal_data(self):
        return (self.anal_actions, self.anal_noise)


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
                 each tensor has shape [b, a, x]. b is batch size, a is num agents,
                 and x is num states (for one agent) for s and s';
                     x is num actions (for one agent) for a
                     x is 1 for r and done
               gamma (float): discount factor
        """

        # extract the elements of the replayed experience row
        obs, actions, rewards, next_obs, dones = experiences
        #print("learn: obs = ", obs)
        #print("       rewards = ", rewards, rewards.shape)
        #print("       next_obs = ", next_obs)
        #print("       dones = ", dones, dones.shape)
        #print("       actions = ", actions)


        # ---------------------------- update critic ---------------------------- #

        # grab the batch of next state vectors for each actor and feed them into the
        # target network to get predicted next actions (each row is actions of all actors)
        target_actions = torch.zeros(self.batch_size, 2*self.action_size, dtype=torch.float)
        for i in range(self.num_agents):
            s = next_obs[:, i, :].to(device)
            target_actions[:, i*self.action_size:(i+1)*self.action_size] = \
                          self.actor_target(s)

        # create a next_states tensor [b, x] where each row represents the states
        # of all agents
        next_states = next_obs.view(self.batch_size, -1).to(device)

        # compute the Q values for the next states/actions from the target model
        q_targets_next = self.critic_target(next_states, target_actions)

        # Compute Q targets for current states (y_i) for this agent
        reward = rewards[:, agent_id, :].squeeze(dim=1).view(self.batch_size, -1).to(device)
        done = dones[:, agent_id, :].squeeze(dim=1).view(self.batch_size, -1).to(device)
        q_targets = reward + gamma*q_targets_next*(1 - done)

        # reshape the observations & actions so that all agents are represented on each row
        all_agents_states = obs.view(self.batch_size, -1).to(device)
        all_agents_actions = actions.view(self.batch_size, -1).to(device)

        # Compute critic loss
        q_expected = self.critic_local(all_agents_states, all_agents_actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        #print("       critic_loss = ", critic_loss)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #

        # Compute actor loss
        actions_pred = torch.zeros(self.batch_size, 2*self.action_size, dtype=torch.float)
        for i in range(self.num_agents):
            s = obs[:, i, :].to(device)
            actions_pred[:, i*self.action_size:(i+1)*self.action_size] = self.actor_local(s)
        actor_loss = -self.critic_local(all_agents_states, actions_pred).mean() #can't detach()

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
