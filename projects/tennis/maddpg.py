# -------- HEY JOHN:



# UPDATE THE DESCRIPTION HERE...


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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from numpy.random import default_rng
import copy

from model         import Actor, Critic
from ou_noise      import OUNoise
from replay_buffer import ReplayBuffer

# initial probability of keeping "bad" episodes (until enough exist to start learning)
BAD_STEP_KEEP_PROB_INIT = 0.04

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Maddpg:
    """Manages the training and execution of multiple agents in the same environment"""

    def __init__(self, state_size, action_size, num_agents, bad_step_prob=0.5, random_seed=0,
                 batch_size=32, buffer_size=1000000, noise_decay=1.0, noise_scale=1.0,
                 buffer_prime_size=1000, learn_every=20, learn_iter=1, lr_actor=0.00001,
                 lr_critic=0.000001, lr_anneal_freq=2000, lr_anneal_mult=0.5, weight_decay=1.0e-5,
                 gamma=0.99, tau=0.001, model_display_step=0):
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
            noise_decay (float):  multiplier on the magnitude of noise; decay is applied each
                                    time step (must be <= 1.0)
            noise_scale (float):  scale factor applied to all noise, regardless of decay state
            buffer_prime_size (int): number of experiences to be stored in the replay buffer
                                    before learning begins, under the influence of the
                                    BAD_STEP_KEEP_PROB_INIT probability
            learn_every (int):    number of time steps between learning sessions
            learn_iter (int):     number of learning iterations that get run during each
                                    learning session
            lr_actor (float):     learning rate for each agent's actor network
            lr_critic (float):    learning rate for each agent's critic network
            lr_anneal_freq (int): number of episodes (NOT time steps!) between annealing steps
            lr_anneal_mult (float): multiplier used to anneal both LRs at each annealing step
            weight_decay (float): decay rate applied to each agent's critic network optimizer
            gamma (float):        future reward discount factor
            tau (float):          target network soft update rate
            model_display_step (int): time step (through all episodes) on which NN weights are
                                    to be printed; if <= 0 then no printing will occur
        """

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.bad_step_keep_prob = min(bad_step_prob, 1.0)
        self.rng = default_rng(random_seed)
        self.batch_size = batch_size
        self.noise_decay = min(noise_decay, 1.0)
        self.noise_scale = noise_scale
        self.learn_every = learn_every
        self.buffer_prime_size = buffer_prime_size
        self.learn_iter = learn_iter
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_anneal_freq = lr_anneal_freq
        self.lr_anneal_mult = lr_anneal_mult
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.tau = tau
        self.model_display_step = model_display_step

        # initialize other internal things
        self.noise_mult = 1.0 # the multiplier that will get decayed
        self.prev_clr = self.lr_critic
        self.learn_control = 0 #counts iterations between learning sessions
        layer1_units = 400
        layer2_units = 256

        # define simple replay memory common to all agents
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, buffer_prime_size,
                                   random_seed)
        self.learning_underway = False

        # create the actor networks
        self.actor_policy = []
        self.actor_target = []
        self.actor_optimizer = []
        self.actor_scheduler = []
        for i in range(num_agents):
            self.actor_policy.append(Actor(state_size, action_size, random_seed,
                                           fc1_units=layer1_units, fc2_units=layer2_units) \
                                          .to(device))
            self.actor_target.append(Actor(state_size, action_size, random_seed,
                                           fc1_units=layer1_units, fc2_units=layer2_units) \
                                          .to(device))
            self.actor_optimizer.append(optim.Adam(self.actor_policy[i].parameters(),
                                                   lr=lr_actor))
            self.actor_scheduler.append(StepLR(self.actor_optimizer[i],
                                               step_size=self.lr_anneal_freq,
                                               gamma=self.lr_anneal_mult))

        # create the common critic networks
        self.critic_policy = Critic(num_agents*state_size, num_agents*action_size,
                                    random_seed, fcs1_units=layer1_units,
                                    fc2_units=layer2_units).to(device)
        self.critic_target = Critic(num_agents*state_size, num_agents*action_size,
                                    random_seed, fcs1_units=layer1_units,
                                    fc2_units=layer2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_policy.parameters(), lr=lr_critic,
                                           weight_decay=weight_decay)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=self.lr_anneal_freq,
                                       gamma=self.lr_anneal_mult)

        # noise process & latches for decay reporting
        self.noise = OUNoise(action_size, random_seed)
        self.noise_level1_reported = False
        self.noise_level2_reported = False


    def reset(self):
        """Resets all agents to an initial state to begin another episode"""

        self.noise.reset()


    def act(self, states, add_noise=True):
        """Computes the action of each agent. Initially, these actions are random,
           to prime the replay buffer until at least one batch-full of experiences is
           stored. At that point learning can begin. From then on, it uses each agent's
           current policy to generate its actions.

           Params:
               states (tuple of float tensors):  the state values for all agents
               add_noise (bool):                 should noise be added to the results?

           Return:  ndarray of actions taken by all agents
        """

        actions = np.zeros((self.num_agents, self.action_size))

        # if learning is underway (replay buffer is sufficiently populated), then compute
        # actions based on the agent policies
        if self.learning_underway:
            for i in range(self.num_agents):

                # get the raw action
                state = torch.from_numpy(states[i]).float().to(device)
                self.actor_policy[i].eval()
                with torch.no_grad():
                    action = self.actor_policy[i](state).cpu().data.numpy()
                self.actor_policy[i].train()

                # add noise if appropriate, and decay it
                if add_noise:
                    n = self.noise.sample() * self.noise_scale
                    action += n * self.noise_mult
                    self.noise_mult *= self.noise_decay
                    if self.noise_mult < 0.2:
                        if not self.noise_level1_reported:
                            print("\n* noise mult = 0.2")
                            self.noise_level1_reported = True
                        if self.noise_mult < 0.0005:
                            self.noise_mult = 0.0005
                            if not self.noise_level2_reported:
                                print("\n* noise mult = 0.0005")
                                self.noise_level2_reported = True


                actions[i, :] = np.clip(action, -1.0, 1.0)

        # else, prime the replay buffer with random actions that uniformly cover the full 
        # range of action space
        else:
            for i in range(self.num_agents):
                actions[i, :] = torch.from_numpy(2.0 * self.rng.random((1, self.num_agents)) - 1.0)

        return actions


    def step(self, obs, actions, rewards, next_obs, dones):
        """Stores a new experience from the environment in replay buffer, if appropriate,
           and advances the agents by one time step.

           Params:
               obs (ndarray of float):      the current state values for all agents, one
                                              row per agent
               actions (ndarray of float):  the current actions from all agents, one row
                                              per agent
               rewards (list of float):     current reward earned from each agent
               next_obs (ndarray of float): est of next time step's states, one row per agent
               dones (list of bool):        for each agent, is episode complete?

           Return:  none
        """

        # set up probability of keeping bad experiences based upon whether the buffer is
        # full enough to start learning
        if len(self.memory) > max(self.batch_size, self.buffer_prime_size):
            threshold = self.bad_step_keep_prob
            self.learning_underway = True
        else:
            threshold = BAD_STEP_KEEP_PROB_INIT

        # if this step got some reward then keep it;
        # if it did not score any points, then use random draw to decide if it's a keeper
        if max(rewards) > 0.0  or  self.rng.random() < threshold:
            self.memory.add(obs, actions, rewards, next_obs, dones)

        # initiate learning on each agent, but only every N time steps
        self.learn_control += 1
        if self.learning_underway:

            # perform the learning if it is time
            if self.learn_control > self.learn_every:
                self.learn_control = 0
                for j in range(self.learn_iter):
                   experiences = self.memory.sample()
                   self.learn(experiences)

            # update learning rate annealing; this is counting episodes, not time steps
            if any(dones):
                for i in range(self.num_agents):
                    self.actor_scheduler[i].step()
                self.critic_scheduler.step()
                clr = self.critic_scheduler.get_lr()[0]
                if clr < self.prev_clr:
                    print("\n* LR annealed: new actor = {:.7f}, critic = {:.7f}" \
                          .format(self.actor_scheduler[0].get_lr()[0], clr))
                    self.prev_clr = clr


    def learn(self, experiences):
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
        """

        # extract the elements of the replayed batch of experiences
        obs, actions, rewards, next_obs, dones = experiences

        #---------- update critic network

        # grab the batch of next state vectors for each actor and feed them into the
        # target network to get predicted next actions (each row is actions of all actors)
        target_actions = torch.zeros(self.batch_size, 2*self.action_size, dtype=torch.float) \
                           .to(device)
        for i in range(self.num_agents):
            s = next_obs[:, i, :].to(device)
            target_actions[:, i*self.action_size:(i+1)*self.action_size] = \
                          self.actor_target[i](s)

        # create a next_states tensor [b, x] where each row represents the states
        # of all agents
        next_states = next_obs.view(self.batch_size, -1).to(device)

        # compute the Q values for the next states/actions from the target model
        q_targets_next = self.critic_target(next_states, target_actions).squeeze()

        # Compute Q targets for current states (y_i) for this agent
        reward = rewards.max(dim=1)[0].squeeze().to(device) #sum rewards from both agents
        done = dones.max(dim=1)[0].squeeze().to(device)     #if either agent indicates done
        q_targets = reward + self.gamma*q_targets_next*(1 - done)

        # reshape the observations & actions so that all agents are represented on each row
        all_agents_states = obs.view(self.batch_size, -1).to(device)
        all_agents_actions = actions.view(self.batch_size, -1).to(device)

        # Compute critic loss
        q_expected = self.critic_policy(all_agents_states, all_agents_actions)
        critic_loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_policy.parameters(), 1)
        self.critic_optimizer.step()

        #---------- update actor networks

        # Compute actor loss
        actions_pred = torch.zeros(self.batch_size, 2*self.action_size, dtype=torch.float)
        for i in range(self.num_agents):
            s = obs[:, i, :].to(device)
            actions_pred[:, i*self.action_size:(i+1)*self.action_size] = self.actor_policy[i](s)
        actor_loss = -self.critic_policy(all_agents_states, actions_pred).mean()

        # Minimize the loss
        for i in range(self.num_agents):
            self.actor_optimizer[i].zero_grad()
            # backpropagate, but clear its buffers only on final loop iteration
            actor_loss.backward(retain_graph=(i < self.num_agents-1))
            torch.nn.utils.clip_grad_norm_(self.actor_policy[i].parameters(), 1)
            self.actor_optimizer[i].step()

        #---------- update target networks

        for i in range(self.num_agents):
            self.soft_update(self.actor_policy[i], self.actor_target[i])
        self.soft_update(self.critic_policy, self.critic_target)


    def soft_update(self, policy_model, target_model):
        """Soft update model parameters - move the target model params a bit closer to those
           of the given policy model.
           θ_target = τ*θ_policy + (1 - τ)*θ_target; where tau < 1

           Params
               policy_model: PyTorch model (weights will be copied from)
               target_model: PyTorch model (weights will be copied to)

        """

        for tgt, pol in zip(target_model.parameters(), policy_model.parameters()):
            tgt.data.copy_(self.tau*pol.data + (1.0-self.tau)*tgt.data)


    def get_memory_stats(self):
        """Gets statistics on the replay buffer memory contents.

           Return:  tuple of (size, good_exp), where size is total number
                      of items in the buffer, and good_exp is the number of those
                      items with a reward that exceeds the threshold of "good".
        """

        return (len(self.memory), self.memory.num_rewards_exceeding_threshold())


    def is_learning_underway(self):
        return self.learning_underway


    def save_checkpoint(self, path, tag, episode):
        """Saves checkpoint files for each of the networks and optimizers.  This version stores
           the 'new' (non-legacy) format.

           Params:
               path (string): directory path where the files will go (if not None, needs to end in /)
               tag (string):  an aribitrary tag to distinguish this set of networks (e.g. test ID)
               episode (int): the episode number

           For info on file structure, see the description of restore_checkpoint(), below.
        """

        checkpoint = {}
        for i in range(self.num_agents):
            key_a = "actor{}".format(i)
            key_oa = "optimizer_actor{}".format(i)
            checkpoint[key_a] = self.actor_policy[i].state_dict()
            checkpoint[key_oa] = self.actor_optimizer[i].state_dict()

        checkpoint["critic"] = self.critic_policy.state_dict()
        checkpoint["optimizer_critic"] = self.critic_optimizer.state_dict()

        # TODO: figure out how to store the buffer also (error on attribute lookup)
        #checkpoint["replay_buffer"] = self.memory
 
        filename = "{}{}_{}.pt".format(path, tag, episode)
        torch.save(checkpoint, filename)


    def restore_checkpoint(self, path_root, tag, episode, legacy=False):
        """Loads data from a checkpoint for continued training.
           MUST be called AFTER defining the MultiDdptAgent objects, optimizers and
           replay buffer.

           Params:
               path_root (string): directory path of the root dir above which the checkpoint
                                     is saved. Assumes all checkpoints are in a subdir named
                                     the same as the tag. Assumed to end in '/'
               tag (string):       tag that distinguishes this set of networks/runs
               episode (int):      the episode number when the checkpoint was stored
               legacy (bool):      should we use the legacy file structure?

               Assumes checkpoint filenames are all <tag>.<run>_[networkname_]<episode>.pt
               where networkname is used only for legacy formats, as those have separate
               files for each network.  Legacy checkpoints are comprised of 4 files, where
               networkname is either "actor0", "actor1", "critic0" or "critic1". Each file
               is just that network.  There is no optimizer or replay buffer stored.  New
               checkpoints are saved as a single file for the entire model, which is
               structured as a dictionary containing the following fields:
                 actor0
                 actor1
                 optimizer_actor0
                 optimizer_actor1
                 critic
                 optimizer_critic
               Each field holds a state_dict.
        """

        #---------- load legacy checkpoints

        if legacy:
            print("\n\n///// WARNING: legacy checkpoint load was requested; the current\n",
                  "solution is incompatible with these checkpoint files.\n")


        #----------- load new style checkpoints

        else:
            print("\n///// WARNING: new style checkpoint loading needs to be upgraded.")

            filename = "{}{}_{}.pt".format(path_root, tag, episode)
            checkpoint = torch.load(filename)

"""
            for i, a in enumerate(self.agents):
                key_a = "actor{}".format(i)
                key_c = "critic{}".format(i)
                key_oa = "optimizer_actor{}".format(i)
                key_oc = "optimizer_critic{}".format(i)
                self.agents[i].actor_local.load_state_dict(checkpoint[key_a])
                self.agents[i].critic_local.load_state_dict(checkpoint[key_c])
                self.agents[i].actor_optimizer.load_state_dict(checkpoint[key_oa])
                self.agents[i].critic_optimizer.load_state_dict(checkpoint[key_oc])
"""

            #self.memory = checkpoint['replay_buffer']

        #print("Checkpoint loaded for {}, episode {}".format(tag, episode))
