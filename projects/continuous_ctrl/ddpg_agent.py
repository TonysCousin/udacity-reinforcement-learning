# This code is based on code provided by Udacity instructor staff
# for the DRL nanodegree program.

import numpy as np
import random
from numpy.random import default_rng
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size; orig 1e4
GAMMA = 0.99            # discount factor; orig 0.99
TAU = 0.001             # for soft update of target parameters; orig 0.001
LR_ACTOR = 0.001       # learning rate of the actor; orig 0.0001
LR_CRITIC = 0.001       # learning rate of the critic; orig 0.001
WEIGHT_DECAY = 1e-5        # L2 weight decay; orig 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DdpgAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, batch_size=32,
                 noise_decay=1.0, learn_every=20, learn_iter=10):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int):     dimension of each state
            action_size (int):    dimension of each action
            random_seed (int):    random seed
            batch_size (int):     the size of each minibatch used for learning
            noise_decay (float):  multiplier on the magnitude of noise; decay is applied each time step (must be <= 1.0)
            learn_every (int):    number of time steps between learning sessions
            learn_iter (int):     number of learning iterations that get run during each learning session
        """

        self.state_size = state_size
        self.action_size = action_size
        random.seed(random_seed)
        self.batch_size = batch_size

        ##### Added by John to help training
        self.noise_mult = 1.0
        self.noise_decay = min(noise_decay, 1.0) #guarantee that this won't make the noise grow
        self.learn_control = 0
        self.learn_every = learn_every
        self.learn_iterations = learn_iter

        # was using 128, 48
        layer1_units = 400
        layer2_units = 256

        # Actor Network (w/ Target Network)
        self.actor_local  = Actor(state_size, action_size, random_seed, fc1_units=layer1_units, fc2_units=layer2_units).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units=layer1_units, fc2_units=layer2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local  = Critic(state_size, action_size, random_seed, fcs1_units=layer1_units, fc2_units=layer2_units).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, fcs1_units=layer1_units, fc2_units=layer2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.batch_size, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        ##### John modified this section for infrequent learning updates
        # Learn, if enough samples are available in memory
        self.learn_control += 1
        if len(self.memory) > self.batch_size  and  self.learn_control % self.learn_every == 0:
            for j in range(self.learn_iterations):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # John added noise decay
        if add_noise:
            action += self.noise.sample() * self.noise_mult
            self.noise_mult *= self.noise_decay
            if self.noise_mult < 0.001:
                self.noise_mult = 0.001

        return np.clip(action, -1, 1)


    def reset(self):
        """Reset the noise generator."""
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + gamma*Q_targets_next*(1 - dones)

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) #John added per lesson suggestion; gradient clipping
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #

        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean() #can't detach() here

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1) #John added
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.rng = default_rng()
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([self.rng.standard_normal() for i in range(len(x))])
        self.state = x + dx
        return self.state

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
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

