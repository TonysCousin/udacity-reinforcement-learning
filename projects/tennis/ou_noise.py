# Defines an agent that uses the DDPG algorithm.
#
# This code is a copy of the code used for my continuous control (Reacher) project.
#
# This code is based on code provided by Udacity instructor staff
# for the DRL nanodegree program.

import numpy as np
import torch
import random
from numpy.random import default_rng
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
