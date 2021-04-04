# Generates a simple Gaussian noise profile

import numpy as np
import torch
from numpy.random import default_rng
import copy

class GaussNoise:

    def __init__(self, size, seed, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.sigma = sigma
        self.rng = default_rng(seed)


    def reset(self):
        """Required for interface compatibility."""

        pass


    def sample(self):
        """Returns a noise sample."""

        dx = self.sigma * np.array([self.rng.standard_normal() for i in range(self.size)])
        return dx
