# This class is from https://github.com/rlcode/per

import random
import numpy as np
from SumTree import SumTree

#############################################################################
# Memory class that represents a Prioritized Experience Replay (PER) buffer
# intended for use in Deep Q Networks (DQN).  This class allows specification
# of a maximum buffer size.  Once it is filled, any new items overwrite the
# oldest remaining ones, without regard to their priority.
#############################################################################

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a


    # Adds a data sample to the PER database
    # error - Bellman equation error between target and current estimate of Q
    # sample - the data object (e.g. a (s, a, r, s') tuple) to be stored
    # Return: none

    def add(self, error, sample):
        p = self._get_priority(error)
        #print("Memory.add: p = ", p) ##### jas
        self.tree.add(p, sample)


    # Pulls n samples from the PER object (e.g. n is batch size) segmenting the probability space equally among them.
    # Return:  batches - a list of the data objects
    #          idxs - a list of the indices to these batches in the SumTree internal storage; these will be needed for calls
    #                 to the update() method
    #          is_weights - a list of the weights (aka delta values) used in the modified Bellman equation when applying PER

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        #print("Memory.sample: n = ", n, ", segment = ", segment) ### jas

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            #print("Memory.sample: i = {}, a = {:.3f}, b = {:.3f}, s = {:.3f}".format(i, a, b, s)) ### jas
            (idx, p, data) = self.tree.get(s)
            #print("Memory.sample: return from tree.get: idx = {}, p = {:.3f}, data = {}".format(idx, p, data)) ### jas
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight


    # Updates the error value of an existing experience object (and propagates it up the SumTree
    # structure).
    # idx - index of the object in the SumTree
    # error - the new value of the error to be stored with this object (and thus update its priority)

    def update(self, idx, error):
        p = self._get_priority(error)
        #print("Memory.update: idx = {}, error = {:.3f}, p = {:.3f}".format(idx, error, p)) ### jas
        self.tree.update(idx, p)
