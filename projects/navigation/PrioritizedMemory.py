# This class is from https://github.com/rlcode/per

import random
import numpy as np
from SumTree import SumTree

#############################################################################
# A memory class that represents a Prioritized Experience Replay (PER) buffer
# intended for use in Deep Q Networks (DQN).  This class allows specification
# of a maximum buffer size.  Once it is filled, any new items overwrite the
# oldest remaining ones, without regard to their priority.
#############################################################################

class PrioritizedMemory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity


    def __len__(self):
        return len(self.tree)


    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a


    # Returns the maximum error value of all experiences stored in the buffer.
    # NB: there is a difference between error (lower-case delta in the paper, and priority).  For efficiency, the
    #     SumTree structure stores priority, but that is not what is being requested here.

    def get_max_error(self):
        return self.tree.get_max_priority() ** (1.0/self.a) - self.e


    # Adds a data sample to the PER database.
    # error - Bellman equation error between target and current estimate of Q
    # sample - the data object (e.g. a (s, a, r, s') tuple) to be stored
    # Return: none

    def add(self, error, sample):
        p = self._get_priority(error)
        #print("Memory.add: p = ", p) ##### jas
        #print("            sample = ", sample)
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


    # Updates the error values of a batch of existing experience objects in the SumTree.
    # idxs - a list of indices of each experience object in the SumTree
    # errors - a list of the error values to be stored with the objects indicated by idxs (updated priorities)

    def update(self, idxs, errors):
        #print("Memory.update: idxs = ", idxs)
        #print("               errors = ", errors)
        for i, e in zip(idxs, errors):
            #print("Memory.update: i = ", i)
            #print("               e = ", e)
            #print("update: i = {}, e = {:.3f}".format(i, e))
            self.update_single_exp(i, e)


    # Updates the error value of an existing experience object (and propagates it up the SumTree
    # structure).
    # idx - index of the object in the SumTree
    # error - the new value of the error to be stored with this object (and thus update its priority)

    def update_single_exp(self, idx, error):
        p = self._get_priority(error)
        #print("Memory.update_single_exp: idx = {}, error = {:.3f}, p = {:.3f}".format(idx, error, p)) ### jas
        self.tree.update(idx, p)