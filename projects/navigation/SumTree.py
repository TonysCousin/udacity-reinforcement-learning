# This class is from https://github.com/rlcode/per

import numpy

##############################################################################
# SumTree class - a binary tree data structure where the parent’s value is the
# sum of its children's.
#
# Each data object is stored associated with a (real) priority value. The sum
# of their priorities defines the size of the priority space.  Objects can 
# then be retrieved (the space sampled) randomly in proportion to their
# relative priorities.  This is an essential capability for Deep Q Learning
# with Prioritized Experience Replay (PER).  Each data object is stored as
# a node in the binary tree structure.
##############################################################################

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        #print("SumTree._propagate: idx = {}, change = {:.3f}, parent = {}".format(idx, change, parent)) ### jas

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        #print("SumTree._retrieve: idx = {}, s = {:.3f}, left = {}, right = {}".format(idx, s, left, right)) ### jas

        if left >= len(self.tree):
            #print("                   len(self.tree) = {}, self.tree[left] = nan".format(len(self.tree))) ### jas
            return idx

        #print("                   len(self.tree) = {}, self.tree[left] = {:.3f}".format(len(self.tree), self.tree[left])) ### jas
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])


    # Returns the sum of priorities of all objects in the tree.
    def total(self):
        #print("SumTree.total: returning ", self.tree[0]) ### jas
        return self.tree[0]


    # Stores a new data object along with its priority.
    # p - priority of the object
    # data - the object to be stored

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        #print("SumTree.add: idx = ", idx) ### jas

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1


    # Updates the priority value of an existing node.
    # idx - index into the "tree" structure (it is implemented as a simple list)
    # p - the new value of priority to be assigned to the node

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)


    # Retrieves a single node from the tree that corresponds to the given sample value.
    # Imagine all of the stored samples lined up, left-to-right, and the width of each is proportional
    # to its priority value.  This line then extends from 0 to the sum of all the nodes' priorities.
    # The sample value, s, is then a point on that line in [0, sum].
    # s - the value of the desired sample (point on the line)
    # Returns:
    #    idx - the index into the list structure used to store the nodes
    #    priority - the priority value of the object that is pointed to by s
    #    data - the object that is pointed to by s

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        #print("SumTree.get: s = {:.3f}, idx = {}, dataIdx = {}".format(s, idx, dataIdx)) ### jas

        return (idx, self.tree[idx], self.data[dataIdx])
