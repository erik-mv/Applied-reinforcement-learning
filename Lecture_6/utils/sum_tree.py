import numpy


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity

        c = self.capacity
        pow = 0
        while c > 1:
            c = (c + 1)//2
            pow += 1

        self.true_capacity = 2 ** pow
        self.tree = numpy.zeros(2 * self.true_capacity - 1)
        self.data = [None for _ in range(capacity)]
        self.n_entries = 0
        self.write = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.true_capacity - 1
        self.update(idx, p)

        self.data[self.write] = data
        self.write = (self.write + 1) % self.capacity

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s * self.total())
        dataIdx = idx - self.true_capacity + 1

        return idx, self.data[dataIdx], self.tree[idx] / self.total()