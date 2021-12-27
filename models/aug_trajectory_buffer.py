import torch

import numpy as np
from random import sample
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('obs', 'instr', 'action', 'reward'))


class goal_storage(object):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.position = 0

    def put(self, state, instr, action, reward):
        self.memory.append(Transition(state, instr, action, reward))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = sample(self.memory, batch_size)
        return Transition(*(zip(*transitions)))

    def __len__(self):
        return len(self.memory)

    def len(self):
        return len(self.memory)