import numpy as np
import random

class RandomAgent:

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent obje

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

    def act(self, state, eps=0.):
        """Choose a random action."""
        return np.random.randint(self.action_size)

    def learn(self):
        """No learning."""
        pass
