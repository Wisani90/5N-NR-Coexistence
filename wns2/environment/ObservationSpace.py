import gym
import numpy as np
from gym import spaces

class ObservationSpace(gym.Space):
    """
    {0,...,1,...,0}

    Example usage:
    self.observation_space = OneHotEncoding(size=4)
    """
    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0
        self.size = size
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        observation = np.random.rand(self.size, 1)
        return observation

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)) and x.size == self.size:
            result = True
            for elem in list(x):
                if (elem < 0 or elem > 1):
                    result = False
            return result
        else:
            return False

    def __repr__(self):
        return "ObservationSpace(%d)" % self.size

    def __eq__(self, other):
        return self.size == other.size