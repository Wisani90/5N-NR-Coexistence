from rl.policy import Policy
import numpy as np


class OneHotPolicy(Policy):

    def __init__(self, eps=.1):
        super().__init__()
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)

        onehot_action = np.zeros(5)
        onehot_action[action] = 1
        action = onehot_action
        return action
