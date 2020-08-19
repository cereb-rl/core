import numpy as np

class EpsilonGreedy:
    def __init__(self, action_space, policy, epsilon=0):
        self.action_space = action_space
        self.policy = policy
        self.epsilon = epsilon
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return self.policy.get_action(state)