import numpy as np

class Episode:
    def __init__(self, horizon=200, policy=None):
        self.horizon = horizon
        self.policy = policy
        self.rewards = []
        self.states = []
        self.actions = []

    def add(self, state, action, next_state, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
