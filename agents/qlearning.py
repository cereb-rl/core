'''
    QLearningAgent.py: Class for QLearningAgent from [Sutton and Barto].    
    Notes:
        - 
'''

# Python Imports
from typing import Dict

import numpy as np
import random
import copy

# Core Imports
from core.agents import BaseAgent
from core.utils.Policy import TabularPolicy

QLEARNING_DEFAULTS: Dict[str, float] = {
    'gamma': 0.9,  # discount factor
    'alpha': 0.9,  # learning rate
    'epsilon': 0.1  # exploration factor
}


class QLearningAgent(BaseAgent):
    """
        Implementation for an Q-Learning Agent [Sutton and Barto]
    """

    def __init__(self, observation_space, action_space, name="Q-Learning Agent", params=None, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name)

        # Policy Setup
        self.starting_policy = starting_policy
        self.policy = self.starting_policy if self.starting_policy else TabularPolicy(self.action_space.n)

        # Hyper-parameters
        self.params = dict(QLEARNING_DEFAULTS)
        if params:
            for key, value in params:
                self.params[key] = value
        self.alpha = self.params['alpha']

    def learn(self, state, reward, done=False):
        """

        :param state:
        :param reward:
        :param done:
        :return:
        """
        if random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()  # Explore action space
        else:
            action = self.policy.get_max_action(state)  # Exploit learned values

        self.update(state, reward)

        # update class
        self.prev_state = state
        self.prev_action = action
        self.prev_reward = reward

        # if done, end episode
        if done:
            self.end_of_episode()

        return action

    def update(self, state, reward):
        """

        :param state:
        :param reward:
        """
        old_value = self.policy.get_q_value(self.prev_state, self.prev_action)
        next_max = self.policy.get_max_q_value(state)

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.policy.set_q_value(self.prev_state, self.prev_action, new_value)

    def dump_policy(self):
        current_policy = copy.deepcopy(self.policy)
        return current_policy

    def __str__(self):
        return str(self.policy)
