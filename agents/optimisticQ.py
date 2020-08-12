'''
    QLearningAgent.py: Class for QLearningAgent from [Sutton and Barto].
    Notes:
        -
'''

# Python Imports
from typing import Dict

import numpy as np
import random

# Core Imports
from core.agents import BaseAgent
from core.utils.Policy import DiscreteTabularPolicy

QLEARNING_DEFAULTS: Dict[str, float] = {
    'gamma': 0.95,  # discount factor
    'alpha': 0.7,  # learning rate
    'epsilon': 1,  # exploration factor
    'max_epsilon': 1,    # Exploration probability at start
    'min_epsilon': 0.01,  # Minimum exploration probability
    'decay_rate': 0.01,    # Exponential decay rate for exploration prob
}


class OptimisticQAgent(BaseAgent):
    """
        Implementation for a Optimistic Q-Learning Agent
    """

    def __init__(self, observation_space, action_space, name="Q-Learning Agent", params={}, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name, params=dict(QLEARNING_DEFAULTS, **params))

        # Hyper-parameters
        self.alpha = self.params['alpha']


        # Policy Setup
        self.starting_policy = starting_policy
        if self.starting_policy:
            self.policy = self.starting_policy
        else:
            self.policy = DiscreteTabularPolicy(self.observation_space, self.action_space, 1/(1-self.gamma))

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
        self.prev_action = action
        BaseAgent.learn(self, state, reward, done)
        return action

    def update(self, state, reward):
        """

        :param state:
        :param reward:
        """
        if self.prev_state is not None and self.prev_action is not None:
            old_value = self.policy.get_q_value(self.prev_state, self.prev_action)
            next_max = self.policy.get_max_q_value(state)

            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            self.policy.set_q_value(self.prev_state, self.prev_action, new_value)

    def end_of_episode(self):
        if self.episode_learn_steps > 0:
            # Reduce epsilon (because we need less and less exploration)
            self.epsilon = self.params['min_epsilon'] + (self.params['max_epsilon'] - self.params['min_epsilon'])*np.exp(-self.params['decay_rate']*self.episode_number)
        BaseAgent.end_of_episode(self)
