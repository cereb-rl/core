'''
    QLearningAgent.py: Class for QLearningAgent from [Sutton and Barto].    
'''

# Python Imports
from typing import Dict

import numpy as np
import random

# Core Imports
from core.agents.base import BaseAgent
from core.agents.policies import EpsilonGreedy, DiscreteTabularPolicy
from core.utils import constants, specs

QLEARNING_CONSTS: Dict[str, float] = {
    'gamma': 0.95,  # discount factor
    'alpha': 1,  # learning rate
    'max_alpha': 1,  # Maximum learning rate
    'min_alpha': 0.01,  # Minimum learning rate
    'epsilon': 1,  # exploration factor
    'max_epsilon': 1,    # Exploration probability at start
    'min_epsilon': 0.01,  # Minimum exploration probability
    'decay_rate': 0.001,    # Exponential decay rate for exploration prob
    'ada_divisor': 25,     # decay rate parameter
}

QLEARNING_SPEC = specs.AgentSpec(
    observation_space=constants.SpaceTypes.DISCRETE,  # observation 
    action_space=constants.SpaceTypes.DISCRETE,
)


class QLearningAgent(BaseAgent):
    """
        Implementation for an Q-Learning Agent [Sutton and Barto]
    """

    def __init__(self, observation_space, action_space, name="Q-Learning Agent", parameters={}, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name, params=dict(QLEARNING_CONSTS, **parameters))

        # Policy Setup
        if starting_policy:
            self.predict_policy = starting_policy
        else:
            self.predict_policy = DiscreteTabularPolicy(self.observation_space, self.action_space, default_value=1/(1-self.gamma))
        self.learn_policy = EpsilonGreedy(
                action_space=self.action_space,
                policy=self.predict_policy,
                epsilon=self.epsilon
            )

    def stepwise_update(self, state, reward):
        if self.prev_state is not None and self.prev_action is not None:
            old_value = self.predict_policy.get_q_value(self.prev_state, self.prev_action)
            next_max = self.predict_policy.get_max_q_value(state)

            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

            # Update Q-table
            self.predict_policy.set_q_value(self.prev_state, self.prev_action, new_value)

    def episodic_update(self):
        # Reduce epsilon (because we need less and less exploration)

        # self.epsilon = max(self.params['min_epsilon'], min(1.0, 1.0 - np.log10((self.episode_number + 1) / self.params['ada_divisor'])))
        self.epsilon = self.params['min_epsilon'] + (self.params['max_epsilon'] - self.params['min_epsilon'])*np.exp(-self.params['decay_rate']*self.episode_number)
        self.learn_policy.epsilon = self.epsilon
        # self.alpha = max(self.params['min_alpha'], min(1.0, 1.0 - np.log10((self.episode_number + 1) / self.params['ada_divisor'])))
        self.alpha = self.params['min_alpha'] + (self.params['max_alpha'] - self.params['min_alpha'])*np.exp(-self.params['decay_rate']*self.episode_number)
