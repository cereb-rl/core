'''
RMaxAgentClass.py: Class for an RMaxAgent from [Strehl, Li and Littman 2009].

Notes:
    - Assumes WLOG reward function codomain is [0,1] (so RMAX is 1.0)
'''

# Python imports.
import random
import numpy as np
#from itertools import product

# Local classes.
from core.agents.base import BaseAgent
from core.utils.models import DiscreteTabularModel
from core.utils.policies import LogisticPolicy
from core.utils.policy_helpers import *

REINFORCE_DEFAULTS: Dict[str, float] = {
    'gamma': 0.95,  # discount factor
    'alpha': 1,  # learning rate
    'max_alpha': 1, # Maximum learning rate
    'min_alpha': 0.01, # Minimum learning rate
    'epsilon': 1,  # exploration factor
    'max_epsilon': 1,    # Exploration probability at start
    'min_epsilon': 0.01,  # Minimum exploration probability
    'decay_rate': 0.001,    # Exponential decay rate for exploration prob
    'ada_divisor': 25,     # decay rate parameter
}

class RMaxAgent(BaseAgent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, observation_space, action_space, name="REINFORCE Agent", extra_params={}, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name, params=dict(REINFORCE_DEFAULTS, **extra_params))

        # Hyper-parameters
        self.alpha = self.params['alpha']
        self.gamma = self.params['gamma']

        # Policy Setup
        self.starting_policy = starting_policy if starting_policy else LogisticPolicy(num_features=4)

        # Model Setup
        self.last_episode = None

        self.reset()

    def learn(self, state, reward, done=False):
        """

        :param state:
        :param reward:
        :param done:
        :return:
        """
        probs = self.policy.action_probs(state)
        action = np.random.choice(np.arange(self.action_space.n), p=probs)

        BaseAgent.learn(self, state, reward, done)
        self.prev_action = action
        return action

    def update(self, state, action, reward, next_state):
        self.last_episode.append((state, action, reward, next_state))

    # def bellman_policy_backup(self, state, action):
    #     new_value = (1-self.gamma)*self.model.known_rewards[state, action] + self.gamma * np.inner(self.model.known_transitions[state, action],self.policy.get_max_q_values())
    #     self.policy.set_q_value(state, action, new_value)
    #
    # def iterate_policy(self, num_steps):
    #     for _ in range(num_steps):
    #         map(self.bellman_policy_backup, product(range(self.observation_space.n), range(self.action_space.n)))

    def vectorized_iterate_policy(self, num_steps):
        for _ in range(num_steps):
            self.policy.q_table = self.model.known_rewards + self.gamma*np.dot(self.model.known_transitions, self.policy.get_max_q_values())

    def end_of_episode(self):
        for transition in self.last_episode:
            state, action, next_state, reward = transition
        BaseAgent.end_of_episode(self)
