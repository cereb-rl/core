'''
RMaxAgentClass.py: Class for an RMaxAgent from [Strehl, Li and Littman 2009].

Notes:
    - Assumes WLOG reward function codomain is [0,1] (so RMAX is 1.0)
'''

# Python imports.
import random
from collections import defaultdict
import numpy as np

# Local classes.
from core.agents.base import BaseAgent
from core.utils.Models import KnownTabularModel
from core.utils.Policy import DiscreteTabularPolicy
from core.utils.policy_helpers import *

MBIE_DEFAULTS = {
    'gamma': 0.95,
    'known_threshold': 2,
    'max_reward': 1,
    'epsilon_one': 0.99,
    'beta': 0.1
}


class MBIEAgent(BaseAgent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, observation_space, action_space, name="MBIE Agent", params=None, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name)

        # Hyper-parameters
        self.params = dict(MBIE_DEFAULTS)
        if params:
            for key, value in params:
                self.params[key] = value
        self.max_reward = self.params['max_reward']
        self.epsilon_one = self.params['epsilon_one']
        self.known_threshold = self.params['known_threshold']
        self.beta = self.params['beta']
        self.gamma = self.params['gamma']
        # self.max_reward = 1 / (1 - self.gamma)

        # Policy Setup
        self.starting_policy = starting_policy
        self.backup_lim = int(np.log(1 / (self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
        self.stepwise_backup_steps = 1
        self.episodic_backup_steps = min(self.backup_lim, 5)

        # Model Setup
        self.model = KnownTabularModel(action_space.n, self.max_reward, self.known_threshold)
        self.reset()

    def reset(self):
        '''
        Summary:
            Resets the agent model and policy back to its tabula rasa config.
        '''
        self.model.reset()
        self.policy = self.starting_policy if self.starting_policy else DiscreteTabularPolicy(self.observation_space,
                                                                                              self.action_space,
                                                                                              self.max_reward)

    def learn(self, state, reward, done=False):
        """

        :param state:
        :param reward:
        :param done:
        :return:
        """
        action = self.policy.get_max_action(state)  # Exploit learned values
        self.update(self.prev_state, self.prev_action, reward, state) # update model and policy

        self.prev_action = action
        BaseAgent.learn(self, state, reward, done)
        return action

    def get_bellman_backup_function(self):
        def update_fn(state, action, policy, model, gamma):
            value = bellman_policy_backup(state, action, policy, model, gamma)
            if self.model.get_count(state, action) > 0:
                value += self.beta / np.sqrt(self.model.get_count(state, action))
            return value
        return update_fn

    def update(self, state, action, reward, next_state):
        if not self.model.is_known(state, action):
            # Add new data points if we haven't seen this s-a enough.
            self.model.update(state, action, reward, next_state)
            iterate_policy(self.policy, self.model, states=range(self.observation_space.n), num_steps=self.stepwise_backup_steps, gamma=self.gamma,
                           update_fn=self.get_bellman_backup_function())

    def end_of_episode(self):
        # Update policy
        iterate_policy(self.policy, self.model, num_steps=self.episodic_backup_steps, gamma=self.gamma, update_fn=self.get_bellman_backup_function())
        BaseAgent.end_of_episode(self)