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

RMAX_DEFAULTS = {
    'gamma': 0.95,
    'known_threshold': 2,
    'max_reward': 1,
    'epsilon_one': 0.99
}

class RMaxAgent(BaseAgent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, observation_space, action_space, name="RMax Agent", params=None, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name)

        # Hyper-parameters
        self.params = dict(RMAX_DEFAULTS)
        if params:
            for key, value in params:
                self.params[key] = value
        self.max_reward = self.params['max_reward']
        self.epsilon_one = self.params['epsilon_one']
        self.known_threshold = self.params['known_threshold']
        self.gamma = self.params['gamma']
        #self.max_reward = 1 / (1 - self.gamma)

        # Policy Setup
        self.starting_policy = starting_policy
        self.backup_lim = int(np.log(1/(self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
        self.stepwise_backup_steps = 1 # self.backup_lim
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
        self.policy = self.starting_policy if self.starting_policy else DiscreteTabularPolicy(self.observation_space, self.action_space, 0.3)

    def learn(self, state, reward, done=False):
        """

        :param state:
        :param reward:
        :param done:
        :return:
        """
        action = self.policy.get_max_action(state)  # Exploit learned values

        self.update(self.prev_state, self.prev_action, reward, state)

        self.prev_action = action
        BaseAgent.learn(self, state, reward, done)
        return action

    def update(self, state, action, reward, next_state):
        if not self.model.is_known(state, action):
            self.model.update(state, action, reward, next_state)
            iterate_policy(self.policy, self.model, states=self.model.get_known_states(), num_steps=self.stepwise_backup_steps, gamma=self.gamma)

    def end_of_episode(self):
        iterate_policy(self.policy, self.model, states=self.model.get_known_states(), num_steps=self.episodic_backup_steps, gamma=self.gamma)
        BaseAgent.end_of_episode(self)
