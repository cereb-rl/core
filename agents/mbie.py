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
from core.utils.models import DiscreteTabularModel
from core.utils.Policy import DiscreteTabularPolicy
from core.utils.policy_helpers import *

MBIE_DEFAULTS = {
    'gamma': 0.95,
    'known_threshold': 10,
    'max_reward': 1,
    'epsilon_one': 0.99,
    'beta': 1,
    'include_eb': True
}


class MBIEAgent(BaseAgent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, observation_space, action_space, name="MBIE Agent", params={}, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name, params=dict(MBIE_DEFAULTS, **params))

        # Hyper-parameters
        self.max_reward = self.params['max_reward']
        self.epsilon_one = self.params['epsilon_one']
        self.known_threshold = self.params['known_threshold']
        self.beta = self.params['beta']
        self.gamma = self.params['gamma']
        # self.max_reward = 1 / (1 - self.gamma)
        self.include_exploration_bonus = self.params['include_eb']

        # Policy Setup
        self.starting_policy = starting_policy if starting_policy else DiscreteTabularPolicy(self.observation_space, self.action_space, default_value=1/(1-self.gamma))
        self.policy = starting_policy
        self.backup_lim = int(np.log(1 / (self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
        self.stepwise_backup_steps = min(self.backup_lim, 10)
        self.episodic_backup_steps = min(self.backup_lim, 0)

        # Model Setup
        self.model = DiscreteTabularModel(observation_space, action_space, default_reward=self.max_reward, limit=self.known_threshold)

        self.reset()

    def learn(self, state, reward, done=False):
        """

        :param state:
        :param reward:
        :param done:
        :return:
        """
        action = self.policy.get_max_action(state)  # Exploit learned values
        if self.prev_state is not None and self.prev_action is not None:
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
            self.vectorized_iterate_policy(state, action, num_steps=self.stepwise_backup_steps)

    def vectorized_iterate_policy(self,state, action, num_steps):
        #known_states = self.model.get_known_states()
        #print(known_states)
        eb = 0 if not self.include_exploration_bonus else self.beta / np.sqrt(self.model.counts[state,action])
        for _ in range(num_steps):
            self.policy.q_table = self.model.rewards + self.gamma*np.dot(self.model.transitions, self.policy.get_max_q_values()) + eb
        #print('policy =',self.policy.q_table)

    def end_of_episode(self):
        # Update policy
        #self.vectorized_iterate_policy(num_steps=self.episodic_backup_steps)
        BaseAgent.end_of_episode(self)
