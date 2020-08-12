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
from core.utils.Policy import DiscreteTabularPolicy
from core.utils.policy_helpers import *

RMAX_DEFAULTS = {
    'gamma': 0.95,
    'known_threshold': 10,
    'max_reward': 1,
    'epsilon_one': 0.99,
    'max_stepwise_backups': 20,
    'max_episodic_backups': 0,
}

class RMaxAgent(BaseAgent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, observation_space, action_space, name="RMax Agent", extra_params={}, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name, params=dict(RMAX_DEFAULTS, **extra_params))

        # Hyper-parameters
        self.max_reward = self.params['max_reward']
        self.epsilon_one = self.params['epsilon_one']
        self.known_threshold = self.params['known_threshold']
        self.gamma = self.params['gamma']

        # Policy Setup
        self.starting_policy = starting_policy if starting_policy else DiscreteTabularPolicy(self.observation_space, self.action_space, default_value=1/(1-self.gamma))
        self.backup_lim = int(np.log(1/(self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
        self.stepwise_backup_steps = min(self.backup_lim, self.params['max_stepwise_backups'])
        self.episodic_backup_steps = min(self.backup_lim, self.params['max_episodic_backups'])
        self.policy_iterations = 0

        # Model Setup
        self.model = DiscreteTabularModel(observation_space, action_space, default_reward=self.max_reward, is_known_mdp=True, limit=self.known_threshold)

        self.reset()

    def learn(self, state, reward, done=False):
        """

        :param state:
        :param reward:
        :param done:
        :return:
        """
        if self.model.is_known_state(state):
            action = self.policy.get_max_action(state)  # Exploit learned values
        else:
            _, action = min((self.model.get_count(state, act), act) for act in range(self.action_space.n))

        if self.prev_state is not None and self.prev_action is not None:
            self.update(self.prev_state, self.prev_action, reward, state)

        BaseAgent.learn(self, state, reward, done)
        self.prev_action = action
        return action

    def update(self, state, action, reward, next_state):
        if not self.model.is_known(state, action):
            self.model.update(state, action, reward, next_state)
            if self.model.is_known_state(state):
                self.vectorized_iterate_policy(num_steps=self.stepwise_backup_steps)

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
        if self.episode_learn_steps:
            self.vectorized_iterate_policy(num_steps=self.episodic_backup_steps)
        BaseAgent.end_of_episode(self)
