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

UBEV_DEFAULTS = {
    'gamma': 0.95,
    'delta': 1,
    'known_threshold': 10,
    'max_reward': 1,
    'epsilon_one': 0.99,
    'max_stepwise_backups': 100,
    'max_episodic_backups': 0,
}

class UBEVAgent(BaseAgent):
    '''
    Implementation for an UBEV Agent [Dann, Lattimore and Brunskill 2018]
    https://arxiv.org/pdf/1703.07710.pdf
    '''

    def __init__(self, observation_space, action_space, name="UBEV Agent", extra_params={}, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name, params=dict(UBEV_DEFAULTS, **extra_params))

        # Hyper-parameters
        self.max_reward = self.params['max_reward']
        self.epsilon_one = self.params['epsilon_one']
        self.known_threshold = self.params['known_threshold']
        self.gamma = self.params['gamma']
        self.delta = self.params['delta']

        # Policy Setup
        self.starting_policy = starting_policy if starting_policy else DiscreteTabularPolicy(self.observation_space, self.action_space, default_value=1/(1-self.gamma))
        self.policy = self.starting_policy
        self.backup_lim = int(np.log(1/(self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
        self.stepwise_backup_steps = min(self.backup_lim, self.params['max_stepwise_backups'])
        self.episodic_backup_steps = min(self.backup_lim, self.params['max_episodic_backups'])
        self.policy_iterations = 0

        # Model Setup
        self.model = DiscreteTabularModel(observation_space, action_space)

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
            self.update(self.prev_state, self.prev_action, reward, state)

        BaseAgent.learn(self, state, reward, done)
        self.prev_action = action
        return action

    def update(self, state, action, reward, next_state):
        self.model.update(state, action, reward, next_state)

    # def bellman_policy_backup(self, state, action):
    #     phi = np.sqrt((2*np.log())/model.get_counts(state, action))
    #     new_value = (1-self.gamma)*self.model.known_rewards[state, action] + self.gamma * np.inner(self.model.known_transitions[state, action],self.policy.get_max_q_values())
    #     self.policy.set_q_value(state, action, new_value)
    #
    # def iterate_policy(self, num_steps):
    #     for _ in range(num_steps):
    #         map(self.bellman_policy_backup, product(range(self.observation_space.n), range(self.action_space.n)))

    def vectorized_iterate_policy(self, num_steps):
        phi = np.sqrt((2*np.log(np.log(np.maximum(np.e, self.model.counts))) + np.log(18*self.observation_space.n*self.action_space.n*self.episode_learn_steps/self.delta))/self.model.counts)
        #print(self.model.counts)
        #print(phi)
        #print(self.episode_learn_steps)
        for t in range(num_steps):
            max_values = self.policy.get_max_q_values()
            v_next = np.dot(self.model.transitions, max_values)
            #print(max_values.reshape(self.observation_space.n, 1))
            #v_max = np.tile(max_values.reshape(self.observation_space.n,1), self.action_space.n)
            #print(np.max(max_values), v_next)
            self.policy.q_table = np.minimum(1, self.model.rewards + phi) + np.minimum(np.max(max_values), v_next + (self.episode_learn_steps - t)*phi)

    def end_of_episode(self):
        if self.episode_learn_steps:
            self.vectorized_iterate_policy(num_steps=self.episode_learn_steps)
            #print(self.policy)
        BaseAgent.end_of_episode(self)
