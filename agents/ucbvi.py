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

UCBVI_DEFAULTS = {
    'gamma': 0.95,
    'known_threshold': 100,
    'max_reward': 20,
    'epsilon_one': 0.99,
    'delta': 0.1
}

class UCBVIAgent(BaseAgent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, observation_space, action_space, name="UCBVI Agent", params=None, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name)

        # Hyper-parameters
        self.params = dict(UCBVI_DEFAULTS)
        if params:
            for key, value in params:
                self.params[key] = value
        self.max_reward = self.params['max_reward']
        self.epsilon_one = self.params['epsilon_one']
        self.known_threshold = self.params['known_threshold']
        self.gamma = self.params['gamma']
        #self.max_reward = 1 / (1 - self.gamma)
        self.delta = self.params['delta']

        # Policy Setup
        self.starting_policy = starting_policy
        self.backup_lim = int(np.log(1/(self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
        self.stepwise_backup_steps = 0
        self.episodic_backup_steps = self.backup_lim
        self.policy_iterations = 0

        # Model Setup
        self.model = DiscreteTabularModel(observation_space, action_space, default_reward=self.max_reward, limit=self.known_threshold)

        # Experience Tracking
        self.last_episode = []
        # self.last_episode_model = KnownTabularModel(action_space.n, self.max_reward, 1)

        self.reset()

    def reset(self):
        '''
        Summary:
            Resets the agent model and policy back to its tabula rasa config.
        '''
        self.model.reset()
        self.policy = self.starting_policy if self.starting_policy else DiscreteTabularPolicy(self.observation_space, self.action_space, default_value=1/(1-self.gamma))

    def update(self, state, action, reward, next_state):
        self.model.update(state, action, reward, next_state)
        self.last_episode.append((state, action, reward, next_state))

    def get_bellman_backup_function(self):
        def backup_fn(state, action, policy, model, gamma):
            H = len(self.last_episode)
            L = np.log(5*(self.observation_space.n*self.action_space.n*self.learn_steps))
            if model.is_known(state, action):
                bonus = 7*H*L*np.sqrt(1/model.get_count(state, action))
                update_value = model.get_reward(state, action) + gamma * np.inner(model.get_transition(state, action), policy.get_values()) + bonus
                return min(policy.get_q_value(state, action), H, update_value)
            else:
                return H
        return backup_fn

    def vectorized_iterate_policy(self, num_steps):
        H = len(self.last_episode)
        L = np.log(5*(self.observation_space.n*self.action_space.n*self.learn_steps)/self.delta)
        #print(L)
        bonus1 = 7*H*L/np.sqrt((1+self.model.counts))
        #print(bonus1)
        for _ in range(num_steps):
            new_q_table = self.model.rewards + self.gamma*np.dot(self.model.transitions, self.policy.get_max_q_values()) + bonus1
            self.policy.q_table = np.minimum(self.policy.q_table, H, new_q_table)
        print(self.policy.q_table)

    def end_of_episode(self):
        self.policy.reset_values()
        if self.episode_learn_steps:
            self.vectorized_iterate_policy(num_steps=len(self.last_episode))
            self.policy_iterations += len(self.last_episode)
        BaseAgent.end_of_episode(self)
