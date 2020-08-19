'''
RMaxAgentClass.py: Class for an RMaxAgent from [Strehl, Li and Littman 2009].

Notes:
    - Assumes WLOG reward function codomain is [0,1] (so RMAX is 1.0)
'''

# Python imports.
import random
import numpy as np
#from itertools import product
from typing import Dict

from gym.spaces import Discrete

# Local classes.
from core.agents.base import BaseAgent
from core.utils.models import DiscreteTabularModel
from core.utils.policies import LinearSoftmaxPolicy
from core.utils.policy_helpers import *

REINFORCE_DEFAULTS: Dict[str, float] = {
    'gamma': 0.99,  # discount factor
    'alpha': 0.000025,  # learning rate
    'max_alpha': 1, # Maximum learning rate
    'min_alpha': 0.01, # Minimum learning rate
    'epsilon': 1,  # exploration factor
    'max_epsilon': 1,    # Exploration probability at start
    'min_epsilon': 0.01,  # Minimum exploration probability
    'decay_rate': 0.001,    # Exponential decay rate for exploration prob
    'ada_divisor': 25,     # decay rate parameter
}

class ReinforceAgent(BaseAgent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, observation_space, action_space, name="REINFORCE Agent", extra_params={}, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name, params=dict(REINFORCE_DEFAULTS, **extra_params))

        # Hyper-parameters
        self.alpha = self.params['alpha']

        # Policy Setup
        num_features = self.observation_space.shape[0]
        if isinstance(observation_space, Discrete):
            num_features = observation_space.n

        self.starting_policy = starting_policy if starting_policy else LinearSoftmaxPolicy(num_features=num_features, num_actions=action_space.n)

        # Model Setup
        self.last_episode_states = []
        self.last_episode_actions = []
        self.last_episode_rewards = []

        self.reset()
    
    def predict(self, state):
        probs = self.policy.pi(state)
        action = np.random.choice(np.arange(self.action_space.n), p=probs)
        #print(self.policy.weights)
        return action

    def learn(self, state, reward, done=False):
        """

        :param state:
        :param reward:
        :param done:
        :return:
        """
        probs = self.policy.pi(state)
        # print(probs)
        action = np.random.choice(np.arange(self.action_space.n), p=probs)
        if self.prev_state is not None and self.prev_action is not None:
            self.update(self.prev_state, self.prev_action, reward)

        BaseAgent.learn(self, state, reward, done)
        self.prev_action = action
        return action

    def update(self, state, action, reward):
        self.last_episode_states.append(state)
        self.last_episode_actions.append(action)
        self.last_episode_rewards.append(reward)


    def update_policy(self):
        # calculate gradients for each action over all observations
        gradients = np.array([self.policy.gradient(ob,action) for ob, action in zip(self.last_episode_states, self.last_episode_actions)])

        #print(gradients.shape)

        # calculate temporaly adjusted, discounted rewards
        discounted_rewards = self.policy.discount_rewards(self.last_episode_rewards, self.gamma)

        # gradients times rewards
        grad_dot = np.dot(discounted_rewards, gradients)
        #print(grad_dot)

        # gradient ascent on parameters
        self.policy.weights += self.alpha*grad_dot
        #self.policy.weights = self.policy.weights / np.linalg.norm(self.policy.weights)
        #print(self.policy.weights)

    def end_of_episode(self):
        self.update_policy()
        self.last_episode_states = []
        self.last_episode_actions = []
        self.last_episode_rewards = []
        BaseAgent.end_of_episode(self)

    # @property
    # def prereq(self):
    #     return ('observation_space'= Box,'action_space'= Box)
