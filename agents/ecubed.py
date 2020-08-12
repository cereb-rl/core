'''
RMaxAgentClass.py: Class for an RMaxAgent from [Strehl, Li and Littman 2009].

Notes:
    - Assumes WLOG reward function codomain is [0,1] (so RMAX is 1.0)
'''

# Python imports.
import random
import numpy as np
from collections import defaultdict

# Local classes.
from core.agents import BaseAgent
from core.utils.Models import KnownTabularModel
from core.utils.Policy import DiscreteTabularPolicy
from core.utils.policy_helpers import *

ECUBED_DEFAULTS = {
    'gamma': 0.95,
    'max_reward': 1,
    'epsilon_one': 0.99
}


class ExploreModel(KnownTabularModel):
    def __init__(self, num_actions, default_reward=1, known_threshold=float('inf')):
        KnownTabularModel.__init__(num_actions, default_reward, known_threshold)

    def get_reward(self, state, action):
        if self.is_known(state, action):
            return 0
        else:
            return self.default_reward


class ECubedAgent(BaseAgent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, observation_space, action_space, name="RMax Agent", extra_params={},
                 starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name)

        # Hyper-parameters
        self.max_reward = self.params['max_reward']
        self.epsilon_one = self.params['epsilon_one']
        self.known_threshold = self.params['known_threshold']
        # self.gamma = self.params['gamma']

        # Agent Trackers
        self.balanced_wandering = True

        # Policy and Model Setup
        self.starting_policy = starting_policy
        self.max_backup_steps = 5
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
        self.explorePolicy = TabularPolicy(self.actions, self.max_reward)
        self.balanced_wandering = True

    def learn(self, state, reward, done=False):
        if self.balanced_wandering:
            # Choose the action we've encountered the least
            _, action = min((self.model.get_counts(state, possible_action), possible_action) for possible_action in
                            range(self.action_space.n))
        else:
            # Compute best action by argmaxing over Q values of all possible s,a pairs
            action = self.policy.get_max_action(state)
        self.update(self.prev_state, self.prev_action, state, reward)
        self.prev_action = action
        BaseAgent.learn(self, state, reward, done)
        return action

    def update(self, state, action, next_state, reward):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates T and R.
        '''
        if state != None and action != None:
            if not self.model.is_known(state, action):
                # Add new data points if we haven't seen this s-a enough.
                self.model.update(state, action, reward, next_state)

                if self.model.is_known(state, action):
                    self.update_policy(state, action)

    def update_policy(self, state, action):
        # print("updating policy")
        # Start updating Q values for subsequent states
        lim = int(np.log(1 / (self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
        for _ in range(1, lim):
            for curr_state in self.model.get_states():
                for curr_action in self.actions:
                    if self.model.is_known(curr_state, curr_action):
                        new_value = self._get_reward(curr_state, curr_action) + (
                                    self.gamma * expected_q_value(curr_state, curr_action, self.policy, self.model))
                        self.policy.set_value(curr_state, curr_action, new_value)
