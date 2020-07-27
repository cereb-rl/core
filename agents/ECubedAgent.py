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
from core.agents.Agent import Agent
from core.utils.Models import KnownTabularModel
from core.utils.Policy import TabularPolicy
from core.utils.policy_helpers import *

AgentParameters = {
    'name': 'RMaxAgent',
    'gamma': 0.95,
    'max_reward': 1,
    'epsilon_one': 0.99
}

class ECubedAgent(Agent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, actions, hyperparameters={}, starting_policy=None):
        Agent.__init__(self, AgentParameters, hyperparameters, starting_policy)
        self.actions = actions
        self.starting_policy = starting_policy
        self.model = KnownTabularModel(len(actions), self.known_threshold)

        self.reset()

    def reset(self):
        '''
        Summary:
            Resets the agent model and policy back to its tabula rasa config.
        '''
        self.model.reset()
        self.exploitPolicy = self.starting_policy if self.starting_policy else TabularPolicy(self.actions)
        self.explorePolicy = TabularPolicy(self.actions, self.max_reward)
        self.balanced_wandering = True

    def select_action(self, state):
        if self.balanced_wandering:
            # Choose the action we've encountered the least
            _, action = min((self.model.get_counts(state, possible_action), possible_action) for possible_action in self.actions)
        else:
            # Compute best action by argmaxing over Q values of all possible s,a pairs
            action = self.policy.get_max_action(state)

        # Update pointers.
        self.prev_action = action
        self.prev_state = state

        return action

    def update(self, state, action, reward, next_state):
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
        #print("updating policy")
        # Start updating Q values for subsequent states
        lim = int(np.log(1/(self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
        for _ in range(1, lim):
            for curr_state in self.model.get_states():
                for curr_action in self.actions:
                    if self.model.is_known(curr_state, curr_action):
                        new_value = self._get_reward(curr_state, curr_action) + (self.gamma * expected_q_value(curr_state, curr_action, self.policy, self.model))
                        self.policy.set_value(curr_state, curr_action, new_value)

    def _get_reward(self, state, action):
        '''
        Args:
            state (any)
            action (any)

        Returns:
            Believed reward of executing @action in @state. If R(s,a) is unknown
            for this s,a pair, return self.max_reward. Otherwise, return the MLE.
        '''

        if self.model.is_known(state, action):
            return self.model.get_reward(state, action)
        else:
            # Otherwise return max_reward.
            return self.max_reward

    def __str__(self):
        return str(self.policy) + "\n" + str(self.model)
