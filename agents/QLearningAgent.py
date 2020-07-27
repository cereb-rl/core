'''
    QLearningAgent.py: Class for QLearningAgent from [Sutton and Barto].
    
    Notes:
        - 
'''

# Python Imports
import numpy as np
import random
import copy

# Core Imports
from core.agents.Agent import Agent
from core.utils.Policy import TabularPolicy

AgentParameters = {
    'name': 'QLearningAgent',
    'gamma': 0.6, # discount factor
    'alpha': 0.1,
    'epsilon': 0.1 # exploration factor
}

class QLearningAgent:
    '''
        Implementation for an Q-Learning Agent [Sutton and Barto]
    '''

    def __init__(self, actions, hyperparameters={}, starting_policy=None):
        Agent.__init__(self, actions, AgentParameters, hyperparameters)
        self.starting_policy = starting_policy
        self.policy = self.starting_policy if self.starting_policy else TabularPolicy(self.actions)
    
    def select_action(self, state):
        '''
        Args:
            state (State)

        Summary:
            Returns action (int)
        '''
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions) # Explore action space
        else:
            action = self.policy.get_max_action(state) # Exploit learned values
        return action

    def update(self, state, action, reward, next_state):
        '''
            Args:
                state (State)
                action (str)
                reward (float)
                next_state (State)

            Summary:
                Updates Policy.
        '''
        old_value = self.policy.get_value(state, action)
        next_max = self.policy.get_max_value(next_state)
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.policy.set_value(state, action, new_value)
    
    def dump_policy(self):
        current_policy = copy.deepcopy(self.policy)
        return current_policy

    def __str__(self):
        return str(self.policy)