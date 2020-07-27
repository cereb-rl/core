import numpy as np
import random
from collections import defaultdict

# Core Imports
from core.utils.Constants import Policy
from core.utils.Models import TabularModel

class TabularPolicy:
    '''
        Implements a tabular Q-function
    '''
    def __init__(self, actions, default_value = 0):
        self.actions = actions
        self.default_value = default_value
        self.q_table = defaultdict(lambda: defaultdict(lambda: default_value)) # S -> A -> default_value
        self.value_table = defaultdict(lambda: default_value) # S -> default_value

    def set_q_value(self, state, action, new_value):
        '''
            Args:
                state (any)
                action (any)
                new_value (float)

            Summary:
                Updates Q function with new_value
        '''
        self.q_table[state][action] = new_value

    def set_value(self, state, new_value):
        '''
            Args:
                state (any)
                action (any)
                new_value (float)

            Summary:
                Updates Value function with new_value
        '''
        self.value_table[state] = new_value

    def get_max_action(self, state):
        '''
            Args:
                state (any)

            Summary:
                Returns action with max q value for state
        '''
        max_value = self.get_max_q_value(state)
        max_actions = list(filter(lambda x: self.q_table[state][x] == max_value, self.q_table[state]))
        if max_actions:
            return random.choice(max_actions)
        else:
            return random.choice(self.actions)
    
    def get_max_q_value(self, state):
        '''
            Args:
                state (any)

            Summary:
                Returns maximum Q value
        '''
        action_values = list(self.q_table[state].values()) + [self.default_value]
        return max(action_values)

    def get_q_value(self, state, action):
        '''
            Args:
                state (any)
                action (any)

            Summary:
                Returns Q value for state-action pair
        '''
        return self.q_table[state][action]

    # def save():
    #     pass
        
    # def __getitem__(self, key):
    #     return self.q_table[key]

    def __str__(self):
        if type(self.q_table) == None:
            return "Empty q-table"
        else:
            return self.q_table.__str__()


# class TabularPolicy:
#     '''
#         Implements a tabular Q-function
#     '''
#     def __init__(self, actions, default_value = 0, initialization: Policy.Init = Policy.Init.ZEROS):
#         self.actions = actions
#         self.q_table = defaultdict(lambda: defaultdict(lambda: default_value))
#         # if initialization == Policy.Init.ONES:
#         #     self.q_table = np.ones([num_states, num_actions])
#         # else:
#         #     self.q_table = np.zeros([num_states, num_actions])

#     # SETTERS

#     def set_value(self, state, action, new_value):
#         self.q_table[state, action] = new_value

#     # GETTERS

#     def get_max_action(self, state):
        
#         return np.argmax(self.q_table[state])
    
#     def get_max_value(self, state):
#         return np.max(self.q_table[state])

#     def get_value(self, state, action):
#         return self.q_table[state, action]

#     def save():
#         pass
        
#     # def __getitem__(self, key):
#     #     return self.q_table[key]

#     def __str__(self):
#         if type(self.q_table) == None:
#             return "Empty q-table"
#         else:
#             return self.q_table.__str__()