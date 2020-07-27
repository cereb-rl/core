''' AgentClass.py: Class for a basic RL Agent '''

# Python imports.
from collections import defaultdict

class Agent(object):
    ''' Abstract Agent class. '''

    def __init__(self, actions, agent_defaults, hyper_parameters = {}):

        # Initialize hyper_parameters as class attributes
        for param in agent_defaults:
            param_value = agent_defaults[param]
            if param in hyper_parameters:
                param_value = hyper_parameters[param]
            setattr(self, param, param_value)

        self.episode_number = 0
        self.prev_state = None
        self.prev_action = None
        self.actions = actions

        print(f'Creating {self.name} on environment with {len(self.actions)} actions')

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        return {}

    def act(self, state, reward):
        '''
        Args:
            state (State): see StateClass.py
            reward (float): the reward associated with arriving in state @state.
        Returns:
            (str): action.
        '''
        pass

    def policy(self, state):
        return self.act(state, 0)

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        self.prev_state = None
        self.prev_action = None
        self.step_number = 0

    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        self.prev_state = None
        self.prev_action = None
        self.episode_number += 1

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def __str__(self):
        return str(self.name)