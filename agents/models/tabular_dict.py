# Python imports.
import random
import numpy as np
from collections import defaultdict


class TabularModel:
    '''
        Implements a tabular MDP model
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        '''
        Summary:
            Resets the model back to its tabula rasa config.
        '''
        self.rewards = defaultdict(lambda: defaultdict(float))  # S --> A --> reward
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # S --> A --> S' --> counts
        self.state_action_counts = defaultdict(lambda: defaultdict(int))  # S --> A --> #rs
        self.prev_state = None
        self.prev_action = None

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (any)
            action (any)
            reward (float)
            next_state (any)

        Summary:
            Updates T and R.
        '''
        if state != None and action != None:
            self.rewards[state][action] += reward
            self.transitions[state][action][next_state] += 1
            self.state_action_counts[state][action] += 1
            self.prev_state = next_state
            self.prev_action = action

    def get_reward(self, state, action):
        '''
        Args:
            state (any)
            action (any)

        Returns:
            MLE
        '''
        if not self.state_action_counts[state][action]:
            return 0
        return float(self.rewards[state][action]) / self.state_action_counts[state][action]

    def get_transition(self, state, action, next_state):
        '''
        Args: 
            state (any)
            action (any)
            next_state (any)

            Returns:
                Empirical probability of transition n(s,a,s')/n(s,a) 
        '''
        if not self.state_action_counts[state][action]:
            return 0
        return self.transitions[state][action][next_state] / self.state_action_counts[state][action]

    def get_count(self, state, action):
        '''
            Args: 
                state (any)
                action (any)

            Returns:
                counts for rewards and transitions
        '''
        return self.state_action_counts[state][action]

    def get_states(self):
        '''
            Args: 

            Returns:
                reward states 
        '''
        return self.rewards.keys()

    def __str__(self):
        return str(self.transitions) + "\n" + str(self.rewards) + "\n" + str(self.state_action_counts)

    # def __getitem__(self, key):
    # def __getitem__(self, key):

class KnownTabularModel(TabularModel):
    '''
        Extends the tabular model to include known states
    '''

    def __init__(self, num_actions, default_reward=1, known_threshold=float('inf')):
        TabularModel.__init__(self)
        self.num_actions = num_actions
        self.default_reward = 1
        self.known_threshold = known_threshold
        self.known_states = set()

    def is_known(self, state, action):
        '''
        Args: 
            state (any)
            action (any)

            Returns:
                True if reward and transition counts are greater than known_threshold
        '''
        return self.state_action_counts[state][action] >= self.known_threshold

    def is_known_state(self, state):
        '''
        Args: 
            state (any)

            Returns:
                True if reward and transition counts for state are greater than known_threshold
        '''
        if state not in self.known_states:
            seen_all_actions = (len(self.state_action_counts[state]) >= self.num_actions)
            seen_enough_actions = all(self.is_known(state, action) for action in self.state_action_counts[state])
            if seen_all_actions and seen_enough_actions:
                self.known_states.add(state)
        return state in self.known_states

    def get_known_states(self):
        '''
        Args: 
            state (any)

            Returns:
                True if reward and transition counts for state are greater than known_threshold
        '''
        return list(self.known_states)

    def get_reward(self, state, action):
        '''
        Args:
            state (any)
            action (any)

        Returns:
            MLE
        '''
        if self.is_known(state, action):
            return TabularModel.get_reward(self, state, action)
        else:
            return self.default_reward

    def update(self, state, action, reward, next_state):
        TabularModel.update(self, state, action, reward, next_state)
        if self.is_known_state(state):
            self.known_states.add(state)
