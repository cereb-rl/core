import numpy as np
from gym.spaces import Space


class DiscreteTabularPolicy:
    '''
        Implements a tabular Q-function for a discrete observation and action space
    '''

    def __init__(self, observation_space: Space, action_space: Space, default_value=0):
        self.observation_space = observation_space
        self.action_space = action_space
        self.default_value = default_value
        self.q_table = np.full(
            shape=(observation_space.n, action_space.n),
            fill_value=default_value,
            dtype=np.float)
        self.value_table = None

        self.reset_values()
    
    def get_action(self, state):
        return self.get_max_action(state)

    def set_q_value(self, state, action, new_value):
        self.q_table[state, action] = new_value

    def set_value(self, state, new_value):
        self.value_table[state] = new_value

    def get_q_value(self, state, action):
        return self.q_table[state, action]

    def get_max_q_value(self, state):
        return np.amax(self.q_table[state])

    def get_max_q_values(self):
        return np.amax(self.q_table, axis=1)

    def get_value(self, state):
        return self.value_table[state]

    def get_max_action(self, state):
        max_q_value = self.get_max_q_value(state)
        return np.random.choice(np.arange(self.action_space.n)[self.q_table[state] == max_q_value])

    def reset_values(self):
        self.value_table = np.full(shape=self.observation_space.n, fill_value=self.default_value, dtype=np.float)

    def __str__(self):
        if type(self.q_table) == None:
            return "Empty q-table"
        else:
            return self.q_table.__str__()