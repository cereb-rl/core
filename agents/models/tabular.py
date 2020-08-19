import numpy as np
from gym.spaces import Space

class DiscreteTabularModel:
    '''
        Implements a tabular MDP model for a discrete state and action space
    '''

    def __init__(self, observation_space: Space, action_space: Space, default_reward = 0, limit=10000):
        self.observation_space = observation_space
        self.action_space = action_space
        self.default_reward = default_reward
        self.known_threshold = limit
        self.limit = limit

        self.counts = None
        self.rewards = None
        self.transitions = None

        self.reset()

    def reset(self):
        # S --> A --> #rs
        self.counts = np.zeros(
            shape=(self.observation_space.n, self.action_space.n),
            dtype=np.int32)
        # S --> A --> average reward
        self.rewards = np.full(
            shape=(self.observation_space.n, self.action_space.n),
            fill_value=self.default_reward,
            dtype=np.float)
        # S --> A --> S' --> probabilities [0,1]
        self.transitions = np.zeros(
            shape=(self.observation_space.n, self.action_space.n, self.observation_space.n),
            dtype=np.float32)

        # Optimistic initialization - P(s', a, s) = 1 if s' == s
        for state in range(self.observation_space.n):
            for action in range(self.action_space.n):
                self.transitions[state,action,state] = 1

    def update(self, state, action, reward, next_state):
        if (state is not None) and (action is not None) and (self.counts[state, action] < self.limit):

            # Update rewards
            self.rewards[state, action] = (self.rewards[state, action]*self.counts[state, action] + reward) / (self.counts[state, action]+1)

            # Update probabilities
            transition_counts = self.transitions[state, action]*self.counts[state, action]
            transition_counts[next_state] += 1
            self.transitions[state, action] = transition_counts/(self.counts[state, action]+1)
            
            # Update counts
            self.counts[state, action] += 1

    def get_reward(self, state, action):
        return self.rewards[state, action]

    def get_transition(self, state, action, next_state):
        # Returns transition probability
        return self.transitions[state, action, next_state]

    def get_count(self, state, action):
        return self.counts[state, action]

    def get_states(self):
        # Returns boolean array representing all the states that have been encountered
        return np.apply_along_axis(np.all, 1, self.counts > 0)

    def is_known(self, state, action):
        return self.counts[state, action] >= self.known_threshold

    def is_known_state(self, state):
        return np.all(self.counts[state] >= self.known_threshold)

    def get_known_states(self):
        return np.apply_along_axis(np.all, 1, self.counts >= self.known_threshold)

    def __str__(self):
        return str(self.transitions) + "\n" + str(self.rewards) + "\n" + str(self.counts)
