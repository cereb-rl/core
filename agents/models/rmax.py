# Python imports
import numpy as np

# Core imports
from core.agents.models.tabular import DiscreteTabularModel

class RMaxModel(DiscreteTabularModel):
    def __init__(self, observation_space, action_space, default_reward, limit):
        self.known_rewards = None
        self.known_transitions = None
        DiscreteTabularModel.__init__(self, observation_space, action_space, default_reward, limit)
    
    def update(self, state, action, reward, next_state):
        if (state is not None) and (action is not None) and (next_state is not None):
            was_not_known = not self.is_known_state(state)

            DiscreteTabularModel.update(self, state, action, reward, next_state)

            # New known (state, action) pair
            if was_not_known and self.is_known_state(state):
                self.known_rewards[state] = self.rewards[state]
                self.known_transitions[state] = self.transitions[state]

    def reset(self):
        DiscreteTabularModel.reset(self)
        self.known_rewards = np.array(self.rewards)
        self.known_transitions = np.array(self.transitions)
        