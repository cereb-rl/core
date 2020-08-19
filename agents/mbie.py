'''
MBIEAgentClass.py: Class for an RMaxAgent from [Strehl, Li and Littman 2009].

Notes:
    - Assumes WLOG reward function codomain is [0,1] (so RMAX is 1.0)
'''

# Python imports.
import random
from collections import defaultdict
import numpy as np

# Local classes.
from core.agents.base import BaseAgent
from core.agents.models import DiscreteTabularModel
from core.agents.policies import ExploreLeastKnown, DiscreteTabularPolicy
from core.utils import constants, specs

MBIE_DEFAULTS = {
    'gamma': 0.95,
    'known_threshold': 5,
    'max_reward': 1,
    'epsilon_one': 0.99,
    'beta': 1,
    'include_eb': True,
    'max_stepwise_backups': 20,
    'max_episodic_backups': 0,
}

MBIE_SPEC = specs.AgentSpec(
    observation_space=constants.SpaceTypes.DISCRETE,
    action_space=constants.SpaceTypes.DISCRETE
)


class MBIEAgent(BaseAgent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, observation_space, action_space, name="MBIE Agent", params={}, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name, params=dict(MBIE_DEFAULTS, **params))

        # Policy Setup
        if starting_policy:
            self.predict_policy = starting_policy
        else:
            self.predict_policy = DiscreteTabularPolicy(self.observation_space, self.action_space, default_value=1/(1-self.gamma))
        self.backup_lim = int(np.log(1/(self.params['epsilon_one'] * (1 - self.gamma))) / (1 - self.gamma))
        self.policy_iterations = 0

        # Model Setup
        self.model = DiscreteTabularModel(observation_space, action_space, default_reward=self.params['max_reward'], limit=self.params['known_threshold'])

        self.learn_policy = self.predict_policy

    def stepwise_update(self, state, reward):
        if not self.model.is_known(self.prev_state, self.prev_action):
            # Add new data points if we haven't seen this s-a enough.
            self.model.update(self.prev_state, self.prev_action, reward, state)
            self.vectorized_iterate_policy(self.prev_state, self.prev_action, num_steps=min(self.params['max_stepwise_backups'], self.backup_lim))

    def episodic_update(self):
        # Update policy
        # self.vectorized_iterate_policy(num_steps=min(self.params['max_episodic_backups'], self.backup_lim))
        return

    def vectorized_iterate_policy(self, state, action, num_steps):
        eb = 0 if not self.params['include_eb'] else self.params['beta']/ np.sqrt(self.model.counts[state,action])
        for _ in range(num_steps):
            self.predict_policy.q_table = self.model.rewards + self.gamma*np.dot(self.model.transitions, self.predict_policy.get_max_q_values()) + eb

    def get_bellman_backup_function(self):
        def update_fn(state, action, policy, model, gamma):
            value = bellman_policy_backup(state, action, policy, model, gamma)
            if self.model.get_count(state, action) > 0:
                value += self.beta / np.sqrt(self.model.get_count(state, action))
            return value
        return update_fn