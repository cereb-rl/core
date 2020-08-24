'''
RMaxAgentClass.py: Class for an RMaxAgent from [Strehl, Li and Littman 2009].

Notes:
    - Assumes WLOG reward function codomain is [0,1] (so RMAX is 1.0)
'''

# Python imports.
import random
import numpy as np
from itertools import product

# Local classes.
from core.agents.base import BaseAgent
from core.agents.models import RMaxModel
from core.agents.policies import ExploreLeastKnown, DiscreteTabularPolicy
from core.utils import constants, specs


RMAX_DEFAULTS = {
    'epsilon': 0,  # There's no exploration in R-Max
    'gamma': 0.95,  # discount factor
    'known_threshold': 5,  # number of occurrences of (state, action) pairs before it is marked as known
    'max_reward': 1,  # maximum reward
    'epsilon_one': 0.99,  #
    'max_stepwise_backups': 20,  # maximum number of backups per experience/transition during training
    'max_episodic_backups': 0,  # maximum number of backups at the end of an episode
}

RMAX_SPEC = specs.AgentSpec(
    observation_space=constants.SpaceTypes.DISCRETE,
    action_space=constants.SpaceTypes.DISCRETE
)

class RMaxAgent(BaseAgent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, observation_space, action_space, name="RMax Agent", parameters={}, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name, params=dict(RMAX_DEFAULTS, **parameters), specs=RMAX_SPEC)

        # Policy Setup
        if starting_policy:
            self.predict_policy = starting_policy
        else:
            self.predict_policy = DiscreteTabularPolicy(self.observation_space, self.action_space, default_value=1/(1-self.gamma))
        self.backup_lim = int(np.log(1/(self.params['epsilon_one'] * (1 - self.gamma))) / (1 - self.gamma))
        self.policy_iterations = 0

        # Model Setup
        self.model = RMaxModel(observation_space, action_space, default_reward=self.params['max_reward'], limit=self.params['known_threshold'])

        self.learn_policy = ExploreLeastKnown(
                action_space=self.action_space,
                policy=self.predict_policy,
                model=self.model
            )

    def stepwise_update(self, state, reward):
        if not self.model.is_known(self.prev_state, self.prev_action):
            self.model.update(self.prev_state, self.prev_action, reward, state)
            if self.model.is_known_state(self.prev_state):
                self.vectorized_iterate_policy(num_steps=min(self.backup_lim, self.params['max_stepwise_backups']))

    def episodic_update(self):
        self.vectorized_iterate_policy(num_steps=self.params['max_episodic_backups'])

    def vectorized_iterate_policy(self, num_steps):
        for _ in range(num_steps):
            assert (self.model.known_rewards < 1).any()
            self.predict_policy.q_table = self.model.known_rewards + self.gamma*np.dot(self.model.known_transitions, self.predict_policy.get_max_q_values())

    def bellman_policy_backup(self, state, action):
        new_value = (1-self.gamma)*self.model.known_rewards[state, action] + self.gamma * np.inner(self.model.known_transitions[state, action], self.predict_policy.get_max_q_values())
        self.predict_policy.set_q_value(state, action, new_value)
    
    def iterate_policy(self, num_steps):
        for _ in range(num_steps):
            map(self.bellman_policy_backup, product(range(self.observation_space.n), range(self.action_space.n)))
