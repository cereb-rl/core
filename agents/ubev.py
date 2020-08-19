'''
UBEVAgentClass.py: Class for an UBEV Agent [Dann, Lattimore and Brunskill 2018].
'''

# Python imports.
import numpy as np

# Local classes.
from core.agents.base import BaseAgent
from core.agents.models import DiscreteTabularModel
from core.agents.policies import DiscreteTabularPolicy
from core.utils import constants, specs

UBEV_DEFAULTS = {
    'gamma': 0.95,
    'delta': 1,
    'max_stepwise_backups': 0,
    'max_episodic_backups': 20,
}

UBEV_SPEC = specs.AgentSpec(
    observation_space=constants.SpaceTypes.DISCRETE,  # observation 
    action_space=constants.SpaceTypes.DISCRETE,
)

class UBEVAgent(BaseAgent):
    '''
    Implementation for an UBEV Agent [Dann, Lattimore and Brunskill 2018]
    https://arxiv.org/pdf/1703.07710.pdf
    '''

    def __init__(self, observation_space, action_space, name="UBEV Agent", parameters={}, starting_policy=None):
        BaseAgent.__init__(self, observation_space, action_space, name, params=dict(UBEV_DEFAULTS, **parameters), specs=UBEV_SPEC)

        # Policy Setup
        if starting_policy:
            self.predict_policy = starting_policy
        else:
            self.predict_policy = DiscreteTabularPolicy(self.observation_space, self.action_space, default_value=1/(1-self.gamma))
        self.learn_policy = self.predict_policy
        self.policy_iterations = 0

        # Model Setup
        self.model = DiscreteTabularModel(observation_space, action_space)

    def stepwise_update(self, state, reward):
        self.model.update(self.prev_state, self.prev_action, reward, state)

    def episodic_update(self):
        self.vectorized_iterate_policy(num_steps=min(self.episode_learn_steps, self.params['max_episodic_backups']))

    def vectorized_iterate_policy(self, num_steps):
        phi = np.sqrt((2*np.log(np.log(np.maximum(np.e, self.model.counts))) + np.log(18*self.observation_space.n*self.action_space.n*self.episode_learn_steps/self.params['delta']))/self.model.counts)
        for t in range(num_steps):
            max_values = self.predict_policy.get_max_q_values()
            v_next = np.dot(self.model.transitions, max_values)
            self.predict_policy.q_table = np.minimum(1, self.model.rewards + phi) + np.minimum(np.max(max_values), v_next + (self.episode_learn_steps - t)*phi)
        self.policy_iterations += 1
