'''
Contains specification objects for agents, environments, etc
'''

from core.utils.constants import SpaceTypes


class AgentSpec:
    '''
    Object containing the specification for an agent
        - observation_space type
        - action_space type

    '''

    observation_space = None
    action_space = None

    def __init__(self, observation_space: SpaceTypes, action_space: SpaceTypes):
        self.observation_space = observation_space
        self.action_space = action_space