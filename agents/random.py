""" RandomAgentClass.py: Class for a randomly acting RL Agent """

# Core imports
from core.agents import BaseAgent


class RandomAgent(BaseAgent):
    """ Class for a random decision maker. """

    def __init__(self, observation_space, action_space, name="Random Agent"):
        BaseAgent.__init__(self, observation_space, action_space, name)