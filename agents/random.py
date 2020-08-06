""" RandomAgentClass.py: Class for a randomly acting RL Agent """

# Python imports.
import random

# Core imports
from core.agents import BaseAgent


class RandomAgent(BaseAgent):
    """ Class for a random decision maker. """

    def __init__(self, observation_space, action_space, name="Random Agent"):
        BaseAgent.__init__(self, observation_space, action_space, name)

    def learn(self, state, reward=None):
        """

        :param state: 
        :param reward: 
        :return: 
        """
        BaseAgent.learn(self, state, reward)
        return self.action_space.sample()
