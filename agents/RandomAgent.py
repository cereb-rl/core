''' RandomAgentClass.py: Class for a randomly acting RL Agent '''

# Python imports.
import random

# Core imports
from core.agents.Agent import Agent

class RandomAgent(Agent):
    ''' Class for a random decision maker. '''

    def __init__(self, num_actions, name=""):
        name = "RandomAgent" if name is "" else name
        Agent.__init__(self, name=name, num_actions=num_actions)

    def choose_action(self, state, reward):
        return random.randrange(self.num_actions)