""" AgentClass.py: Class for a basic RL Agent """

# Python imports

# External imports
from gym.spaces import Space


class BaseAgent(object):
    """ Abstract Agent class. """
    name: str

    # Standard Parameters
    epsilon: int
    gamma: int

    episode_number: int
    step_number: int

    def __init__(self, observation_space: Space, action_space: Space, name="BaseAgent", epsilon=0.9, gamma=0.6):
        self.observation_space = observation_space
        self.action_space = action_space

        # Naming Agent
        self.name = name

        # Standard Parameters
        self.epsilon = epsilon
        self.gamma = gamma

        # Setup State Trackers
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.episode_number = 0
        self.step_number = 0

        print(f'Creating {self.name} for environment with {self.observation_space} space and {self.action_space} actions')

    def learn(self, state, reward, done=False):
        """

        :param state:
        :param reward:
        :param done:
        """
        pass

    def predict(self, state):
        """

        :param state:
        """
        pass

    def start_of_episode(self, state):
        """

        :param state:
        :return: action
        """
        self.prev_state = state
        self.prev_action = self.action_space.sample()
        return self.prev_action

    def end_of_episode(self):
        """
        ends episode
        """
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.episode_number += 1

    def reset(self):
        """
        Wipes the agent
        """
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.step_number = 0
        self.episode_number = 0

    def set_name(self, name):
        """

        :param name: str
        """
        self.name = name

    @property
    def get_name(self):
        """

        :return: agent's name
        """
        return self.name

    def __str__(self):
        return str(self.name)
