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

    prev_reward: float

    def __init__(self, observation_space: Space, action_space: Space, name="BaseAgent", epsilon=0.4, gamma=0.6):
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
        self.episode_number = 0
        self.learn_steps = 0
        self.predict_steps = 0
        self.episode_learn_steps = 0

        # Setup policy and model
        self.policy = None
        self.model = None

        print(f'Creating {self.name} for environment with {self.observation_space} space and {self.action_space} actions')

    def learn(self, state, reward=None, done=False):
        """

        :param state:
        :param reward:
        :param done:
        """
        # update class
        self.prev_state = state
        self.prev_reward = reward

        # if done, end episode
        if done:
            self.end_of_episode()

        self.learn_steps += 1
        self.episode_learn_steps += 1
        pass

    def predict(self, state):
        """

        :param state:
        """
        if self.policy:
            return self.policy.get_max_action(state)
        return self.action_space.sample()

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
        self.episode_learn_steps = 0

    def reset(self):
        """
        Wipes the agent
        """
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.episode_number = 0
        self.learn_steps = 0
        self.predict_steps = 0
        self.episode_learn_steps = 0

        if self.policy:
            self.policy.reset()
        if self.model:
            self.model.reset()

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

    @property
    def get_policy(self):
        """

        :return:
        """
        return self.policy

    def __str__(self):
        return self.name + "\n" + str(self.policy) + "\n" + str(self.model)
