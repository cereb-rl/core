""" AgentClass.py: Class for a basic RL Agent """

# Python imports

# External imports
from gym.spaces import Space, Discrete

class BaseAgent(object):
    """ Abstract Agent class. """
    alpha = 0.5,
    gamma = 1,

    def __init__(self, observation_space: Space, action_space: Space, name="BaseAgent", params={'gamma': 0.95}, specs=None):

        # Check that we have valid inputs
        if specs:
            self.validate(observation_space, action_space, specs)

        self.observation_space = observation_space
        self.action_space = action_space

        # Naming Agent
        self.name = name

        # Standard Parameters
        self.params = params
        if 'alpha' in params:
            self.alpha = params['alpha']
        if 'gamma' in params:
            self.gamma = self.params['gamma']
        if 'epsilon' in params:
            self.epsilon = self.params['epsilon']

        # Setup State Trackers
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.episode_number = 0
        self.total_learn_steps = 0
        self.total_predict_steps = 0
        self.episode_learn_steps = 0

        # Setup policy and model
        self.learn_policy = None
        self.predict_policy = None
        self.model = None

        print(f'Creating {self.name} for environment with {self.observation_space} space and {self.action_space} actions')

    def learn(self, state, reward=None, done=False):
        """

        :param state:
        :param reward:
        :param done:
        """
        # do stepwise update
        self.stepwise_update(state, reward)

        # update class
        self.prev_state = state
        self.prev_reward = reward

        # update action
        if self.learn_policy:
            self.prev_action = self.learn_policy.get_action(state)
        else:
            self.prev_action = self.action_space.sample()

        self.episode_learn_steps += 1

        # if done, end episode
        if done:
            self.end_of_episode()

        return self.prev_action
    
    def stepwise_update(self, state, reward):
        """
        agent implements this
        """
        pass

    def predict(self, state):
        """

        :param state:
        """
        if self.predict_policy:
            return self.predict_policy.get_action(state)
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
        if self.episode_learn_steps:
            self.episodic_update()

        self.episodic_update()
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.episode_number += 1
        self.total_learn_steps += self.episode_learn_steps
        self.episode_learn_steps = 0
    
    def episodic_update(self):
        """
        agent implements this
        """
        pass

    def reset(self):
        """
        Wipes the agent
        """
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.episode_number = 0
        self.total_learn_steps = 0
        self.total_predict_steps = 0
        self.episode_learn_steps = 0

        self.learn_policy = None
        self.predict_policy = None
        
        if self.model:
            self.model.reset()
    
    def validate(self, observation_space, action_space, specs):
        pass


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
        return self.predict_policy

    def __str__(self):
        return self.name + "\n" + str(self.policy) + "\n" + str(self.model)
