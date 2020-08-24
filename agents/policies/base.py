class BasePolicy:
    """ Base Policy """

    def __init__(self, action_space):
        """

        :param action_space: OpenAI Gym Space object
        """
        self.action_space = action_space

    def get_action(self, state):
        """

        :param state: current environment state
        :return: action based on the policy
        """
        return self.action_space.sample()
