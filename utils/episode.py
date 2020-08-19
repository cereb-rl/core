class Episode:
    """ Represents a single episode """

    def __init__(self, state, observation, gamma=1.0):
        """
        Create an episode with the start_state state and start_observation observation
        :param state: start state
        :param observation: start observation
        """

        self._term = False

        self._states = [state]
        self._actions = []
        self._rewards = []
        self._observations = [observation]
        self._return = 0.0
        self._discount = 1.0
        self.gamma = gamma

    def add(self, action, reward, new_obs, new_state):
        """
        Add the result of a single action. Episodes are created incrementally so this action is being taken on the
        last state/observation in self.states which is nonempty
        :param action: Action taken
        :param reward: Reward received for taking that action
        :param new_obs:  New observation
        :param new_state: New state
        :return:
        """

        if self._term:
            raise AssertionError("Cannot add to a terminate episode")

        self._actions.append(action)
        self._rewards.append(reward)
        self._observations.append(new_obs)
        self._states.append(new_state)

        self._return = self._return + self._discount * reward
        self._discount = self._discount * self.gamma

    def terminate(self):
        """
        Terminate an episode
        :return:
        """
        self._term = True

    def get_states(self):
        """
        :return: All states in this episode
        """
        return self._states

    def get_actions(self):
        """
        :return: All actions in this episode
        """
        return self._actions

    def get_rewards(self):
        """
        :return: All rewards in this episode
        """
        return self._rewards

    def get_observations(self):
        """
        :return: All observations in this episode
        """
        return self._observations

    def get_return(self):
        """
        :return: Return the total discounted return in this episode
        """
        return self._return

    def get_state_action_pairs(self):
        """
        :return: Return an iterator of state and actions
        """

        return zip(self._states[:-1], self._actions)

    def get_transitions(self):
        """
        :return: Return all transition in this episode
        """

        return zip(self._states[:-1], self._actions, self._states[1:])

    def get_len(self):
        """
        :return returns the length of the episode which is the number of actions taken
        """
        return len(self._actions)