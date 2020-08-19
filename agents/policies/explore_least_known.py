class ExploreLeastKnown:
    def __init__(self, action_space, policy, model):
        self.action_space = action_space
        self.policy = policy
        self.model = model
    
    def get_action(self, state):
        if self.model.is_known_state(state):
            action = self.policy.get_max_action(state)  # Exploit learned values
        else:
            _, action = min((self.model.get_count(state, act), act) for act in range(self.action_space.n))
        return action