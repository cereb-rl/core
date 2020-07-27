from core.utils.Policy import TabularPolicy
from core.utils.Models import TabularModel

def expected_q_value(state, action, policy: TabularPolicy, model: TabularModel, gamma = 1):
    '''
        Args:
            state (any)
            action (any)
            policy (TabularPolicy)
            model (TabularModel)

        Summary:
            Returns Expected Q value based on model using policy
    '''
    return model.get_reward(state, action) + gamma * sum((model.get_transition(state, action, next_state) * policy.get_max_q_value(next_state)) for next_state in model.get_states())

def expected_value(state, policy: TabularPolicy, model: TabularModel, gamma = 1):
    '''
        Args:
            state (any)
            model (TabularModel)

        Summary:
            Returns Expected value function based on model using policy
    '''
    return max(expected_q_value(state, action, policy, model, gamma) for action in policy.actions)

def iterate_policy(policy: TabularPolicy, model: TabularModel, num_steps, gamma):
    '''
        Args:
            policy (TabularPolicy)
            model (TabularModel)

        Summary:
            Returns policy after num_steps value iterations
    '''
    for t in range(num_steps):
        for state in model.get_states():
            for action in policy.actions:
                policy(state, action, expected_q_value(state, action, policy, model, gamma))
    return policy

def iterate_value(policy: TabularPolicy, model: TabularModel, num_steps):
    '''
        Args:
            state (any)
            model (TabularModel)

        Summary:
            Returns Expected Q value based on model using policy
    '''
    return policy