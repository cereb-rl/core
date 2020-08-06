from core.utils.Policy import TabularPolicy
from core.utils.Models import TabularModel


def bellman_policy_backup(state, action, policy: TabularPolicy, model: TabularModel, gamma=1):
    return model.get_reward(state, action) + gamma * sum(
        (model.get_transition(state, action, next_state) * policy.get_max_q_value(next_state)) for next_state in
        model.get_states())


def bellman_backup(state, policy: TabularPolicy, model: TabularModel, gamma=1):
    '''
        Args:
            state (any)
            model (TabularModel)

        Summary:
            Returns Expected value function based on model using policy
    '''
    return max(bellman_policy_backup(state, action, policy, model, gamma) for action in policy.actions)


def iterate_policy(policy: TabularPolicy, model: TabularModel, num_steps, gamma=1, states=None, update_fn=bellman_policy_backup):
    states = states if states else model.get_states()
    actions = range(policy.action_space.n)
    for _ in range(num_steps):
        for state in states:
            for action in actions:
                policy.set_q_value(state, action, update_fn(state, action, policy, model, gamma))
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
