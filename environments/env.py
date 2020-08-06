import gym

gym_envs = {
    'Taxi-v3': {
        'reward_norm': (10,30),  # (a,b) such that reward = (reward + a)/b
        'cycle_freq': 10,
        'expected_steps_per_episode': 1000
    },
    'FrozenLake-v0': {
        'reward_norm': (0,1),
        'cycle_freq': 100,
        'expected_steps_per_episode': 8
    },
    'FrozenLake8x8-v0': {
        'reward_norm': (0,1),
        'cycle_freq': 100,
        'expected_steps_per_episode': 8
    }
}

class GymEnvWrapper:
    def __init__(self, env_name):
        self.env = gym.make(env_name).env
        self.env_name = env_name
        self.env_spec = gym_envs[env_name]

    def reset(self):
        self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.normalize_reward(reward), done, info

    def normalize_reward(self, reward):
        x, y = self.env_spec['reward_norm']
        # if reward == 20:
        #     return 20
        # if reward == -1:
        #     return 0
        return (reward + x)/y

    def get_param(self, param_name):
        return self.env_spec[param_name]

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


