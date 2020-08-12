import functools
import math

import numpy as np
import gym

GYM_ENV_WRAPPERS = {
    'default': [

    ],
    'Taxi-v3': [
        lambda env: LinearNormalizeRewardWrapper(env, -10, 20)
    ],
    'Roulette-v0': [
        lambda env: LinearNormalizeRewardWrapper(env, -1, 36)
    ],
    'CartPole-v0': [
        lambda env: BoxToDiscreteWrapper(env, buckets=[4, 4, 6, 12],
                                         l_bound=[env.observation_space.low[0], -0.5, env.observation_space.low[2],
                                                  -math.radians(50)],
                                         u_bound=[env.observation_space.high[0], 0.5, env.observation_space.high[2],
                                                  math.radians(50)])
    ],
    'MountainCar-v0': [
        lambda env: LinearNormalizeRewardWrapper(env, -1, 0),
        lambda env: BoxToDiscreteWrapper(env, buckets=[50,10],
                                         l_bound=[env.observation_space.low[0], env.observation_space.low[1]],
                                         u_bound=[env.observation_space.high[0],env.observation_space.high[1]])
    ],
    "maze-random-10x10-plus-v0": [
        lambda env: LinearNormalizeRewardWrapper(env, -0.001, 1),
        lambda env: MapActionWrapper(env, ['N', 'E', 'S', 'W']),
        lambda env: BoxToDiscreteWrapper(env, buckets=[10,10],
                                         l_bound=[0,0],
                                         u_bound=[9,9])
    ],
    "maze-random-30x30-plus-v0": [
        lambda env: LinearNormalizeRewardWrapper(env, -0.00011111111111111112, 1),
        lambda env: MapActionWrapper(env, ['N', 'E', 'S', 'W']),
        lambda env: BoxToDiscreteWrapper(env, buckets=[30,30],
                                         l_bound=[0,0],
                                         u_bound=[29,29])
    ],
    "maze-random-100x100-v0": [
        lambda env: LinearNormalizeRewardWrapper(env, -0.00011111111111111112, 1),
        lambda env: MapActionWrapper(env, ['N', 'E', 'S', 'W']),
        lambda env: BoxToDiscreteWrapper(env, buckets=[100,100],
                                         l_bound=[0,0],
                                         u_bound=[99,99])
    ]
}


def getWrappedEnv(env_name):
    env = gym.make(env_name).env
    wrappers = GYM_ENV_WRAPPERS['default']
    if env_name in GYM_ENV_WRAPPERS:
        wrappers = GYM_ENV_WRAPPERS[env_name]
    compose = lambda x, y: y(x)
    return functools.reduce(compose, wrappers, env)


class BoxToDiscreteWrapper(gym.ObservationWrapper):
    def __init__(self, env, buckets, l_bound, u_bound):
        gym.ObservationWrapper.__init__(self, env)
        assert isinstance(env.observation_space, gym.spaces.Box), \
            "Should only be used to wrap Box envs."
        self.buckets = np.array(buckets, dtype=np.int32)
        self.lower_bounds = np.array(l_bound)
        self.upper_bounds = np.array(u_bound)
        assert (env.observation_space.shape == self.buckets.shape == self.lower_bounds.shape == self.upper_bounds.shape)
        self.n = np.prod(self.buckets)
        self.observation_space = gym.spaces.Discrete(self.n)

    def observation(self, obs):
        ''' discretise the continuous state into buckets '''
        ratios = (obs - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
        new_obs = np.array(np.ceil(np.multiply(self.buckets, ratios)), dtype=np.int32)
        new_obs = np.minimum(np.maximum(new_obs, np.ones(shape=self.buckets.shape, dtype=np.int32)), self.buckets)
        # Mapping obs list to integer based on buckets
        new_obs_int, c_prod = 0, 1
        for i in range(self.buckets.shape[0]):
            new_obs_int += c_prod * (new_obs[i] - 1)
            c_prod *= self.buckets[i]
        return new_obs_int


class DiscreteToBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete), \
            "Should only be used to wrap Discrete envs."
        self.n = self.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (self.n,))

    def observation(self, obs):
        new_obs = np.zeros(self.n)
        new_obs[obs] = 1
        return new_obs


class LinearNormalizeRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, l_bound=0, u_bound=1):
        super().__init__(env)
        # l, r = env.reward_range
        self.lower_bound = l_bound
        self.upper_bound = u_bound

    def reward(self, rew):
        #print(rew)
        return (rew - self.lower_bound) / (self.upper_bound - self.lower_bound)

class MapActionWrapper(gym.ActionWrapper):
    def __init__(self, env, action_map):
        super().__init__(env)
        self.action_map = action_map

    def action(self, act):
        # modify act
        return self.action_map[act]

class VectorizedEnvWrapper(gym.Wrapper):
    def __init__(self, make_env, num_envs=1):
        super().__init__(make_env())
        self.num_envs = num_envs
        self.envs = [make_env() for env_index in range(num_envs)]

    def reset(self):
        return np.asarray([env.reset() for env in self.envs])

    def reset_at(self, env_index):
        return self.envs[env_index].reset()

    def step(self, actions):
        next_states, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            next_state, reward, done, info = env.step(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.asarray(next_states), np.asarray(rewards), \
            np.asarray(dones), np.asarray(infos)


class GymWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # modify obs
        return obs


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, rew):
        # modify rew
        return rew


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, act):
        # modify act
        return act
