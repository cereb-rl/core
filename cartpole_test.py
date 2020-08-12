'''
Inspired from
https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947

In this example, the state space is continuous.
To use Q-Learning for the cartpole environment, it is necessary
to discretize the continuous state space to a number of buckets.
'''

import gym
import numpy as np
import math
import matplotlib.pyplot as plt

# CREATE ENVIRONMENT
env = gym.make('CartPole-v0').env
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
print("Action space size: ", n_actions)
print("State space size: ", n_states)

print('states high value:')
print(env.observation_space.high[0])
print(env.observation_space.high[1])
print(env.observation_space.high[2])
print(env.observation_space.high[3])

print('states low value:')
print(env.observation_space.low[0])
print(env.observation_space.low[1])
print(env.observation_space.low[2])
print(env.observation_space.low[3])


# DOWN-SCALE THE FEATURE SPACE TO DISCRETE RANGE
buckets = (1, 1, 6, 12)     # define the number of buckets for each state value (x, x', theta, theta')

# define upper and lower bounds for each state value
# note: setting the bucket to 1 for the first 2 numbers is equivalent to ignoring these parameters
upper_bounds = [
        env.observation_space.high[0],
        0.5,
        env.observation_space.high[2],
        math.radians(50)
        ]
lower_bounds = [
        env.observation_space.low[0],
        -0.5,
        env.observation_space.low[2],
        -math.radians(50)]


# HYPERPARAMETERS
n_episodes = 10000           # Total train episodes
n_steps = 10000               # Max steps per episode
min_alpha = 0.1             # learning rate
min_epsilon = 0.1           # exploration rate
gamma = 1                   # discount factor
ada_divisor = 25            # decay rate parameter for alpha and epsilon

# INITIALISE Q MATRIX
Q = np.zeros(shape=(2304,2))
print(np.shape(Q))

def discretize(obs):
    ''' discretise the continuous state into buckets '''
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def disc(obs, env):
    buckets=np.array([4, 4, 12, 12])
    l_bound=np.array([env.observation_space.low[0], -0.5, env.observation_space.low[2],-math.radians(50)])
    u_bound=np.array([env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)])
    ''' discretise the continuous state into buckets '''
    ratios = (obs - l_bound) / (u_bound - l_bound)
    new_obs = np.array(np.ceil(np.multiply(buckets, ratios)), dtype=np.int32)
    new_obs = np.minimum(np.maximum(new_obs, np.ones(shape=buckets.shape, dtype=np.int32)), buckets)
    # Mapping obs list to integer based on buckets
    # print(new_obs)
    new_obs_int, c_prod = 0, 1
    for i in range(buckets.shape[0]):
        new_obs_int += c_prod * (new_obs[i] - 1)
        c_prod *= buckets[i]
    #print(new_obs_int)
    return new_obs_int

def epsilon_policy(state, epsilon):
    ''' choose an action using the epsilon policy '''
    exploration_exploitation_tradeoff = np.random.random()
    if exploration_exploitation_tradeoff <= epsilon:
        action = env.action_space.sample()  # exploration
    else:
        action = np.argmax(Q[state])   # exploitation
    return action

def greedy_policy(state):
    ''' choose an action using the greedy policy '''
    return np.argmax(Q[state])

def update_q(current_state, action, reward, new_state, alpha):
    #print(current_state)
    ''' update the Q matrix with the Bellman equation '''
    Q[current_state][action] = (1-alpha)* Q[current_state][action] + alpha * (reward + gamma * np.max(Q[new_state]))

def get_epsilon(t):
    ''' decrease the exploration rate at each episode '''
    return max(min_epsilon, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))

def get_alpha(t):
    ''' decrease the learning rate at each episode '''
    return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))


# TRAINING PHASE
rewards = []

for episode in range(n_episodes):
    current_state = env.reset()
    current_state = disc(current_state, env)

    alpha = get_alpha(episode)
    epsilon = get_epsilon(episode)

    episode_rewards = 0

    for t in range(n_steps):
        # env.render()
        action = epsilon_policy(current_state, epsilon)
        new_state, reward, done, _ = env.step(action)
        new_state = disc(new_state, env)
        update_q(current_state, action, reward, new_state, alpha)
        current_state = new_state

        # increment the cumulative reward
        episode_rewards += reward

        # at the end of the episode
        if done:
            print('Episode:{}/{} finished with a total reward of: {}'.format(episode, n_episodes, episode_rewards))
            break
    #print(Q)
    # append the episode cumulative reward to the reward list
    rewards.append(episode_rewards)


# PLOT RESULTS
x = range(n_episodes)
plt.plot(x, rewards)
plt.xlabel('episode')
plt.ylabel('Training cumulative reward')
plt.savefig('Q_learning_CART.png', dpi=300)
plt.show()

# TEST PHASE
for episode in range(30):
    current_state = env.reset()
    current_state = disc(current_state, env)
    episode_rewards = 0


    for t in range(n_steps):
        env.render()
        action = greedy_policy(current_state)
        new_state, reward, done, _ = env.step(action)
        new_state = disc(new_state, env)
        update_q(current_state, action, reward, new_state, alpha)
        current_state = new_state
        episode_rewards += reward

        # at the end of the episode
        if done:
            print('Test episode finished with a total reward of: {}'.format(episode_rewards))
            break

env.close()
