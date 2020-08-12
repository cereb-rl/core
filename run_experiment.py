from core.experiments.Experiment import Experiment
from core.environments import getWrappedEnv
from core.agents import RandomAgent, QLearningAgent, RMaxAgent, MBIEAgent, UCBVIAgent, UBEVAgent
import gym
import gym_maze

envs = ["MountainCar-v0","maze-random-10x10-plus-v0",'CartPole-v0', 'Taxi-v3', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'GuessingGame-v0', 'Roulette-v0', 'HotterColder-v0', "maze-random-30x30-plus-v0", "maze-random-100x100-v0"]
env = getWrappedEnv(envs[0])
print(env.observation_space)
observation_space = env.observation_space
action_space = env.action_space
randomAgent = RandomAgent(observation_space, action_space)
qlearningAgent = QLearningAgent(observation_space, action_space)
rmaxAgent = RMaxAgent(observation_space, action_space)
mbieAgent = MBIEAgent(observation_space, action_space)
ubevAgent = UBEVAgent(observation_space, action_space)
# ucbviAgent = UCBVIAgent(env.observation_space, env.action_space)
agents = [randomAgent, qlearningAgent, rmaxAgent, mbieAgent, ubevAgent] #ucbviAgent
exp = Experiment(env=env, agents=agents, visuals=True)
exp.train(1000)
exp.train(100, True)
print("done")
