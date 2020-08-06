from core.experiments.Experiment import Experiment
from core.environments import GymEnvWrapper
from core.agents import RandomAgent, QLearningAgent, RMaxAgent, MBIEAgent
import gym

#env = ['Taxi-v3', 'FrozenLake-v0', 'FrozenLake8x8-v0']
env = GymEnvWrapper("Taxi-v3")
randomAgent = RandomAgent(env.observation_space, env.action_space)
qlearningAgent = QLearningAgent(env.observation_space, env.action_space)
rmaxAgent = RMaxAgent(env.observation_space, env.action_space)
#mbieAgent = MBIEAgent(env.observation_space, env.action_space)
exp = Experiment(env=env, agents=[randomAgent, qlearningAgent, rmaxAgent], visuals=True)
exp.train(1000)
exp.train(100, True)
print("done")
