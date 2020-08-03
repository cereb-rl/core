from core.experiments.Experiment import Experiment
from core.agents import RandomAgent, QLearningAgent
import gym

env = gym.make("Taxi-v3").env
randomAgent = RandomAgent(env.observation_space, env.action_space)
qlearningAgent = QLearningAgent(env.observation_space, env.action_space)
#rmaxAgent = RMaxAgent(env.observation_space, env.action_space)
exp = Experiment(env_name="Taxi-v3", agents=[randomAgent, qlearningAgent], visuals=True)
exp.train(1000)
print("done")
