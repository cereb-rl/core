import gym
import numpy as np, math, sys

# Core imports
from .visdom import VisdomDisplay

class Experiment:
    """
    A simple experiment class
    """

    def __init__(self, exp_name="Experiment1", env_name="Taxi-v3", env=None, agents=None, verbose=False, visuals=False):
        """

        :type agents: List[BaseAgent]
        """
        self.exp_name = exp_name
        if env:
            self.env = env
        else:
            self.env = gym.make(env_name).env
        self.env_name = self.env.unwrapped.spec.id
        self.agents = agents if agents else []
        agent_names = [agent.get_name for agent in agents]
        agent_names_str: str = ', '.join(agent_names)
        print(f'Starting {self.exp_name} on {self.env_name} environment with {agent_names_str}')

        self.verbose = verbose
        self.visuals = visuals
        if self.visuals:
            print("has visuals")
            self.visdom = VisdomDisplay(exp_name=self.exp_name, env_name=self.env_name, agent_names=agent_names)

    def train(self, num_episodes: int, eval=False):
        """

        :param num_episodes:
        :return:
        """
        self.visdom.new_training()
        if not self.agents:
            print(f'{self.exp_name} has no agents to train. \n Please add agents to experiment.')
        total_steps = [0 for _ in self.agents]
        total_rewards = [0 for _ in self.agents]
        loop_count = 0
        for i in range(len(self.agents)):
            print(f"Starting training on {self.agents[i].get_name}")
        for ep in range(num_episodes):
            loop_rewards = [0 for _ in self.agents]
            loop_steps = [0 for _ in self.agents]
            for i in range(len(self.agents)):
                steps, rewards = self.run_single_episode(self.agents[i], not eval)
                rewards = rewards/steps
                loop_steps[i] = steps
                loop_rewards[i] = rewards

                total_rewards[i] += rewards
                total_steps[i] += steps
                #total_episodes[i] += 1

                if ep % (num_episodes/100) == 0:
                    print(f'Episode: {ep} for agent {self.agents[i].get_name} with total reward of {rewards*steps}', end="\r", flush=False)
                    # print('agent epsilon',self.agents[i].epsilon)
                loop_count += 1

            self.visdom.add_episode(rewards=loop_rewards, steps=loop_steps, count=loop_count+1)
        print()
        print("Training finished.\n")

        for i in range(len(self.agents)):
            print(f"Results after {num_episodes} episodes:")
            print(f"Average timesteps per episode: {total_steps[i] / num_episodes}")
            print(f"Average rewards per episode: {total_rewards[i] / num_episodes}")
            print()


    def run_single_episode(self, agent, is_train=False, render=False, max_steps=200):
        """

        :param is_train: Is this a training episode?
        :param agent: Agent to be trained
        :return: total number of steps and total rewards obtained
        """
        state = self.env.reset()
        action = agent.start_of_episode(state)
        steps, rewards = 0, 0
        done = False
        if render:
            self.env.render()

        while not done and steps <= max_steps:
            state, reward, done, info = self.env.step(action)
            if is_train:
                action = agent.learn(state, reward)
            else:
                action = agent.predict(state)

            steps += 1
            rewards += reward

        agent.end_of_episode()

        if self.verbose:
            print(f"Results after {steps} timesteps:")
            print(f"Average reward: {reward / steps}")

        return steps, rewards

    def add_agent(self, agent):
        """

        :param agent:
        """
        if self.agents:
            self.agents.append(agent)
        else:
            self.agents = [agent]
        print(f'Adding {agent} to {self}')

    def __str__(self):
        return self.exp_name
