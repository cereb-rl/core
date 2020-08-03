from IPython.display import clear_output
from time import sleep
import gym
import visdom
import numpy as np


class Experiment:
    """
    A simple experiment class
    """

    def __init__(self, exp_name="Experiment1", env_name="Taxi-v3", env=None, agents=None, verbose=False, visuals=False):
        """

        :type agents: List[BaseAgent]
        """
        self.env_name = env_name
        if env:
            self.env = env
        else:
            self.env = gym.make(env_name).env
        self.agents = agents if agents else []
        self.exp_name = exp_name
        agent_names: str = ', '.join(agent.get_name for agent in agents)
        print(f'Starting {self.exp_name} on {self.env_name} environment with {agent_names}')

        self.verbose = verbose
        self.visuals = visuals
        if self.visuals:
            print("has visuals")
            self.visdom = visdom.Visdom(env=exp_name)
            self.agents_visdom = self.visdom.line(
                Y=[[0 for _ in self.agents]],
                X=[[0 for _ in self.agents]],
                opts=dict(
                    width=800,
                    height=800,
                    xlabel='Steps',
                    ylabel='Episode Rewards',
                    title='Episodic Reward Plot',
                    legend=[agent.get_name for agent in self.agents]
                ),
            )

    # def train(self, num_steps: int):
    #     """
    #
    #     :param num_steps:
    #     :return:
    #     """
    #     if not self.agents:
    #         print(f'{self.exp_name} has no agents to train. \n Please add agents to experiment.')
    #     for agent in self.agents:
    #         print(f"Starting training on {agent.get_name}")
    #         self.train_single_agent(agent, num_steps)
    def train(self, num_episodes: int):
        """

        :param num_episodes:
        :return:
        """
        if not self.agents:
            print(f'{self.exp_name} has no agents to train. \n Please add agents to experiment.')
        total_steps = [0 for _ in self.agents]
        total_rewards = [0 for _ in self.agents]
        loop_count = 0
        for i in range(len(self.agents)):
            print(f"Starting training on {self.agents[i].get_name}")
        for ep in range(num_episodes):
            loop_rewards = [0 for _ in self.agents]
            for i in range(len(self.agents)):
                # if total_steps[i] >= num_steps:
                #     continue
                steps, rewards = self.run_single_episode(self.agents[i], True)
                loop_rewards[i] += rewards/steps

                total_rewards[i] += rewards
                total_steps[i] += steps
                #total_episodes[i] += 1

                if ep % 100 == 0:
                    clear_output(wait=True)
                    print(f'Episode: {ep} for agent {self.agents[i].get_name}')
                loop_count += 1
            self.visdom.line(
                Y=[loop_rewards],
                X=[[loop_count+1 for _ in self.agents]],
                win=self.agents_visdom,
                update='append'
            )

        print("Training finished.\n")

        for i in range(len(self.agents)):
            print(f"Results after {ep} episodes:")
            print(f"Average timesteps per episode: {total_steps[i] / ep}")
            print(f"Average rewards per episode: {total_rewards[i] / ep}")
            print()


    def train_single_agent(self, agent, num_steps: int):
        """

        :param agent:
        :param num_steps:
        """
        total_steps, num_episodes, total_reward = 0, 0, 0
        while total_steps < num_steps:
            steps, rewards = self.run_single_episode(agent, True)

            total_reward += rewards
            total_steps += steps
            num_episodes += 1

            if num_episodes % 100 == 0:
                clear_output(wait=True)
            print(f'Episode: {num_episodes}')

        print("Training finished.\n")

        print(f"Results after {num_episodes} episodes:")
        print(f"Average timesteps per episode: {total_steps / num_episodes}")
        print(f"Average rewards per episode: {total_reward / num_episodes}")
        print()
        return agent

    def run_single_episode(self, agent, is_train=False):
        """

        :param is_train: Is this a training episode?
        :param agent: Agent to be trained
        :return: total number of steps and total rewards obtained
        """
        state = self.env.reset()
        action = agent.start_of_episode(state)
        steps, rewards = 0, 0
        done = False
        frames = []  # for animation

        while not done:
            state, reward, done, info = self.env.step(action)
            if is_train:
                action = agent.learn(state, reward)
            else:
                action = agent.predict(state)
            if self.verbose:
                # Put each rendered frame into dict for animation
                frames.append({
                    'frame': self.env.render(mode='ansi'),
                    'state': state,
                    'action': action,
                    'reward': reward
                }
                )

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

    def print_frames(frames):
        for i, frame in enumerate(frames):
            clear_output(wait=True)
            print(frame)
            print(frame['frame'])
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(.1)

    def __str__(self):
        return self.exp_name
