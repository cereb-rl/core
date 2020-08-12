from IPython.display import clear_output
from time import sleep
import gym
import visdom
import numpy as np, math, sys

def disc(obs, env):
    return obs
    buckets=np.array([10,10])
    l_bound=np.array([0,0])
    u_bound=np.array([9,9])
    ''' discretise the continuous state into buckets '''
    ratios = (obs + abs(l_bound)) / (u_bound - l_bound)
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
        print(f'Starting {self.exp_name} on {self.env.unwrapped.spec.id} environment with {agent_names}')

        self.verbose = verbose
        self.visuals = visuals
        if self.visuals:
            print("has visuals")
            self.visdom = visdom.Visdom(env=self.exp_name)

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
    def train(self, num_episodes: int, eval=False):
        """

        :param num_episodes:
        :return:
        """
        if not self.agents:
            print(f'{self.exp_name} has no agents to train. \n Please add agents to experiment.')
        self.rewards_visdom = self.visdom.line(
                Y=[[0 for _ in self.agents]],
                X=[[0 for _ in self.agents]],
                opts=dict(
                    width=800,
                    height=800,
                    xlabel='Episodes',
                    ylabel='Episode Rewards',
                    title=f'Episodic Reward Plot on {self.env.unwrapped.spec.id}',
                    legend=[agent.get_name for agent in self.agents]
                ),
            )
        self.steps_visdom = self.visdom.line(
            Y=[[0 for _ in self.agents]],
            X=[[0 for _ in self.agents]],
            opts=dict(
                width=800,
                height=800,
                xlabel='Episodes',
                ylabel=f'Number of Steps',
                title=f'Episodic Step Plot on {self.env.unwrapped.spec.id}',
                legend=[agent.get_name for agent in self.agents]
            ),
        )
        # self.render_visdom = self.visdom.text(u'''<h1>Hello Visdom</h1><br>Visdom is a visual tool developed by Facebook specifically for <b>PyTorch</b>.
        #            It has been used internally for a long time and was opened in March 2017.
        #            Visdom is very lightweight, but it has very powerful features that support almost all scientific computing visualization tasks ''',
        #  win='visdom',
        #  opts={'title': u'Introduction to visdom'}
        #  )
        total_steps = [0 for _ in self.agents]
        total_rewards = [0 for _ in self.agents]
        loop_count = 0
        for i in range(len(self.agents)):
            print(f"Starting training on {self.agents[i].get_name}")
        for ep in range(num_episodes):
            loop_rewards = [0 for _ in self.agents]
            loop_steps = [0 for _ in self.agents]
            for i in range(len(self.agents)):
                # if total_steps[i] >= num_steps:
                #     continue
                steps, rewards = self.run_single_episode(self.agents[i], not eval)
                rewards = rewards/steps
                loop_steps[i] = steps
                loop_rewards[i] = rewards

                total_rewards[i] += rewards
                total_steps[i] += steps
                #total_episodes[i] += 1

                if ep % (num_episodes/50) == 0:
                    clear_output(wait=True)
                    print(f'Episode: {ep} for agent {self.agents[i].get_name}')
                    # print('agent epsilon',self.agents[i].epsilon)
                loop_count += 1
            self.visdom.line(
                Y=[loop_rewards],
                X=[[loop_count+1 for _ in self.agents]],
                win=self.rewards_visdom,
                update='append'
            )
            self.visdom.line(
                Y=[loop_steps],
                X=[[loop_count+1 for _ in self.agents]],
                win=self.steps_visdom,
                update='append'
            )

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
        action = agent.start_of_episode(disc(state, self.env))
        steps, rewards = 0, 0
        done = False
        frames = []  # for animation
        if render:
                self.env.render()

        while not done and steps <= max_steps:
            state, reward, done, info = self.env.step(action)
            if is_train:
                action = agent.learn(disc(state, self.env), reward)
            else:
                action = agent.predict(disc(state, self.env))
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
