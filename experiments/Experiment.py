
import random
from IPython.display import clear_output
from time import sleep
import gym
import numpy as np
import random
from core.agents.QLearningAgent import QLearningAgent
from core.agents.RMaxAgent import RMaxAgent
from core.agents.ECubedAgent import ECubedAgent

class Experiment:
    def __init__(self):
        pass

    def train(self):
        total_steps, total_penalties = 0, 0
        episodes = 50

        env = gym.make("Taxi-v3").env
        agent = MDIEAgent(list(range(env.action_space.n)))
        print("here")
        for i in range(episodes):
            #print(i)
            steps, penalties = self.run_single_epsiode(env, agent)

            total_penalties += penalties
            total_steps += steps

            if i % 100 == 0:
                clear_output(wait=True)
            print(f"Episode: {i}")

        print("Training finished.\n")

        print(f"Results after {episodes} episodes:")
        print(f"Average timesteps per episode: {total_steps / episodes}")
        print(f"Average penalties per episode: {total_penalties / episodes}")

        return agent

    def run_single_epsiode(self, env, agent, verbose=False):
        state = env.reset()
        steps, penalties, reward = 0, 0, 0
        done = False
        frames = [] # for animation
        #action = 0
        #states = [State(i) for i in range(500)]
        prev_state = state
        
        while not done:
            action = agent.select_action(state)
            prev_state = state
            state, reward, done, info = env.step(action)
            # agent
            # if done:
            #     states[state].terminal = True
            agent.update(prev_state, action, reward, state)
            #print(prev_state, action, state, reward, done, info)
            #print(agent)
            if verbose:
                # Put each rendered frame into dict for animation
                frames.append({
                    'frame': env.render(mode='ansi'),
                    'state': state,
                    'action': action,
                    'reward': reward
                    }
                )

            if reward == -10:
                penalties += 1

            steps += 1
        #print(agent)
        
        if verbose:
            print(f"Results after {steps} timesteps:")
            print(f"Average penalties: {penalties / steps}")
            #print_frames(frames)

        return steps, penalties

    def test(self, agent):
        # Hyperparameters
        alpha = 0.1
        gamma = 0.6
        epsilon = 0.1

        total_steps, total_penalties = 0, 0
        episodes = 100

        env = gym.make("Taxi-v3").env

        self.run_single_epsiode(env, agent, True)


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
