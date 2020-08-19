import visdom

class VisdomDisplay:
    '''
    A class for managing experiments displayed on visdom
    '''

    def __init__(self, exp_name='Sample Experiment', env_name='Sample Environment', agent_names=['Random Agent']):
        self.exp_name = exp_name
        self.env_name = env_name
        self.agent_names = agent_names
        self.visdom = visdom.Visdom(env=self.exp_name)
        self.rewards = None
        self.steps = None

    def add_episode(self, rewards, steps, count):
        self.visdom.line(
                Y=[rewards],
                X=[[count for _ in self.agent_names]],
                win=self.rewards,
                update='append'
            )
        self.visdom.line(
            Y=[steps],
            X=[[count for _ in self.agent_names]],
            win=self.steps,
            update='append'
        )

    def add_step(self):
        pass
    
    def new_training(self):
        self.rewards = self.visdom.line(
                Y=[[0 for _ in self.agent_names]],
                X=[[0 for _ in self.agent_names]],
                opts=dict(
                    width=800,
                    height=800,
                    xlabel='Episodes',
                    ylabel='Episode Rewards',
                    title=f'Episodic Reward Plot on {self.env_name}',
                    legend=self.agent_names
                ),
            )
        self.steps = self.visdom.line(
            Y=[[0 for _ in self.agent_names]],
            X=[[0 for _ in self.agent_names]],
            opts=dict(
                width=800,
                height=800,
                xlabel='Episodes',
                ylabel=f'Number of Steps',
                title=f'Episodic Step Plot on {self.env_name}',
                legend=self.agent_names
            ),
        )