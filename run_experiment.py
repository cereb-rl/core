from core.experiments.Experiment import Experiment

class State:
    terminal = False

    def __init__(self, num):
        self.num = num

    def is_terminal(self):
        return self.terminal

exp = Experiment()   
trained_agent = exp.train()
exp.test(trained_agent)
print("done")