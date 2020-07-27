from enum import Enum, auto

class Policy:
    class Init(Enum):
        ONES = auto()
        ZEROS = auto()
        RANDOM = auto() 

class QLearning(Enum):
    alpha = 0.1
    gamma = 0.5
    epsilon=0.01