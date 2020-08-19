from enum import Enum, auto

class SpaceTypes:
    ANY = 0
    DISCRETE = 1
    FLATBOX = 2
    BOX = 3
    OTHER = 4

class Policy:
    class Init(Enum):
        ONES = auto()
        ZEROS = auto()
        RANDOM = auto() 

class QLearning(Enum):
    alpha = 0.1
    gamma = 0.5
    epsilon=0.01