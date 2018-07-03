import random
from abc import ABC, abstractmethod
from typing import Optional
from gym import Space

class Agent(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, state, action):
        pass

    @abstractmethod
    def observe(self, state, action, reward, done):
        pass


class RandomAgent(Agent):
    def __init__(self, action_space: Space, seed: Optional[int]=None):
        super().__init__()
        self.random = random.Random(seed)
        self.action_space = action_space

    def select_action(self, state, action):
        return self.action_space.sample()

    def observe(self, state, action, reward, done):
        # a random agent does not learn in response to observations
        pass
