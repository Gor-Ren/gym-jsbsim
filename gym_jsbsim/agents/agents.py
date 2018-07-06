import random
import gym
from abc import ABC, abstractmethod
from typing import Optional


class Agent(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def observe(self, state, action, reward, done):
        pass


class RandomAgent(Agent):
    """ An agent that selects random actions.

    The Random object making selection is gym.np_random used by the
    Space.sample() method. Its seed is set by gym.
    """
    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def act(self, _):
        return self.action_space.sample()

    def observe(self, state, action, reward, done):
        # this agent type does not learn in response to observations
        pass


class ConstantAgent(Agent):
    def __init__(self, action_space: gym.spaces.Box):
        super().__init__()
        self.constant_action = (action_space.low + action_space.high) / 2

    def act(self, _):
        return self.constant_action

    def observe(self, state, action, reward, done):
        # this agent type does not learn in response to observations
        pass
