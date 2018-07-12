import gym
import numpy as np
from gym_jsbsim.simulation import Simulation
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def act(self, state) -> np.ndarray:
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
        self.constant_action = (action_space.low + action_space.high) / 2

    def act(self, _):
        return self.constant_action

    def observe(self, state, action, reward, done):
        # this agent type does not learn in response to observations
        pass


class RepeatAgent(Agent):

    def __init__(self, action_space: gym.spaces.Box, action_names, sim: Simulation):
        """
        An agent which tries not to change the position of a JSBSim aircraft
        control surfaces.

        This is a bit of a hack of an agent intended for debugging. It requires
        access to the Simulation object to extract previous commands and repeat
        them.

        :param action_space: a Box object, the action space of the environment
        :param action_names: list of strs, the JSBSim property name of each
            action variable in order of their position in action arrays
        :param sim: the Simulation being used
        """
        if len(action_space.low) != len(action_names):
            raise ValueError('action_space and action_names should be same size')
        super().__init__()
        self.action_names = action_names
        self.sim = sim

    def act(self, state):
        action = np.array([self.sim[name] for name in self.action_names])
        return action

    def observe(self, state, action, reward, done):
        # this agent type does not learn in response to observations
        pass