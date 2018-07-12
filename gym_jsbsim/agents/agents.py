import gym
import numpy as np
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


class ConstantChangeNothingAgent(Agent):

    def __init__(self, action_space: gym.spaces.Box, state_indices_for_actions):
        """
        An agent which tries not to change the position of a JSBSim aircraft
        control surfaces.

        It will read control surface positions from the state, and return actions
        with appropriate commands of the same value.

        :param action_space: a Box object, the action space of the environment
        :param state_indices_for_actions: sequence of ints, the same length as
           the number of action variables, actions will be selected by
           looking up the state value at this index
        """
        if len(state_indices_for_actions) != len(action_space.low):
            raise ValueError('every action variable must be provided with a '
                             'corresponding state variable to copy value from\n'
                             f'received {len(state_indices_for_actions)} indices, '
                             f'expected {len(action_space.low)}')
        super().__init__()
        self.state_indices_for_actions = state_indices_for_actions

    def act(self, state):
        action = np.array([state[i] for i in self.state_indices_for_actions])
        return action

    def observe(self, state, action, reward, done):
        # this agent type does not learn in response to observations
        pass