import gym
import numpy as np
from typing import List
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def act(self, state) -> np.ndarray:
        ...

    @abstractmethod
    def observe(self, state, action, reward, done):
        ...


class RandomAgent(Agent):
    """ An agent that selects random actions.

    The Random object making selection is gym.np_random used by the
    Space.sample() method. Its seed is set by gym.
    """
    def __init__(self, action_space: gym.Space):
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
    action_to_state_name_map = {
        'fcs/aileron-cmd-norm': 'fcs/left-aileron-pos-norm',
        'fcs/elevator-cmd-norm': 'fcs/elevator-pos-norm',
        'fcs/throttle-cmd-norm': 'fcs/throttle-pos-norm',
        'fcs/rudder-cmd-norm': 'fcs/rudder-pos-norm'
    }

    def __init__(self, action_space: gym.spaces.Box, action_names, state_names):
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
        self.state_indices_for_actions = self._get_state_indices_for_actions(action_names,
                                                                             state_names)
        # we should have an index for every action variable
        assert len(self.state_indices_for_actions) == len(action_space.low)

    def _get_state_indices_for_actions(self, action_names: List[str], state_names: List[str]):
        """ Given a list of action properties, finds which indices in state arrays
        that they correspond to.

        For example, the action 'fcs/rudder-cmd-norm' would look for
        'fcs/rudder-pos-norm' in the state names.

        :param action_names: list of str, the JSBSim property names for all actions
        :param state_names: list of str, the JSBSim property names for all states
        :return: sequence of ints, the same length as the number of action variables,
            actions can be selected by looking up the state value at this index
        """
        result = []
        for action_name in action_names:
            search_state_name = self.action_to_state_name_map[action_name]
            index = state_names.index(search_state_name)
            result.append(index)
        return result

    def act(self, state):
        action = np.array([state[i] for i in self.state_indices_for_actions])
        return action

    def observe(self, state, action, reward, done):
        # this agent type does not learn in response to observations
        pass