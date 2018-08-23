import gym_jsbsim.properties as prp
from abc import ABC, abstractmethod
from typing import Tuple, Union

State = 'tasks.FlightTask.State'  # alias for type hint


class Reward(object):
    """
    Immutable class storing an RL reward.

    We decompose rewards into tuples of component values, reflecting contributions
    from different goals. Separate tuples are maintained for the base (non-shaping)
    components and the shaping components to allow analysis.

    Scalar reward values are retrieved by calling .reward() or non_shaping_reward().
    The scalar value is the mean of the components.
    """

    def __init__(self, base_reward_elements: Tuple, shaping_reward_elements: Tuple):
        self.base_reward_elements = base_reward_elements
        self.shaping_reward_elements = shaping_reward_elements
        if not self.base_reward_elements:
            raise ValueError('base reward cannot be empty')

    def reward(self) -> float:
        """ Returns scalar reward value by taking mean of all reward elements """
        sum_reward = sum(self.base_reward_elements) + sum(self.shaping_reward_elements)
        num_reward_components = len(self.base_reward_elements) + len(self.shaping_reward_elements)
        return sum_reward / num_reward_components

    def non_shaping_reward(self) -> float:
        """ Returns scalar non-shaping reward by taking mean of base reward elements. """
        return sum(self.base_reward_elements) / len(self.base_reward_elements)

    def is_shaping(self):
        return bool(self.shaping_reward_elements)


class RewardComponent(ABC):
    """ Interface for RewardComponent, an object which calculates one component value of a Reward """

    @abstractmethod
    def calculate(self, state: State, last_state: State, is_terminal: bool) -> float:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...


class AbstractComponent(RewardComponent, ABC):
    def __init__(self, name: str, prop: prp.BoundedProperty,
                 state_variables: Tuple[prp.BoundedProperty]):
        """
        Constructor.

        :param name: the uniquely identifying name of this component
        :param prop: the BoundedProperty for which a value will be retrieved
            from the State
        :param state_variables: the state variables corresponding to each State element
        """
        self.name = name
        self.index = state_variables.index(prop)

    def get_name(self):
        return self.name


class TerminalComponent(AbstractComponent):
    """
    A sparse reward component which gives a reward on terminal condition.

    The reward is equal to the terminal value of some state property relative
    to a maximum value.
    """

    def __init__(self, name: str, prop: prp.BoundedProperty,
                 state_variables: Tuple[prp.BoundedProperty],
                 max_target: float):
        """
        Constructor.

        :param name: the uniquely identifying name of this component
        :param prop: the BoundedProperty for which a value will be retrieved
            from the State
        :param max_target: the maximum value the property can take, against
            which reward is calculated as a fraction
        """
        super().__init__(name, prop, state_variables)
        self.max_target = max_target

    def calculate(self, state: State, _: State, is_terminal: bool):
        if is_terminal:
            value = state[self.index]
            return value / self.max_target
        else:
            return 0.0


class ComplementComponent(AbstractComponent, ABC):
    """
    Calculates rewards based on a normalised error complement.

    Normalising an error takes some absolute difference |value - target| and
    transforms it to the interval [0,1], where 0 is no error and 1 is +inf error.

    We then take the complement of this (1-normalised_error) and use it
    as a reward component.
    """

    def __init__(self, name: str, prop: prp.BoundedProperty,
                 state_variables: Tuple[prp.BoundedProperty],
                 target: Union[int, float, prp.Property, prp.BoundedProperty],
                 scaling_factor: Union[float, int]):
        super().__init__(name, prop, state_variables)
        self._set_target(target, state_variables)
        self.scaling_factor = scaling_factor

    def _set_target(self, target: Union[int, float, prp.Property, prp.BoundedProperty],
                    state_variables: Tuple[prp.BoundedProperty]) -> None:
        """
        Sets the target value or an index for retrieving it from States

        Depending on how target is specified, it may either be a constant, or a
        Property's value that needs to be retrieved from the State.
        """
        if isinstance(target, float) or isinstance(target, int):
            self.constant_target = True
            self.target = target
        elif isinstance(target, prp.Property) or isinstance(target, prp.BoundedProperty):
            self.constant_target = False
            self.target_index = state_variables.index(target)

    def is_constant_target(self):
        return self.constant_target

    def error_complement(self, state: State) -> float:
        """
        Calculates the 'goodness' of a State given we want the compare_property
        to be some target_value. The target value may be a constant or
        retrieved from another property in the state.

        The 'goodness' of the state is given by 1 - normalised_error, i.e. the
        error's complement.
        """
        if self.is_constant_target():
            target = self.target
        else:
            # else we have to look it up from the state
            target = state[self.target_index]
        value = state[self.index]
        error = abs(value - target)
        normalised_error = self._normalise_error(error, self.scaling_factor)
        return 1 - normalised_error

    @staticmethod
    def _normalise_error(absolute_error: float, error_scaling: float):
        """
        Given an error in the interval [0, +inf], returns a normalised error in [0, 1]

        The normalised error asymptotically approaches 1 as absolute_error -> +inf.

        The parameter error_scaling is used to scale for magnitude.
        When absolute_error == error_scaling, the normalised error is equal to 0.5
        """
        if absolute_error < 0:
            raise ValueError(f'Error to be normalised must be non-negative '
                             f': {absolute_error}')
        scaled_error = absolute_error / error_scaling
        return scaled_error / (scaled_error + 1)


class StepFractionComponent(ComplementComponent):
    """
    Rewards based on a property's closeness to a target value each timestep.

    Reward is equal to error_complement / episode_timesteps, therefore a
    this component sums to 1.0 over a perfect episode.
    """

    def __init__(self, name: str, prop: prp.BoundedProperty,
                 state_variables: Tuple[prp.BoundedProperty],
                 target: Union[int, float, prp.Property, prp.BoundedProperty],
                 scaling_factor: Union[float, int],
                 episode_timesteps: int):
        """
        Constructor.

        :param name: the name of this component used for __repr__, e.g.
            'altitude_keeping'
        :param prop: the Property for which a value will be retrieved
            from the State
        :param target: the target value for the property, or the Property from
            which the target value will be retrieved
        :param scaling_factor: the property value is scaled down by this amount.
            The RewardComponent outputs 0.5 when the value equals this factor
        :param episode_timesteps: the number of timesteps in each episode
        """
        super().__init__(name, prop, state_variables, target, scaling_factor)
        self.episode_timesteps = episode_timesteps

    def calculate(self, state: State, _: State, is_terminal: bool):
        error_complement = self.error_complement(state)
        return error_complement / self.episode_timesteps


class ShapingComponent(ComplementComponent):
    """ A potential-based shaping reward component using error complements """

    def __init__(self, name: str, prop: prp.BoundedProperty,
                 state_variables: Tuple[prp.BoundedProperty],
                 target: Union[int, float, prp.Property, prp.BoundedProperty],
                 scaling_factor: Union[float, int]):
        """
        Constructor.

        :param name: the name of this component used for __repr__, e.g.
            'altitude_keeping'
        :param prop: the Property for which a value will be retrieved
            from the State
        :param target: the target value for the property, or the Property from
            which the target value will be retrieved
        :param scaling_factor: the property value is scaled down by this amount.
            The RewardComponent outputs 0.5 when the error equals this factor
        """
        super().__init__(name, prop, state_variables, target, scaling_factor)

    def get_potential(self, state: State, is_terminal):
        if is_terminal:
            return 0
        else:
            return self.error_complement(state)

    def calculate(self, state: State, last_state: State, is_terminal: bool):
        potential = self.get_potential(state, is_terminal)
        last_potential = self.get_potential(last_state, False)
        return potential - last_potential
