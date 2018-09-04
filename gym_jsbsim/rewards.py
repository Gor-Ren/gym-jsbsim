import gym_jsbsim.properties as prp
from abc import ABC, abstractmethod
from typing import Tuple, Union
from gym_jsbsim.utils import reduce_reflex_angle_deg

State = 'tasks.FlightTask.State'  # alias for type hint


class Reward(object):
    """
    Immutable class storing an RL reward.

    We decompose rewards into tuples of component values, reflecting contributions
    from different goals. Separate tuples are maintained for the assessment (non-shaping)
    components and the shaping components. It is intended that the

    Scalar reward values are retrieved by calling .reward() or non_shaping_reward().
    The scalar value is the mean of the components.
    """

    def __init__(self, base_reward_elements: Tuple, shaping_reward_elements: Tuple):
        self.base_reward_elements = base_reward_elements
        self.shaping_reward_elements = shaping_reward_elements
        if not self.base_reward_elements:
            raise ValueError('base agent_reward cannot be empty')

    def agent_reward(self) -> float:
        """ Returns scalar reward value by taking mean of all reward elements """
        sum_reward = sum(self.base_reward_elements) + sum(self.shaping_reward_elements)
        num_reward_components = len(self.base_reward_elements) + len(self.shaping_reward_elements)
        return sum_reward / num_reward_components

    def assessment_reward(self) -> float:
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

    @abstractmethod
    def get_potential(self, state: State, is_terminal) -> float:
        ...

    @abstractmethod
    def is_potential_difference_based(self) -> bool:
        ...


class NormalisedComponent(RewardComponent, ABC):
    """
    Base implementation of a RewardComponent implementing common methods.

    All potentials of subclasses should be normalised in [0.0, 1.0]
    """
    POTENTIAL_BASED_DIFFERENCE_TERMINAL_VALUE = 0.0

    def __init__(self,
                 name: str,
                 prop: prp.BoundedProperty,
                 state_variables: Tuple[prp.BoundedProperty],
                 target: Union[int, float, prp.Property, prp.BoundedProperty],
                 potential_difference_based: bool):
        """
        Constructor.

        :param name: the uniquely identifying name of this component
        :param prop: the BoundedProperty for which a value will be retrieved
            from the State
        :param state_variables: the state variables corresponding to each State element
            that this component will be passed.
        :param is_potential_based: True if reward is based on a potential difference
            between prev_state and state (AKA potential based shaping reward) else
            False (and reward depends only on the potential of current state).
        """
        self.name = name
        self.state_index_of_value = state_variables.index(prop)
        self.potential_difference_based = potential_difference_based
        self._set_target(target, state_variables)

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

    def calculate(self, state: State, prev_state: State, is_terminal: bool):
        """
        Calculates the value of this RewardComponent.

        If this component is potential difference based, its value is the
        difference in potentials between prev_state and state. Otherwise its
        value is the potential of state.
        """
        if self.potential_difference_based:
            # reward is a potential difference of state, prev_state
            reward = self.get_potential(state, is_terminal) - self.get_potential(prev_state, False)
        else:
            reward = self.get_potential(state, is_terminal)
        return reward

    def is_constant_target(self):
        return self.constant_target

    def get_name(self) -> str:
        return self.name

    def is_potential_difference_based(self) -> bool:
        return self.potential_difference_based


class ErrorComponent(NormalisedComponent, ABC):
    """
    Calculates rewards based on a normalised error complement from a target value.

    Normalising an error takes some absolute difference |value - target| and
    transforms it to the interval [0,1], where 1.0 is no error and 0.0 is inf error.
    """

    def get_potential(self, state: State, is_terminal) -> float:
        """
        Calculates the 'goodness' of a State given we want the compare_property
        to be some target_value. The target value may be a constant or
        retrieved from another property in the state.

        The 'goodness' of the state is given in the interval [-1,0], where 0
        corresponds to zero error, and -1 corresponds to inf error.
        """
        if is_terminal and self.potential_difference_based:
            return self.POTENTIAL_BASED_DIFFERENCE_TERMINAL_VALUE

        if self.is_constant_target():
            target = self.target
        else:
            # else we have to look it up from the state
            target = state[self.target_index]
        value = state[self.state_index_of_value]
        error = abs(value - target)
        return 1 - self._normalise_error(error)

    @abstractmethod
    def _normalise_error(self, absolute_error: float) -> float:
        """
        Given an error in the interval [0, +inf], returns a normalised error in [0, 1]

        The normalised error asymptotically approaches 1 as absolute_error -> +inf.

        The parameter error_scaling is used to scale for magnitude.
        When absolute_error == error_scaling, the normalised error is equal to 0.5
        """
        ...


class AsymptoticErrorComponent(ErrorComponent):
    """
    A reward component which gives a negative reward that asymptotically approaches -1
    as the error to the desired value approaches +inf. This is convenient for not having
    to worry about the bounds on the absolute error value.
    """

    def __init__(self,
                 name: str,
                 prop: prp.BoundedProperty,
                 state_variables: Tuple[prp.BoundedProperty],
                 target: Union[int, float, prp.Property, prp.BoundedProperty],
                 is_potential_based: bool,
                 scaling_factor: Union[float, int]):
        """
        Constructor.

        :param scaling_factor: the property value is scaled down by this amount.
            Shaping potential is at 0.5 when the error equals this factor.
        """
        super().__init__(name, prop, state_variables, target, is_potential_based)
        self.scaling_factor = scaling_factor

    def _normalise_error(self, absolute_error: float):
        return normalise_error_asymptotic(absolute_error, self.scaling_factor)


class AngularAsymptoticErrorComponent(AsymptoticErrorComponent):
    """
    A potential-based shaping reward component.

    Potential is based asymptotically on the  size of the error between a
    property of interest and its target. The error can be unbounded in
    magnitude.

    Values must be in units of degrees. Errors are reduced to the interval
    (-180, 180] before processing.
    """

    def _normalise_error(self, angular_error: float):
        """
        Given an angle off of a target direction in degrees, calculates a
        normalised error in [0,1]. The angular error is firstly transformed
        to interval [-180,180] to account for the fact the agent can turn
        left or right to face the target.

        :param angular_error: float, angle off target in degrees
        :return: float, normalised error in [0,1]
        """
        reduced_angle_error = abs(reduce_reflex_angle_deg(angular_error))
        return super()._normalise_error(reduced_angle_error)


class LinearErrorComponent(ErrorComponent):
    """
    A potential-based shaping reward component.

    Potential is based linearly on the size of the error between a property of
    interest and its target. The error must be in the interval [0, scaling_factor].
    """

    def __init__(self,
                 name: str,
                 prop: prp.BoundedProperty,
                 state_variables: Tuple[prp.BoundedProperty],
                 target: Union[int, float, prp.Property, prp.BoundedProperty],
                 is_potential_based: bool,
                 scaling_factor: Union[float, int]):
        """
        Constructor.

        :param scaling_factor: the max size of the difference between prop and
            target. Minimum potential (0.0) occurs when error is
            max_error_size or greater.
        """
        super().__init__(name, prop, state_variables, target, is_potential_based)
        self.scaling_factor = scaling_factor

    def _normalise_error(self, absolute_error: float):
        return normalise_error_linear(absolute_error, self.scaling_factor)


def normalise_error_asymptotic(absolute_error: float, scaling_factor: float) -> float:
    """
    Given an error in the interval [0, +inf], returns a normalised error in [0, 1]

    The normalised error asymptotically approaches 1 as absolute_error -> +inf.

    The parameter scaling_factor is used to scale for magnitude.
    When absolute_error == scaling_factor, the normalised error is equal to 0.5
    """
    if absolute_error < 0:
        raise ValueError(f'Error to be normalised must be non-negative '
                         f': {absolute_error}')
    scaled_error = absolute_error / scaling_factor
    return scaled_error / (scaled_error + 1)


def normalise_error_linear(absolute_error: float, max_error: float) -> float:
    """
    Given an absolute error in [0, max_error], linearly normalises error in [0, 1]

    If absolute_error exceeds max_error, it is capped back to max_error
    """
    if absolute_error < 0:
        raise ValueError(f'Error to be normalised must be non-negative '
                         f': {absolute_error}')
    elif absolute_error > max_error:
        return 1.0
    else:
        return absolute_error / max_error


class RewardStub(Reward):
    def __init__(self, agent_reward_value: float, assessment_reward_value: float):
        assert isinstance(agent_reward_value, float)
        assert isinstance(assessment_reward_value, float)
        self.agent_reward_value = agent_reward_value
        self.assessment_reward_value = assessment_reward_value

    def agent_reward(self) -> float:
        return self.agent_reward_value

    def assessment_reward(self) -> float:
        return self.assessment_reward_value

    def is_shaping(self):
        return True