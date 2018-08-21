import gym_jsbsim.properties as prp
from collections import namedtuple
from abc import ABC, abstractmethod
from gym_jsbsim.tasks import FlightTask
from typing import Tuple, Union, Iterable, Dict


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
        assert bool(self.base_reward_elements)  # don't allow empty

    def reward(self) -> float:
        """ Returns scalar reward value by taking mean of all reward elements """
        sum_reward = sum(self.base_reward_elements) + sum(self.shaping_reward_elements)
        num_reward_components = len(self.base_reward_elements) + len(self.base_reward_elements)
        return sum_reward / num_reward_components

    def non_shaping_reward(self) -> float:
        """ Returns scalar non-shaping reward by taking mean of base reward elements. """
        return sum(self.base_reward_elements) / len(self.base_reward_elements)

    def is_shaping(self):
        return bool(self.shaping_reward_elements)


class Assessor(ABC):
    """ Interface for Assessors which calculate Rewards from States. """

    @abstractmethod
    def assess(self, state: FlightTask.State, last_state: FlightTask.State,
               is_terminal: bool) -> Reward:
        """ Calculates reward from environment's state, previous state and terminal condition """
        ...


class AssessorImpl(Assessor):
    """
    Determines the Reward from a state transitions.

    Initialised with RewardComponents and ShapingComponents which allow
    calculation of the base (non-shaping) and shaping rewards respectively.
    """

    def __init__(self, base_components: Iterable['RewardComponent'],
                 shaping_components: Iterable['ShapingComponent'] = ()):
        """
        :param base_components: RewardComponents from which Reward is to be calculated
        :param shaping_components: ShapingComponents from which the shaping
            reward is to be calculated, or an empty tuple for no shaping
        """
        self.base_components = base_components
        self.shaping_components = shaping_components

    def assess(self, state: FlightTask.State, last_state: FlightTask.State,
               is_terminal: bool) -> Reward:
        return Reward(self._base_rewards(state, last_state, is_terminal),
                      self._shaping_rewards(state, last_state, is_terminal))

    def _base_rewards(self, state: FlightTask.State, last_state: FlightTask.State,
                      is_terminal: bool) -> Tuple[float, ...]:
        return tuple(cmp.calculate(state, last_state, is_terminal) for cmp in self.base_components)

    def _shaping_rewards(self, state: FlightTask.State, last_state: FlightTask.State,
                         is_terminal: bool) -> Tuple[float, ...]:
        return tuple(cmp.calculate(state, last_state, is_terminal) for cmp in self.shaping_components)


class SequentialAssessor(AssessorImpl, ABC):
    """
    Abstract class that calculates a shaping Reward from state transitions,
    where the shaping component values depend on previous component's values.

    Concrete subclasses should implement _apply_dependents(), which modifies
    the 'normal' component potentials to account for dependents
    """

    def __init__(self, base_components: Iterable['RewardComponent'],
                 shaping_components: Iterable['ShapingComponent'] = (),
                 shaping_dependency_map: Dict['ShapingComponent', 'ShapingComponent'] = {}):
        """
        :param base_components: RewardComponents from which the non-shaping
            part of the Reward is to be calculated
        :param shaping_components: ShapingComponents from which the shaping
            reward is to be calculated, or an empty tuple for no shaping
        :param shaping_dependency_map: maps components with sequential
            dependencies to their dependent components, defaults to
            no dependencies
        """
        super().__init__(base_components, shaping_components)
        self.shaping_dependency_map = shaping_dependency_map.copy()
        self.Potential = namedtuple('Potential', [cmp.name for cmp in self.shaping_components])

    def _shaping_rewards(self, state: FlightTask.State, last_state: FlightTask.State,
                         is_terminal: bool) -> Tuple[float, ...]:
        potentials = self.Potential(cmp.get_potential(state, is_terminal) for cmp in self.shaping_components)
        last_potentials = self.Potential(cmp.get_potential(last_state, False) for cmp in self.shaping_components)

        seq_potentials = self._apply_dependents(potentials)
        seq_last_potentials = self._apply_dependents(last_potentials)
        return tuple(pot - last_pot for pot, last_pot in zip(seq_potentials, seq_last_potentials))

    @abstractmethod
    def _apply_dependents(self, potentials: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Modifies potentials to account for dependent components.

        :param potentials: the normal component potential values
        :return: a collection of component potentials, transformed to account
            for sequential dependencies
        """
        ...


class ContinuousSequentialAssessor(SequentialAssessor):

    def _apply_dependents(self, potentials: Tuple[float, ...]):
        ...


class DiscontinuousSequentialAssessor(SequentialAssessor):

    def _apply_dependents(self, potentials: Tuple[float, ...]):
        ...


class RewardComponent(ABC):
    """ Interface for RewardComponent, an object which calculates one component value of a Reward """

    @abstractmethod
    def calculate(self, state: FlightTask.State, last_state: FlightTask.State, is_terminal: bool) -> float:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...


class TerminalComponent(RewardComponent):
    """ A sparse reward component which gives a reward on terminal condition. """

    def __init__(self, name: str, prop: prp.BoundedProperty, max_target: float):
        """
        Constructor.

        :param name: the uniquely identifying name of this component
        :param prop: the BoundedProperty for which a value will be retrieved
            from the State
        :param max_target: the maximum value the property can take, against
            which reward is calculated as a fraction
        """
        self.name = name
        self.prop = prop
        self.max_target = max_target

    def calculate(self, state: FlightTask.State, _: FlightTask.State, is_terminal: bool):
        if is_terminal:
            value = state.__getattribute__(self.prop.name)
            return value / self.max_target
        else:
            return 0.0

    def get_name(self):
        return self.name


class ComplementComponent(RewardComponent, ABC):
    """
    Calculates rewards based on a normalised error complement.

    Normalising an error takes some absolute difference |value - target| and
    transforms it to the interval [0,1], where 0 is no error and 1 is +inf error.

    We then take the complement of this (1-normalised_error) and use it
    as a reward component.
    """

    def error_complement(self, state: FlightTask.State, compare_property: prp.Property,
                         target: Union[prp.Property, float, int], error_scaling: float) -> float:
        """
        Calculates the 'goodness' of a State given we want the compare_property
        to be some target_value. The target value may be a constant or
        retrieved from another property in the state.

        The 'goodness' of the state is given by 1 - normalised_error, i.e. the
        error's complement.
        """
        if isinstance(target, float) or isinstance(target, int):
            return self._error_complement_constant(state, compare_property, target, error_scaling)
        elif isinstance(target, prp.Property) or isinstance(target, prp.BoundedProperty):
            return self._error_complement_property(state, compare_property, target, error_scaling)
        else:
            raise ValueError(f'target must be a float, int or Property: {target}')

    def _error_complement_constant(self, state: FlightTask.State, compare_property: prp.Property,
                                   target_value: float, error_scaling: float) -> float:
        value = state.__getattribute__(compare_property.name)
        error = abs(value - target_value)
        normalised_error = self._normalise_error(error, error_scaling)
        return 1 - normalised_error

    def _error_complement_property(self, state: FlightTask.State, compare_property: prp.Property,
                                   target_property: prp.Property, error_scaling: float) -> float:
        target_value = state.__getattribute__(target_property.name)
        return self.error_complement(state, compare_property, target_value, error_scaling)

    @staticmethod
    def _normalise_error(absolute_error: float, error_scaling: float):
        """
        Given an error in the interval [0, +inf], returns a normalised error in [0, 1]

        The normalised error asymptotically approaches 1 as absolute_error -> +inf.

        The parameter error_scaling is used to scale for magnitude.
        When absolute_error == error_scaling, the normalised error is equal to 0.75
        """
        if absolute_error < 0:
            raise ValueError(f'Error to be normalised must be non-negative '
                             f'(use abs()): {absolute_error}')
        scaled_error = absolute_error / error_scaling
        return (scaled_error / (scaled_error + 1)) ** 0.5


class StepFractionComponent(ComplementComponent):
    """
    Rewards based on a property's closeness to a target value each timestep.

    Reward is equal to error_complement / episode_timesteps, therefore a
    this component sums to 1.0 over a perfect episode.
    """

    def __init__(self, name: str, prop: prp.Property, target: float,
                 scaling_factor: float, episode_timesteps: int):
        """
        Constructor.

        :param name: the name of this component used for __repr__, e.g.
            'altitude_keeping'
        :param prop: the Property for which a value will be retrieved
            from the State
        :param target: the target value from the property
        :param scaling_factor: the property value is scaled down by this amount.
            The RewardComponent outputs 0.5 when the value equals this factor
        :param episode_timesteps: the number of timesteps in each episode
        """
        self.name = name
        self.prop = prop
        self.target = target
        self.scaling_factor = scaling_factor
        self.episode_timesteps = episode_timesteps

    def calculate(self, state: FlightTask.State, _: FlightTask.State, is_terminal: bool):
        error_complement = self.error_complement(state,
                                                 self.prop,
                                                 self.target,
                                                 self.scaling_factor)
        return error_complement / self.episode_timesteps

    def get_name(self) -> str:
        return self.name


class ShapingComponent(ComplementComponent):
    """ A potential-based shaping reward component using error complements """

    def __init__(self, name: str, prop: prp.Property, target: Union[float, prp.Property],
                 scaling_factor: float):
        """
        Constructor.

        :param name: the name of this component used for __repr__, e.g.
            'altitude_keeping'
        :param prop: the Property for which a value will be retrieved
            from the State
        :param target: the target value from the property
        :param scaling_factor: the property value is scaled down by this amount.
            The RewardComponent outputs 0.75 when the error equals this factor
        """
        self.name = name
        self.prop = prop
        self.target = target
        self.scaling_factor = scaling_factor

    def get_potential(self, state: FlightTask.State, is_terminal):
        if is_terminal:
            return 0
        else:
            return self.error_complement(state, self.prop, self.target, self.scaling_factor)

    def calculate(self, state: FlightTask.State, last_state: FlightTask.State, is_terminal: bool):
        potential = self.get_potential(state, is_terminal)
        last_potential = self.get_potential(last_state, False)
        return potential - last_potential

    def get_name(self) -> str:
        return self.name
