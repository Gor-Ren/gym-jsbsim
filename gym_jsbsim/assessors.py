from abc import ABC, abstractmethod
from collections.__init__ import namedtuple
from typing import Iterable, Tuple, Dict
from gym_jsbsim.rewards import State, Reward, RewardComponent, ShapingComponent


class Assessor(ABC):
    """ Interface for Assessors which calculate Rewards from States. """

    @abstractmethod
    def assess(self, state: State, last_state: State, is_terminal: bool) -> Reward:
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
        self.base_components = tuple(base_components)
        self.shaping_components = tuple(shaping_components)
        if not self.base_components:
            raise ValueError('base reward components cannot be empty')

    def assess(self, state: State, last_state: State, is_terminal: bool) -> Reward:
        return Reward(self._base_rewards(state, last_state, is_terminal),
                      self._shaping_rewards(state, last_state, is_terminal))

    def _base_rewards(self, state: State, last_state: State, is_terminal: bool) -> Tuple[float, ...]:
        return tuple(cmp.calculate(state, last_state, is_terminal) for cmp in self.base_components)

    def _shaping_rewards(self, state: State, last_state: State, is_terminal: bool) -> Tuple[float, ...]:
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

    def _shaping_rewards(self, state: State,
                         last_state: State,
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
