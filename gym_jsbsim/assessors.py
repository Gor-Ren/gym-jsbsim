from gym_jsbsim import utils
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Dict
from gym_jsbsim.rewards import State, Reward, RewardComponent, ShapingComponent


class Assessor(ABC):
    """ Interface for Assessors which calculate Rewards from States. """

    @abstractmethod
    def assess(self, state: State, prev_state: State, is_terminal: bool) -> Reward:
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

    def assess(self, state: State, prev_state: State, is_terminal: bool) -> Reward:
        return Reward(self._base_rewards(state, prev_state, is_terminal),
                      self._shaping_rewards(state, prev_state, is_terminal))

    def _base_rewards(self, state: State, last_state: State, is_terminal: bool) -> Tuple[
        float, ...]:
        return tuple(cmp.calculate(state, last_state, is_terminal) for cmp in self.base_components)

    def _shaping_rewards(self, state: State, last_state: State, is_terminal: bool) -> Tuple[
        float, ...]:
        return tuple(
            cmp.calculate(state, last_state, is_terminal) for cmp in self.shaping_components)


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
        self.all_dependant_indices = self._get_sequential_indices(shaping_dependency_map)

    def _get_sequential_indices(self,
                                shaping_dependency_map: Dict) -> Tuple[Tuple[int, ...]]:
        """
        Given a map of shaping components to their sequentially dependant
        components, determines the indices of the dependant components'
        potentials and stores them in a tuple such that the ith collection of
        indices corresponds to the ith reward component's dependants.
        """
        all_dependants = []
        for component in self.shaping_components:
            dependant_comps = shaping_dependency_map.get(component, ())
            dependant_indices = tuple(self.shaping_components.index(cmp) for cmp in dependant_comps)
            all_dependants.append(dependant_indices)
        return tuple(all_dependants)

    def _shaping_rewards(self, state: State,
                         last_state: State,
                         is_terminal: bool) -> Tuple[float, ...]:
        potentials = tuple(cmp.get_potential(state, is_terminal) for cmp in self.shaping_components)
        last_potentials = tuple(cmp.get_potential(last_state, False) for cmp in self.shaping_components)

        seq_potentials = self._apply_dependents(potentials)
        seq_last_potentials = self._apply_dependents(last_potentials)
        return tuple(pot - last_pot for pot, last_pot in zip(seq_potentials, seq_last_potentials))

    @abstractmethod
    def _apply_dependents(self, potentials: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Modifies potentials to account for dependant components.

        :param potentials: the normal component potential values
        :return: a collection of component potentials, transformed to account
            for sequential dependencies
        """
        ...


class ContinuousSequentialAssessor(SequentialAssessor):
    """
    A sequential assessor in which shaping components with dependents have their potential
    reduced according to their dependent's potentials through multiplication.

    For example a component with a "base" potential of 0.8 and a dependent component at
    0.5 have a sequential potential of 0.8 * 0.5 = 0.4.
    """

    def _apply_dependents(self, potentials: Tuple[float, ...]):
        sequential_potentials = []
        for base_potential, dependant_indices in zip(potentials, self.all_dependant_indices):
            # multiply each base potential value by all dependants
            sequential_factor = utils.product(potentials[i] for i in dependant_indices)
            sequential_potentials.append(base_potential * sequential_factor)
        return tuple(sequential_potentials)


class DiscontinuousSequentialAssessor(SequentialAssessor):
    """
    A sequential assessor in which shaping components with dependents are given zero
    potential until their dependent's potentials exceed a critical value.

    For example a component with a "base" potential of 0.8 and a dependent
    component with potential < critical has sequential potential 0.0. Once
    the dependent potential >= critical, the dependent component potential
    returns to 0.8.
    """
    DEPENDANT_REQD_POTENTIAL = 0.85

    def _apply_dependents(self, potentials: Tuple[float, ...]):
        sequential_potentials = []
        for base_potential, dependant_indices in zip(potentials, self.all_dependant_indices):
            # multiply each base potential value by all dependants
            dependancies_met = all(potentials[i] > self.DEPENDANT_REQD_POTENTIAL
                                 for i in dependant_indices)
            if dependancies_met:
                sequential_potential = base_potential
            else:
                sequential_potential = 0.0
            sequential_potentials.append(sequential_potential)
        return tuple(sequential_potentials)
