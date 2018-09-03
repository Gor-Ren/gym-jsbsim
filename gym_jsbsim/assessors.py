from gym_jsbsim import utils
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Dict
from gym_jsbsim.rewards import State, Reward, RewardComponent


class Assessor(ABC):
    """ Interface for Assessors which calculate Rewards from States. """

    @abstractmethod
    def assess(self, state: State, prev_state: State, is_terminal: bool) -> Reward:
        """ Calculates reward from environment's state, previous state and terminal condition """
        ...


class AssessorImpl(Assessor):
    """
    Determines the Reward from a state transitions.

    Initialised with RewardComponents which allow
    calculation of the base (non-shaping) and shaping rewards respectively.
    """

    def __init__(self, base_components: Iterable['RewardComponent'],
                 potential_based_components: Iterable['RewardComponent'] = ()):
        """
        :param base_components: RewardComponents from which Reward is to be calculated
        :param potential_based_components: RewardComponents from which a potential-based
            reward component is to be calculated from
        """
        self.base_components = tuple(base_components)
        self.potential_components = tuple(potential_based_components)
        if not self.base_components:
            raise ValueError('base reward components cannot be empty')

    def assess(self, state: State, prev_state: State, is_terminal: bool) -> Reward:
        return Reward(self._base_rewards(state, prev_state, is_terminal),
                      self._potential_based_rewards(state, prev_state, is_terminal))

    def _base_rewards(self, state: State, last_state: State, is_terminal: bool) -> Tuple[
        float, ...]:
        return tuple(cmp.calculate(state, last_state, is_terminal) for cmp in self.base_components)

    def _potential_based_rewards(self, state: State, last_state: State, is_terminal: bool) -> Tuple[
        float, ...]:
        return tuple(
            cmp.calculate(state, last_state, is_terminal) for cmp in
            self.potential_components)


class SequentialAssessor(AssessorImpl, ABC):
    """
    Abstract class that allows base and potential components to be assigned
    dependencies of other components, such that they are affected by the
    other's values.

    Concrete subclasses should implement _apply_dependents(), which modifies
    the 'normal' component potentials to account for dependents
    """

    def __init__(self, base_components: Iterable['RewardComponent'],
                 potential_components: Iterable['RewardComponent'] = (),
                 base_dependency_map: Dict['RewardComponent', Tuple['RewardComponent', ...]] = {},
                 potential_dependency_map: Dict['RewardComponent', Tuple['RewardComponent', ...]] = {}):
        """
        :param base_components: RewardComponents from which the non-shaping
            part of the Reward is to be calculated
        :param potential_components: ErrorComponents from which the shaping
            reward is to be calculated, or an empty tuple for no shaping
        :param base_dependency_map: maps base components with sequential
            dependencies to their dependent components, defaults to
            no dependencies
        :param potential_dependency_map: maps potential components with sequential
            dependencies to their dependent components, defaults to
            no dependencies
        """
        super().__init__(base_components, potential_components)
        self.base_dependent_indices = self._get_sequential_indices(self.base_components,
                                                                   base_dependency_map)
        self.potential_dependant_indices = self._get_sequential_indices(self.potential_components,
                                                                        potential_dependency_map)

    @staticmethod
    def _get_sequential_indices(components: Tuple['RewardComponent'],
                                dependency_map: Dict) -> Tuple[Tuple[int, ...]]:
        """
        Given a collection of components, and map of components to their
        sequential dependents, determines the indices of the dependant components'
        potentials and stores them in a tuple such that the ith collection of
        indices corresponds to the ith reward component's dependants.
        """
        all_dependants = []
        for component in components:
            dependant_comps = dependency_map.get(component, ())
            dependant_indices = tuple(components.index(cmp) for cmp in dependant_comps)
            all_dependants.append(dependant_indices)
        return tuple(all_dependants)

    def _base_rewards(self, state: State, prev_state: State, is_terminal: bool) -> Tuple[
        float, ...]:
        normal_potentials = super()._base_rewards(state, prev_state, is_terminal)
        return self._apply_dependents(normal_potentials, self.base_dependent_indices)

    def _potential_based_rewards(self, state: State, prev_state: State,
                                 is_terminal: bool) -> Tuple[float, ...]:
        potentials = tuple(
            cmp.get_potential(state, is_terminal) for cmp in self.potential_components)
        prev_potentials = tuple(
            cmp.get_potential(prev_state, False) for cmp in self.potential_components)

        seq_potentials = self._apply_dependents(potentials, self.potential_dependant_indices)
        seq_last_potentials = self._apply_dependents(prev_potentials,
                                                     self.potential_dependant_indices)
        return tuple(pot - last_pot for pot, last_pot in zip(seq_potentials, seq_last_potentials))

    @abstractmethod
    def _apply_dependents(self, potentials: Tuple[float, ...],
                          dependent_indices: Tuple[Tuple[int, ...]]) -> Tuple[float, ...]:
        """
        Modifies potentials to account for dependant components.

        :param potentials: the normal component potential values
        :param dependent_indices: tuple of tuples containing indices, such that
            the ith element contains a tuple of all dependent's of ith component
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

    def _apply_dependents(self, potentials: Tuple[float, ...],
                          dependant_indices: Tuple[Tuple[int, ...]]) -> Tuple[float, ...]:
        sequential_potentials = []
        for base_potential, indices in zip(potentials, dependant_indices):
            # multiply each base potential value by all dependants
            sequential_factor = utils.product(potentials[i] for i in indices)
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

    def _apply_dependents(self, potentials: Tuple[float, ...],
                          dependant_indices: Tuple[Tuple[int, ...]]) -> Tuple[float, ...]:
        sequential_potentials = []
        for base_potential, indices in zip(potentials, dependant_indices):
            # multiply each base potential value by all dependants
            dependancies_met = all(potentials[i] > self.DEPENDANT_REQD_POTENTIAL for i in indices)
            if dependancies_met:
                sequential_potential = base_potential
            else:
                sequential_potential = 0.0
            sequential_potentials.append(sequential_potential)
        return tuple(sequential_potentials)
