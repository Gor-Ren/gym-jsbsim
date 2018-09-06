import warnings
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

    Initialised with RewardComponents which allow calculation of the base
    (policy-influencing) and shaping rewards (non-policy-influencing) rewards respectively.
    """

    def __init__(self, base_components: Iterable['RewardComponent'],
                 potential_based_components: Iterable['RewardComponent'] = (),
                 positive_rewards: bool = False):
        """
        :param base_components: RewardComponents from which Reward is to be calculated
        :param potential_based_components: RewardComponents from which a potential-based
            reward component is to be calculated from
        :param positive_rewards: True if rewards should be in [0.0, 1.0] (0.0 corresp. to
            worst behaviour), else rewards will be in [-1.0, 0.0] with 0.0 corresp. to
            perfect behaviour. Has no effect one potential difference based components.
        """
        self.base_components = tuple(base_components)
        self.potential_components = tuple(potential_based_components)
        self.positive_rewards = positive_rewards
        if not self.base_components:
            raise ValueError('base reward components cannot be empty')
        if any(cmp.is_potential_difference_based() for cmp in self.base_components):
            raise ValueError('base rewards must be non potential based in this implementation')
            # because of the positive_rewards logic
        if not all(cmp.is_potential_difference_based() for cmp in self.potential_components):
            warnings.warn(f'Potential component not is_potential_difference_based()')

    def assess(self, state: State, prev_state: State, is_terminal: bool) -> Reward:
        """ Calculates a Reward from the state transition. """
        return Reward(self._base_rewards(state, prev_state, is_terminal),
                      self._potential_based_rewards(state, prev_state, is_terminal))

    def _base_rewards(self, state: State, prev_state: State, is_terminal: bool) -> Tuple[
        float, ...]:
        cmp_values = (cmp.calculate(state, prev_state, is_terminal) for cmp in self.base_components)
        if self.positive_rewards:
            return tuple(cmp_values)
        else:
            return tuple(value - 1 for value in cmp_values)

    def _potential_based_rewards(self, state: State, last_state: State, is_terminal: bool) -> Tuple[
        float, ...]:
        return tuple(
            cmp.calculate(state, last_state, is_terminal) for cmp in self.potential_components)


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
                 potential_dependency_map: Dict[
                     'RewardComponent', Tuple['RewardComponent', ...]] = {},
                 positive_rewards: bool = False):
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
        super().__init__(base_components, potential_components, positive_rewards)
        self.base_dependency_map = base_dependency_map
        self.potential_dependency_map = potential_dependency_map

    def _base_rewards(self, state: State, prev_state: State, is_terminal: bool) -> Tuple[
        float, ...]:
        potentials = tuple(cmp.get_potential(state, is_terminal) for cmp in self.base_components)
        seq_discounts = self._get_sequential_discounts(state,
                                                       is_terminal,
                                                       self.base_components,
                                                       self.base_dependency_map)

        seq_values = (pot * discount for pot, discount in zip(potentials, seq_discounts))
        if self.positive_rewards:
            return tuple(seq_values)
        else:
            return tuple(value - 1 for value in seq_values)

    def _potential_based_rewards(self, state: State, prev_state: State,
                                 is_terminal: bool) -> Tuple[float, ...]:
        potentials = tuple(cmp.get_potential(state, is_terminal)
                           for cmp in self.potential_components)
        prev_potentials = tuple(cmp.get_potential(prev_state, False)
                                for cmp in self.potential_components)

        discounts = self._get_sequential_discounts(state,
                                                   is_terminal,
                                                   self.potential_components,
                                                   self.potential_dependency_map)
        prev_discounts = self._get_sequential_discounts(prev_state,
                                                        False,
                                                        self.potential_components,
                                                        self.potential_dependency_map)

        seq_potentials = (p * d for p, d in zip(potentials, discounts))
        seq_prev_potentials = (p * d for p, d in zip(prev_potentials, prev_discounts))
        return tuple(pot - prev_pot for pot, prev_pot in zip(seq_potentials, seq_prev_potentials))

    @abstractmethod
    def _get_sequential_discounts(self, state: State, is_terminal: bool,
                                  components: Iterable['RewardComponent'],
                                  dependency_map: Dict['RewardComponent', Tuple]) -> Tuple[
        float, ...]:
        """
        Calculates a discount factor in [0,1] from each component's dependencies.

        The dependencies may reduce that component's values because they are not
        yet met. A component with no dependencies has discount factor 1.0.

        :param state: the state that discount factor is to be evaluated at
        :param is_terminal: whether the transition to state was terminal
        :param components: the RewardComponents to be assessed for discounting
        :param dependency_map: a map of RewardComponents to their dependent
            RewardComponents.
        :return: tuple of floats, discount factors in [0,1], corresponding to
            same order as 'components' input
        """
        ...


class ContinuousSequentialAssessor(SequentialAssessor):
    """
    A sequential assessor in which shaping components with dependents have their potential
    reduced according to their dependent's potentials through multiplication.

    For example a component with a "base" potential of 0.8 and a dependent component at
    0.5 have a sequential potential of 0.8 * 0.5 = 0.4.
    """

    def _get_sequential_discounts(self, state: State, is_terminal: bool,
                                  components: Iterable['RewardComponent'],
                                  dependency_map: Dict[
                                      'RewardComponent', Tuple['RewardComponent', ...]]) -> Tuple[
        float, ...]:
        discounts = []
        for component in components:
            dependents = dependency_map.get(component, ())
            dependent_potentials = (dep.get_potential(state, is_terminal) for dep in dependents)
            discount = utils.product(pot for pot in dependent_potentials)
            discounts.append(discount)
        return tuple(discounts)
