import collections
import math
import numpy as np
import gym_jsbsim.properties as prp
from gym_jsbsim import utils
from gym_jsbsim.simulation import Simulation
from typing import Tuple, Callable, Union


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


class Assessor(object):
    """
    Interface for a Assessor, a callable object which calculates a Reward.

    A Reward is decomposed into two tuples of component values. An Assessor
    stores two corresponding tuples of functions, which calculate these values
    from environment state.

    An assessor object can be called thus:
        rewarder(state, last_state, is_terminal)
    and will return a Reward object.
    """
    def __init__(self, base_reward_functions: Tuple[Callable, ...],
                 shaping_reward_functions: Tuple[Callable, ...]=()):
        """
        Constructor. Sets the functions that will be used to calculate reward components.

        :param base_reward_functions: collection of functions r(s, s', is_terminal)
        :param shaping_reward_functions: optional, collection of functions f(s, s', is_terminal)
            defaults to empty tuple resulting in no reward shaping
        """
        self.base_reward_functions = base_reward_functions
        self.shaping_reward_functions = shaping_reward_functions

    def __call__(self, state: np.ndarray, last_state: np.ndarray, is_terminal: bool) -> Reward:
        """ Calculates reward from environment's state, previous state and terminal condition """
        return Reward(tuple(R(state, last_state, is_terminal) for R in self.base_reward_functions),
                      tuple(F(state,last_state, is_terminal) for F in self.shaping_reward_functions))


class RewardFunctionFactory(object):
    

    def get_property_closeness_to_target_function(controlled_prop: prp.Property, target_prop: prp.Property,
                                                  scale_factor: float) -> Callable:
        """ Returns a function on States that calculates how close a controlled property is to its target.

        :param controlled_prop: the property in State that is to be examined
        :param target_prop: the property in State which is the target value, at
            which the returned function outputs 1.0
        :param scale_factor: float, the returned function outputs 0.25 when there
            is this error to target. Used to account for units of measure etc.
        :return: a function: State -> float, calculating how close the controlled prop is to target
        """
        def normalised_goodness_function(state: Tuple[float, ...]) -> float:
            """
            Calculates how "good" a state is considering we wish one state
            variable to be equal to a target state variable.

            Returns 1.0 when controlled variable is equal to target value, and
            asymptotically approaches 0.0 as their absolute error approaches
            infinity.

            :param state: State namedtuple, the environment state
            :return: float, how close the controlled prop is to target, 1.0 if equal, reducing to 0.0
            """
            actual_value = state.__getattribute__(controlled_prop)
            target_value = state.__getattribute__(target_prop)
            abs_error = abs(actual_value - target_value)
            return 1 - normalise_unbounded_error(abs_error, scale_factor)

        return normalised_goodness_function


    def get_property_closeness_to_constant_function(controlled_prop: prp.Property, target_value: float,
                                                    scale_factor: float) -> Callable:
        """ Returns a function on States that calculates how close a controlled property is to its target.

        :param controlled_prop: the property in State that is to be examined
        :param target_value: the target for controlled_prop, at which returned function outputs 1.0
        :param scale_factor: float, the difference at which the returned function outputs 0.75
        :return: a function: State -> float, calculating how close the controlled prop is to target
        """
        def normalised_goodness_function(state: Tuple[float, ...]) -> float:
            """
            Calculates how "good" a state is considering we wish one state
            variable to be equal to a constant target value.

            Returns 1.0 when controlled variable is equal to target value, and
            asymptotically approaches 0.0 as their absolute error approaches
            infinity.

            :param state: State namedtuple, the environment state
            :return: float, how close the controlled prop is to target, 1.0 if equal, reducing to 0.0
            """
            actual_value = state.__getattribute__(controlled_prop)
            abs_error = abs(actual_value - target_value)
            return 1 - normalise_unbounded_error(abs_error, scale_factor)

        return normalised_goodness_function


class HeadingControlRewarder(Assessor):
    """
    Abstract Assessor for heading control tasks.

    Contains common methods for calculating rewards or potentials for re-use by
    inheriting classes.
    """
    FEET_PER_METRE = 3.28084
    ALTITUDE_ERROR_SCALING_FT = 100  # approximate order of magnitude of altitude error from target
    # namedtuple for more informative printing of reward tuples

    def __init__(self, max_distance_ft: float, max_timesteps: int):
        self.max_distance_ft = max_distance_ft
        self.max_timesteps = max_timesteps
        # these attrs are assigned on .reset():
        self.target_heading_deg = None
        self.target_altitude_ft = None
        self.initial_position = None

    def reset(self, target_heading_deg: float, target_altitude_ft: float,
              initial_latitude: float, initial_longitude: float) -> None:
        self.target_heading_deg = target_heading_deg
        self.target_altitude_ft = target_altitude_ft
        self.initial_position = utils.GeodeticPosition(initial_latitude, initial_longitude)

    def _altitude_keeping_reward(self, sim: Simulation) -> float:
        """
        Calculates the reward for the agent being at its current altitude
        for a single timestep.

        The agent receives maximum reward of +1 if the difference between
        target_altitude_ft and its altitude in the sim is zero. Each timestep
        it can receive 1 / max_timesteps altitude keeping reward.
        """
        altitude_ft = sim[prp.altitude_sl_ft]
        altitude_error_ft = abs(self.target_altitude_ft - altitude_ft)
        norm_error = utils.normalise_unbounded_error(altitude_error_ft, self.ALTITUDE_ERROR_SCALING_FT)
        return (1 - norm_error) / self.max_timesteps

    def _distance_travelled_reward(self, sim: Simulation, is_terminal: bool) -> float:
        if is_terminal:
            return self._normalised_distance_travelled(sim)
        else:
            return 0.0

    def _normalised_distance_travelled(self, sim: Simulation):
        parallel_dist_travelled_ft = self._parallel_distance_travelled_ft(sim)
        return parallel_dist_travelled_ft / self.max_distance_ft

    def _parallel_distance_travelled_ft(self, sim: Simulation):
        """ Calculates distance travelled in the sim parallel to the target heading """
        final_position = utils.GeodeticPosition.from_sim(sim)
        distance_travelled_m = sim[prp.dist_travel_m]
        heading_travelled_deg = self.initial_position.heading_deg_to(final_position)

        heading_error_rad = math.radians(heading_travelled_deg - self.target_heading_deg)

        parallel_distance_m = distance_travelled_m * math.cos(heading_error_rad)
        return parallel_distance_m


class HeadingControlBaseRewarder(HeadingControlRewarder):
    """
    Calculates the base (non-shaping) reward components for a heading control task.

    The reward consists of two components:
        1. every timestep, (1 - altitude_error) / max_timesteps reward is given
        2. 0 if non-terminal, else (parallel_distance_target_heading / distance_max)
    """
    RewardTuple = collections.namedtuple('RewardTuple',
                                         ['altitude_keeping_reward', 'distance_travel_reward'])

    def __call__(self, sim: Simulation, is_terminal: bool) -> Tuple:
        return self.RewardTuple(self._altitude_keeping_reward(sim),
                                self._distance_travelled_reward(sim, is_terminal))


class HeadingControlSimpleShapingRewarder(HeadingControlRewarder):
    """
    Calculates a shaping reward of a single component based on distance moved.

    Equation: normalised_d - normalised_d_last_step
    """
    RewardTuple = collections.namedtuple('RewardTuple',
                                         ['distance_shaping_reward'])
    last_potential = None

    def reset(self, target_heading_deg: float, target_altitude_ft: float,
              initial_latitude: float, initial_longitude: float) -> None:
        super().reset(target_heading_deg, target_altitude_ft, initial_latitude, initial_longitude)
        self.last_potential = 0.0  # zero distance travelled at timestep zero

    def __call__(self, sim: Simulation, is_terminal: bool):
        """ Returns a single reward component, the shaping reward for distance travelled """
        return self.RewardTuple(self._distance_shaping_reward(sim, is_terminal))

    def _distance_shaping_reward(self, sim: Simulation, is_terminal: bool) -> float:
        """ Calculates a potential based reward from incremental distance travelled """
        if is_terminal:
            # terminal potential is always zero
            new_potential = 0.0
        else:
            new_potential = self._normalised_distance_travelled(sim)

        shaping_reward_component = self.last_potential - new_potential
        self.last_potential = new_potential
        return shaping_reward_component


def normalise_unbounded_error(absolute_error: float, error_scaling: float):
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
