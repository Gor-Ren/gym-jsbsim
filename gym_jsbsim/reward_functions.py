import collections
import math
from gym_jsbsim import utils
from gym_jsbsim.simulation import Simulation
from typing import Tuple
from abc import ABC, abstractmethod


class ShapingReward(object):
    """
    Immutable class storing an RL reward.

    We decompose rewards into tuples of component values, reflecting contributions
    from different goals. Separate tuples are maintained for the base (non-shaping)
    components and the shaping components to allow better analysis.

    Scalar reward values can be retrieved from a ShapingReward by calling .reward()
    or non_shaping_reward() as required.
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


class Rewarder(ABC):
    """
    Interface for a Rewarder, a callable object which calculates components of a reward signal.

    We decompose the reward into a tuple of component values. A Rewarder
    calculates each component from a simulation state and returns the tuple.

    When called,
        rewarder(simulation, is_terminal)
    should return a tuple of reward components.
    """
    @abstractmethod
    def __call__(self, sim: Simulation, is_terminal: bool) -> Tuple:
        ...

    def reset(self, *args):
        """ Called at the end of each episode, reset() should clear any state.

        Default behaviour is to do nothing.
        """
        pass


class EmptyRewarder(object):
    """
    A Rewarder which always returns empty tuple.

    Use this as your shaping rewarder when you don't want to use shaping.
    """
    def __call__(self, _, __) -> Tuple:
        """ Given any Simulation and is_terminal condition, returns empty tuple. """
        return ()


class HeadingControlRewarder(ABC, Rewarder):
    """
    Abstract Rewarder for heading control tasks.

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
        altitude_ft = sim['position/h-sl-ft']
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
        distance_travelled_m = sim['position/distance-from-start-mag-mt']
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
