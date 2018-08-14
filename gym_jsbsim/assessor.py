import abc
import math
from gym_jsbsim import utils
from gym_jsbsim.simulation import Simulation
from typing import Tuple


class Assessor(abc.ABC):
    """
    Interface for an Assessor, an object which calculates an agent's reward

    An Assessor implements the calculate_reward() method.
    """
    def __init__(self):
        pass

    def calculate_reward(self, sim: Simulation, is_terminal: bool):
        raise NotImplementedError


class HeadingControlBaseAssessor(Assessor):
    FEET_PER_METRE = 3.28084
    ALTITUDE_ERROR_SCALING_FT = 100  # approximate order of magnitude of altitude error from target

    def __init__(self, target_heading_deg: float, target_altitude_ft: float,
                 initial_latitude: float, initial_longitude: float,
                 max_distance_ft: float, max_timesteps: int):
        super().__init__()
        self.target_heading_deg = target_heading_deg
        self.target_altitude_ft = target_altitude_ft
        self.initial_position = utils.GeodeticPosition(initial_latitude, initial_longitude)
        self.max_distance_ft = max_distance_ft
        self.max_timesteps = max_timesteps

    def calculate_reward(self, sim: Simulation, is_terminal: bool):
        if is_terminal:
            return self._terminal_reward(sim)
        else:
            return self._non_terminal_reward(sim)

    def _terminal_reward(self, sim):
        return self._normalised_distance_travelled(sim) + self._altitude_keeping_reward_step(sim)

    def _non_terminal_reward(self, sim):
        return self._altitude_keeping_reward_step(sim)

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

    def _altitude_keeping_reward_step(self, sim):
        """
        Calculates the reward for the agent being at its current altitude
        for a single timestep.

        The agent receives maximum reward if the difference between
        target_altitude_ft and its altitude in the sim is zero. Each timestep
        it can receive 1 / max_timesteps altitude keeping reward.
        """
        altitude_ft = sim['position/h-sl-ft']
        altitude_error_ft = abs(self.target_altitude_ft - altitude_ft)
        norm_error = utils.normalise_unbounded_error(altitude_error_ft, self.ALTITUDE_ERROR_SCALING_FT)
        return (1 - norm_error) / self.max_timesteps
