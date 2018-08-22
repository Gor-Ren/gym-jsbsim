import gym
import numpy as np
import random
import types
import math
import gym_jsbsim.properties as prp
from gym_jsbsim import utils
from collections import namedtuple
from gym_jsbsim.simulation import Simulation
from gym_jsbsim import rewards
from gym_jsbsim.properties import BoundedProperty, Property
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Tuple, NamedTuple, Type


class Task(ABC):
    """
    Interface for Tasks, modules implementing specific environments in JSBSim.

    A task defines its own state space, action space, termination conditions and reward function.
    """

    @abstractmethod
    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) \
            -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Calculates step reward and termination from an agent observation.

        :param sim: a Simulation, the simulation from which to extract state
        :param action: sequence of floats, the agent's last action
        :param sim_steps: number of JSBSim integration steps to perform following action
            prior to making observation
        :return: tuple of (observation, reward, done, info) where,
            observation: np.ndarray, agent's observation of the environment state
            reward: float, the reward for that step
            done: bool, True if the episode is over else False
            info: dict, optional, containing diagnostic info for debugging etc.
        """

    ...

    @abstractmethod
    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        """
        Initialise any state/controls and get first state observation from reset sim.

        :param sim: Simulation, the environment simulation
        :return: np array, the first state observation of the episode
        """
        ...

    @abstractmethod
    def get_initial_conditions(self) -> Optional[Dict[Property, float]]:
        """
        Returns dictionary mapping initial episode conditions to values.

        Episode initial conditions (ICs) are defined by specifying values for
        JSBSim properties, represented by their name (string) in JSBSim.

        JSBSim uses a distinct set of properties for ICs, beginning with 'ic/'
        which differ from property names during the simulation, e.g. "ic/u-fps"
        instead of "velocities/u-fps". See https://jsbsim-team.github.io/jsbsim/

        :return: dict mapping string for each initial condition property to
            initial value, a float, or None to use Env defaults
        """
        ...

    @abstractmethod
    def get_observation_space(self) -> gym.Space:
        """ Get the task's observation Space object """
        ...

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """ Get the task's action Space object """
        ...


class FlightTask(Task, ABC):
    """
    Abstract superclass for flight tasks.

    Concrete subclasses should implement the following:
        state_variables attribute: tuple of Propertys, the task's state representation
        action_variables attribute: tuple of Propertys, the task's actions
        get_initial_conditions(): returns dict mapping InitialPropertys to initial values
        _is_done(): determines episode termination
        (optional) _new_episode(): performs any control input/initialisation on episode reset
        (optional) _update_custom_properties: updates any custom properties in the sim
    """
    base_state_variables = (prp.altitude_sl_ft, prp.pitch_rad, prp.roll_rad,
                            prp.u_fps, prp.v_fps, prp.w_fps,
                            prp.p_radps, prp.q_radps, prp.r_radps,
                            prp.aileron_left, prp.aileron_right, prp.elevator,
                            prp.rudder)
    base_initial_conditions = types.MappingProxyType(  # MappingProxyType makes dict immutable
        {prp.initial_altitude_ft: 5000,
         prp.initial_terrain_altitude_ft: 0.00000001,
         prp.initial_longitude_geoc_deg: -2.3273,
         prp.initial_latitude_geod_deg: 51.3781  # corresponds to UoBath
         }
    )
    state_variables: Tuple[BoundedProperty, ...]
    action_variables: Tuple[BoundedProperty, ...]
    assessor: 'rewards.Assessor'
    State: Type[NamedTuple]

    def __init__(self, assessor: 'rewards.Assessor') -> None:
        self.last_state = None
        self.assessor = assessor
        self._make_state_class()

    def _make_state_class(self) -> None:
        """ Creates a namedtuple for readable State data """
        illegal_chars, translate_to = '\-/', '___'
        legal_translate = str.maketrans(illegal_chars, translate_to)
        # get list of state property names, containing legal chars only
        legal_attribute_names = [prop.name.translate(legal_translate) for prop in self.state_variables]
        self.State = namedtuple('State', legal_attribute_names)

    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) \
            -> Tuple[np.ndarray, float, bool, Dict]:
        # input actions
        for prop, command in zip(self.action_variables, action):
            sim[prop] = command

        # run simulation
        for _ in range(sim_steps):
            sim.run()

        self._update_custom_properties(sim)
        state = self.State(sim[prop] for prop in self.state_variables)
        done = self._is_done(state, sim[prp.sim_time_s])
        reward = self.assessor.assess(state, self.last_state, done)
        self.last_state = state
        info = {'reward': reward}

        return np.array(state), reward.reward(), done, info

    def _update_custom_properties(self, sim: Simulation) -> None:
        """ Calculates any custom properties which change every timestep. """
        pass

    @abstractmethod
    def _is_done(self, state: Tuple[float, ...], episode_time: float) -> bool:
        """ Determines whether the current episode should terminate.

        :param state: the last state observation
        :param episode_time: the episode time in seconds
        :return: True if the episode should terminate else False
        """
        ...

    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        self._new_episode(sim)
        self._update_custom_properties(sim)
        state = self.State(sim[prop] for prop in self.state_variables)
        self.last_state = state
        return np.ndarray(state)

    def _new_episode(self, sim: Simulation) -> None:
        """
        This method is called at the start of every episode. It is used to set
        the value of any controls or environment properties not already defined
        in the task's initial conditions.

        By default it simply starts the aircraft engines.
        """
        sim.start_engines()

    @abstractmethod
    def get_initial_conditions(self) -> Dict[Property, float]:
        ...

    def get_observation_space(self) -> gym.Space:
        state_lows = np.array([state_var.min for state_var in self.state_variables])
        state_highs = np.array([state_var.max for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    def get_action_space(self) -> gym.Space:
        action_lows = np.array([act_var.min for act_var in self.action_variables])
        action_highs = np.array([act_var.max for act_var in self.action_variables])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float')


class SteadyLevelFlightTask(FlightTask):
    """
    A task in which the agent must perform steady, level flight maintaining its
    current heading.
    """
    MAX_TIME_SECS = 15
    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    INITIAL_HEADING_DEG = 270
    target_heading_deg = BoundedProperty('max_target/heading-deg', 'desired heading [deg]',
                                         prp.heading_deg.min, prp.heading_deg.max)
    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd)

    def __init__(self, assessor: 'rewards.Assessor', max_distance_m: float):
        self.distance_parallel_to_heading_m = BoundedProperty('max_target/dist-parallel-heading-m',
                                                              'distance travelled parallel to max_target heading [m]',
                                                              0, max_distance_m)
        self.extra_state_variables = (prp.heading_deg, self.target_heading_deg, self.distance_parallel_to_heading_m)
        self.state_variables = FlightTask.base_state_variables + self.extra_state_variables
        super().__init__(assessor)

    def _make_target_values(self):
        """
        Makes a tuple representing the desired state of the aircraft.

        :return: tuple of triples (property, target_value, gain) where:
            property: str, the name of the property in JSBSim
            target_value: number, the desired value to be controlled to
            gain: number, by which the error between actual and max_target value
                 is multiplied to calculate reward
        """
        PROPORTIONAL_TO_DERIV_RATIO = 2  # how many times lower are rate terms vs absolute terms
        RAD_TO_DEG = 57
        ALT_GAIN = 0.1
        ALT_RATE_GAIN = ALT_GAIN / PROPORTIONAL_TO_DERIV_RATIO
        ROLL_GAIN = 1 * RAD_TO_DEG
        ROLL_RATE_GAIN = ROLL_GAIN / PROPORTIONAL_TO_DERIV_RATIO
        HEADING_GAIN = 1
        HEADING_RATE_GAIN = HEADING_GAIN * RAD_TO_DEG / PROPORTIONAL_TO_DERIV_RATIO
        PITCH_RATE_GAIN = 1 * RAD_TO_DEG / PROPORTIONAL_TO_DERIV_RATIO

        start_alt_ft = self.get_initial_conditions()[prp.initial_altitude_ft]
        target_values = (
            (prp.altitude_sl_ft, start_alt_ft, ALT_GAIN),
            (prp.altitude_rate_fps, 0, ALT_RATE_GAIN),
            (prp.roll_rad, 0, ROLL_GAIN),
            (prp.p_radps, 0, ROLL_RATE_GAIN),
            (prp.heading_deg, self.INITIAL_HEADING_DEG, HEADING_GAIN),
            (prp.r_radps, 0, HEADING_RATE_GAIN),
            # absolute pitch is implicitly controlled through altitude, so just control rate
            (prp.q_radps, 0, PITCH_RATE_GAIN)
        )

        return target_values

    def get_initial_conditions(self) -> Dict[Property, float]:
        extra_conditions = {prp.initial_u_fps: 150,
                            prp.initial_v_fps: 0,
                            prp.initial_w_fps: 0,
                            prp.initial_p_radps: 0,
                            prp.initial_q_radps: 0,
                            prp.initial_r_radps: 0,
                            prp.initial_roc_fpm: 0,
                            prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
                            }
        return {**self.base_initial_conditions, **extra_conditions}

    def _update_custom_properties(self, sim: Simulation) -> None:
        self._update_parallel_distance_travelled(sim, self.INITIAL_HEADING_DEG)

    def _update_parallel_distance_travelled(self, sim: Simulation, target_heading_deg: float) -> None:
        """
        Calculates how far aircraft has travelled from initial position parallel to max_target heading

         Stores result in Simulation as custom property.
         """
        current_position = utils.GeodeticPosition.from_sim(sim)
        heading_travelled_deg = self.initial_position.heading_deg_to(current_position)
        heading_error_rad = math.radians(heading_travelled_deg - target_heading_deg)

        distance_travelled_m = sim[prp.dist_travel_m]
        parallel_distance_travelled_m = math.cos(heading_error_rad) * distance_travelled_m
        sim[self.distance_parallel_to_heading_m] = parallel_distance_travelled_m

    def _is_done(self, state: Tuple[float, ...], episode_time: float) -> bool:
        return episode_time > self.MAX_TIME_SECS

    def _new_episode(self, sim: Simulation) -> None:
        sim.start_engines()
        sim[prp.throttle_cmd] = self.THROTTLE_CMD
        sim[prp.mixture_cmd] = self.MIXTURE_CMD

        sim[self.target_heading_deg] = self._get_target_heading()
        self.initial_position = utils.GeodeticPosition.from_sim(sim)

    def _get_target_heading(self) -> float:
        # use the same, initial heading every episode
        return self.INITIAL_HEADING_DEG


class HeadingControlTask(SteadyLevelFlightTask):
    """
    A task in which the agent must make a turn and fly level on a desired heading.
    """

    def _make_target_values(self):
        """ Sets an attribute specifying the desired state of the aircraft.

        target_values is a tuple of triples of format
            (property, target_value, gain) where:
            property: str, the name of the property in JSBSim
            target_value: number, the desired value to be controlled to
            gain: number, by which the error between actual and max_target value
                 is multiplied to calculate reward
        """
        super_target_values = super()._make_target_values()
        heading_target_index = 4
        old_heading_triple = super_target_values[heading_target_index]
        random_target_heading = random.uniform(prp.heading_deg.min, prp.heading_deg.max)
        new_heading_triple = (old_heading_triple[0], random_target_heading, old_heading_triple[2])

        target_values = []
        for i, target_triple in enumerate(super_target_values):
            if i != heading_target_index:
                target_values.append(target_triple)
            else:
                target_values.append(new_heading_triple)
        assert len(target_values) == len(super_target_values)
        return target_values

    def get_initial_conditions(self) -> [Dict[Property, float]]:
        initial_conditions = super().get_initial_conditions()
        random_heading = random.uniform(prp.heading_deg.min, prp.heading_deg.max)
        initial_conditions[prp.initial_heading_deg] = random_heading
        return initial_conditions

    def _update_custom_properties(self, sim: Simulation) -> None:
        self._update_parallel_distance_travelled(sim, sim[self.target_heading_deg])

    def _get_target_heading(self) -> float:
        # select a random heading each episode
        return random.uniform(self.target_heading_deg.min,
                              self.target_heading_deg.max)
