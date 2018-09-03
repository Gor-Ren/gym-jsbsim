import gym
import numpy as np
import random
import types
import math
import enum
import warnings
from collections import namedtuple
import gym_jsbsim.properties as prp
from gym_jsbsim import assessors, rewards, utils
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.properties import BoundedProperty, Property
from gym_jsbsim.aircraft import Aircraft
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Tuple, NamedTuple, Type


class Task(ABC):
    """
    Interface for Tasks, modules implementing specific environments in JSBSim.

    A task defines its own state space, action space, termination conditions and agent_reward function.
    """

    @abstractmethod
    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) \
            -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Calculates new state, reward and termination.

        :param sim: a Simulation, the simulation from which to extract state
        :param action: sequence of floats, the agent's last action
        :param sim_steps: number of JSBSim integration steps to perform following action
            prior to making observation
        :return: tuple of (observation, reward, done, info) where,
            observation: array, agent's observation of the environment state
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
    def get_state_space(self) -> gym.Space:
        """ Get the task's state Space object """
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
        _is_terminal(): determines episode termination
        (optional) _new_episode_init(): performs any control input/initialisation on episode reset
        (optional) _update_custom_properties: updates any custom properties in the sim
    """
    INITIAL_ALTITUDE_FT = 5000
    base_state_variables = (prp.altitude_sl_ft, prp.pitch_rad, prp.roll_rad,
                            prp.u_fps, prp.v_fps, prp.w_fps,
                            prp.p_radps, prp.q_radps, prp.r_radps,
                            prp.aileron_left, prp.aileron_right, prp.elevator,
                            prp.rudder)
    base_initial_conditions = types.MappingProxyType(  # MappingProxyType makes dict immutable
        {prp.initial_altitude_ft: INITIAL_ALTITUDE_FT,
         prp.initial_terrain_altitude_ft: 0.00000001,
         prp.initial_longitude_geoc_deg: -2.3273,
         prp.initial_latitude_geod_deg: 51.3781  # corresponds to UoBath
         }
    )
    state_variables: Tuple[BoundedProperty, ...]
    action_variables: Tuple[BoundedProperty, ...]
    assessor: assessors.Assessor
    State: Type[NamedTuple]

    def __init__(self, assessor: assessors.Assessor, debug: bool=False) -> None:
        self.last_state = None
        self.assessor = assessor
        self._make_state_class()
        self.debug = debug

    def _make_state_class(self) -> None:
        """ Creates a namedtuple for readable State data """
        # get list of state property names, containing legal chars only
        legal_attribute_names = [prop.get_legal_name() for prop in
                                 self.state_variables]
        self.State = namedtuple('State', legal_attribute_names)

    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) \
            -> Tuple[NamedTuple, float, bool, Dict]:
        # input actions
        for prop, command in zip(self.action_variables, action):
            sim[prop] = command

        # run simulation
        for _ in range(sim_steps):
            sim.run()

        self._update_custom_properties(sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        done = self._is_terminal(sim)
        reward = self.assessor.assess(state, self.last_state, done)
        if self.debug:
            self._validate_state(state, done, action, reward)
        self.last_state = state
        info = {'reward': reward}

        return state, reward.agent_reward(), done, info

    def _validate_state(self, state, done, action, reward):
        if any(math.isnan(el) for el in state):  # float('nan') in state doesn't work!
            msg = (f'Invalid state encountered!\n'
                         f'State: {state}\n'
                         f'Prev. State: {self.last_state}\n'
                         f'Action: {action}\n'
                         f'Terminal: {done}\n'
                         f'Reward: {reward}')
            warnings.warn(msg, RuntimeWarning)

    def _update_custom_properties(self, sim: Simulation) -> None:
        """ Calculates any custom properties which change every timestep. """
        pass

    @abstractmethod
    def _is_terminal(self, sim: Simulation) -> bool:
        """ Determines whether the current episode should terminate.

        :param sim: the current simulation
        :return: True if the episode should terminate else False
        """
        ...

    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        self._new_episode_init(sim)
        self._update_custom_properties(sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        self.last_state = state
        return state

    def _new_episode_init(self, sim: Simulation) -> None:
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

    def get_state_space(self) -> gym.Space:
        state_lows = np.array([state_var.min for state_var in self.state_variables])
        state_highs = np.array([state_var.max for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    def get_action_space(self) -> gym.Space:
        action_lows = np.array([act_var.min for act_var in self.action_variables])
        action_highs = np.array([act_var.max for act_var in self.action_variables])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float')


class HeadingControlTask(FlightTask):
    """
    A task in which the agent must perform steady, level flight maintaining its
    initial heading.
    """
    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    INITIAL_HEADING_DEG = 270
    DEFAULT_EPISODE_TIME_S = 200.
    ALTITUDE_SCALING_FT = 25
    HEADING_ERROR_SCALING_DEG = 7.5
    ROLL_ERROR_SCALING_RAD = 0.125  # approx. 7.5 deg
    MAX_ALTITUDE_DEVIATION_FT = 1000  # terminate if altitude error exceeds this
    Shaping = enum.Enum.__call__('Shaping', ['OFF', 'BASIC', 'ADDITIVE', 'SEQUENTIAL_CONT',
                                             'SEQUENTIAL_DISCONT'])
    target_heading_deg = BoundedProperty('target/heading-deg', 'desired heading [deg]',
                                         prp.heading_deg.min, prp.heading_deg.max)
    heading_error_deg = BoundedProperty('target/heading-error-deg',
                                        'error to desired heading [deg]', -180, 180)
    altitude_error_ft = BoundedProperty('target/altitude-error-ft',
                                        'error to desired altitude [ft]',
                                        prp.altitude_sl_ft.min,
                                        prp.altitude_sl_ft.max)
    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd)

    def __init__(self, shaping_type: Shaping, step_frequency_hz: float, aircraft: Aircraft,
                 episode_time_s: float = DEFAULT_EPISODE_TIME_S):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        """
        self.max_time_s = episode_time_s
        self.episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.aircraft = aircraft

        self.distance_parallel_m = BoundedProperty('position/dist-parallel-heading-m',
                                                   'distance travelled parallel to target heading [m]',
                                                   0, aircraft.get_max_distance_m(self.max_time_s))
        self.extra_state_variables = (
            self.altitude_error_ft, self.heading_error_deg, self.distance_parallel_m)
        self.state_variables = FlightTask.base_state_variables + self.extra_state_variables
        assessor = self.make_assessor(shaping_type)
        super().__init__(assessor)

    def make_assessor(self, shaping: Shaping) -> assessors.AssessorImpl:
        base_components = self._make_base_reward_components()
        shaping_components = self._make_shaping_components(shaping)
        return self._select_assessor(base_components, shaping_components, shaping)

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        target_altitude = self.base_initial_conditions[prp.initial_altitude_ft]
        base_components = (
            rewards.TerminalComponent('distance_travel', self.distance_parallel_m,
                                      self.state_variables, self.distance_parallel_m.max),
            rewards.StepFractionComponent('altitude_keeping', prp.altitude_sl_ft,
                                          self.state_variables,
                                          target_altitude, self.ALTITUDE_SCALING_FT,
                                          self.episode_steps)
        )
        return base_components

    def _make_shaping_components(self, shaping: Shaping) -> Tuple[rewards.ShapingComponent, ...]:
        distance_shaping = rewards.LinearShapingComponent('dist_travel_shaping',
                                                          self.distance_parallel_m,
                                                          self.state_variables,
                                                          self.distance_parallel_m.max,
                                                          self.distance_parallel_m.max)
        altitude_error = rewards.AsymptoticShapingComponent('altitude_error',
                                                            self.altitude_error_ft,
                                                            self.state_variables,
                                                            0,
                                                            self.ALTITUDE_SCALING_FT)
        if shaping is self.Shaping.OFF:
            shaping_components = ()
        elif shaping is self.Shaping.BASIC:
            shaping_components = (distance_shaping, altitude_error)
        else:
            shaping_components = (
                distance_shaping,
                altitude_error,
                rewards.AsymptoticShapingComponent('heading_error',
                                                   self.heading_error_deg,
                                                   self.state_variables,
                                                   0,
                                                   self.HEADING_ERROR_SCALING_DEG),
                rewards.AsymptoticShapingComponent('wings_level', prp.roll_rad,
                                                   self.state_variables,
                                                   0, self.ROLL_ERROR_SCALING_RAD),
            )

        return shaping_components

    def _select_assessor(self, base_components: Tuple[rewards.RewardComponent, ...],
                         shaping_components: Tuple[rewards.ShapingComponent, ...],
                         shaping: Shaping) -> assessors.AssessorImpl:
        if shaping is self.Shaping.OFF or shaping is self.Shaping.BASIC or shaping is self.Shaping.ADDITIVE:
            return assessors.AssessorImpl(base_components, shaping_components)
        else:
            dist_travel, altitude_error, heading_error, wings_level = shaping_components
            # worry about control in this order: correct altitude, correct heading, wings level,
            #   distance travelled
            dependency_map = {dist_travel: (altitude_error, heading_error, wings_level),
                              wings_level: (heading_error, altitude_error),
                              heading_error: (altitude_error,)}
            if shaping is self.Shaping.SEQUENTIAL_CONT:
                return assessors.ContinuousSequentialAssessor(base_components, shaping_components,
                                                              dependency_map)
            elif shaping is self.Shaping.SEQUENTIAL_DISCONT:
                return assessors.SequentialAssessor(base_components, shaping_components,
                                                    dependency_map)

    def get_initial_conditions(self) -> Dict[Property, float]:
        extra_conditions = {prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
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
        self._update_parallel_distance_travelled(sim)
        self._update_heading_error(sim)
        self._update_altitude_error(sim)

    def _update_parallel_distance_travelled(self, sim: Simulation) -> None:
        """
        Calculates how far aircraft has travelled from initial position parallel to max_target heading

        Stores result in Simulation as custom property.
        """
        current_position = prp.GeodeticPosition.from_sim(sim)
        heading_travelled_deg = self.initial_position.heading_deg_to(current_position)
        target_heading_deg = sim[self.target_heading_deg]
        heading_error_rad = math.radians(heading_travelled_deg - target_heading_deg)

        distance_travelled_m = sim[prp.dist_travel_m]
        parallel_distance_travelled_m = math.cos(heading_error_rad) * distance_travelled_m
        sim[self.distance_parallel_m] = parallel_distance_travelled_m

    def _update_heading_error(self, sim: Simulation):
        heading_deg = sim[prp.heading_deg]
        target_heading_deg = sim[self.target_heading_deg]
        error_deg = utils.reduce_reflex_angle_deg(heading_deg - target_heading_deg)
        sim[self.heading_error_deg] = error_deg

    def _update_altitude_error(self, sim: Simulation):
        altitude_ft = sim[prp.altitude_sl_ft]
        target_altitude_ft = self._get_target_altitude()
        error_ft = altitude_ft - target_altitude_ft
        sim[self.altitude_error_ft] = error_ft

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        episode_time = sim[prp.sim_time_s]
        altitude_error_ft = sim[self.altitude_error_ft]

        over_time = math.isclose(episode_time, self.max_time_s) or episode_time > self.max_time_s
        altitude_out_of_bounds = abs(altitude_error_ft) > self.MAX_ALTITUDE_DEVIATION_FT
        return over_time or altitude_out_of_bounds

    def _new_episode_init(self, sim: Simulation) -> None:
        sim.start_engines()
        sim[prp.throttle_cmd] = self.THROTTLE_CMD
        sim[prp.mixture_cmd] = self.MIXTURE_CMD

        sim[self.target_heading_deg] = self._get_target_heading()
        self.initial_position = prp.GeodeticPosition.from_sim(sim)

    def _get_target_heading(self) -> float:
        # use the same, initial heading every episode
        return self.INITIAL_HEADING_DEG

    def _get_target_altitude(self) -> float:
        return self.INITIAL_ALTITUDE_FT

    def get_props_to_output(self) -> Tuple:
        return (prp.u_fps, prp.altitude_sl_ft, self.altitude_error_ft, prp.heading_deg,
                self.target_heading_deg, self.heading_error_deg, prp.dist_travel_m,
                self.distance_parallel_m, prp.roll_rad)


class TurnHeadingControlTask(HeadingControlTask):
    """
    A task in which the agent must make a turn from a random initial heading,
    and fly level to a random target heading.
    """

    def get_initial_conditions(self) -> [Dict[Property, float]]:
        initial_conditions = super().get_initial_conditions()
        random_heading = random.uniform(prp.heading_deg.min, prp.heading_deg.max)
        initial_conditions[prp.initial_heading_deg] = random_heading
        return initial_conditions

    def _get_target_heading(self) -> float:
        # select a random heading each episode
        return random.uniform(self.target_heading_deg.min,
                              self.target_heading_deg.max)
