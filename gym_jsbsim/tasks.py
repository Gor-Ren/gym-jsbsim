import gym
import numpy as np
import random
import types
import gym_jsbsim.properties as prp
from gym_jsbsim.properties import Property, InitialProperty
from gym_jsbsim.simulation import Simulation
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Tuple


class Task(ABC):
    """
    Interface for Tasks, modules implementing specific environments in JSBSim.

    A task defines its own state space, action space, termination conditions and reward function.
    """
    @abstractmethod
    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int)\
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
        Get first state observation from reset sim. Tasks may initialise any book keeping.

        :param sim: Simulation, the environment simulation
        :return: array, the first state observation of the episode
        """
        ...

    @abstractmethod
    def get_initial_conditions(self) -> Optional[Dict[str, float]]:
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
        state_variables attribute: the task's state representation
        action_variables attribute: the task's actions
        get_initial_conditions(): returns dict mapping InitialPropertys to initial values
        _calculate_reward(): determines the step reward
        _is_done(): determines episode termination
        _input_initial_controls(): (optional) sets simulation values at start of episode
    """
    base_state_variables = (prp.altitude_ft, prp.pitch_rad, prp.roll_rad,
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
    state_variables: Tuple[Property, ...]
    action_variables: Tuple[Property, ...]

    def __init__(self, use_shaped_reward: bool=True) -> None:
        """ Constructor

        :param use_shaped_reward: use potential based reward shaping if True
        """
        self.use_shaped_reward = use_shaped_reward
        self.last_reward = None

    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int)\
            -> Tuple[np.ndarray, float, bool, Dict]:
        # input actions
        for prop, command in zip(self.action_variables, action):
            sim[prop.name] = command

        # run simulation
        for _ in range(sim_steps):
            sim.run()

        obs = [sim[prop.name] for prop in self.state_variables]
        reward = self._calculate_reward(sim)
        if self.use_shaped_reward:
            shaped_reward = reward - self.last_reward
            self.last_reward = reward
            reward = shaped_reward
        done = self._is_done(sim)
        info = {'sim_time': sim.get_sim_time()}

        return np.array(obs), reward, done, info

    @abstractmethod
    def _calculate_reward(self, sim: Simulation) -> float:
        """ Calculates the reward from the simulation state.

        :param sim: Simulation, the environment simulation
        :return: a number, the reward for the timestep
        """
        ...

    @abstractmethod
    def _is_done(self, sim: Simulation) -> bool:
        """ Determines whether the current episode should terminate.

        :param sim: Simulation, the environment simulation
        :return: True if the episode should terminate else False
        """
        ...

    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        self._input_initial_controls(sim)
        state = [sim[prop.name] for prop in self.state_variables]
        if self.use_shaped_reward:
            self.last_reward = self._calculate_reward(sim)
        return np.array(state)

    def _input_initial_controls(self, sim: Simulation) -> None:
        """
        This method is called at the start of every episode. It is used to set
        the value of any controls or environment properties not already defined
        in the task's initial conditions.

        By default it simply starts the aircraft engines.
        """
        sim.start_engines()

    @abstractmethod
    def get_initial_conditions(self) -> Dict[InitialProperty, float]:
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
    MIN_ALT_FT = 1000
    TOO_LOW_REWARD = -10
    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    INITIAL_HEADING_DEG = 270

    extra_state_variables = (prp.heading_deg,)
    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd)

    def __init__(self, ):
        super().__init__()
        self.state_variables = super().base_state_variables + self.extra_state_variables
        self.target_values = self._make_target_values()

    def _make_target_values(self):
        """
        Makes a tuple representing the desired state of the aircraft.

        :return: tuple of triples (property, target_value, gain) where:
            property: str, the name of the property in JSBSim
            target_value: number, the desired value to be controlled to
            gain: number, by which the error between actual and target value
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
            ('position/h-sl-ft', start_alt_ft, ALT_GAIN),
            ('velocities/h-dot-fps', 0, ALT_RATE_GAIN),
            ('attitude/roll-rad', 0, ROLL_GAIN),
            ('velocities/phidot-rad_sec', 0, ROLL_RATE_GAIN),
            ('attitude/psi-deg', self.INITIAL_HEADING_DEG, HEADING_GAIN),
            ('velocities/psidot-rad_sec', 0, HEADING_RATE_GAIN),
            # absolute pitch is implicitly controlled through altitude, so just control rate
            ('velocities/thetadot-rad_sec', 0, PITCH_RATE_GAIN)
        )

        return target_values

    def get_initial_conditions(self) -> Dict[InitialProperty, float]:
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

    def _calculate_reward(self, sim: Simulation) -> float:
        reward = 0
        for prop, target_value, gain in self.target_values:
            reward -= abs(target_value - sim[prop]) * gain
        too_low = sim[prp.altitude_ft.name] < self.MIN_ALT_FT
        if too_low:
            reward += self.TOO_LOW_REWARD
        return reward

    def _is_done(self, sim: Simulation) -> bool:
        time_out = sim.get_sim_time() > self.MAX_TIME_SECS
        too_low = sim[prp.altitude_ft.name] < self.MIN_ALT_FT
        return time_out or too_low

    def _input_initial_controls(self, sim: Simulation) -> None:
        # start engines and trims for steady, level flight
        sim.start_engines()
        sim[prp.throttle_cmd.name] = self.THROTTLE_CMD
        sim[prp.mixture_cmd.name] = self.MIXTURE_CMD


class HeadingControlTask(SteadyLevelFlightTask):
    """
    A task in which the agent must make a turn and fly level on a desired heading.
    """
    HEADING_MIN = 0
    HEADING_MAX = 360
    target_heading_deg = Property('target/heading-deg', 'desired heading [deg]',
                                  prp.heading_deg.min, prp.heading_deg.max)
    extra_state_variables = (prp.heading_deg, target_heading_deg)

    def _make_target_values(self):
        """ Sets an attribute specifying the desired state of the aircraft.

        target_values is a tuple of triples of format
            (property, target_value, gain) where:
            property: str, the name of the property in JSBSim
            target_value: number, the desired value to be controlled to
            gain: number, by which the error between actual and target value
                 is multiplied to calculate reward
        """
        super_target_values = super()._make_target_values()
        heading_target_index = 4
        old_heading_triple = super_target_values[heading_target_index]
        random_target_heading = random.uniform(self.HEADING_MIN, self.HEADING_MAX)
        new_heading_triple = (old_heading_triple[0], random_target_heading, old_heading_triple[2])

        target_values = []
        for i, target_triple in enumerate(super_target_values):
            if i != heading_target_index:
                target_values.append(target_triple)
            else:
                target_values.append(new_heading_triple)
        assert len(target_values) == len(super_target_values)
        return target_values

    def get_initial_conditions(self) -> [Dict[InitialProperty, float]]:
        initial_conditions = super().get_initial_conditions()
        random_heading = random.uniform(prp.heading_deg.min, prp.heading_deg.max)
        initial_conditions[prp.initial_heading_deg] = random_heading
        return initial_conditions

    def _input_initial_controls(self, sim: Simulation) -> None:
        sim[self.target_heading_deg.name] = random.uniform(self.target_heading_deg.min,
                                                           self.target_heading_deg.max)
        return super()._input_initial_controls(sim)
