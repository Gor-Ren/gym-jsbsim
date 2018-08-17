import gym
import math
import random
import numpy as np
from gym_jsbsim.simulation import Simulation
import abc
from typing import Optional, Sequence, Dict, Tuple


class Task(abc.ABC):
    """
    Interface for Tasks, modules implementing specific environments in JSBSim.

    A task defines its own state space, action space, termination conditions and reward function.
    """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        """
        Get first state observation from reset sim. Tasks may initialise any book keeping.

        :param sim: Simulation, the environment simulation
        :return: array, the first state observation of the episode
        """
        ...

    @abc.abstractmethod
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

    @abc.abstractmethod
    def get_observation_space(self) -> gym.Space:
        """ Get the task's observation Space object """
        ...

    @abc.abstractmethod
    def get_action_space(self) -> gym.Space:
        """ Get the task's action Space object """
        ...



class FlightTask(Task, abc.ABC):
    """
    Abstract superclass for several flight tasks.


    """
    task_state_variables = None  # should be specified by concrete implementations
    base_state_variables = (
        dict(name='position/h-sl-ft', description='altitude above mean sea level [ft]',
             high=85000, low=-1400),
        # altitude limits max 85 kft (highest an SR-71 Blackbird got to)
        #   and min of Black Sea
        dict(name='attitude/pitch-rad', description='pitch [rad]',
             high=0.5 * math.pi, low=-0.5 * math.pi),
        dict(name='attitude/roll-rad', description='roll [rad]',
             high=math.pi, low=-math.pi),
        # limits assume pitch and roll have same limits as Euler angles theta and phi,
        #   as per Aircraft Control and Simulation 3rd Edn p. 12
        dict(name='velocities/u-fps', description='body frame x-axis velocity; positive forward [ft/s]',
             high=2200, low=-2200),
        dict(name='velocities/v-fps', description='body frame y-axis velocity; positive right [ft/s]',
             high=2200, low=-2200),
        dict(name='velocities/w-fps',
             description='body frame z-axis velocity; positive down [ft/s]',
             high=2200, low=-2200),
        # note: limits assume no linear velocity will exceed approx. +- Mach 2
        dict(name='velocities/p-rad_sec', description='roll rate [rad/s]',
             high=2 * math.pi, low=-2 * math.pi),
        dict(name='velocities/q-rad_sec', description='pitch rate [rad/s]',
             high=2 * math.pi, low=-2 * math.pi),
        dict(name='velocities/r-rad_sec', description='yaw rate [rad/s]',
             high=2 * math.pi, low=-2 * math.pi),
        # note: limits assume no angular velocity will exceed ~1 revolution/s
        dict(name='fcs/left-aileron-pos-norm', description='left aileron position, normalised',
             high=1, low=-1),
        dict(name='fcs/right-aileron-pos-norm', description='right aileron position, normalised',
             high=1, low=-1),
        dict(name='fcs/elevator-pos-norm', description='elevator position, normalised',
             high=1, low=-1),
        dict(name='fcs/rudder-pos-norm', description='rudder position, normalised',
             high=1, low=-1),
        dict(name='fcs/throttle-pos-norm', description='throttle position, normalised',
             high=1, low=0),
    )
    base_action_variables = (
        dict(name='fcs/aileron-cmd-norm', description='aileron commanded position, normalised',
             high=1.0, low=-1.0),
        dict(name='fcs/elevator-cmd-norm', description='elevator commanded position, normalised',
             high=1.0, low=-1.0),
        dict(name='fcs/rudder-cmd-norm', description='rudder commanded position, normalised',
             high=1.0, low=-1.0),
        dict(name='fcs/throttle-cmd-norm', description='throttle commanded position, normalised',
             high=1.0, low=0.0),
    )
    base_initial_conditions = {'ic/h-sl-ft': 5000,
                               'ic/terrain-elevation-ft': 0.00000001,
                               'ic/long-gc-deg': -2.3273,
                               'ic/lat-geod-deg': 51.3781,  # corresponds to UoBath
                               }

    def __init__(self, task_name: Optional[str], use_shaped_reward: bool=True):
        """ Constructor

        :param task_name: str, the name of the task
        :param use_shaped_reward: use potential based reward shaping if True
        """
        self.task_name = task_name
        self.state_variables = self.get_full_state_variables()
        self.action_variables = self.get_full_action_variables()
        self.state_names = tuple([state_var['name'] for state_var in self.state_variables])
        self.action_names = tuple([act_var['name'] for act_var in self.action_variables])
        self.use_shaped_reward = use_shaped_reward
        self.last_reward = None

    def __repr__(self):
        return f'<FlightTask {self.task_name}>'

    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) -> Tuple:
        # input actions
        for var, command in zip(self.action_names, action):
            sim[var] = command

        # run simulation
        for _ in range(sim_steps):
            sim.run()

        obs = [sim[prop] for prop in self.state_names]
        reward = self._calculate_reward(sim)
        if self.use_shaped_reward:
            shaped_reward = reward - self.last_reward
            self.last_reward = reward
            reward = shaped_reward
        done = self._is_done(sim)
        info = {'sim_time': sim.get_sim_time()}

        return np.array(obs), reward, done, info

    def _calculate_reward(self, sim: Simulation):
        """ Calculates the reward from the simulation state.

        :param sim: Simulation, the environment simulation
        :return: a number, the reward for the timestep
        """
        raise NotImplementedError

    def _is_done(self, sim: Simulation):
        """ Determines whether the current episode should terminate.

        :param sim: Simulation, the environment simulation
        :return: True if the episode should terminate else False
        """
        raise NotImplementedError

    def get_full_state_variables(self):
        """ Returns tuple of information on all of the task's state variables.

        Each state variable is defined by a dict with the following entries:
            'source': 'jsbsim' or 'task', where to get the variable's value.
                If 'jsbsim', value is retrieved from JSBSim using 'name'. If
                'function', a function which will calculate it when input the
                last observation.
            'name': str, the property name of the variable in JSBSim, e.g.
                "velocities/u-fps"
            'description': str, a description of what the variable represents,
                and its unit, e.g. "body frame x-axis velocity [ft/s]"
            'high': number, the upper range of this variable, or +inf if
                unbounded
            'low': number, the lower range of this variable, or -inf

        An environment may have some default state variables which are commonly
        used; these are input as base_state_vars matching the same format. The
        task may choose to omit some or all of these base variables.

        The order of variables in the returned tuple corresponds to their order
        in the observation array extracted from the environment.

        :return: tuple of dicts, each dict having a 'source', 'name',
            'description', 'high' and 'low' value
        """
        if self.task_state_variables is None:
            raise NotImplementedError(f'task {self} has not defined task '
                                      'state variables')
        else:
            return self.base_state_variables + self.task_state_variables

    def get_full_action_variables(self):
        """ Returns collection of all task's action variables.

        Each action variable is defined by a dict with the following entries:
            'name': str, the property name of the variable in JSBSim, e.g.
                "fcs/aileron-cmd-norm"
            'description': str, a description of what the variable represents,
                and its unit, e.g. "aileron commanded position, normalised"
            'high': number, the upper range of this variable, or +inf if
                unbounded
            'low': number, the lower range of this variable, or -inf

        An environment may have some default action variables which are commonly
        used. The default behaviour is for the task to use all of these.
        Alternative behaviour can be achieved by overriding this method.

        The order of variables in the returned tuple corresponds to their order
        in the action array passed to the environment by the agent.

        :return: tuple of dicts, each dict having a 'name',
            'description', 'high' and 'low' value
        """
        return self.base_action_variables

    def get_initial_conditions(self) -> Optional[Dict[str, float]]:
        raise NotImplementedError()

    def get_observation_space(self):
        state_lows = np.array([state_var['low'] for state_var in self.state_variables])
        state_highs = np.array([state_var['high'] for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    def get_action_space(self):
        action_lows = np.array([act_var['low'] for act_var in self.action_variables])
        action_highs = np.array([act_var['high'] for act_var in self.action_variables])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float')

    def observe_first_state(self, sim: Simulation):
        self._input_initial_controls(sim)
        state = [sim[prop] for prop in self.state_names]
        if self.use_shaped_reward:
            self.last_reward = self._calculate_reward(sim)
        return np.array(state)

    def _input_initial_controls(self, sim: Simulation):
        """
        This method is called at the start of every episode. It is used to set
        the value of any controls or environment properties not already defined
        in the task's initial conditions.

        By default it simply starts the aircraft engines.
        """
        sim.start_engines()


class SteadyLevelFlightTask(FlightTask):
    """
    A task in which the agent must perform steady, level flight maintaining its
    current heading.
    """
    task_state_variables = (dict(name='attitude/psi-deg',
                                 description='heading [deg]',
                                 high=360, low=0),
                            )
    MAX_TIME_SECS = 15
    MIN_ALT_FT = 1000
    TOO_LOW_REWARD = -10
    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    INITIAL_HEADING_DEG = 270

    def __init__(self, task_name='SteadyLevelFlightTask-v2'):
        super().__init__(task_name)
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

        initial_altitude_ft = self.get_initial_conditions()['ic/h-sl-ft']
        target_values = (
            ('position/h-sl-ft', initial_altitude_ft, ALT_GAIN),
            ('velocities/h-dot-fps', 0, ALT_RATE_GAIN),
            ('attitude/roll-rad', 0, ROLL_GAIN),
            ('velocities/phidot-rad_sec', 0, ROLL_RATE_GAIN),
            ('attitude/psi-deg', self.INITIAL_HEADING_DEG, HEADING_GAIN),
            ('velocities/psidot-rad_sec', 0, HEADING_RATE_GAIN),
            # absolute pitch is implicitly controlled throught altitude, so just control rate
            ('velocities/thetadot-rad_sec', 0, PITCH_RATE_GAIN)
        )

        return target_values

    def get_initial_conditions(self) -> Optional[Dict[str, float]]:
        """ Returns dictionary mapping initial episode conditions to values.

        The aircraft is initialised at a random orientation and velocity.

        :return: dict mapping string for each initial condition property to
            a float, or None to use Env defaults
        """
        initial_conditions = {'ic/u-fps': 150,
                              'ic/v-fps': 0,
                              'ic/w-fps': 0,
                              'ic/p-rad_sec': 0,
                              'ic/q-rad_sec': 0,
                              'ic/r-rad_sec': 0,
                              'ic/roc-fpm': 0,  # rate of climb
                              'ic/psi-true-deg': self.INITIAL_HEADING_DEG,  # heading
                              }
        return {**self.base_initial_conditions, **initial_conditions}

    def get_full_action_variables(self):
        """ Returns information defining all action variables for this task.

        For steady level flight the agent controls ailerons and elevator.
        Throttle and rudder are set in the initial conditions and maintained at a
        constant value.

        :return: tuple of dicts, each dict having a 'source', 'name',
            'description', 'high' and 'low' key
        """
        all_action_vars = super().get_full_action_variables()
        desired_action_var_names = ['fcs/aileron-cmd-norm',
                                    'fcs/elevator-cmd-norm',
                                    'fcs/rudder-cmd-norm']
        action_vars = tuple(var for var in all_action_vars if var['name'] in desired_action_var_names)
        assert len(action_vars) == len(desired_action_var_names)
        return action_vars

    def _calculate_reward(self, sim: Simulation):
        """ Calculates the reward from the simulation state.

        For this task the agent is required to maintain its initial altitude
        and heading.

        :param sim: Simulation, the environment simulation
        :return: a number, the reward for the timestep
        """
        reward = 0
        for prop, target_value, gain in self.target_values:
            reward -= abs(target_value - sim[prop]) * gain
        too_low = sim['position/h-sl-ft'] < self.MIN_ALT_FT
        if too_low:
            reward += self.TOO_LOW_REWARD
        return reward

    def _is_done(self, sim: Simulation):
        """ Determines whether the current episode should terminate.

        :param sim: Simulation, the environment simulation
        :return: True if the episode should terminate else False
        """
        time_out = sim['simulation/sim-time-sec'] > self.MAX_TIME_SECS
        too_low = sim['position/h-sl-ft'] < self.MIN_ALT_FT
        return time_out or too_low

    def _input_initial_controls(self, sim: Simulation):
        """ Sets control inputs for start of episode.

        :param sim: Simulation, the environment simulation
        """
        # start engines and trims for steady, level flight
        sim.start_engines()
        sim['fcs/throttle-cmd-norm'] = self.THROTTLE_CMD
        sim['fcs/mixture-cmd-norm'] = self.MIXTURE_CMD


class HeadingControlTask(SteadyLevelFlightTask):
    """
    A task in which the agent must make a turn and fly level on a desired heading.
    """
    HEADING_MIN = 0
    HEADING_MAX = 360
    task_state_variables = (dict(name='attitude/psi-deg',
                                 description='heading [deg]',
                                 low=HEADING_MIN, high=HEADING_MAX),
                            dict(name='target/heading-deg',
                                 description='desired heading [deg]',
                                 low=HEADING_MIN, high=HEADING_MAX)
                            )

    def __init__(self, task_name='HeadingControlTask-v0'):
        super().__init__(task_name)

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

    def get_initial_conditions(self) -> Optional[Dict[str, float]]:
        """ Returns dictionary mapping initial episode conditions to values.

        The aircraft is initialised at a random orientation and velocity.

        :return: dict mapping string for each initial condition property to
            a float, or None to use Env defaults
        """
        initial_conditions = super().get_initial_conditions()
        random_heading = random.uniform(self.HEADING_MIN, self.HEADING_MAX)
        initial_conditions['ic/psi-true-deg'] = random_heading
        return initial_conditions

    def _input_initial_controls(self, sim: Simulation):
        sim['target/heading-deg'] = random.uniform(self.HEADING_MIN, self.HEADING_MAX)
        return super()._input_initial_controls(sim)
