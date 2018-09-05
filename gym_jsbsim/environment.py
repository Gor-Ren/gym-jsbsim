import gym
import numpy as np
from gym_jsbsim.tasks import Shaping, HeadingControlTask
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.visualiser import FigureVisualiser, FlightGearVisualiser
from gym_jsbsim.aircraft import Aircraft, cessna172P
from typing import Type, Tuple, Dict


class JsbSimEnv(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An JsbSimEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.

    ATTRIBUTION: this class implements the OpenAI Gym Env API. Method
    docstrings have been adapted or copied from the OpenAI Gym source code.
    """
    JSBSIM_DT_HZ: int = 60  # JSBSim integration frequency
    metadata = {'render.modes': ['human', 'flightgear']}

    def __init__(self, task_type: Type[HeadingControlTask], aircraft: Aircraft = cessna172P,
                 agent_interaction_freq: int = 5, shaping: Shaping=Shaping.STANDARD):
        """
        Constructor. Inits some internal state, but JsbSimEnv.reset() must be
        called first before interacting with environment.

        :param task_type: the Task subclass for the task agent is to perform
        :param aircraft: the JSBSim aircraft to be used
        :param agent_interaction_freq: int, how many times per second the agent
            should interact with environment.
        :param shaping: a HeadingControlTask.Shaping enum, what type of agent_reward
            shaping to use (see HeadingControlTask for options)
        """
        if agent_interaction_freq > self.JSBSIM_DT_HZ:
            raise ValueError('agent interaction frequency must be less than '
                             'or equal to JSBSim integration frequency of '
                             f'{self.JSBSIM_DT_HZ} Hz.')
        self.sim: Simulation = None
        self.sim_steps_per_agent_step: int = self.JSBSIM_DT_HZ // agent_interaction_freq
        self.aircraft = aircraft
        self.task = task_type(shaping, agent_interaction_freq, aircraft)
        # set Space objects
        self.observation_space: gym.spaces.Box = self.task.get_state_space()
        self.action_space: gym.spaces.Box = self.task.get_action_space()
        # set visualisation objects
        self.figure_visualiser: FigureVisualiser = None
        self.flightgear_visualiser: FlightGearVisualiser = None
        self.step_delay = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        :param action: the agent's action, with same length as action variables.
        :return:
            state: agent's observation of the current environment
            reward: amount of reward returned after previous action
            done: whether the episode has ended, in which case further step() calls are undefined
            info: auxiliary information, e.g. full reward shaping data
        """
        if not (action.shape == self.action_space.shape):
            raise ValueError('mismatch between action and action space size')

        state, reward, done, info = self.task.task_step(self.sim, action, self.sim_steps_per_agent_step)
        return np.array(state), reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        :return: array, the initial observation of the space.
        """
        init_conditions = self.task.get_initial_conditions()
        if self.sim:
            self.sim.reinitialise(init_conditions)
        else:
            self.sim = self._init_new_sim(self.JSBSIM_DT_HZ, self.aircraft, init_conditions)

        state = self.task.observe_first_state(self.sim)

        if self.flightgear_visualiser:
            self.flightgear_visualiser.configure_simulation_output(self.sim)

        return np.array(state)

    def _init_new_sim(self, dt, aircraft, initial_conditions):
        return Simulation(sim_frequency_hz=dt,
                          aircraft=aircraft,
                          init_conditions=initial_conditions)

    def render(self, mode='flightgear', flightgear_blocking=True):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        :param mode: str, the mode to render with
        :param flightgear_blocking: waits for FlightGear to load before
            returning if True, else returns immediately
        """
        if mode == 'human':
            if not self.figure_visualiser:
                self.figure_visualiser = FigureVisualiser(self.sim,
                                                          self.task.get_props_to_output())
            self.figure_visualiser.plot(self.sim)
        elif mode == 'flightgear':
            if not self.flightgear_visualiser:
                self.flightgear_visualiser = FlightGearVisualiser(self.sim,
                                                                  self.task.get_props_to_output(),
                                                                  flightgear_blocking)
            self.flightgear_visualiser.plot(self.sim)
        else:
            super().render(mode=mode)

    def close(self):
        """ Cleans up this environment's objects

        Environments automatically close() when garbage collected or when the
        program exits.
        """
        if self.sim:
            self.sim.close()
        if self.figure_visualiser:
            self.figure_visualiser.close()
        if self.flightgear_visualiser:
            self.flightgear_visualiser.close()

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        gym.logger.warn("Could not seed environment %s", self)
        return


class NoFGJsbSimEnv(JsbSimEnv):
    """
    An RL environment for JSBSim with rendering to FlightGear disabled.

    This class exists to be used for training agents where visualisation is not
    required. Otherwise, restrictions in JSBSim output initialisation cause it
    to open a new socket for every single episode, eventually leading to
    failure of the network.
    """
    metadata = {'render.modes': ['human']}

    def _init_new_sim(self, dt: float, aircraft: Aircraft, initial_conditions: Dict):
        return Simulation(sim_frequency_hz=dt,
                          aircraft=aircraft,
                          init_conditions=initial_conditions,
                          allow_flightgear_output=False)

    def render(self, mode='human', flightgear_blocking=True):
        if mode == 'flightgear':
            raise ValueError('flightgear rendering is disabled for this class')
        else:
            super().render(mode, flightgear_blocking)
