import gym
import numpy as np
from JsbSimInstance import JsbSimInstance
from typing import Tuple

class JsbSimEnv(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    A JsbSimEnv is instantiated with a TaskModule that implements a specific
    aircraft control task through additional task-related observation/action
    variables and reward calculation.

    The following API methods will be implemented between JsbSimEnv:
        step
        reset
        render
        close
        seed

    Along with the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    ATTRIBUTION: this class is based on the OpenAI Gym Env API. Method
    docstrings have been taken from the OpenAI API and modified where required.
    """
    sim: JsbSimInstance = None
    agent_step_skip: int = None
    observation_space: gym.spaces.Box = None
    action_space: gym.spaces.Box = None
    observation_names: Tuple[str] = None
    action_names: Tuple[str] = None

    def __init__(self, dt: float=1/120, agent_step_skip: int=12):
        """


        :param dt: float, the JSBSim integration timestep in seconds. Defaults
            to 1/120, i.e. 120 Hz
        :param agent_interaction_freq: int, how many JSBSim steps should pass
            between agent observation/action steps.
        """
        self.sim = JsbSimInstance(dt=dt)
        self.agent_step_skip = agent_step_skip
        self.init_spaces()
        # TODO: set self.reward_range

    def init_spaces(self):
        base_state_variables = (
            {'name': 'attitude/pitch-rad',
             'description': 'pitch [rad]',
             'high': 0.5,
             'low': -0.5,},
            {'name': 'attitude/roll-rad',
             'description': 'roll [rad]',
             'high': 1,
             'low': -1,},
            # limits assume pitch and roll have same limits as Euler angles theta and phi,
            #   as per Aircraft Control and Simulation 3rd Edn p. 12
            {'name': 'velocities/u-fps',
             'description': 'body frame x-axis velocity; positive forward [ft/s]',
             'high': 2200,
             'low': -2200,},
            {'name': 'velocities/v-fps',
             'description': 'body frame y-axis velocity; positive right [ft/s]',
             'high': 2200,
             'low': -2200,},
            {'name': 'velocities/w-fps',
             'description': 'body frame z-axis velocity; positive down [ft/s]',
             'high': 2200,
             'low': -2200, },
            # note: limits assume no linear velocity will exceed approx. +- Mach 2
            {'name': 'velocities/p-rad_sec',
             'description': 'roll rate [rad/s]',
             'high': 31,
             'low': -31, },
            {'name': 'velocities/q-rad_sec',
             'description': 'pitch rate [rad/s]',
             'high': 31,
             'low': -31, },
            {'name': 'velocities/r-rad_sec',
             'description': 'yaw rate [rad/s]',
             'high': 31,
             'low': -31, },
            # note: limits assume no angular velocity will exceed ~5 revolution/s
            {'name': 'fcs/left-aileron-pos-norm',
             'description': 'left aileron position, normalised',
             'high': 1,
             'low': -1, },
            {'name': 'fcs/right-aileron-pos-norm',
             'description': 'right aileron position, normalised',
             'high': 1,
             'low': -1, },
            {'name': 'fcs/elevator-pos-norm',
             'description': 'elevator position, normalised',
             'high': 1,
             'low': -1, },
            {'name': 'fcs/rudder-pos-norm',
             'description': 'rudder position, normalised',
             'high': 1,
             'low': -1, },
            {'name': 'fcs/throttle-pos-norm',
             'description': 'throttle position, normalised',
             'high': 1,
             'low': 0, },
        )

        # TODO: merge in TaskModule state vars
        state_variables = base_state_variables + ()

        # TODO: action variables should come from TaskModule
        action_variables = (
            {'name': 'fcs/aileron-cmd-norm',
             'description': 'right aileron position, normalised',
             'high': 1.0,
             'low': -1.0, },
            {'name': 'fcs/elevator-cmd-norm',
             'description': 'elevator position, normalised',
             'high': 1.0,
             'low': -1.0, },
            {'name': 'fcs/rudder-cmd-norm',
             'description': 'rudder position, normalised',
             'high': 1.0,
             'low': -1.0, },
            {'name': 'fcs/throttle-cmd-norm',
             'description': 'throttle position, normalised',
             'high': 1.0,
             'low': 0.0, },
        )

        # create Space objects
        state_lows = np.array([state_var['low'] for state_var in state_variables])
        state_highs = np.array([state_var['high'] for state_var in state_variables])
        self.observation_space = gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

        action_lows = np.array([act_var['low'] for act_var in action_variables])
        action_highs = np.array([act_var['high'] for act_var in action_variables])
        self.action_space = gym.spaces.Box(low=action_lows, high=action_highs, dtype='float')

        # store variable names for getting/setting in the simulation
        self.observation_names = tuple([state_var['name'] for state_var in state_variables])
        self.action_names = tuple([act_var['name'] for act_var in action_variables])

    def step(self, action: np.array):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action: array, the agent's action represented by one value per action variable
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError


    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        raise NotImplementedError


    def render(self, mode='human'):
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
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError


    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return


    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
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