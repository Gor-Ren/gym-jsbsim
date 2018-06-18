import jsbsim
import os
import gym


root_dir = os.path.abspath("/home/gordon/Apps/jsbsim-code")

fdm = jsbsim.FGFDMExec(root_dir=root_dir)

print(fdm)

class JsbSimEnv(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    This class contains common functionality, such as
    instantiating the FDM and exchanging actions and observations. A JsbSimEnv
    is instantiated with a TaskModule that implements a specific
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
    base_actions = 0
    base_observations = 7  # TODO: decide observation space

    def __init__(self, task_module: 'TaskModule'):
        self.task_module = task_module


class TaskModule(object):
    """
    A class encapsulating task-specific elements of a JSBSim environment. A
    TaskModule contains information on the task to be carried out, including:
        the task name
        additional state and action variables
        a method for calculating the task reward
        a method for rendering task objectives
    """
    def __init__(self, task_name: str):
        self.task_name = task_name

    def reset_task(self):
        pass

    def get_obs_space(self):
        pass

    def get_action_space(self):
        pass

    def render_task(self):
        pass



