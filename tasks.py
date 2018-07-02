import gym
from abc import ABC
from typing import Optional, Sequence, Mapping, Dict, Tuple, Any, Union
Num = Union[int, float]  # a useful type


class TaskModule(ABC):
    task_name = None
    seed = None

    def __init__(self, task_name: str, seed: Optional[int]=None):
        """ Constructor """
        self.task_name = task_name
        self.seed = seed

    def __repr__(self):
        return f'<TaskModule {self.task_name}>'

    def task_step(self, observation) -> Tuple:
        """ Calculates step reward and termination from an agent observation.

        :param observation: list of floats, the agent's state
        :return: tuple of (reward, done) where,
            reward: float, the reward for that step
            done: bool, True if the episode is over else False
        """
        raise NotImplementedError()

    def get_task_state_variables(self, base_state_vars):
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

        :param base_state_vars: tuple of dicts, the default state variables
        :return: tuple of dicts, each dict having a 'source', 'name',
            'description', 'high' and 'low' value
        """
        raise NotImplementedError()

    def get_task_action_variables(self, base_action_vars):
        """ Returns collection of task-specific action variables.

        Each action variable is defined by a dict with the following entries:
            'source': 'jsbsim' or 'task', where to get the variable's value.
                If 'jsbsim', value is retrieved from JSBSim using 'name'. If
                'function', a function which will calculate it when input the
                last observation.
            'name': str, the property name of the variable in JSBSim, e.g.
                "fcs/aileron-cmd-norm"
            'description': str, a description of what the variable represents,
                and its unit, e.g. "aileron commanded position, normalised"
            'high': number, the upper range of this variable, or +inf if
                unbounded
            'low': number, the lower range of this variable, or -inf

        An environment may have some default action variables which are commonly
        used; these are input as base_action_vars matching the same format. The
        task may choose to omit some or all of these base variables.

        The order of variables in the returned tuple corresponds to their order
        in the action array passed to the environment by the agent.

        :param base_action_vars: tuple of dicts, the default action variables
        :return: tuple of dicts, each dict having a 'source', 'name',
            'description', 'high' and 'low' value
        """
        raise NotImplementedError()

    def get_initial_conditions(self) -> Optional[Dict[str, float]]:
        """ Returns dictionary mapping initial episode conditions to values.

        Episode initial conditions (ICs) are defined by specifying values for
        JSBSim properties, represented by their name (string) in JSBSim.

        JSBSim uses a distinct set of properties for ICs, beginning with 'ic/'
        which differ from property names during the simulation, e.g. "ic/u-fps"
        instead of "velocities/u-fps"

        JSBSim properties can be found in its C++ API documentation, see
        https://jsbsim-team.github.io/jsbsim/

        :return: dict mapping string for each initial condition property to
            initial value, a float
        """
        gym.logger.warn('Task did not provide set of ICs; using default.')
        return None


class DummyTask(TaskModule):
    def task_step(self, observation: Tuple) -> Tuple:
        return 0, True

    def get_task_state_variables(self, base_state_vars):
        return base_state_vars

    def get_task_action_variables(self, base_action_vars):
        return base_action_vars

    def __init__(self):
        super().__init__('DummyTask')
