from gym_jsbsim.tasks import TaskModule
from typing import Optional


class TaskStub(TaskModule):
    """ A minimal task module for testing. """
    task_state_variables = ()

    def __init__(self, task_name: Optional[str]='TaskStub'):
        super().__init__(task_name)

    def _calculate_reward(self, _):
        return 0

    def _is_done(self, _):
        return False

    def get_initial_conditions(self):
        return None


class SimStub(dict):

    def run(self):
        pass

    def start_engines(self):
        pass

    def trim(self, _):
        self['fcs/pitch-trim-cmd-norm'] = 0.01
        self['fcs/elevator-cmd-norm'] = 0.0

    def get_sim_time(self) -> float:
        """ Gets the simulation time from JSBSim, a float. """
        return self['simulation/sim-time-sec']

    @staticmethod
    def make_valid_state_stub(task: TaskModule):
        """
        A factory method for SimStubs with valid state values for a task.

        Returns a SimStub containing keys of the names of state properties
        mapped to valid values at the midpoint of the allowable range of
        each property.

        :param task: a TypeModule, the task the SimStub should conform to
        :return: a SimStub configured with valid state for the task
        """
        sim = SimStub()
        for prop in task.get_full_state_variables():
            name = prop['name']
            low = prop['low']
            high = prop['high']
            sim[name] = (low + high) / 2  # take halfway value
        sim['simulation/sim-time-sec'] = 1.0
        return sim
