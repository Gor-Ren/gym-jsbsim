from gym_jsbsim.tasks import FlightTask
from gym_jsbsim.properties import aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd
from typing import Optional


class TaskStub(FlightTask):
    """ A minimal task module for testing. """

    def __init__(self):
        super().__init__()
        self.state_variables = super().base_state_variables
        self.action_variables = (aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd)

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
    def make_valid_state_stub(task: FlightTask):
        """
        A factory method for SimStubs with valid state values for a task.

        Returns a SimStub containing keys of the names of state properties
        mapped to valid values at the midpoint of the allowable range of
        each property.

        :param task: a TypeModule, the task the SimStub should conform to
        :return: a SimStub configured with valid state for the task
        """
        sim = SimStub()
        for prop in task.get_state_variables():
            name = prop['name']
            low = prop['low']
            high = prop['high']
            sim[name] = (low + high) / 2  # take halfway value
        sim['simulation/sim-time-sec'] = 1.0
        return sim

class DefaultSimStub(object):
    """
    A stub for a Sim object which never throws KeyErrors when retrieving
    properties; a default value is always returned instead.
    """
    def __init__(self, default_value=5.0):
        self.default_value = default_value
        self.properties = {}

    def __getitem__(self, item):
        return self.properties.get(item, self.default_value)

    def run(self):
        pass

    def start_engines(self):
        pass