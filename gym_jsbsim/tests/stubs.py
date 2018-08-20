from gym_jsbsim.tasks import FlightTask
import gym_jsbsim.properties as prp
from typing import Optional


class TaskStub(FlightTask):
    """ A minimal task module for testing. """

    def __init__(self):
        super().__init__()
        self.state_variables = super().base_state_variables
        self.action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd, prp.throttle_cmd)

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

    def get_sim_time(self) -> float:
        """ Gets the simulation time from JSBSim, a float. """
        return self[prp.sim_time_s]

    def __setitem__(self, prop, value):
        return super().__setitem__(prop.name, value)

    def __getitem__(self, prop):
        return super().__getitem__(prop.name)

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
        for prop in task.state_variables:
            name = prop.name
            low = prop.min
            high = prop.max
            sim[name] = (low + high) / 2  # take halfway value
        sim[prp.sim_time_s] = 1.0
        return sim

class DefaultSimStub(object):
    """
    A stub for a Sim object which never throws KeyErrors when retrieving
    properties; a default value is always returned instead.
    """
    def __init__(self, default_value=5.0):
        self.default_value = default_value
        self.properties = {}

    def __getitem__(self, prop):
        return self.properties.get(prop.name, self.default_value)

    def __setitem__(self, prop, value):
        self.properties[prop.name] = value

    def run(self):
        pass

    def start_engines(self):
        pass