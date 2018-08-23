import collections
from gym_jsbsim.tasks import FlightTask
from gym_jsbsim.rewards import State, Reward, RewardComponent
from gym_jsbsim.assessors import Assessor
import gym_jsbsim.properties as prp
from typing import Tuple, NamedTuple, Iterable


class AssessorStub(Assessor):

    def assess(self, state: State, last_state: State, is_terminal: bool) -> Reward:
        base_reward = (0,)
        shaping_reward = ()
        return Reward(base_reward, shaping_reward)


class FlightTaskStub(FlightTask):
    """ A minimal task module for testing. """
    test_property1 = prp.BoundedProperty('test_property1', 'dummy property for testing', -1, 1)
    test_property2 = prp.BoundedProperty('test_property2', 'dummy property for testing', -1, 1)

    def __init__(self):
        self.state_variables = (self.test_property1, self.test_property2)
        self.action_variables = (prp.aileron_cmd, prp.elevator_cmd)
        super().__init__(AssessorStub())

    def _is_terminal(self, state: Tuple[float, ...], episode_time: float) -> bool:
        return False

    def get_initial_conditions(self):
        return self.base_initial_conditions

    def get_state(self, value1: float, value2: float):
        """ Returns a State of this class' test properties populated with input values """
        return self.State(value1, value2)

    def get_dummy_state_and_properties(self, values: Iterable[float]) -> Tuple[
        NamedTuple, Tuple[prp.Property, ...]]:
        """
        given a collection of floats, creates dummy Properties for each value
        and inits a State
        """
        dummy_properties = tuple(prp.Property('test_prop' + str(i), '') for i in range(len(values)))
        DummyState = collections.namedtuple('DummyState', [prop.name for prop in dummy_properties])
        return DummyState(*values), dummy_properties


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


class ConstantRewardComponentStub(RewardComponent):
    """ A RewardComponent which always returns a constant value s"""
    default_return = 1.0

    def __init__(self, return_value: float = default_return):
        self.return_value = return_value

    def calculate(self, _: State, __: State, ___: bool):
        return self.return_value

    def get_return_value(self):
        return self.return_value

    def get_name(self):
        return str(self)
