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
    """ A minimal Task for testing. """
    test_property1 = prp.BoundedProperty('test_property1', 'dummy property for testing', -1, 1)
    test_property2 = prp.BoundedProperty('test_property2', 'dummy property for testing', -1, 1)

    def __init__(self, *_):
        self.state_variables = (self.test_property1, self.test_property2)
        self.action_variables = (prp.aileron_cmd, prp.elevator_cmd)
        super().__init__(AssessorStub())

    def _is_terminal(self, _: Tuple[float, ...], __: float) -> bool:
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


class BasicFlightTask(FlightTask):
    """ A Task with basic but realistic state and action space. """
    def __init__(self, *_):
        self.state_variables = super().base_state_variables
        self.action_variables = (prp.aileron_cmd, prp.rudder_cmd, prp.elevator_cmd)
        super().__init__(AssessorStub())

    def _is_terminal(self, _: Tuple[float, ...], __: float) -> bool:
        return False

    def get_initial_conditions(self):
        return self.base_initial_conditions


class SimStub(dict):

    def run(self):
        pass

    def start_engines(self):
        self[prp.engine_running] = 1.0

    def get_sim_time(self) -> float:
        """ Gets the simulation time from JSBSim, a float. """
        return self[prp.sim_time_s]

    def __setitem__(self, prop: prp.Property, value: float):
        return super().__setitem__(prop.name, value)

    def __getitem__(self, prop: prp.Property) -> float:
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
        for initial_prop, value in task.get_initial_conditions().items():
            sim[initial_prop] = value
        for prop in task.state_variables:
            typical_value = (prop.min + prop.max) / 2
            sim[prop] = typical_value
        sim[prp.sim_time_s] = 1.0
        sim[prp.lat_geod_deg] = 33.3
        sim[prp.lng_geoc_deg] = 44.4
        sim[prp.dist_travel_m] = 2.0
        return sim


class TransitioningSimStub(object):
    """
    A dummy Simulation object which holds two SimStub states, which it
    transitions between when run() or reset().
    """

    def __init__(self, initial_sim: SimStub, next_sim: SimStub):
        self.initial_sim = initial_sim
        self.next_sim = next_sim
        self.current_sim = self.initial_sim

    def reset(self):
        self.current_sim = self.initial_sim

    def run(self):
        self.current_sim = self.next_sim

    def __setitem__(self, prop: prp.Property, value: float):
        self.current_sim[prop] = value

    def __getitem__(self, prop: prp.Property) -> float:
        return self.current_sim[prop]

    def start_engines():
        self[prp.engine_running] = 1.0


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
    """ A RewardComponent which always returns a constant value """
    default_return = 1.0

    def __init__(self, return_value: float = default_return):
        self.return_value = return_value

    def calculate(self, _: State, __: State, ___: bool):
        return self.return_value

    def get_return_value(self):
        return self.return_value

    def get_name(self):
        return str(self)
