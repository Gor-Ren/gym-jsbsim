import unittest
import collections
from typing import Type, NamedTuple
from gym_jsbsim.assessors import AssessorImpl, ContinuousSequentialAssessor, \
    DiscontinuousSequentialAssessor
from gym_jsbsim.tests import stubs as stubs


class TestAssessorImpl(unittest.TestCase):

    def get_class_under_test(self):
        return AssessorImpl

    def get_assessor(self, *args, **kwargs):
        assessor_class = self.get_class_under_test()
        return assessor_class(*args, **kwargs)

    @staticmethod
    def get_dummy_state_class() -> Type[NamedTuple]:
        return collections.namedtuple('State', ['test_prop1', 'test_prop2'])

    def get_dummy_state(self, value1: float = 0.0, value2: float = 1.0):
        State = self.get_dummy_state_class()
        return State(value1, value2)

    def test_init_throws_error_on_empty_base_components(self):
        base_components = ()

        with self.assertRaises(ValueError):
            _ = self.get_assessor(base_components)

    def test_init_throws_error_on_empty_base_components_non_empty_shaping_components(self):
        base_components = ()
        shaping_components = (stubs.ConstantRewardComponentStub(),)

        with self.assertRaises(ValueError):
            _ = self.get_assessor(base_components, shaping_components)

    def test_calculate_single_base_component(self):
        component = stubs.ConstantRewardComponentStub()
        assessor = self.get_assessor(base_components=(component,))
        state = self.get_dummy_state()

        reward = assessor.assess(state, state, True)

        expected_shaping_reward = component.get_return_value()
        expected_non_shaping_reward = expected_shaping_reward  # should be same because not shaping
        self.assertAlmostEqual(expected_shaping_reward, reward.agent_reward())
        self.assertAlmostEqual(expected_non_shaping_reward, reward.assessment_reward())

    def test_calculate_multiple_base_components(self):
        reward_values = [1, 2, 4]
        components = tuple(stubs.ConstantRewardComponentStub(val) for val in reward_values)
        assessor = self.get_assessor(components)
        state = self.get_dummy_state_class()

        reward = assessor.assess(state, state, True)

        expected_shaping_reward = sum(reward_values) / len(reward_values)
        expected_non_shaping_reward = expected_shaping_reward  # should be same because not shaping
        self.assertAlmostEqual(expected_shaping_reward, reward.agent_reward())
        self.assertAlmostEqual(expected_non_shaping_reward, reward.assessment_reward())

    def test_calculate_with_shaping_components(self):
        base_reward_vals = [0, 1]
        shaping_reward_vals = [2, 3]
        base_components = tuple(stubs.ConstantRewardComponentStub(val) for val in base_reward_vals)
        shape_components = tuple(
            stubs.ConstantRewardComponentStub(val) for val in shaping_reward_vals)
        assessor = self.get_assessor(base_components, shape_components)
        state = self.get_dummy_state()

        reward = assessor.assess(state, state, True)

        expected_shaping_reward = (sum(base_reward_vals + shaping_reward_vals) /
                                   len(base_reward_vals + shaping_reward_vals))
        expected_non_shaping_reward = sum(base_reward_vals) / len(base_reward_vals)
        self.assertAlmostEqual(expected_shaping_reward, reward.agent_reward())
        self.assertAlmostEqual(expected_non_shaping_reward, reward.assessment_reward())


class TestContinuousSequentialAssessor(TestAssessorImpl):

    def get_class_under_test(self):
        return ContinuousSequentialAssessor

    def test_calculate_with_shaping_components(self):
        num_state_vars = 3
        DummyState, props = stubs.FlightTaskStub.get_dummy_state_class_and_properties(
            num_state_vars)
        base_reward = 0
        base_component = stubs.ConstantRewardComponentStub(0)

        # create two states with a component that will recognise as low and high potential resp.
        state_low_potential = DummyState(*(1.0 for _ in range(num_state_vars)))
        state_high_potential = DummyState(*(2.0 for _ in range(num_state_vars)))
        low_potential, high_potential = 0.5, 1.0
        potential_map = {state_low_potential: low_potential,
                         state_high_potential: high_potential}
        shape_component = stubs.PotentialComponentStub(potential_map)

        assessor = self.get_assessor((base_component,), (shape_component,))

        # if non-terminal, expect to see reward equal to potential increase
        terminal = False
        reward = assessor.assess(state_high_potential, state_low_potential, terminal)

        expected_shaping_reward = (base_reward + (high_potential - low_potential)) / 2
        expected_non_shaping_reward = base_reward

        self.assertAlmostEqual(expected_shaping_reward, reward.agent_reward())
        self.assertAlmostEqual(expected_non_shaping_reward, reward.assessment_reward())

        # if terminal, expect to see reward as if terminal step potential was zero
        terminal = True
        terminal_potential = 0.0
        reward = assessor.assess(state_high_potential, state_low_potential, terminal)

        expected_shaping_reward = (base_reward + (terminal_potential - low_potential)) / 2
        expected_non_shaping_reward = base_reward

        self.assertAlmostEqual(expected_shaping_reward, reward.agent_reward())
        self.assertAlmostEqual(expected_non_shaping_reward, reward.assessment_reward())

    def assess_reward_for_potential_change_with_dependency(self,
                                                           state_potential: float,
                                                           prev_state_potential: float,
                                                           dependency_potential: float):
        """
        Calculates the reward given we transition from prev_state_potential to
        state_potential, and a dependant component unchanged at dependency_potential.

        Step is non-terminal and base reward is zero.
        """
        num_state_vars = 3
        terminal = False

        DummyState, props = stubs.FlightTaskStub.get_dummy_state_class_and_properties(
            num_state_vars)
        base_component = stubs.ConstantRewardComponentStub(0)
        # create two states with a component that will recognise as low and high potential resp.
        state = DummyState(*(1.0 for _ in range(num_state_vars)))
        prev_state = DummyState(*(2.0 for _ in range(num_state_vars)))
        potential_map = {state: state_potential,
                         prev_state: prev_state_potential}

        # make components
        shape_component = stubs.PotentialComponentStub(potential_map)
        dependency_potential_map = {key: dependency_potential for key in potential_map}
        dependant_shape_component = stubs.PotentialComponentStub(dependency_potential_map)

        dependency_map = {shape_component: (dependant_shape_component,)}

        assessor = self.get_assessor((base_component,),
                                     (shape_component, dependant_shape_component),
                                     dependency_map)

        return assessor.assess(state, prev_state, terminal)

    def test_calculate_with_shaping_components_and_dependency_at_zero(self):
        low_potential = 0.5
        high_potential = 1.0
        dependent_potential = 0.0
        reward = self.assess_reward_for_potential_change_with_dependency(low_potential,
                                                                         high_potential,
                                                                         dependent_potential)

        # we expect to have had a potential improvement, but because dependent component is at
        #   zero potential, no reward is given
        expected_reward = 0.0

        self.assertAlmostEqual(expected_reward, reward.agent_reward())

    def test_calculate_with_shaping_components_and_dependency_at_fraction(self):
        low_potential = 0.5
        high_potential = 1.0
        dependent_potential = 0.75
        reward = self.assess_reward_for_potential_change_with_dependency(high_potential,
                                                                         low_potential,
                                                                         dependent_potential)

        total_reward_values = (high_potential - low_potential) * dependent_potential
        # there are 3 components (incl. the dependant component and base
        #   component with value 0) so divide
        averaged_reward_values = total_reward_values / (
                len(reward.shaping_reward_elements) + len(reward.base_reward_elements))

        self.assertAlmostEqual(averaged_reward_values, reward.agent_reward())

    def test_calculate_with_shaping_components_and_dependency_at_one(self):
        low_potential = 0.5
        high_potential = 1.0
        dependent_potential = 1.0
        reward = self.assess_reward_for_potential_change_with_dependency(high_potential,
                                                                         low_potential,
                                                                         dependent_potential)

        total_reward_values = (high_potential - low_potential) * dependent_potential
        # there are 3 components (incl. the dependant component and base
        #   component with value 0) so divide
        averaged_reward_values = total_reward_values / (
                len(reward.shaping_reward_elements) + len(reward.base_reward_elements))

        self.assertAlmostEqual(averaged_reward_values, reward.agent_reward())
