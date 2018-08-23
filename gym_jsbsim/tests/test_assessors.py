import unittest
from collections import __init__
from typing import Type, NamedTuple

from gym_jsbsim.assessors import AssessorImpl
from gym_jsbsim.tests import stubs as stubs


class TestAssessorImpl(unittest.TestCase):

    @staticmethod
    def get_dummy_state_class() -> Type[NamedTuple]:
        return collections.namedtuple('State', ['test_prop1', 'test_prop2'])

    def get_dummy_state(self, value1: float = 0.0, value2: float = 1.0):
        State = self.get_dummy_state_class()
        return State(value1, value2)

    def test_init_throws_error_on_empty_base_components(self):
        base_components = ()

        with self.assertRaises(ValueError):
            _ = AssessorImpl(base_components)

    def test_init_throws_error_on_empty_base_components_non_empty_shaping_components(self):
        base_components = ()
        shaping_components = (stubs.ConstantRewardComponentStub(),)

        with self.assertRaises(ValueError):
            _ = AssessorImpl(base_components, shaping_components)

    def test_calculate_single_base_component(self):
        component = stubs.ConstantRewardComponentStub()
        assessor = AssessorImpl(base_components=(component,))
        state = self.get_dummy_state()

        reward = assessor.assess(state, state, True)

        expected_shaping_reward = component.get_return_value()
        expected_non_shaping_reward = expected_shaping_reward  # should be same because not shaping
        self.assertAlmostEqual(expected_shaping_reward, reward.reward())
        self.assertAlmostEqual(expected_non_shaping_reward, reward.non_shaping_reward())

    def test_calculate_multiple_base_components(self):
        reward_values = [1, 2, 4]
        components = tuple(stubs.ConstantRewardComponentStub(val) for val in reward_values)
        assessor = AssessorImpl(components)
        state = self.get_dummy_state_class()

        reward = assessor.assess(state, state, True)

        expected_shaping_reward = sum(reward_values) / len(reward_values)
        expected_non_shaping_reward = expected_shaping_reward  # should be same because not shaping
        self.assertAlmostEqual(expected_shaping_reward, reward.reward())
        self.assertAlmostEqual(expected_non_shaping_reward, reward.non_shaping_reward())

    def test_calculate_with_shaping_components(self):
        base_reward_vals = [0, 1]
        shaping_reward_vals = [2, 3]
        base_components = tuple(stubs.ConstantRewardComponentStub(val) for val in base_reward_vals)
        shape_components = tuple(stubs.ConstantRewardComponentStub(val) for val in shaping_reward_vals)
        assessor = AssessorImpl(base_components, shape_components)
        state = self.get_dummy_state()

        reward = assessor.assess(state, state, True)

        expected_shaping_reward = (sum(base_reward_vals + shaping_reward_vals) /
                                   len (base_reward_vals + shaping_reward_vals))
        expected_non_shaping_reward = sum(base_reward_vals) / len(base_reward_vals)
        self.assertAlmostEqual(expected_shaping_reward, reward.reward())
        self.assertAlmostEqual(expected_non_shaping_reward, reward.non_shaping_reward())