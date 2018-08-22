import unittest
import sys
from abc import ABC, abstractmethod
from gym_jsbsim.rewards import Reward, TerminalComponent, StepFractionComponent, ShapingComponent, ComplementComponent
from gym_jsbsim.tests.stubs import FlightTaskStub
from typing import Type


class TestReward(unittest.TestCase):

    def test_init_error_on_empty_base_reward(self):
        empty_base_reward = ()
        shaping_reward = (1, 2)

        with self.assertRaises(ValueError):
            _ = Reward(empty_base_reward, shaping_reward)

    def test_base_reward_reward(self):
        base_reward = (1, 2)
        shaping_reward = ()

        reward = Reward(base_reward, shaping_reward)

        expected_r = sum(base_reward) / len(base_reward)
        self.assertAlmostEqual(expected_r, reward.reward())

    def test_base_reward_non_shaping_reward(self):
        base_reward = (1, 2)
        shaping_reward = ()

        reward = Reward(base_reward, shaping_reward)

        expected_non_shaping_r = sum(base_reward) / len(base_reward)
        self.assertAlmostEqual(expected_non_shaping_r, reward.non_shaping_reward())

    def test_shaping_reward_reward(self):
        base_reward = (1, 2)
        shaping_reward = (3, 4)

        reward = Reward(base_reward, shaping_reward)

        expected_r = sum(base_reward + shaping_reward) / len(base_reward + shaping_reward)
        self.assertAlmostEqual(expected_r, reward.reward())

    def test_shaping_reward_non_shaping_reward(self):
        base_reward = (1, 2)
        shaping_reward = (3, 4)

        reward = Reward(base_reward, shaping_reward)

        expected_non_shaping_r = sum(base_reward) / len(base_reward)
        self.assertAlmostEqual(expected_non_shaping_r, reward.non_shaping_reward())

    def test_base_reward_is_not_shaping(self):
        base_reward = (1, 2)
        shaping_reward = ()

        reward = Reward(base_reward, shaping_reward)

        self.assertFalse(reward.is_shaping())

    def test_shaping_reward_is_shaping(self):
        base_reward = (1, 2)
        shaping_reward = (3, 4)

        reward = Reward(base_reward, shaping_reward)

        self.assertTrue(reward.is_shaping())


class TestTerminalComponent(unittest.TestCase):
    dummy_task = FlightTaskStub()
    default_state_vars = dummy_task.state_variables
    default_property_index = 0
    default_State = dummy_task.State
    default_max_value = 10
    default_name = 'test_component'

    @staticmethod
    def get_component(name=default_name, property_index=default_property_index,
                      state_vars=default_state_vars, max_value=default_max_value):
        return TerminalComponent(name, state_vars[property_index], state_vars, max_value)

    def test_get_name(self):
        name = 'should_be_returned'

        component = self.get_component(name=name)

        self.assertEqual(name, component.get_name())

    def test_calculate_returns_zero_when_non_terminal(self):
        high_value_state = self.dummy_task.get_state(self.default_max_value, self.default_max_value)
        component = self.get_component()

        self.assertAlmostEqual(0.0, component.calculate(high_value_state, high_value_state, False))

    def test_calculate_returns_zero_terminal(self):
        zero_reward_state = self.dummy_task.get_state(0, 0)
        component = self.get_component()

        self.assertAlmostEqual(0.0, component.calculate(zero_reward_state, zero_reward_state, True))

    def test_calculate_returns_zero_bad_state_good_last_state(self):
        zero_reward_state = self.dummy_task.get_state(0, 0)
        high_value_state = self.dummy_task.get_state(self.default_max_value, self.default_max_value)
        component = self.get_component()

        self.assertAlmostEqual(0.0, component.calculate(zero_reward_state, high_value_state, True))

    def test_calculate_returns_half_state_is_halfway_good(self):
        half_to_target_state = self.dummy_task.get_state(self.default_max_value / 2, 0)
        zero_reward_state = self.dummy_task.get_state(0, 0)
        component = self.get_component()

        self.assertAlmostEqual(0.5, component.calculate(half_to_target_state, zero_reward_state, True))

    def test_calculate_returns_one_state_at_max_value(self):
        perfect_reward_state = self.dummy_task.get_state(self.default_max_value, 0)
        zero_reward_state = self.dummy_task.get_state(0, 0)
        component = self.get_component()

        self.assertAlmostEqual(1.0, component.calculate(perfect_reward_state, zero_reward_state, True))

    def test_calculate_returns_half_second_property(self):
        component = self.get_component(property_index=1)
        half_to_target_state = self.dummy_task.get_state(0, self.default_max_value / 2)

        self.assertAlmostEqual(0.5, component.calculate(half_to_target_state, (), True))


class AbstractTestComplementComponent(unittest.TestCase, ABC):
    dummy_task = FlightTaskStub()
    default_state_vars = dummy_task.state_variables
    default_property_index = 0
    default_target_index = 1
    default_target_value = 5.
    default_State = dummy_task.State
    default_scaling_factor = 1
    default_name = 'test_component'
    extra_kwargs = dict()

    def setUp(self):
        self.COT = self.get_class_under_test()

    @abstractmethod
    def get_class_under_test(self) -> Type[ComplementComponent]:
        ...

    def get_default_perfect_state(self):
        """ Returns a state where the controlled property matches the target exactly """
        return self.dummy_task.get_state(self.default_target_value, self.default_target_value)

    def get_default_midling_state(self):
        """ Returns a state where the error (value - target) is equal to the scaling factor """
        value = self.default_target_value + self.default_scaling_factor
        return self.dummy_task.get_state(value, self.default_target_value)

    def get_default_rubbish_state(self):
        """ Returns a state where the error between value and target is very very large """
        value = sys.float_info.max
        return self.dummy_task.get_state(value, self.default_target_value)

    def get_component_target_property(self,
                                      name=default_name,
                                      property_index=default_property_index,
                                      state_vars=default_state_vars,
                                      target_index=default_target_index,
                                      scaling_factor=default_scaling_factor):
        return self.COT(name, state_vars[property_index], state_vars,
                        state_vars[target_index], scaling_factor, **self.extra_kwargs)

    def get_component_target_constant(self,
                                      name=default_name,
                                      property_index=default_property_index,
                                      state_vars=default_state_vars,
                                      target_value=default_target_value,
                                      scaling_factor=default_scaling_factor):
        return self.COT(name, state_vars[property_index], state_vars,
                        target_value, scaling_factor, **self.extra_kwargs)


class TestStepFractionComponent(AbstractTestComplementComponent):
    default_episode_timesteps = 10
    extra_kwargs = dict(episode_timesteps=default_episode_timesteps)

    def get_class_under_test(self):
        return StepFractionComponent

    def test_calculate_perfect_state_returns_one_terminal(self):
        target_value = 2
        perfect_state = self.dummy_task.get_state(target_value, target_value)
        terminal = True

        component = self.get_component_target_constant(target_value=target_value)

        expect_reward = 1.0 / self.default_episode_timesteps
        self.assertAlmostEqual(expect_reward, component.calculate(perfect_state, perfect_state, terminal))

    def test_calculate_perfect_state_returns_one_non_terminal(self):
        target_value = 0.3
        perfect_state = self.dummy_task.get_state(target_value, target_value)
        terminal = False

        component = self.get_component_target_constant(target_value=target_value)

        expect_reward = 1.0 / self.default_episode_timesteps
        self.assertAlmostEqual(expect_reward, component.calculate(perfect_state, perfect_state, terminal))

    def test_calculate_high_error_state_returns_near_zero_terminal(self):
        target_value = 0
        actual_value = sys.float_info.max  # a very big number
        terrible_state = self.dummy_task.get_state(actual_value, actual_value)
        terminal = True

        component = self.get_component_target_constant(target_value=target_value)

        self.assertAlmostEqual(0.0, component.calculate(terrible_state, terrible_state, terminal))

    def test_calculate_high_error_state_returns_near_zero_non_terminal(self):
        target_value = 0
        actual_value = sys.float_info.max  # a very big number
        terrible_state = self.dummy_task.get_state(actual_value, actual_value)
        terminal = False

        component = self.get_component_target_constant(target_value=target_value)

        self.assertAlmostEqual(0.0, component.calculate(terrible_state, terrible_state, terminal))

    def test_calculate_mid_error_state_returns_mid_reward(self):
        target_value = 0
        scaling_factor = 100
        actual_value = scaling_factor
        midling_state = self.dummy_task.get_state(actual_value, actual_value)

        component = self.get_component_target_constant(target_value=target_value, scaling_factor=scaling_factor)

        expected_reward_at_scaling_factor = 0.5 / self.default_episode_timesteps
        self.assertAlmostEqual(expected_reward_at_scaling_factor,
                               component.calculate(midling_state, midling_state, False))

    def test_calculate_not_affected_by_last_state(self):
        perfect_state = self.dummy_task.get_state(self.default_target_value, self.default_target_value)
        terrible_last_state = self.dummy_task.get_state(sys.float_info.max, sys.float_info.max)

        component = self.get_component_target_constant()

        expected_reward = 1.0 / self.default_episode_timesteps
        self.assertAlmostEqual(expected_reward, component.calculate(perfect_state, terrible_last_state, True))

    def test_calculate_with_target_property(self):
        target_value = 1
        middle_irrelevant_value = 0.0
        controlled_value = 1
        state, props = self.dummy_task.get_dummy_state_and_properties([target_value,
                                                                       middle_irrelevant_value,
                                                                       controlled_value])
        # use property index instead of a constant value
        target_index = 0

        component = self.get_component_target_property(property_index=2, target_index=target_index, state_vars=props)

        expected_reward = 1.0 / self.default_episode_timesteps
        self.assertAlmostEqual(expected_reward, component.calculate(state, None, True))


class TestShapingComponent(AbstractTestComplementComponent):

    def get_class_under_test(self):
        return ShapingComponent

    def test_calculate_get_potential_perfect_state_non_terminal(self):
        target_value = 0
        actual_value = target_value
        state = self.dummy_task.get_state(actual_value, target_value)
        terminal = False

        component: ShapingComponent = self.get_component_target_constant(target_value=target_value)

        expected_potential = 1.0
        self.assertAlmostEqual(expected_potential, component.get_potential(state, terminal))

    def test_calculate_get_potential_perfect_state_terminal(self):
        target_value = 0
        actual_value = target_value
        state = self.dummy_task.get_state(actual_value, target_value)
        terminal = True

        component: ShapingComponent = self.get_component_target_constant(target_value=target_value)

        expected_potential = 0.0
        self.assertAlmostEqual(expected_potential, component.get_potential(state, terminal))

    def test_calculate_get_potential_terrible_state(self):
        target_value = 0
        actual_value = sys.float_info.max
        state = self.dummy_task.get_state(actual_value, target_value)
        terminal = False

        component: ShapingComponent = self.get_component_target_constant(target_value=target_value)

        expected_potential = 0.0
        self.assertAlmostEqual(expected_potential, component.get_potential(state, terminal))

    def test_calculate_get_potential_midway_value(self):
        target_value = 0
        actual_value = self.default_scaling_factor  # should produce potential of 0.5
        state = self.dummy_task.get_state(actual_value, target_value)
        terminal = False

        component: ShapingComponent = self.get_component_target_constant(target_value=target_value)

        expected_potential = 0.5
        self.assertAlmostEqual(expected_potential, component.get_potential(state, terminal))

    def test_calculate_same_states_non_terminal(self):
        last_state = self.get_default_perfect_state()
        state = self.get_default_perfect_state()
        terminal = False

        component = self.get_component_target_constant()

        self.assertAlmostEqual(0.0, component.calculate(state, last_state, terminal))

    def test_calculate_same_states_terminal(self):
        last_state = self.get_default_perfect_state()
        state = self.get_default_perfect_state()
        terminal = True

        component = self.get_component_target_constant()

        self.assertAlmostEqual(-1.0, component.calculate(state, last_state, terminal))

    def test_calculate_improved_state_non_terminal(self):
        last_state = self.get_default_midling_state()
        state = self.get_default_perfect_state()
        terminal = False

        component = self.get_component_target_constant()

        self.assertAlmostEqual(0.5, component.calculate(state, last_state, terminal))

    def test_calculate_worsening_state_non_terminal(self):
        last_state = self.get_default_perfect_state()
        state = self.get_default_rubbish_state()
        terminal = False

        component = self.get_component_target_constant()

        self.assertAlmostEqual(-1.0, component.calculate(state, last_state, terminal))
