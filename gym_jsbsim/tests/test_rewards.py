import unittest
import sys
from abc import ABC, abstractmethod
from gym_jsbsim.rewards import Reward, AsymptoticErrorComponent, ErrorComponent, \
    LinearErrorComponent, \
    AngularAsymptoticErrorComponent
import gym_jsbsim.tests.stubs as stubs
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
        self.assertAlmostEqual(expected_r, reward.agent_reward())

    def test_base_reward_non_shaping_reward(self):
        base_reward = (1, 2)
        shaping_reward = ()

        reward = Reward(base_reward, shaping_reward)

        expected_non_shaping_r = sum(base_reward) / len(base_reward)
        self.assertAlmostEqual(expected_non_shaping_r, reward.assessment_reward())

    def test_shaping_reward_reward(self):
        base_reward = (1, 2)
        shaping_reward = (3, 4)

        reward = Reward(base_reward, shaping_reward)

        expected_r = sum(base_reward + shaping_reward) / len(base_reward + shaping_reward)
        self.assertAlmostEqual(expected_r, reward.agent_reward())

    def test_shaping_reward_non_shaping_reward(self):
        base_reward = (1, 2)
        shaping_reward = (3, 4)

        reward = Reward(base_reward, shaping_reward)

        expected_non_shaping_r = sum(base_reward) / len(base_reward)
        self.assertAlmostEqual(expected_non_shaping_r, reward.assessment_reward())

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


class AbstractTestErrorComponent(unittest.TestCase, ABC):
    dummy_task = stubs.FlightTaskStub()
    default_state_vars = dummy_task.state_variables
    default_property_index = 0
    default_target_index = 1
    default_target_value = 5.
    default_State = dummy_task.State
    default_scaling_factor = 1
    default_shaping = True
    default_name = 'test_component'
    extra_kwargs = dict()

    def setUp(self):
        self.COT = self.get_class_under_test()

    @abstractmethod
    def get_class_under_test(self) -> Type[ErrorComponent]:
        ...

    def get_default_perfect_state(self):
        """ Returns a state where the controlled property matches the target exactly """
        return self.dummy_task.get_state(self.default_target_value, self.default_target_value)

    def get_default_middling_state(self):
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
                                      shaping=default_shaping,
                                      scaling_factor=default_scaling_factor):
        return self.COT(name, state_vars[property_index], state_vars,
                        target_value, shaping, scaling_factor, **self.extra_kwargs)


class TestAsymptoticErrorComponent(AbstractTestErrorComponent):
    PERFECT_POTENTIAL = 1.0
    SCALING_FACTOR_POTENTIAL = 0.5
    INF_ERROR_POTENTIAL = 0.0

    def get_class_under_test(self):
        return AsymptoticErrorComponent

    def test_calculate_get_potential_perfect_state_non_terminal(self):
        target_value = 0
        actual_value = target_value
        state = self.dummy_task.get_state(actual_value, target_value)
        terminal = False

        component: AsymptoticErrorComponent = self.get_component_target_constant(
            target_value=target_value)

        expected_potential = self.PERFECT_POTENTIAL
        self.assertAlmostEqual(expected_potential, component.get_potential(state, terminal))

    def test_calculate_get_potential_perfect_state_terminal(self):
        target_value = 0
        actual_value = target_value
        state = self.dummy_task.get_state(actual_value, target_value)
        shaping = True
        terminal = True

        component: AsymptoticErrorComponent = self.get_component_target_constant(
            target_value=target_value, shaping=shaping)

        expected_potential = component.POTENTIAL_BASED_DIFFERENCE_TERMINAL_VALUE
        self.assertAlmostEqual(expected_potential, component.get_potential(state, terminal))

    def test_calculate_get_potential_terrible_state(self):
        target_value = 0
        actual_value = sys.float_info.max
        state = self.dummy_task.get_state(actual_value, target_value)
        terminal = False

        component: AsymptoticErrorComponent = self.get_component_target_constant(
            target_value=target_value)

        expected_potential = self.INF_ERROR_POTENTIAL
        self.assertAlmostEqual(expected_potential, component.get_potential(state, terminal))

    def test_calculate_get_potential_midway_value(self):
        target_value = 0
        actual_value = self.default_scaling_factor  # should produce potential of 0.5
        state = self.dummy_task.get_state(actual_value, target_value)
        terminal = False

        component: AsymptoticErrorComponent = self.get_component_target_constant(
            target_value=target_value)

        expected_potential = self.SCALING_FACTOR_POTENTIAL
        self.assertAlmostEqual(expected_potential, component.get_potential(state, terminal))

    def test_calculate_same_perfect_states_shaping_non_terminal(self):
        last_state = self.get_default_perfect_state()
        state = self.get_default_perfect_state()
        terminal = False

        component = self.get_component_target_constant(shaping=True)

        potential_diff = self.PERFECT_POTENTIAL - self.PERFECT_POTENTIAL
        self.assertAlmostEqual(potential_diff, component.calculate(state, last_state, terminal))

    def test_calculate_same_perfect_states_shaping_terminal(self):
        last_state = self.get_default_perfect_state()
        state = self.get_default_perfect_state()
        terminal = True

        component = self.get_component_target_constant(shaping=True)

        potential_diff = component.POTENTIAL_BASED_DIFFERENCE_TERMINAL_VALUE - self.PERFECT_POTENTIAL
        self.assertAlmostEqual(potential_diff, component.calculate(state, last_state, terminal))

    def test_calculate_improved_state_shaping_non_terminal(self):
        last_state = self.get_default_middling_state()
        state = self.get_default_perfect_state()
        terminal = False

        component = self.get_component_target_constant(shaping=True)

        potential_diff = self.PERFECT_POTENTIAL - self.SCALING_FACTOR_POTENTIAL
        self.assertAlmostEqual(potential_diff, component.calculate(state, last_state, terminal))

    def test_calculate_worsening_state_non_terminal(self):
        last_state = self.get_default_perfect_state()
        state = self.get_default_rubbish_state()
        terminal = False

        component = self.get_component_target_constant()

        potential_diff = self.INF_ERROR_POTENTIAL - self.PERFECT_POTENTIAL
        self.assertAlmostEqual(potential_diff, component.calculate(state, last_state, terminal))


class TestLinearShapingComponent(AbstractTestErrorComponent):
    PERFECT_POTENTIAL = 1.0
    MIDDLE_POTENTIAL = 0.5
    SCALING_FACTOR_POTENTIAL = 0.0

    def get_class_under_test(self):
        return LinearErrorComponent

    def get_default_middling_state(self):
        """
        Returns a state where the error (value - target) is equal to
        half of the scaling factor, corresponding to normalised error of 0.5
        """
        value = self.default_target_value - 0.5 * self.default_scaling_factor
        return self.dummy_task.get_state(value, self.default_target_value)

    def get_default_rubbish_state(self):
        """
        Returns a state where the error between value and target is at maximum,
        i.e. equal to the scaling factor.
        """
        value = self.default_target_value - self.default_scaling_factor
        return self.dummy_task.get_state(value, self.default_target_value)

    def test_calculate_get_potential_perfect_state_non_terminal(self):
        state = self.get_default_perfect_state()
        terminal = False

        component: LinearErrorComponent = self.get_component_target_constant()
        self.assertIsInstance(component, LinearErrorComponent)

        self.assertAlmostEqual(self.PERFECT_POTENTIAL, component.get_potential(state, terminal))

    def test_calculate_get_potential_perfect_state_terminal(self):
        state = self.get_default_perfect_state()
        terminal = True

        component: LinearErrorComponent = self.get_component_target_constant()

        self.assertAlmostEqual(component.POTENTIAL_BASED_DIFFERENCE_TERMINAL_VALUE,
                               component.get_potential(state, terminal))

    def test_calculate_get_potential_rubbish_state(self):
        state = self.get_default_rubbish_state()
        terminal = False

        component: LinearErrorComponent = self.get_component_target_constant()

        self.assertAlmostEqual(self.SCALING_FACTOR_POTENTIAL,
                               component.get_potential(state, terminal))

    def test_calculate_get_potential_midway_value(self):
        state = self.get_default_middling_state()
        terminal = False

        component: LinearErrorComponent = self.get_component_target_constant()

        self.assertAlmostEqual(self.MIDDLE_POTENTIAL, component.get_potential(state, terminal))

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

        component = self.get_component_target_constant(shaping=True)

        pot_difference = component.POTENTIAL_BASED_DIFFERENCE_TERMINAL_VALUE - self.PERFECT_POTENTIAL
        self.assertAlmostEqual(pot_difference, component.calculate(state, last_state, terminal))

    def test_calculate_improved_state_shaping_non_terminal(self):
        last_state = self.get_default_middling_state()
        state = self.get_default_perfect_state()
        terminal = False

        component = self.get_component_target_constant(shaping=True)

        self.assertAlmostEqual(0.5, component.calculate(state, last_state, terminal))

    def test_calculate_worsening_state_non_terminal(self):
        last_state = self.get_default_perfect_state()
        state = self.get_default_rubbish_state()
        terminal = False

        component = self.get_component_target_constant()

        self.assertAlmostEqual(-1.0, component.calculate(state, last_state, terminal))


class TestAngularAsmyptoticErrorComponent(unittest.TestCase):
    dummy_task = stubs.FlightTaskStub()
    default_state_vars = dummy_task.state_variables
    default_property_index = 0
    default_target_index = 1
    default_target_value = 90.
    default_State = dummy_task.State
    default_scaling_factor = 10
    default_name = 'test_component'

    PERFECT_POTENTIAL = 1.0
    SCALING_FACTOR_POTENTIAL = 0.5
    INF_ERROR_POTENTIAL = 0.0

    @staticmethod
    def get_component_constant(name=default_name,
                               property_index=default_property_index,
                               state_vars=default_state_vars,
                               target_value=default_target_value,
                               scaling_factor=default_scaling_factor) -> AngularAsymptoticErrorComponent:
        return AngularAsymptoticErrorComponent(name=name,
                                               prop=state_vars[property_index],
                                               state_variables=state_vars,
                                               target=target_value,
                                               is_potential_based=True,
                                               scaling_factor=scaling_factor)

    def test_get_potential_value_equals_target_terminal(self):
        value = self.default_target_value
        target = self.default_target_value
        state = self.dummy_task.get_state(value, target)
        terminal = True

        component = self.get_component_constant()

        self.assertAlmostEqual(0.0, component.get_potential(state, terminal))

    def test_get_potential_value_equals_target_non_terminal(self):
        value = self.default_target_value
        target = self.default_target_value
        state = self.dummy_task.get_state(value, target)
        terminal = False

        component = self.get_component_constant()

        self.assertAlmostEqual(self.PERFECT_POTENTIAL, component.get_potential(state, terminal))

    def test_get_potential_value_scaling_factor_off_target(self):
        target = self.default_target_value
        value_left = target - self.default_scaling_factor
        value_right = target + self.default_scaling_factor
        state_left_of_target = self.dummy_task.get_state(value_left, target)
        state_right_of_target = self.dummy_task.get_state(value_right, target)
        terminal = False

        component = self.get_component_constant()

        self.assertAlmostEqual(self.SCALING_FACTOR_POTENTIAL,
                               component.get_potential(state_left_of_target, terminal))
        self.assertAlmostEqual(self.SCALING_FACTOR_POTENTIAL,
                               component.get_potential(state_right_of_target, terminal))

    def test_get_potential_equal_across_360_degrees(self):
        target = 0
        value_left_negative = target - 1.5
        value_left_positive = 360 - target - 1.5  # equivalent
        value_right = target + 1.5
        state_left_of_target_negative = self.dummy_task.get_state(value_left_negative, target)
        state_left_of_target_positive = self.dummy_task.get_state(value_left_positive, target)
        state_right_of_target = self.dummy_task.get_state(value_right, target)
        terminal = False

        component = self.get_component_constant(target_value=target)
        potential_left_negative = component.get_potential(state_left_of_target_negative, terminal)
        potential_left_positive = component.get_potential(state_left_of_target_positive, terminal)
        potential_right = component.get_potential(state_right_of_target, terminal)

        self.assertLess(potential_left_negative, self.PERFECT_POTENTIAL)
        self.assertLess(potential_left_positive, self.PERFECT_POTENTIAL)
        self.assertLess(potential_right, self.PERFECT_POTENTIAL)
        self.assertAlmostEqual(potential_left_negative, potential_left_positive)
        self.assertAlmostEqual(potential_left_negative, potential_right)
